"""Build the standalone Kaggle notebook for compact neural Track 2 training."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip() + "\n"))


md("""
# RadarML Lite Track 2 — compact complex-STFT neural classifier

This notebook trains a **neural network from random initialization**, with gradient
descent and backpropagation. Its inference is **one network forward pass** over an
explicit time–frequency representation. It contains **no waveform-template bank,
nearest-neighbor lookup, exhaustive configuration search or previous-prediction input**.

The architecture combines learned local complex-STFT features, a small global
phase-sensitive projection and a lightweight CNN over STFT power. Preserving complex
information addresses a limitation of the original power-only spectrogram CNN.
There are approximately **2.48 million trainable parameters / 10 MB of FP32 weights**.
This is a new experiment: **0.996 is a target, not an established accuracy claim**.
The earlier template-bank score does not transfer to this network.

**Kaggle:** attach the same expanded `radar-ml` dataset **with JSON sidecars**, the
published `manifests` dataset and the released `blind-test/participants` dataset.
Enable a GPU and **Run All**. Data/manifest/participants paths are auto-detected;
manual examples matching your previous notebook are in the next cell. The smaller
compact `recorded_dataset` is also supported. No Internet or extra code upload is needed.

The default mode trains, selects a checkpoint, evaluates all **171,600 validation
examples**, exports the model and inference script, and generates a submission on
**CPU**, including measured STFT time. The training budget is **10.5 hours**, leaving
room within your 12-hour session for validation/export/inference. Actual duration
and performance depend on hardware and I/O; the code cannot enforce Kaggle's session
lifetime. It saves epoch checkpoints so an interrupted run can continue.

The organizer's model-freeze restriction was removed according to your update.
Inference is designed for the newer restriction: **model weights + blind signals only**.
Architecture and measured runtime are disclosed; the organizer determines acceptance.
""")
code('''
from pathlib import Path
import copy
import importlib
import json
import shutil
import subprocess
import sys

MODE = "train_and_submit"       # "train_and_submit", "train", "infer"
RUN_NAME = "track2_neural_v1"    # use a new name for a new architecture/configuration
DATA_FORMAT = "auto"            # "auto", "expanded", "compact"
DATA_ROOT = None
MANIFEST_PATH = None
BLIND_ROOT = None
# Known paths from your previous run, if you prefer to set them explicitly:
# DATA_ROOT = Path("/kaggle/input/datasets/hadimanes/radar-ml/expanded (2)/expanded")
# MANIFEST_PATH = Path("/kaggle/input/datasets/hadimanes/manifests")
# BLIND_ROOT = Path("/kaggle/input/datasets/hadimanes/blind-test/participants")

WORK_ROOT = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path.cwd()
RUN_DIR = WORK_ROOT / RUN_NAME
RESUME_FROM = None              # extracted *_resume.zip folder from a saved session
MODEL_PATH = None               # only for MODE="infer": exported neural model.pt

TRAIN_BATCH_SIZE = 256          # reduce to 128 if GPU memory is insufficient
WORKERS = 2                    # use 0 if notebook multiprocessing causes trouble
TRAINING_BUDGET_HOURS = 10.5
REMOVE_TRAINING_CACHE_AFTER_EXPORT = True

INFERENCE_DEVICE = "cpu"        # CPU by default for the organizer's timing requirement
INFERENCE_BATCH_SIZE = 128
INFERENCE_CPU_THREADS = 4
BENCHMARK_SAMPLES = 1000         # real one-signal-at-a-time latency, separate from batching
SUBMISSION_LABEL_COLUMN = "label"  # matches your latest submission; use "predicted_label" if required
PREDICTIONS_PATH = WORK_ROOT / "submission_track2_neural.csv"

assert MODE in {"train_and_submit", "train", "infer"}
assert Path(RUN_NAME).name == RUN_NAME and RUN_NAME not in {"", ".", ".."}
TRAIN_MODEL = MODE in {"train", "train_and_submit"}
GENERATE_SUBMISSION = MODE in {"train_and_submit", "infer"}
if TRAIN_MODEL and RESUME_FROM is not None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("resume.pt", "best.pt", "run_config.json", "history.json", "best_selection_validation.json"):
        source = Path(RESUME_FROM) / name
        destination = RUN_DIR / name
        if source.exists() and not destination.exists():
            shutil.copy2(source, destination)
if not TRAIN_MODEL:
    assert MODEL_PATH is not None, "Set MODEL_PATH to the new neural model.pt for inference-only mode"
print("Mode:", MODE, "| run:", RUN_DIR)
''')
md("""
## Embedded code

All source is included. The first module retains the original validated file-identity
and split helpers, including FSK filename-collision recovery. It is used for development
only. The second module is the **standalone inference script** and model definition.
The third module handles training and is excluded from the inference export.

The exported inference script does not import the development modules, open training
files, require the training cache, or read an earlier submission. Its inputs are just
`model.pt` and the anonymous blind package (signals, index and submission template).
""")
for name in ("radarml_track2_runtime.py", "radarml_track2_neural_infer.py", "radarml_track2_neural_train.py"):
    code(f"%%writefile {name}\n" + (ROOT / "track2" / name).read_text())

md("""
## Training setup and representation

The input is converted to a **256-frequency-bin × 66-time-frame complex STFT**.
Windows are 256-sample rectangular blocks, hop 256, with right zero padding and no
centering. This explicit time–frequency transform preserves every input sample.
Both real and imaginary parts feed learned phase-sensitive neural features. A small
separable-convolution branch also learns modulation shapes from log STFT power.
No filename, waveform-configuration identity, pulse ID or requested-SNR label enters
the network. Per-pulse RMS, noise and clipping measurements come from the signal.

All **52,800 training source pulses** are eligible. The trainer caches their original
recorded waveforms, or their +30 dB variants if using expanded inputs: **3.55 GB** of
training-only int16 data. It then generates **independent noisy views at all 13 SNRs**
during training, following the supplied power windows and int16 clipping convention.
It does **not** fit only clean pulses. It also does not read every stored noisy variant
from the 57 GB expansion; the independent views are training-pulse augmentations.
This avoids repeatedly reading 57 GB and permits fresh noise on repeated visits.

Class-balanced sampling draws **105,600 examples per epoch**, with replacement.
Every source pulse is eligible; an epoch does not promise to visit every pulse.
After two warm-up epochs (+30 through −15 dB), all 13 SNRs are sampled, with twice
as much sampling weight at −20/−25/−30 dB to focus on the earlier model's weak region.
Paired clean/noisy training combines cross-entropy and feature consistency.

AdamW, mixed-precision training, gradient clipping, warm-up/cosine learning rates and
EMA weights are used. The selected export is **one network**, chosen between raw and
EMA checkpoints using validation. There is no inference ensemble or template teacher.
""")
code('''
import numpy as np
import torch
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path.cwd()))
import radarml_track2_runtime as helpers
import radarml_track2_neural_infer as neural
import radarml_track2_neural_train as training
helpers = importlib.reload(helpers)
neural = importlib.reload(neural)
training = importlib.reload(training)
torch.set_num_threads(INFERENCE_CPU_THREADS)

CONFIG = copy.deepcopy(training.DEFAULT_CONFIG)
CONFIG.update(batch_size=TRAIN_BATCH_SIZE, workers=WORKERS, max_run_hours=TRAINING_BUDGET_HOURS)
# Make experiment changes before starting. Keep settings unchanged when resuming.
# CONFIG["epochs"] = 80
# CONFIG["model_kwargs"]["projection_rank"] = 64
# CONFIG["samples_per_epoch"] = 105600

if TRAIN_MODEL:
    DATA_ROOT = training.find_data_root(DATA_ROOT, DATA_FORMAT)
    manifest = helpers.find_manifest(DATA_ROOT, MANIFEST_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Development data:", DATA_ROOT)
    print("Published manifest:", manifest)
    print(json.dumps(neural.hardware(device), indent=2))
    print(json.dumps(CONFIG, indent=2))
    if device.type != "cuda":
        raise RuntimeError("Enable a Kaggle GPU before full training. CPU inference is supported after training.")
else:
    print("Inference-only: development data is not loaded.")
''')
md("""
## Train, validate and export

The first four validation pulse indices per recording, across all 13 SNRs, select
raw/EMA checkpoints using **mean balanced accuracy per requested SNR**, with the
lowest-three-SNR tie-break. No validation pulse fits weights. The selected checkpoint
then receives full validation on **171,600 examples**, plus an audit on **137,280**
examples excluding all selection source pulses. Do not tune using the audit.

Checkpoint files are saved after each epoch. If interrupted during an epoch, rerunning
with the same configuration resumes from the previous saved epoch. The history shows
how many examples were actually drawn and the requested training SNR counts.
The time-budget check can stop a partial epoch; that partial epoch is recorded and
evaluated before export. Early stopping can end the run before 80 epochs.

A fresh run takes time to verify FSK upload identities and build its training cache.
Once caching is complete, it prints epoch/batch loss and validation progress.
The exported `model.pt` contains **neural weights and architecture settings only**.
No raw/averaged training waveform, cached example or old submission is exported.
""")
code('''
if TRAIN_MODEL:
    try:
        full_report = training.train(DATA_ROOT, manifest, RUN_DIR, CONFIG, kind=DATA_FORMAT)
    except KeyboardInterrupt:
        print("Interrupted. The preceding completed epoch is saved in resume.pt.")
        print("Save the notebook outputs before ending the session; resume requires best.pt and run_config.json too.")
        raise
    model_zip, resume_zip = training.package(RUN_DIR)
    MODEL_PATH = RUN_DIR / "export/model.pt"
    print("MODEL:", MODEL_PATH, f"({MODEL_PATH.stat().st_size / 1e6:.2f} MB)")
    print("MODEL + INFERENCE CODE:", model_zip)
    print("RESUME CHECKPOINTS (not for submission):", resume_zip)
    if REMOVE_TRAINING_CACHE_AFTER_EXPORT and (RUN_DIR / "training_cache").exists():
        shutil.rmtree(RUN_DIR / "training_cache")
        print("Training cache removed after export; it is not part of any submission artifact.")
    from IPython.display import FileLink, display
    display(FileLink(str(Path(model_zip).relative_to(WORK_ROOT))))
    display(FileLink(str(Path(resume_zip).relative_to(WORK_ROOT))))
    display(FileLink(str(MODEL_PATH.relative_to(WORK_ROOT))))
    display(FileLink(str((RUN_DIR / "export/radarml_track2_neural_infer.py").relative_to(WORK_ROOT))))
''')
md("""
## Review measured accuracy

These are this neural network's **actual measured validation scores**, not the earlier
matched-filter model's scores. Reaching 0.996 cannot be assumed. If the result is
lower, preserve the reported score; never substitute the template bank's result.
Plots are saved outside the model file. Blind labels are not available to this notebook.
""")
code('''
if TRAIN_MODEL:
    export = RUN_DIR / "export"
    full = json.loads((export / "validation_report.json").read_text())
    audit = json.loads((export / "audit_validation.json").read_text())
    summary = json.loads((export / "training_summary.json").read_text())
    assert full["samples"] == 171600 and audit["samples"] == 137280
    print("NEURAL MODEL FULL VALIDATION:", full["mean_balanced_accuracy"])
    print("SEPARATE AUDIT:", audit["mean_balanced_accuracy"])
    print("Selected epoch:", full["selected_epoch"], "variant:", full["selected_variant"])
    print("Model parameters:", summary["parameters"], "model MB:", summary["model_bytes"] / 1e6)
    print("Requested SNR | balanced accuracy")
    for row in full["per_snr"]:
        print(f"{row['requested_snr_db']:+3d} | {row['balanced_accuracy']:.6f}")
    plots = RUN_DIR / "plots"
    plots.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4))
    for label, report in [("Full validation", full), ("Audit excluding selection", audit)]:
        ax.plot([r["requested_snr_db"] for r in report["per_snr"]],
                [r["balanced_accuracy"] for r in report["per_snr"]], "o-", label=label)
    ax.set(xlabel="Requested SNR (dB)", ylabel="Balanced accuracy", ylim=(0, 1.01))
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
    fig.savefig(plots / "validation_snr.png", dpi=160); plt.show()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, snr in zip(axes, (-20, -25, -30)):
        row = next(r for r in full["per_snr"] if r["requested_snr_db"] == snr)
        cm = np.array(row["confusion_matrix"], dtype=float)
        ax.imshow(cm / cm.sum(1, keepdims=True), vmin=0, vmax=1, cmap="Blues")
        ax.set(title=f"Requested {snr:+d} dB", xlabel="Predicted", ylabel="True")
        ax.set_xticks(range(7), neural.CLASSES, rotation=45, ha="right")
        ax.set_yticks(range(7), neural.CLASSES)
    fig.tight_layout(); fig.savefig(plots / "low_snr_confusions.png", dpi=160); plt.show()
''')
md("""
## Standalone inference-script cell — generate submission and measure CPU latency

This is the same script that goes to the organizer. It reads **only `model.pt` and
the blind package**. The model includes its architecture settings, so no training
configuration, training cache, validation report, previous predictions, or optimizer
checkpoint is required. It emits seven class logits in one learned forward pass.

The default CPU run measures **1,000 individual signals**, then regenerates predictions
for all **171,600** samples. Single-signal latency includes shard access, int16 I/Q
conversion, the entire STFT, neural inference and class selection. It reports mean,
median, p95, p99 and maximum. Batched elapsed time per sample is reported separately
as throughput. One-time model loading is excluded and reported separately.
A 10 ms reference is shown for context; the organizer's exact timing protocol governs.

The output columns default to `sample_id,label`, matching your latest saved CSV.
Set `SUBMISSION_LABEL_COLUMN="predicted_label"` if the organizer requests that header.
No previous submission is opened to infer the column name or make predictions.

**Standalone command**, using the exported script and neural weights:

```bash
python radarml_track2_neural_infer.py --model model.pt \\
  --participants /path/to/participants --output submission.csv \\
  --device cpu --threads 4 --label-column label
```
""")
code('''
if GENERATE_SUBMISSION:
    if TRAIN_MODEL:
        model_path = RUN_DIR / "export/model.pt"
        script_path = RUN_DIR / "export/radarml_track2_neural_infer.py"
    else:
        model_path = Path(MODEL_PATH).expanduser().resolve()
        script_path = model_path.parent / "radarml_track2_neural_infer.py"
    if not script_path.is_file():
        raise FileNotFoundError("Keep the exported inference script beside model.pt")
    participants = neural.find_participants(BLIND_ROOT)
    command = [sys.executable, str(script_path), "--model", str(model_path),
               "--participants", str(participants), "--output", str(PREDICTIONS_PATH),
               "--device", INFERENCE_DEVICE, "--batch-size", str(INFERENCE_BATCH_SIZE),
               "--threads", str(INFERENCE_CPU_THREADS), "--workers", "0",
               "--benchmark-samples", str(BENCHMARK_SAMPLES),
               "--label-column", SUBMISSION_LABEL_COLUMN]
    subprocess.run(command, check=True)
    report_path = PREDICTIONS_PATH.with_name(PREDICTIONS_PATH.name + ".inference.json")
    print(json.loads(report_path.read_text()))
    from IPython.display import FileLink, display
    display(FileLink(str(PREDICTIONS_PATH.relative_to(WORK_ROOT))))
    display(FileLink(str(report_path.relative_to(WORK_ROOT))))
else:
    print("Submission generation skipped because MODE='train'.")
''')
md("""
## Resume and report honestly

The `*_resume.zip` contains training weights/optimizer checkpoints, not cached source
pulses. Extract it as a private Kaggle dataset, set `RESUME_FROM`, retain the exact
configuration and Run All. The training-only cache is rebuilt from the supplied data.
Kaggle must retain your outputs for a later session; an abruptly killed session can
lose files that were never saved. If you need a new experiment, change `RUN_NAME`.

For final submission, use the **exported neural model and inference script**, not
`resume.pt`, the old `bank.pt`, the original CNN's script, or the training cache.
The model ZIP includes measured validation evidence and method/hardware details.

The implementation uses standard APIs documented by
[PyTorch STFT](https://docs.pytorch.org/docs/stable/generated/torch.functional.stft.html),
[automatic mixed precision](https://docs.pytorch.org/docs/stable/notes/amp_examples.html),
and [EMA model averaging](https://docs.pytorch.org/docs/stable/generated/torch.optim.swa_utils.AveragedModel.html).
Data attribution: Ritchie, White and Hosford (2025),
[RadarML](https://doi.org/10.5522/04/30752767.v1). Organizer helpers are GPL-3.0.
AI coding assistance: OpenAI Codex. No pretrained weights or external training data.
""")
notebook = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
    "accelerator": "GPU", "kaggle": {"accelerator": "gpu", "isInternetEnabled": False},
})
nbf.validate(notebook)
path = ROOT / "status/radar-track2-neural.ipynb"
nbf.write(notebook, path)
print(path)
