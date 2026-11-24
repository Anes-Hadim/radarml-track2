"""Build the self-contained Kaggle notebook from the reviewable runtime source."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip() + "\n"))


md("""
# RadarML Lite — Track 2 / B: time–frequency classification

Train a **from-scratch residual CNN on two-sided STFT power views** of the supplied
I/Q pulses. Predict exactly `barker`, `cw`, `fmcw`, `fsk`, `hyp`, `nlfm`, `quad`.
This notebook reads your **already expanded Kaggle dataset**; it never expands the
recorded dataset or downloads signals, pretrained weights, or external training data.
Your original `synapseon.ipynb` is preserved.

**Kaggle:** attach your expanded dataset **with its JSON sidecars**, and the original
`split_manifest.csv` (the existing manifests dataset in your first notebook is fine).
Enable a GPU. No internet or extra repository files are needed: the runtime code is
embedded below. Check the paths in the next cell. Use **Run All**, then download the
ZIP from `/kaggle/working`. The final cell is gated for blind inference after release.

The expanded development dataset is **training + validation**, not the blind test.
There are 52,800 training and 13,200 validation source pulses; all 13 noisy versions
retain their source pulse's published assignment. No validation pulse enters training,
including the final run. Full validation contains 171,600 examples.

**What changed from your first notebook:** all 80 training pulses and all 13 SNRs are
eligible by default; class-balanced sampling controls each epoch's work; two STFT
resolutions and residual blocks replace the small CNN; checkpoint selection uses
validation only; the selected checkpoint receives full validation; the freeze bundle
includes executable code, weights, settings, dependency versions and SHA256 hashes.
No validation score or winning-performance claim is made until you actually train.
""")
code("""
from pathlib import Path
import copy
import json
import sys
import time

MODE = "train"                 # "train" or "infer"
PROFILE = "full"               # "full" or "pilot"; pilot still ends with full validation
RUN_NAME = "track2_full_seed42" # change for every new training experiment

# Paths carried over from your first notebook; edit if Kaggle mounts them differently.
DATA_ROOT = Path("/kaggle/input/datasets/hadimanes/radar-ml/expanded")
MANIFEST_PATH = Path("/kaggle/input/datasets/yacinesm/manifests")
WORK_ROOT = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path.cwd()
OUTPUT_DIR = WORK_ROOT / "radarml_track2" / RUN_NAME

# After freezing, upload/extract the frozen ZIP as a separate private Kaggle dataset.
# FROZEN_BUNDLE is the extracted directory containing model.pt and frozen.sha256.json.
FROZEN_BUNDLE = None
BLIND_ROOT = None              # organizer's extracted blind package, after release only
PREDICTIONS_PATH = WORK_ROOT / "predictions.csv"

# Storage/runtime choices; they do not feed into the model.
INFERENCE_BATCH_SIZE = 64
INFERENCE_WORKERS = 2

assert MODE in {"train", "infer"} and PROFILE in {"full", "pilot"}
assert MODE != "train" or BLIND_ROOT is None, "Keep blind data out of development."
assert MODE != "infer" or (FROZEN_BUNDLE is not None and BLIND_ROOT is not None), "Set both frozen and blind paths."
assert Path(RUN_NAME).name == RUN_NAME and RUN_NAME not in {"", ".", ".."}
print("Mode:", MODE, "| Output:", OUTPUT_DIR)
""")
md("""
## Rules and evidence checked

The workflow follows the supplied **SynapsON Radar ML Lite.pdf**, pp. 1–2; **Radar ML
Briefing.pdf**, slides 36, 44–50 and 52–54; `hackathon/README.md`, `EVENT_RULES.md`,
`participant_loader.py`, `score_submission.py`, and `docs/participant_notes.md`.
The introduction paper describes the larger original RadarML archive; its three
receivers, 1,000 pulses and 3 dB noise steps **do not replace the Lite specifications**.

Track 2 is Track B, the **time–frequency** track. The model gets signal values only.
Filenames determine development labels and storage lookup; requested SNR groups
evaluation results. Neither becomes a model feature. Barker and FSK subtypes each
map to one family. Standard libraries and AI assistance are allowed with attribution;
external learned weights and extra labeled data are excluded.

**Deadline discrepancy:** briefing slide 51 proposes **10–17 October 2026, deadline
17 October at noon**, without a timezone. The repository gives a relative Day 0–6
timetable (freeze Day 4 at 17:00, blind release Day 5 at 09:00, final Day 6 at 17:00,
Europe/London), and says organizers announce the calendar mapping. Follow their
latest event announcement for the actual freeze/release/submission deadlines and
timezone; these materials alone do not resolve the discrepancy. Do not access the
blind signals before the official release, or tune any part of the pipeline afterward.

**Metric:** mean of seven-class balanced accuracy at each of the 13 requested SNRs.
The lowest-three-SNR mean at −20, −25 and −30 dB is also saved for the tie-break.
All plot axes say *requested* SNR; clipping and unattainable targets affect achieved SNR.
""")
md("""
## Embedded runtime — file identity and published split

The next cells write a local Python module. They are part of this notebook, so you
do not need to upload a separate code file. Run all of them before importing it.
FSK filenames can collide when Kaggle sanitizes `#`: the lookup retained from your
notebook uses the JSON's original source identity and array SHA256 to recover the
pairing. Uploaded suffixes such as `(19)` are not pulse or subtype identifiers.
Hashing the FSK files reads a substantial part of the dataset once; progress is printed.
""")
source = (ROOT / "track2/radarml_track2_runtime.py").read_text()
blocks = source.split("# %% ")
code("%%writefile radarml_track2_runtime.py\n" + blocks[0] + "# " + blocks[1])
for block in blocks[2:]:
    title = block.splitlines()[0]
    md("### " + title)
    code("%%writefile -a radarml_track2_runtime.py\n\n# " + block)

md("""
## Configure the run

`full` makes all **686,400 training variants eligible**. Each epoch draws 105,600
examples with replacement using inverse class frequencies; this balances classes
and SNRs in expectation and does not promise to visit every variant in one epoch.
Twenty epochs is a starting budget, with validation early stopping. Increase
`samples_per_epoch` or `epochs` only during development if your GPU/time budget allows.

Each epoch selects on the first four published validation pulse indices per recording
at **all 13 SNRs** (34,320 examples). That intermediate score is clearly labeled as a
selection-subset score. After selection, the saved best model is evaluated on **all
20 validation pulses and all 13 SNRs**. Set `selection_pulses_per_file=20` if you want
full validation at every epoch. The final full report is always produced.

`pilot` uses eight training pulses per recording, two short epochs and one selection
validation pulse per recording, followed by full validation. Use a fresh `RUN_NAME`
when moving from pilot to full. The notebook refuses to overwrite a frozen run.
""")
code("""
import importlib
sys.path.insert(0, str(Path.cwd()))
import radarml_track2_runtime as rt
rt = importlib.reload(rt)

import numpy as np
import torch
import matplotlib.pyplot as plt

CONFIG = copy.deepcopy(rt.DEFAULT_CONFIG)
if PROFILE == "pilot":
    CONFIG.update(train_pulses_per_file=8, epochs=2, samples_per_epoch=6600,
                  selection_pulses_per_file=1, patience=2)

# Make development-only changes here, before you freeze.
# CONFIG["batch_size"] = 32   # lower if GPU memory is insufficient
# CONFIG["workers"] = 0      # useful if notebook multiprocessing causes trouble
# CONFIG["epochs"] = 30
# CONFIG["selection_pulses_per_file"] = 20

if MODE == "train":
    if not torch.cuda.is_available():
        print("CPU available; for the full run, enable a Kaggle GPU accelerator.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(json.dumps(rt.hardware_info(device), indent=2))
    print(json.dumps(CONFIG, indent=2))
else:
    print("Inference will execute the hashed runtime from FROZEN_BUNDLE.")
""")
md("""
## Validate data and view training-only spectrograms

This checks the original manifest checksum, 660 recordings, 80/20 assignments,
source pulse indices, one ADC0 channel, 120 MS/s, int16 array layout, and each JSON's
SNR ordering. It uses memory maps and never loads 57.66 GB into RAM.

The representation uses the entire 140 µs record. It creates centered, zero-padded,
two-sided Hann STFTs at 256 and 1,024 samples, with a 64-sample hop. Power is scaled
by a **per-pulse** noise estimate from STFT frames fully within the fixed 0–20 µs
acquisition window, followed by `log1p` and averaging to 256 × 256. There are no
learned preprocessing statistics to fit on validation or test data. The two views
trade temporal detail against spectral detail; pooling reduces the output resolution.
""")
code("""
if MODE == "train":
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(f"Choose a new RUN_NAME; this run already has artifacts: {OUTPUT_DIR}")
    rt.seed_everything(CONFIG["seed"])
    manifest_path = rt.find_manifest(DATA_ROOT, MANIFEST_PATH)
    preparation_start = time.perf_counter()
    manifest_rows, file_info = rt.prepare_development(DATA_ROOT, manifest_path)
    preparation_seconds = time.perf_counter() - preparation_start
    train_preview_examples = rt.build_examples(manifest_rows, file_info, "train", 1, [0])
    preview_data = rt.DevelopmentDataset(train_preview_examples, file_info)
    preview_indices = [next(i for i, e in enumerate(train_preview_examples) if e[3] == c) for c in range(7)]
    signals = torch.stack([preview_data[i][0] for i in preview_indices])
    representation = rt.TwoResolutionSTFT(**{k: CONFIG["model_kwargs"][k]
                                            for k in ("n_ffts", "hop_length", "image_size")})
    with torch.inference_mode():
        images = representation(signals).numpy()
    fig, axes = plt.subplots(2, 7, figsize=(18, 6), sharex=True, sharey=True)
    # Same colour scale for all examples/resolutions; no per-image autoscaling.
    for resolution, n_fft in enumerate(CONFIG["model_kwargs"]["n_ffts"]):
        for cls, label in enumerate(rt.CLASSES):
            axes[resolution, cls].imshow(images[cls, resolution], origin="lower", aspect="auto",
                                         extent=[0, 140, -60, 60], vmin=0, vmax=8, cmap="magma")
            axes[resolution, cls].set_title(f"{label}, FFT {n_fft}")
            axes[resolution, cls].set_xlabel("Time (µs)")
        axes[resolution, 0].set_ylabel("Baseband frequency (MHz)")
    fig.suptitle("Published training pulses at requested 0 dB — log1p power / noise floor")
    fig.tight_layout()
    plt.show()
""")
md("""
## Train, select, evaluate and freeze

The CNN has 1,463,655 parameters at the default settings and uses GroupNorm,
residual blocks and dropout. AdamW, cosine learning-rate decay, small label smoothing
and gradient clipping train it from random initialization. Mixed precision is used
for CUDA CNN training; STFT computation and validation/inference stay in float32.
The model receives only the I/Q tensor. Labels supervise the loss; requested SNR
is passed only to metric accumulation. No blind input is read in training mode.

The best **saved checkpoint** receives the final evaluation. A full run can take
hours; batch progress is printed. The ZIP below is the concrete freeze submission:
weights, preprocessing/config, code, dependencies, hashes, validation evidence,
method summary and inference command. Retain it before blind-test release.
""")
code("""
if MODE == "train":
    report = rt.train_and_freeze(
        DATA_ROOT, manifest_path, OUTPUT_DIR, CONFIG,
        prepared=(manifest_rows, file_info, preparation_seconds))
    import shutil
    freeze_zip = shutil.make_archive(str(OUTPUT_DIR) + "_frozen", "zip", root_dir=OUTPUT_DIR)
    print("DOWNLOAD THIS FREEZE ZIP:", freeze_zip)
    print("ZIP SHA256:", rt.lookup_sha256(Path(freeze_zip)))
    from IPython.display import FileLink, display
    display(FileLink(freeze_zip))
""")
md("""
## Full published validation evidence

These results come from the saved selected checkpoint on **171,600 validation
examples**, never from the blind test. `validation_report.json` includes every raw
confusion matrix (rows true, columns predicted), per-class recall and per-SNR balanced
accuracy. It also reports measured inference time including data loading and STFTs.
The selection-subset scores and training history are separate artifacts.
""")
code("""
if MODE == "train":
    report = json.loads((OUTPUT_DIR / "validation_report.json").read_text())
    assert report["full_published_validation"] and report["samples"] == 171600
    print(f"Full validation mean balanced accuracy: {report['mean_balanced_accuracy']:.6f}")
    print(f"Lowest-three-SNR mean: {report['lowest_three_mean_balanced_accuracy']:.6f}")
    print(f"Inference including STFT: {report['seconds_per_pulse']*1000:.3f} ms/pulse")
    print("Requested SNR (dB) | Balanced accuracy")
    for row in report["per_snr"]:
        print(f"{row['requested_snr_db']:+3d} | {row['balanced_accuracy']:.6f}")
    plot_dir = WORK_ROOT / "radarml_track2_plots" / RUN_NAME
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot([p["requested_snr_db"] for p in report["per_snr"]],
            [p["balanced_accuracy"] for p in report["per_snr"]], "o-")
    ax.axhline(1/7, color="gray", linestyle="--", label="Constant-class reference: 1/7")
    ax.set(xlabel="Requested SNR (dB)", ylabel="Seven-class balanced accuracy", ylim=(0, 1))
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "validation_snr_curve.png", dpi=160)
    plt.show()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, snr in zip(axes, (30, 0, -30)):
        entry = next(p for p in report["per_snr"] if p["requested_snr_db"] == snr)
        matrix = np.array(entry["confusion_matrix"], dtype=float)
        normalized = matrix / matrix.sum(axis=1, keepdims=True)
        ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
        ax.set(title=f"Requested {snr:+d} dB", xlabel="Predicted", ylabel="True")
        ax.set_xticks(range(7), rt.CLASSES, rotation=45, ha="right")
        ax.set_yticks(range(7), rt.CLASSES)
    fig.tight_layout()
    fig.savefig(plot_dir / "validation_confusions.png", dpi=160)
    plt.show()
    print((OUTPUT_DIR / "method_summary.txt").read_text())
    print("Plots saved separately:", plot_dir, "| frozen bundle remains unchanged.")
""")
md("""
## Blind inference — only after freeze and official release

1. Download the frozen ZIP above and keep its original hash record. Upload the
   extracted bundle to a **private** Kaggle dataset; do not edit its hashed files.
2. After the organizer releases the blind package, verify the package's announced
   checksum before extraction/upload. The repository currently records ZIP SHA256
   `7d13639eea4aa936ad43f815d97b9d0a60cbaaf8fade6a472306c75a70223eab`; use the
   organizer's announcement if they release a different version. Do not publish
   blind signals or send them to external analysis/prediction services.
3. Change only `MODE="infer"`, `FROZEN_BUNDLE` and `BLIND_ROOT` in the first settings
   cell; the paths point to the frozen directory and extracted blind package. Run All.
   Development data and manifests are not needed in inference mode. Batch size and
   workers are storage/runtime controls; the frozen model runs in eval mode, FP32.
4. The last cell verifies **all frozen artifact hashes before executing the frozen
   code**. It loads anonymous `(N,33600)` shards, checks all 171,600 IDs against the
   organizer template, and writes exactly `sample_id,predicted_label`, one row per ID.
   `predictions.csv.inference.json` records timing, hardware and hashes. It cannot
   calculate blind accuracy because labels are organizer-only.

The standalone command is inside the ZIP as `inference_command.txt`:

```bash
python radarml_track2_runtime.py infer --bundle . \\
  --blind-root /path/to/blind_test_participants --output /path/to/predictions.csv
```

Submit that prediction CSV, the frozen pipeline and its identifier/hashes, validation
evidence, method summary and blind inference report to the organizer's announced
destination. Identify the team/track in the submission message, keeping the prediction
header exact. No retraining, test-driven model selection, calibration, threshold changes
or preprocessing fitting is permitted after freeze. Execution-only repairs require
organizer agreement and a recorded diff.
""")
code("""
if MODE == "infer":
    import hashlib
    import subprocess
    frozen = Path(FROZEN_BUNDLE).resolve()
    hashes = json.loads((frozen / "frozen.sha256.json").read_text())
    required = {"model.pt", "config.json", "radarml_track2_runtime.py", "requirements.txt"}
    assert required.issubset(hashes), "Incomplete frozen bundle"
    for name, expected in hashes.items():
        if Path(name).name != name:
            raise ValueError("Invalid artifact path in hash manifest")
        digest = hashlib.sha256()
        with (frozen / name).open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError(f"Frozen artifact changed: {name}")
    command = [sys.executable, str(frozen / "radarml_track2_runtime.py"), "infer",
               "--bundle", str(frozen), "--blind-root", str(Path(BLIND_ROOT)),
               "--output", str(PREDICTIONS_PATH),
               "--batch-size", str(INFERENCE_BATCH_SIZE), "--workers", str(INFERENCE_WORKERS)]
    subprocess.run(command, check=True)
    print(json.loads(PREDICTIONS_PATH.with_name(PREDICTIONS_PATH.name + ".inference.json").read_text()))
    from IPython.display import FileLink, display
    display(FileLink(str(PREDICTIONS_PATH)))
else:
    print("Blind inference skipped. Keep BLIND_ROOT unset during development.")
""")
nb = nbf.v4.new_notebook(cells=cells)
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python", "version": "3.11"},
               "radarml": {"track": "B", "track_number": 2, "self_contained": True}}
nbf.validate(nb)
destination = ROOT / "radarml_track2_kaggle.ipynb"
nbf.write(nb, destination)
print(destination, len(cells), "cells")
