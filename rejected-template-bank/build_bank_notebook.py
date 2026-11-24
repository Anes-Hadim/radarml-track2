"""Build the self-contained Kaggle notebook for the Fourier template classifier."""
from pathlib import Path
import json
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip() + "\n"))


md("""
# RadarML Lite — Track 2: coherent frequency-domain learning

**Goal:** improve the old spectrogram CNN's low-SNR performance by keeping the complex
Fourier phase information. This classifier learns one mean waveform per supplied
training configuration, then computes frequency-domain matched-filter likelihoods.
It is a fitted statistical model, rather than an epoch-based neural network.
There are **no pretrained weights, outside waveforms or blind-data inputs in training**.

Training-only inspection found repeatable acquisition phase and timing in this dataset.
That makes coherent integration a promising alternative to a larger power-spectrogram
CNN. The notebook evaluates eight scoring alternatives using validation only, selects
one, and evaluates the selected pipeline on all **171,600** validation examples.
An additional audit report excludes every source pulse used for selection.
**No leaderboard score, including 0.995, is guaranteed.**

**Kaggle setup:** attach your existing expanded `radar-ml` dataset and the published
`manifests` dataset. Keep the expanded JSON sidecars. Alternatively attach the smaller
`recorded_dataset` with its sidecars; noise variants are generated on demand using the
supplied expander's algorithm, without writing a 57 GB expansion. Enable a GPU;
one GPU is used. Internet is unnecessary. Also attach the released **participants**
dataset. The default mode trains, freezes, then generates your submission in the
same Kaggle run. Check the paths below and **Run All**.

The run is designed to fit within your **12-hour GPU budget**, and will often finish
much sooner than CNN training. It fits finite template parameters directly, so adding
epochs would not improve this estimator. Larger compute alone does not ensure better
accuracy. Training caches, intermediate validation reports and final bundles are saved.

**Event eligibility:** this is a new experiment, separate from your original frozen
model. The repository requires organizer approval to replace a model after the freeze.
Use this notebook as development work only within the applicable event permissions.
""")
reference_root = ROOT / "status/track2_bank_development/frozen"
if (reference_root / "validation_report.json").is_file():
    measured = json.loads((reference_root / "validation_report.json").read_text())
    audit = json.loads((reference_root / "audit_validation.json").read_text())
    low = measured["per_snr"][-1]["balanced_accuracy"]
    md(f"""
## Measured local reference — 15 October 2026

This exact implementation was fitted on all **52,800 original recorded training
pulses** and evaluated using the published validation identities and the supplied
expander's deterministic noise algorithm. It did not read the participant package.

| Evaluation | Examples | Mean balanced accuracy over all 13 SNRs |
| --- | ---: | ---: |
| Full published validation | {measured['samples']:,} | **{measured['mean_balanced_accuracy']:.8f}** |
| Audit excluding selection source pulses | {audit['samples']:,} | **{audit['mean_balanced_accuracy']:.8f}** |

Selected rule: `{measured['candidate']}`. Full-validation balanced accuracy at
**−30 dB: {low:.4%}**. The old notebook's full-validation mean was 0.89787946.
These are validation results, not blind leaderboard scores. The audit excludes
selection pulses but still uses the same known configurations and capture protocol.

The notebook automatically locates your attached development data. With **expanded**
data, it fits templates from the +30 dB variants, which can give slightly different results. For the closest
reproduction of this reference, attach `recorded_dataset` and set `DATA_ROOT`
accordingly with `DATA_FORMAT="compact"`. Both formats keep the same 80/20 split.
""")

code('''
from pathlib import Path
import copy
import hashlib
import importlib
import json
import shutil
import subprocess
import sys
import time

MODE = "train_and_submit"   # "train_and_submit", "train" (no submission), or "infer"
RUN_NAME = "track2_fourier_bank_v1"
DATA_FORMAT = "auto"         # "auto", "expanded" or "compact"
# None automatically locates your attached development data and published manifest.
# Set explicit paths only if multiple development datasets are attached.
# Your original radar-1.ipynb used:
# DATA_ROOT = Path("/kaggle/input/datasets/hadimanes/radar-ml/expanded (2)/expanded")
# MANIFEST_PATH = Path("/kaggle/input/datasets/hadimanes/manifests")
DATA_ROOT = None
MANIFEST_PATH = None
WORK_ROOT = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path.cwd()
RUN_DIR = WORK_ROOT / RUN_NAME

# To continue a run in another Kaggle session, upload its *_resume.zip as a private
# dataset, extract it, and point here to the directory containing run_settings.json.
RESUME_FROM = None

# Only needed for MODE="infer": extracted frozen directory containing bank.pt.
# In train_and_submit mode, the notebook uses the model it just trained and froze.
FROZEN_BUNDLE = None
# None automatically finds the participant package attached under /kaggle/input.
# If multiple packages are attached, set the directory explicitly, for example:
# BLIND_ROOT = Path("/kaggle/input/your-participants-dataset/participants")
BLIND_ROOT = None
PREDICTIONS_PATH = WORK_ROOT / "submission_track2.csv"

BATCH_SIZE = 128              # lower to 32 or 64 if GPU memory is limited
WORKERS = 2                  # use 0 if notebook multiprocessing causes problems
assert MODE in {"train_and_submit", "train", "infer"}
TRAIN_MODEL = MODE in {"train", "train_and_submit"}
GENERATE_SUBMISSION = MODE in {"train_and_submit", "infer"}
assert Path(RUN_NAME).name == RUN_NAME and RUN_NAME not in {"", ".", ".."}
if TRAIN_MODEL:
    if RESUME_FROM is not None and not RUN_DIR.exists():
        shutil.copytree(Path(RESUME_FROM), RUN_DIR)
else:
    assert FROZEN_BUNDLE is not None, "Set FROZEN_BUNDLE for inference-only mode"
print("Mode:", MODE, "| run:", RUN_DIR)
''')
md("""
## Self-contained source

These cells write the tested runtime and the original data helpers into the working
directory. Nothing needs to be installed from GitHub. The unchanged data helpers keep
FSK upload-name collision recovery, the published manifest checksum, the official
seven labels and the anonymous submission format. Kaggle may rename files containing
`#`; JSON source identities plus array hashes recover the correct pairing.
""")
for name in ("radarml_track2_runtime.py", "radarml_track2_bank.py"):
    source = (ROOT / "track2" / name).read_text()
    code(f"%%writefile {name}\n" + source)

md("""
## Configure fitting and validation

All **80 training source pulses per recording** fit each complex mean template.
Compact inputs use the original recorded training pulses. Expanded inputs use only
the **+30 dB training variants** for fitting, to avoid averaging synthetic noise into
the templates. Every training source pulse retains its published assignment.

The fixed acquisition window is samples **2400:14640** (20–122 µs), identical for
all templates and unrelated to filenames. The noise estimate uses samples 0:2400.
Template construction preserves stable observed phase. For training groups with phase
lock below 0.98, it aligns training pulses before averaging; the phase-invariant
alternative can then be selected by validation. There is no blind fitting or averaging.

Each input is transformed to a full **complex FFT**, preserving phase. Eight fixed
scoring alternatives compare it with the learned bank: known phase, known phase with
an approximate clipping correction, marginalized unknown phase, or free amplitude;
each uses either maximum likelihood or an equal-class-prior template mixture.
The scoring alternatives share one bank; they are not eight neural networks.

The first four published validation indices per recording, at all 13 SNRs, select
the rule using the organizer's mean balanced accuracy, with the specified low-SNR
tie-break. Validation never updates template values or learned calibration.
The selected rule then receives full validation, plus an audit excluding those
selection source pulses. Do not choose a new rule using the audit report.
""")
code('''
import numpy as np
import torch
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path.cwd()))
import radarml_track2_runtime as helpers
import radarml_track2_bank as bank
helpers = importlib.reload(helpers)
bank = importlib.reload(bank)
torch.set_num_threads(4)

CONFIG = copy.deepcopy(bank.DEFAULT_CONFIG)
CONFIG.update(batch_size=BATCH_SIZE, workers=WORKERS)
# Leave the published 80/20 source split intact. Do not alter CONFIG when resuming.
# Use a new RUN_NAME for a genuinely new experiment.


def find_development_root(explicit=None, kind="auto", search_root=Path("/kaggle/input")):
    """Locate a complete development release by array filenames, without loading signals."""
    if kind not in {"auto", "compact", "expanded"}:
        raise ValueError("DATA_FORMAT must be auto, compact or expanded")
    scope = Path(explicit if explicit is not None else search_root).expanduser().resolve()
    if not scope.is_dir():
        raise FileNotFoundError(f"Development-data directory does not exist: {scope}. "
                                "Attach the dataset or leave DATA_ROOT=None for automatic detection.")
    counts = {}
    for path in scope.rglob("*.npy"):
        name = helpers.canonical_filename(path.name)
        fmt = ("compact" if name.endswith("_recorded.npy") else
               "expanded" if name.endswith("_single_channel.npy") else None)
        if fmt is None or (kind != "auto" and fmt != kind):
            continue
        for parent in path.parents:
            if not parent.is_relative_to(scope):
                break
            key = (parent, fmt)
            counts[key] = counts.get(key, 0) + 1
    complete = [(path, fmt) for (path, fmt), count in counts.items() if count == 660]
    # Prefer the deepest complete directory, rather than its dataset-owner ancestors.
    complete = [(path, fmt) for path, fmt in complete if not any(
        other != path and other.is_relative_to(path) and other_fmt == fmt
        for other, other_fmt in complete)]
    if not complete:
        folders = "\\n".join(str(path) for path in sorted(scope.iterdir()) if path.is_dir())
        raise FileNotFoundError(
            f"No complete development dataset found under {scope}. Expected 660 "
            "*_single_channel.npy or *_recorded.npy arrays with their JSON sidecars. "
            "Attach the expanded radar-ml dataset or compact recorded_dataset. "
            f"Available folders:\\n{folders or '(none)'}"
        )
    if len(complete) != 1:
        choices = "\\n".join(f"{path} ({fmt})" for path, fmt in sorted(complete))
        raise ValueError("Multiple development datasets found; set DATA_ROOT and "
                         f"DATA_FORMAT explicitly:\\n{choices}")
    return complete[0][0]


if TRAIN_MODEL:
    DATA_ROOT = find_development_root(DATA_ROOT, DATA_FORMAT)
    print("Development data:", DATA_ROOT)
    manifest = helpers.find_manifest(DATA_ROOT, MANIFEST_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(json.dumps(helpers.hardware_info(device), indent=2))
    print(json.dumps(CONFIG, indent=2))
    if device.type == "cpu":
        print("Enable a GPU for faster FFT/template scoring. CPU execution is supported.")
    print("Manifest:", manifest)
else:
    print("Only the frozen runtime will execute inference.")
''')
md("""
## Fit, select, validate and export

Progress is saved after **each recording's fitted template**. If this cell is
interrupted, rerun it with the same settings to reuse completed templates. If GPU
memory is insufficient, start a new RUN_NAME with a smaller batch size, or retain
settings and free GPU memory before resuming. Template fitting is followed by
34,320-example selection and 171,600-example full validation.

The runtime verifies input identity hashes before reusing cached training parameters.
Reading the expanded arrays for hashing can take several minutes. It memory-maps
the data and does not load all recordings into RAM. Full FP32 complex operations
are used; mixed precision is deliberately avoided in coherent likelihood scoring.

The final freeze directory contains `bank.pt`, executable inference code, settings,
dependency versions, training provenance, selection results, full/audit validation,
confusion matrices, a method summary and SHA256 hashes. A completed frozen run is
verified and reused on rerun. The earlier `radar-1.ipynb` model remains separate.
""")
code('''
if TRAIN_MODEL:
    try:
        full_report = bank.train_and_freeze(DATA_ROOT, manifest, RUN_DIR, CONFIG, kind=DATA_FORMAT)
    except KeyboardInterrupt:
        print("Interrupted. Completed training templates are saved; rerun this cell to resume.")
        resume_zip = shutil.make_archive(str(RUN_DIR) + "_resume", "zip", root_dir=RUN_DIR)
        print("Download this before ending the session:", resume_zip)
        raise
    frozen = RUN_DIR / "frozen"
    frozen_zip = shutil.make_archive(str(RUN_DIR) + "_frozen", "zip", root_dir=frozen)
    print("DOWNLOAD FROZEN MODEL:", frozen_zip)
    print("ZIP SHA256:", helpers.lookup_sha256(Path(frozen_zip)))
    from IPython.display import FileLink, display
    display(FileLink(str(Path(frozen_zip).relative_to(WORK_ROOT))))
''')
md("""
## Inspect measured validation evidence

The full score is the mean of class-balanced accuracy at **all 13 requested SNRs**.
Class counts are unequal; ordinary accuracy is a different metric. Reports contain
raw confusion matrices (true rows, predicted columns), per-class recall and requested
SNR curves. The separate audit uses 16 validation source pulses per recording,
excluding the four used for model selection. It is still development validation.
""")
code('''
if TRAIN_MODEL:
    frozen = RUN_DIR / "frozen"
    full = json.loads((frozen / "validation_report.json").read_text())
    audit = json.loads((frozen / "audit_validation.json").read_text())
    candidates = json.loads((frozen / "selection_candidates.json").read_text())
    assert full["samples"] == 171600 and audit["samples"] == 137280
    print("Candidate selection results (validation subset):")
    for name, value in sorted(candidates.items(), key=lambda item: -item[1]["mean_balanced_accuracy"]):
        print(f"  {name:26s} {value['mean_balanced_accuracy']:.8f}")
    print("Selected:", full["candidate"])
    print(f"FULL VALIDATION: {full['mean_balanced_accuracy']:.8f}")
    print(f"SEPARATE AUDIT:  {audit['mean_balanced_accuracy']:.8f}")
    print(f"Lowest three SNRs: {full['lowest_three_mean_balanced_accuracy']:.8f}")
    print("Requested SNR | balanced accuracy | ordinary accuracy")
    for row in full["per_snr"]:
        print(f"{row['requested_snr_db']:+3d} dB | {row['balanced_accuracy']:.6f} | {row['accuracy']:.6f}")
    plot_dir = RUN_DIR / "plots"
    plot_dir.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4))
    for label, report in [("Full validation", full), ("Audit excluding selection", audit)]:
        ax.plot([r["requested_snr_db"] for r in report["per_snr"]],
                [r["balanced_accuracy"] for r in report["per_snr"]], "o-", label=label)
    ax.axhline(0.995, color="gray", linestyle="--", label="0.995 reference")
    ax.set(xlabel="Requested SNR (dB)", ylabel="Seven-class balanced accuracy", ylim=(0, 1.01))
    ax.legend(); ax.grid(alpha=.25); fig.tight_layout()
    fig.savefig(plot_dir / "validation_snr_curve.png", dpi=160)
    plt.show()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, snr in zip(axes, (-20, -25, -30)):
        row = next(r for r in full["per_snr"] if r["requested_snr_db"] == snr)
        matrix = np.array(row["confusion_matrix"], dtype=float)
        ax.imshow(matrix / matrix.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
        ax.set(title=f"Requested {snr:+d} dB", xlabel="Predicted", ylabel="True")
        ax.set_xticks(range(7), helpers.CLASSES, rotation=45, ha="right")
        ax.set_yticks(range(7), helpers.CLASSES)
    fig.tight_layout(); fig.savefig(plot_dir / "low_snr_confusions.png", dpi=160)
    plt.show()
    print((frozen / "method_summary.txt").read_text())
''')
md("""
## Save a resumable archive

Run this cell before stopping a session. Upload the extracted archive as a private
Kaggle dataset, set `RESUME_FROM` to its root, and retain the same run name/settings.
A Kaggle session killed without saving its output may lose its working directory;
the notebook cannot recover files that Kaggle did not retain. Use **Save Version**
to retain output artifacts. The frozen archive is the one needed for inference.
""")
code('''
if TRAIN_MODEL:
    resume_zip = shutil.make_archive(str(RUN_DIR) + "_resume", "zip", root_dir=RUN_DIR)
    print("Resume archive:", resume_zip)
    from IPython.display import FileLink, display
    display(FileLink(str(Path(resume_zip).relative_to(WORK_ROOT))))
''')
md("""
## Submission inference from the frozen bank

**Train and submit in one Kaggle run (default):** leave `MODE="train_and_submit"`
and attach the released participant package as a Kaggle dataset. This cell uses
`RUN_DIR / "frozen"`, the model produced above, and automatically locates the
participant package under `/kaggle/input`, including nested `participants` folders.
Participant signals are first opened here, after fitting, selection and freezing.
If multiple packages are attached, set `BLIND_ROOT` explicitly in the settings cell.

**Submit using an already trained model:** use `MODE="infer"` and point
`FROZEN_BUNDLE` at its extracted frozen directory. Attach the participant dataset;
training data is unnecessary. This mode skips fitting and validation.

**Training only:** use `MODE="train"` to skip submission generation.

This cell checks every frozen artifact hash **before executing the saved code**.
It checks all 171,600 IDs, follows template order, and writes
`submission_track2.csv` with exactly `sample_id,predicted_label`, plus a hardware,
timing and provenance report. All signal processing is included in inference timing.
It cannot compute a blind score without the organizers' private labels.

**Use this notebook or the new bundle's command for the new model.** The previous
`status/generate_submission.py` belongs to the old CNN bundle and does not load this
new bank. Standalone inference, from the extracted new frozen directory:

```bash
python radarml_track2_bank.py infer --bundle . --blind-root /path/to/participants \\
  --output /path/to/submission_track2.csv
```
""")
code('''
def find_participants_root(explicit=None, search_root=Path("/kaggle/input")):
    """Find a unique package containing its template and anonymous index."""
    root = Path(explicit if explicit is not None else search_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Participant dataset directory does not exist: {root}")
    # An explicit path may point directly to the test/ directory.
    if explicit is not None and (root / "index.csv").is_file():
        if (root / "submission_template.csv").is_file():
            return root
        if (root.parent / "submission_template.csv").is_file():
            return root.parent
    candidates = set()
    for template in root.rglob("submission_template.csv"):
        package = template.parent
        if (package / "test/index.csv").is_file() or (package / "index.csv").is_file():
            candidates.add(package.resolve())
    if not candidates:
        raise FileNotFoundError(
            f"No participant package found under {root}. Attach the extracted participants "
            "dataset with submission_template.csv and test/index.csv, or set BLIND_ROOT."
        )
    if len(candidates) > 1:
        choices = "\\n".join(str(path) for path in sorted(candidates))
        raise ValueError(f"Multiple participant packages found; set BLIND_ROOT to one:\\n{choices}")
    return candidates.pop()


if GENERATE_SUBMISSION:
    frozen = (RUN_DIR / "frozen" if TRAIN_MODEL else Path(FROZEN_BUNDLE)).resolve()
    hashes = json.loads((frozen / "frozen.sha256.json").read_text())
    required = {"bank.pt", "config.json", "radarml_track2_bank.py", "radarml_track2_runtime.py"}
    if not required.issubset(hashes):
        raise ValueError("Incomplete frozen bank bundle")
    for name, expected in hashes.items():
        if Path(name).name != name:
            raise ValueError("Invalid artifact path")
        actual = hashlib.sha256((frozen / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Frozen artifact changed: {name}")
    participants = find_participants_root(BLIND_ROOT)
    print("Frozen model:", frozen)
    print("Participant package:", participants)
    subprocess.run([sys.executable, str(frozen / "radarml_track2_bank.py"), "infer",
                    "--bundle", str(frozen), "--blind-root", str(participants),
                    "--output", str(PREDICTIONS_PATH), "--batch-size", str(BATCH_SIZE),
                    "--workers", str(WORKERS)], check=True)
    print(json.loads(PREDICTIONS_PATH.with_name(PREDICTIONS_PATH.name + ".inference.json").read_text()))
    from IPython.display import FileLink, display
    display(FileLink(str(PREDICTIONS_PATH.relative_to(WORK_ROOT))))
    display(FileLink(str(PREDICTIONS_PATH.with_name(PREDICTIONS_PATH.name + ".inference.json").relative_to(WORK_ROOT))))
    print("Download your submission CSV:", PREDICTIONS_PATH)
else:
    print("Submission generation skipped because MODE='train'.")
''')
md("""
## Interpretation and references

This approach is tailored to **known waveform configurations and repeatable capture
phase/timing**. A high score here does not establish performance on new waveforms,
receivers, time offsets or arbitrary phase rotations. The phase-invariant alternatives
are included for validation comparison; the coherent option may be strongest for
this acquisition protocol. Requested SNR differs from achieved SNR, and low-SNR int16
clipping changes the noise distribution. The clipping correction is approximate.

No training-source filename, index, requested-SNR tag or metadata enters prediction.
Training grouping manages the supplied recordings and target families; the fitted
weights are waveform estimates. Inference uses only each input's I/Q-derived Fourier
coefficients, fixed acquisition constants and the frozen training-template bank.
No validation/test pulse is used as a template. No archive search recovers blind labels.

- Ritchie, White and Hosford (2025), [RadarML](https://doi.org/10.5522/04/30752767.v1).
- [Radar pulse compression / matched filtering](https://www.mathworks.com/help/signal/ug/radar-pulse-compression.html).
- [PyTorch complex FFT documentation](https://docs.pytorch.org/docs/stable/generated/torch.fft.fft.html).
- Supplied `hackathon/EVENT_RULES.md`, loader, scorer and compact-data expander.
- Code reuse: organizer GPL-3.0 helpers and the prior local runtime. AI assistance: OpenAI Codex.
""")

notebook = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
    "accelerator": "GPU",
    "kaggle": {"accelerator": "gpu", "isInternetEnabled": False},
})
nbf.validate(notebook)
path = ROOT / "status/radar-track2-strong.ipynb"
nbf.write(notebook, path)
print(path)
