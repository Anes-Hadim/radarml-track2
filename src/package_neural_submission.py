"""Package the downloaded neural model, CPU inference script and plain documentation.

Development utility only. PDF generation requires reportlab; inference does not.
"""
import json
from pathlib import Path
import shutil
import zipfile
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, PageBreak

ROOT = Path(__file__).resolve().parents[1]
EXPORT = ROOT / "results-2/track2_neural_v1/export"


def main():
    summary = json.loads((EXPORT / "training_summary.json").read_text())
    validation = json.loads((EXPORT / "validation_report.json").read_text())
    audit = json.loads((EXPORT / "audit_validation.json").read_text())
    hardware = summary["hardware"]
    cfg = json.loads((EXPORT / "training_config.json").read_text())
    original_timing = json.loads((ROOT / "results-2/submission_track2_neural.csv.inference.json").read_text())
    timing = original_timing["single_signal_latency"]
    for destination in (EXPORT / "radarml_track2_neural_infer.py",
                        ROOT / "results-2/radarml_track2_neural_infer.py"):
        shutil.copy2(ROOT / "track2/neural_submission.py", destination)
    (EXPORT / "requirements.txt").write_text("numpy>=1.26,<3\ntorch>=2.4,<3\n", encoding="utf-8")

    method = [
        ("Model", f"Track 2 seven-class neural classifier, trained from random initialization. "
         f"Classes: {', '.join(summary['classes'])}. {summary['parameters']:,} learned parameters; "
         f"model.pt is {summary['model_bytes']:,} bytes ({summary['model_bytes']/1e6:.2f} MB). "
         "best.pt and export/model.pt contain the same selected checkpoint. The final script uses CPU float32."),
        ("Signal representation and network", "Each signal contains 16,800 complex samples (interleaved int16 I/Q). "
         "I/Q values are divided by 32,768, then RMS normalized. A two-sided complex STFT uses 256-sample "
         "rectangular windows, hop 256 and right zero padding, producing 256 frequency bins by 66 time frames. "
         "The network combines global and convolutional phase-sensitive features from real/imaginary STFT values, "
         "a small separable-convolution branch over log STFT power, and four signal-derived statistics. "
         "A learned classifier produces seven logits; argmax selects the label. "
         "One forward pass is used per signal. No template bank, nearest-neighbor search or stored training "
         "waveforms are used at inference. IDs and filenames locate signals and order output; they are not model features."),
        ("Training", f"The published 80/20 pulse split was used. {summary['training_source_pulses']:,} training source pulses "
         f"were eligible for class-balanced sampling; {summary['training_draws']:,} examples were drawn over "
         f"{summary['epochs_completed']} epochs. Independent noise augmentation covers all 13 SNRs from +30 to -30 dB "
         "after two warm-up epochs, with extra sampling weight on the lowest SNRs. "
         "AdamW, mixed precision, label smoothing, paired clean/noisy cross-entropy and feature consistency were used. "
         "Validation selected the checkpoint; validation signals did not fit model weights. "
         f"The export is epoch {validation['selected_epoch']}, variant {validation['selected_variant']}. "
         "No external training data or pretrained weights were used."),
        ("Training hardware and software", f"Kaggle, {hardware['gpu']} GPU; {hardware['cpu']} CPU architecture, "
         f"{hardware['logical_cpus']} logical CPUs and {hardware['threads']} PyTorch CPU threads. "
         f"Training batch size {cfg['batch_size']}. Python {hardware['python'].split()[0]}, "
         f"PyTorch {hardware['torch']}, NumPy {hardware['numpy']}; Linux. "
         f"Recorded fitting time: {summary['fit_seconds']/60:.1f} minutes, excluding preparation and final evaluation. "
         "The saved report does not identify the CPU model or host RAM capacity."),
        ("Measured validation", f"Mean balanced accuracy across the 13 requested SNRs: "
         f"{validation['mean_balanced_accuracy']*100:.2f}% on {validation['samples']:,} examples. "
         f"Audit excluding checkpoint-selection source pulses: {audit['mean_balanced_accuracy']*100:.2f}% "
         f"on {audit['samples']:,} examples. These are validation results, not blind-test accuracy. "
         "The earlier template-bank score does not apply to this model."),
    ]
    usage = """HOW TO RUN

Requires Python 3.10 or newer, NumPy and PyTorch. No training code is required.
Keep radarml_track2_neural_infer.py and model.pt; the paths may contain spaces
when quoted. From the export folder, install the two runtime dependencies:

  python -m pip install -r requirements.txt

Exactly two path arguments are required:

  python radarml_track2_neural_infer.py --model "/path/to/model.pt" --participants "/path/to/participants"

The blind path can be the participants folder, its test subfolder, or the
blind-test folder containing participants. It must include index.csv and the
referenced int16 .npy shards. If submission_template.csv is present in the
package, only its sample IDs are used to preserve the official order.

Output defaults to submission.csv in the directory where you run the command.
Optional settings (neither is required):

  --output "/path/to/submission.csv" --threads 4

Use --help to display the available arguments. CPU inference is always used.
On this Mac, the Python installation with PyTorch is:

  /Library/Frameworks/Python.framework/Versions/3.12/bin/python3

From the Radar_ML_lite project folder, the following uses the existing files:

  /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 \\
    results-2/track2_neural_v1/export/radarml_track2_neural_infer.py \\
    --model results-2/track2_neural_v1/export/model.pt \\
    --participants status/participants

OUTPUTS AND TIMING

submission.csv
  Exactly two columns: sample_id,label. It never outputs an s column.
  Existing files at the selected output path are replaced after inference.

submission.csv.latency.csv
  Columns: sample_id,inference_ms. One actual measurement for EVERY submitted
  signal, in submission order. No batched time is divided across signals.

submission.csv.inference.json
  Mean, median, p95, p99 and maximum single-signal latency, fraction <=10 ms,
  sample count, model size, CPU/software details, model setup time and total
  elapsed time. Progress and the final timing summary are also printed.

Timing is CPU-only, batch size 1. Each signal's timer starts before reading
its memory-mapped shard row and converting int16 I/Q, and ends after STFT,
the complete neural network and label selection. One-time index/weight
loading, 20 warm-up passes, progress printing and CSV writing are excluded
from individual latency; separate elapsed totals are reported. OS cache
state is uncontrolled, so this is not a guaranteed cold-disk benchmark.
Timing depends on CPU, thread count and machine load. The 10 ms statistic
is a reference, not a statement of organizer acceptance.

Only this script, the model weights and the released blind package are read
to make predictions. Training data, development helpers, validation files,
optimizer checkpoints and previous submissions are not inference inputs.
The model weights and architecture have not been changed by this cleanup.
"""
    prior = (f"Original Kaggle CPU benchmark (earlier inference script): {timing['samples']:,} individually "
             f"timed signals, mean {timing['mean_ms']:.2f} ms, median {timing['median_ms']:.2f} ms, "
             f"p95 {timing['p95_ms']:.2f} ms. This was a 4-logical-CPU x86_64 environment with 4 PyTorch "
             "threads and included I/Q conversion and STFT. Its batched throughput result is a separate metric. "
             "The rewritten script measures every signal separately on the machine running it.")
    local_report = ROOT / "results-2/submission_track2_neural_cpu.csv.inference.json"
    local_measurement = None
    if local_report.is_file():
        local = json.loads(local_report.read_text())
        measured = local["single_signal_latency"]
        local_measurement = (
            f"Rewritten script, full local CPU run: {local['samples']:,} individually timed signals; "
            f"mean {measured['mean_ms']:.3f} ms, median {measured['median_ms']:.3f} ms, "
            f"p95 {measured['p95_ms']:.3f} ms and p99 {measured['p99_ms']:.3f} ms. "
            f"Hardware: {local['hardware']['machine']}, {local['hardware']['logical_cpus']} logical CPUs, "
            f"{local['hardware']['threads']} PyTorch threads, {local['hardware']['platform']}; "
            f"PyTorch {local['hardware']['torch']}. These measurements include I/Q conversion and STFT. "
            "They describe this machine and run, not a guaranteed latency on other hardware.")
    text = "RADARML TRACK 2 - METHOD AND INFERENCE\n\n"
    text += "\n\n".join(title.upper() + "\n" + body for title, body in method)
    text += "\n\n" + usage + "\nPREVIOUS CPU MEASUREMENT\n" + prior + "\n"
    if local_measurement:
        text += "\nLOCAL CPU MEASUREMENT\n" + local_measurement + "\n"
    (EXPORT / "method_summary.txt").write_text(text, encoding="utf-8")

    # Plain black text on white A4 pages: no decorative graphics or branding.
    body_style = ParagraphStyle("Body", fontName="Helvetica", fontSize=10, leading=13,
                                textColor=colors.black, alignment=TA_LEFT, spaceAfter=8)
    heading_style = ParagraphStyle("Heading", parent=body_style, fontName="Helvetica-Bold", spaceBefore=5)
    title_style = ParagraphStyle("Title", parent=heading_style, fontSize=13, leading=17, spaceAfter=12)
    code_style = ParagraphStyle("Code", fontName="Courier", fontSize=8, leading=11, spaceAfter=10)
    story = [Paragraph("RadarML Track 2: complex-STFT neural classifier", title_style)]
    for title, body in method:
        story.extend([Paragraph(escape(title), heading_style), Paragraph(escape(body), body_style)])
    story.extend([PageBreak(), Paragraph("Inference and timing", title_style)])
    for title, body in [
        ("Required files", "The standalone inference script reads model.pt and the released blind package. "
         "It requires Python 3.10 or newer, NumPy and PyTorch. No training files, development helpers or "
         "previous predictions are required. The architecture and trained weights are unchanged."),
        ("Usage", "Run from the folder containing the script. Only the model and blind-data paths are required. "
         "Quote paths containing spaces. The blind path can point to participants, participants/test, "
         "or the blind-test folder containing participants. It must contain index.csv and its signal shards."),
    ]:
        story.extend([Paragraph(title, heading_style), Paragraph(escape(body), body_style)])
    command_text = (
        'python -m pip install -r requirements.txt\n\n'
        'python radarml_track2_neural_infer.py \\\n'
        '  --model "/path/to/model.pt" \\\n'
        '  --participants "/path/to/participants"\n\n'
        '# Optional settings:\n'
        '# --output "/path/to/submission.csv" --threads 4')
    story.append(Paragraph("<br/>".join(escape(line).replace(" ", "&nbsp;")
                                         for line in command_text.split("\n")), code_style))
    for title, body in [
        ("Outputs", "The default submission.csv is written in the current directory with exactly sample_id,label. "
         "A separate submission.csv.latency.csv contains sample_id,inference_ms for every signal. "
         "submission.csv.inference.json records mean, median, p95, p99, maximum, the fraction within 10 ms, "
         "CPU/software details, model setup time and elapsed totals. Existing selected output files are replaced."),
        ("Timing definition", "CPU float32, one signal per forward pass. Each measured interval includes "
         "memory-mapped shard access, int16 I/Q loading/conversion, the entire STFT, neural inference and class selection. "
         "It excludes one-time index and model loading, 20 warm-up passes, logging and output writing. "
         "No batched time is divided across samples. OS cache state is uncontrolled. "
         "Timings vary with hardware, thread count and system load."),
        ("Previous CPU measurement", prior),
        *([("Local CPU measurement", local_measurement)] if local_measurement else []),
        ("Interpretation", "The model performs a learned neural forward pass. It does not search training-waveform "
         "templates. Validation accuracy and timing are measured separately. Blind accuracy is unknown; "
         "acceptance depends on the organizer's rules and timing protocol. The detailed command examples and "
         "training configuration are included in method_summary.txt and the saved training reports."),
    ]:
        story.extend([Paragraph(title, heading_style), Paragraph(escape(body), body_style)])
    document = SimpleDocTemplate(str(EXPORT / "model_writeup.pdf"), pagesize=A4,
                                 rightMargin=45, leftMargin=45, topMargin=40, bottomMargin=40,
                                 title="RadarML Track 2 model write-up", author="")
    document.build(story)
    package = ROOT / "results-2/track2_neural_v1_submission.zip"
    names = ("model.pt", "radarml_track2_neural_infer.py", "requirements.txt", "method_summary.txt", "model_writeup.pdf")
    with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            archive.write(EXPORT / name, name)
    print(f"Method summary: {EXPORT / 'method_summary.txt'}")
    print(f"PDF: {EXPORT / 'model_writeup.pdf'}")
    print(f"Submission package: {package}")


if __name__ == "__main__":
    main()
