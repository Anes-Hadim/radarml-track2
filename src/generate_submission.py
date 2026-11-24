#!/usr/bin/env python3
"""Regenerate the results/ Fourier-bank submission on CPU and measure inference.

Run from the repository root with a Python environment containing NumPy and PyTorch:
    python3 status/generate_submission.py

Defaults: results/track2_fourier_bank_v1/frozen, status/participants, and
results/submission_track2_cpu.csv. The original Kaggle CSV is preserved.
Reports BOTH single-signal latency and batched throughput, including input FFTs.
Default CPU execution packs the same complex inner product into one real matrix
multiply; --implementation frozen retains the original complex implementation.
The 10 ms reference is configurable; it is not a certification of event eligibility.
"""
import argparse
import csv
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "results"
DEFAULT_BUNDLE = RESULTS / "track2_fourier_bank_v1/frozen"


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Must be at least 1")
    return number


def nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("Must be at least 0")
    return number


def positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("Must be finite and greater than zero")
    return number


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def verify_manifest(root, manifest_name):
    root = Path(root).expanduser().resolve()
    entries = json.loads((root / manifest_name).read_text(encoding="utf-8"))
    if not isinstance(entries, dict) or not entries:
        raise ValueError(f"Invalid checksum manifest: {root / manifest_name}")
    for index, (relative, expected) in enumerate(entries.items(), 1):
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Missing file or invalid checksum path: {relative}")
        if sha256(path) != expected:
            raise ValueError(f"Checksum mismatch: {path}")
        if index % 10 == 0 or index == len(entries):
            print(f"Verified {root.name}: {index}/{len(entries)} files", flush=True)
    return entries


def load_frozen_runtime(bundle):
    """Import only verified source from the selected bundle, with no code edits."""
    sys.path.insert(0, str(bundle))
    for name in ("radarml_track2_runtime", "radarml_track2_bank"):
        existing = sys.modules.get(name)
        if existing is not None and Path(existing.__file__).resolve() != bundle / f"{name}.py":
            raise RuntimeError("A different model runtime is already imported; run this script in a fresh Python process")
    runtime = importlib.import_module("radarml_track2_bank")
    if Path(runtime.__file__).resolve() != bundle / "radarml_track2_bank.py":
        raise RuntimeError("Imported a runtime outside the selected frozen bundle")
    runtime.verify_bundle(bundle)
    return runtime


def load_cpu_model(runtime, bundle, torch):
    cfg = json.loads((bundle / "config.json").read_text(encoding="utf-8"))
    if (cfg.get("model_family") != "complex_fft_template_bank" or cfg.get("track") != "B" or
            cfg.get("classes") != list(runtime.CLASSES) or cfg.get("inference_precision") != "float32" or
            cfg.get("selected_candidate") not in runtime.CANDIDATES):
        raise ValueError("Unexpected frozen Fourier-bank settings")
    runtime.data.seed_everything(cfg["seed"])
    checkpoint = torch.load(bundle / "bank.pt", map_location="cpu", weights_only=True)
    model = runtime.FourierBank(checkpoint).to(torch.device("cpu")).eval()
    if any(buffer.device.type != "cpu" for buffer in model.buffers()):
        raise RuntimeError("CPU-only inference was not established")
    return model, cfg


def make_cpu_executor(frozen_model, candidate, torch):
    """Pack Re(<FFT(x), FFT(t)>) into one real GEMM; keep input FFT and scoring."""
    mode, pool = candidate.split(":")
    if mode not in {"coherent", "coherent_clip"} or pool not in {"max", "mixture"}:
        raise ValueError("Packed-real execution supports coherent scores only; use --implementation frozen")

    class PackedRealCoherent(torch.nn.Module):
        def __init__(self, original):
            super().__init__()
            # Frozen kernels contain conj(FFT(template)). Re(z * conj(t)) equals
            # Re(z)*Re(t) + Im(z)*Im(t); the minus below undoes the conjugation.
            weights = torch.cat((original.kernels.real.T, -original.kernels.imag.T), 1).contiguous()
            self.register_buffer("packed_weights", weights)
            self.register_buffer("energy", original.energy)
            self.groups = original.groups

        def forward(self, signals, requested_candidate):
            if requested_candidate != candidate:
                raise ValueError("Executor must use the frozen candidate")
            z = torch.complex(signals[:, 0].float(), signals[:, 1].float())
            spectrum = torch.fft.fft(z, norm="ortho")
            features = torch.cat((spectrum.real, spectrum.imag), 1)
            real_dot = torch.nn.functional.linear(features, self.packed_weights)
            noise = z[:, :2400].abs().square().mean(1, keepdim=True).clamp_min(1e-10)
            if mode == "coherent_clip":
                clipped = ((signals <= -1.) | (signals >= 32767. / 32768.)).float().mean((1, 2))[:, None]
                alpha = (1. - clipped).clamp(0.1, 1.)
                scores = (2. * alpha * real_dot - alpha.square() * self.energy) / noise
            else:
                scores = (2. * real_dot - self.energy) / noise
            if pool == "max":
                classes = [scores[:, idx].amax(1) for idx in self.groups]
            else:
                classes = [torch.logsumexp(scores[:, idx], 1) - math.log(len(idx)) for idx in self.groups]
            return torch.stack(classes, 1)

    return PackedRealCoherent(frozen_model).eval()


def verify_cpu_executor(frozen, executor, dataset, candidate, torch, np):
    """Check the execution rewrite against the frozen implementation, without labels."""
    indices = np.linspace(0, len(dataset) - 1, min(32, len(dataset)), dtype=np.int64)
    with torch.inference_mode():
        signals = torch.stack([dataset[int(index)] for index in indices])
        expected = frozen(signals, candidate)
        actual = executor(signals, candidate)
    # Likelihoods can subtract nearly equal terms; a near-zero losing class
    # score needs an error bound scaled by this input's overall score magnitude.
    scaled_error = (actual - expected).abs() / expected.abs().amax(1, keepdim=True).clamp_min(1.)
    if not torch.isfinite(actual).all() or float(scaled_error.max()) > 2e-4:
        raise RuntimeError("Packed-real scores disagree with the frozen implementation")
    differences = int((expected.argmax(1) != actual.argmax(1)).sum())
    if differences:
        raise RuntimeError("Packed-real predictions disagree with frozen predictions in the execution check")
    return dict(samples=len(indices), different_predictions=differences,
                max_absolute_score_difference=float((actual - expected).abs().max()),
                max_row_scaled_score_difference=float(scaled_error.max()),
                row_scaled_score_tolerance=2e-4,
                note="Same FFT, templates and likelihood; floating-point summation order differs")


def latency_statistics(milliseconds, target_ms, np):
    values = np.asarray(milliseconds, dtype=np.float64)
    if not len(values) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Invalid latency measurements")
    return dict(samples=len(values), mean_ms=float(values.mean()), median_ms=float(np.median(values)),
                p95_ms=float(np.percentile(values, 95)), p99_ms=float(np.percentile(values, 99)),
                min_ms=float(values.min()), max_ms=float(values.max()),
                target_ms=target_ms, fraction_at_or_below_target=float(np.mean(values <= target_ms)),
                mean_at_or_below_target=bool(values.mean() <= target_ms),
                p95_at_or_below_target=bool(np.percentile(values, 95) <= target_ms))


def benchmark_single_signals(model, dataset, rows, candidate, samples, warmup, target_ms, torch, np):
    """Measure actual batch-size-one latency; include dataset reads and conversion."""
    count = min(samples, len(dataset))
    if not count:
        raise ValueError("Cannot benchmark an empty dataset")
    indices = np.linspace(0, len(dataset) - 1, count, dtype=np.int64)
    measurements = []
    start = time.perf_counter()
    with torch.inference_mode():
        for i in range(warmup):
            signal = dataset[int(indices[i % count])].unsqueeze(0)
            model(signal, candidate).argmax(1).item()
        for step, index in enumerate(indices, 1):
            tick = time.perf_counter_ns()
            signal = dataset[int(index)].unsqueeze(0)
            model_start = time.perf_counter_ns()
            model(signal, candidate).argmax(1).item()
            stop = time.perf_counter_ns()
            measurements.append(dict(sample_id=rows[int(index)]["sample_id"],
                                     end_to_end_ms=(stop - tick) / 1e6,
                                     model_with_fft_ms=(stop - model_start) / 1e6))
            if step == 1 or step % 250 == 0:
                print(f"Single-signal CPU benchmark: {step:,}/{count:,}", flush=True)
    report = dict(batch_size=1, warmup_signals_excluded=warmup,
                  selection="Evenly spaced anonymous index positions; no labels used",
                  wall_seconds_including_warmup=time.perf_counter() - start,
                  end_to_end=latency_statistics([r["end_to_end_ms"] for r in measurements], target_ms, np),
                  model_with_fft=latency_statistics([r["model_with_fft_ms"] for r in measurements], target_ms, np),
                  timing_scope="End-to-end: shard access, int16 I/Q conversion/scaling, tensor creation, "
                               "input FFT, noise/clipping estimates, all template scores, class reduction and argmax. "
                               "Model-with-FFT excludes shard access and I/Q conversion. "
                               "Both exclude one-time model loading/template FFTs. OS file-cache state is uncontrolled.")
    return report, measurements


def infer_all(model, dataset, candidate, cfg, batch_size, workers, torch):
    options = dict(batch_size=batch_size, num_workers=workers, shuffle=False, pin_memory=False,
                   generator=torch.Generator().manual_seed(cfg["seed"]))
    loader = torch.utils.data.DataLoader(dataset, **options)
    predictions = []
    # Includes iterator/worker startup, loading, int16 conversion and input FFTs.
    start = time.perf_counter()
    with torch.inference_mode():
        for step, signals in enumerate(loader, 1):
            if signals.device.type != "cpu":
                raise RuntimeError("Unexpected non-CPU input")
            predictions.extend(model(signals, candidate).argmax(1).tolist())
            if step == 1 or step % 100 == 0:
                elapsed = time.perf_counter() - start
                print(f"CPU inference: {len(predictions):,}/{len(dataset):,}; "
                      f"average {elapsed / len(predictions) * 1000:.3f} ms/signal", flush=True)
    elapsed = time.perf_counter() - start
    if len(predictions) != len(dataset):
        raise RuntimeError("Incomplete CPU inference")
    report = dict(samples=len(predictions), batch_size=batch_size, workers=workers,
                  inference_seconds=elapsed, seconds_per_pulse=elapsed / len(predictions),
                  milliseconds_per_signal=elapsed / len(predictions) * 1000,
                  signals_per_second=len(predictions) / elapsed,
                  timing_scope="Full submission pass with warm model: iterator/worker startup, shard access, "
                               "int16 I/Q conversion/scaling, input FFT, noise/clipping estimates, all template "
                               "scores, class reduction, argmax and prediction collection. Excludes model/index "
                               "loading, checksum verification, warm-up benchmark and CSV writing. "
                               "Elapsed time divided by sample count is throughput, not individual latency.")
    return predictions, report


def prediction_column(fieldnames):
    if fieldnames not in (["sample_id", "predicted_label"], ["sample_id", "label"]):
        raise ValueError("Expected columns sample_id,predicted_label or sample_id,label")
    return fieldnames[1]


def write_submission(output, rows, template_ids, predictions, classes, label_column="predicted_label"):
    if len(rows) != len(predictions) or len(set(template_ids)) != len(template_ids):
        raise ValueError("Incomplete predictions or duplicate template IDs")
    by_id = {r["sample_id"]: classes[p] for r, p in zip(rows, predictions)}
    if len(by_id) != len(rows) or set(by_id) != set(template_ids):
        raise ValueError("Submission IDs do not match the template")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".partial")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample_id", label_column])
        writer.writerows((sid, by_id[sid]) for sid in template_ids)
    with temporary.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        saved = list(reader)
        if (reader.fieldnames != ["sample_id", label_column] or
                [r["sample_id"] for r in saved] != template_ids or
                any(r[label_column] not in classes for r in saved)):
            raise ValueError("Serialized CSV failed validation")
    temporary.replace(output)
    return by_id


def compare_reference(path, predictions, classes):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        column = prediction_column(reader.fieldnames)
        rows = list(reader)
    by_id = {r["sample_id"]: r[column] for r in rows}
    if (len(by_id) != len(rows) or by_id.keys() != predictions.keys() or
            any(label not in classes for label in by_id.values())):
        raise ValueError("Reference CSV has duplicate/invalid labels or mismatched IDs")
    differences = sum(predictions[sid] != by_id[sid] for sid in predictions)
    return dict(path=str(path), sha256=sha256(path), samples=len(rows), label_column=column,
                different_predictions=differences,
                identical_predictions=differences == 0,
                note="Prediction agreement only; this is not an accuracy score")


def main(argv=None):
    start = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--participants", type=Path, default=HERE / "participants")
    parser.add_argument("--output", type=Path, default=RESULTS / "submission_track2_cpu.csv")
    parser.add_argument("--batch-size", type=positive_int, default=128)
    parser.add_argument("--workers", type=nonnegative_int, default=0)
    parser.add_argument("--threads", type=positive_int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--implementation", choices=["packed-real", "frozen"], default="packed-real",
                        help="CPU execution: equivalent real GEMM (default) or original complex GEMM")
    parser.add_argument("--benchmark-samples", type=positive_int, default=1000,
                        help="Signals measured individually with batch size 1 (default: 1000)")
    parser.add_argument("--warmup", type=nonnegative_int, default=20)
    parser.add_argument("--target-ms", type=positive_float, default=10.,
                        help="Reference threshold, not an official pass/fail rule (default: 10 ms)")
    parser.add_argument("--reference", type=Path,
                        help="Compare prediction labels with an existing submission (never used by the model)")
    parser.add_argument("--label-column", choices=["auto", "label", "predicted_label"], default="auto",
                        help="Output column; auto preserves the reference CSV format, else uses predicted_label")
    parser.add_argument("--benchmark-only", action="store_true", help="Measure individual latency without a full submission pass")
    parser.add_argument("--check-only", action="store_true", help="Verify model, data and IDs without inference")
    args = parser.parse_args(argv)
    bundle, participants, output = (p.expanduser().resolve() for p in (args.bundle, args.participants, args.output))
    reference = args.reference.expanduser().resolve() if args.reference else RESULTS / "submission_track2.csv"
    for root in (bundle, participants):
        if output.is_relative_to(root):
            parser.error("Choose an output outside the frozen model and participant directories")
    if reference.resolve() == output:
        parser.error("Choose an output different from the existing reference submission")
    if args.reference and not reference.is_file():
        raise FileNotFoundError(f"Requested reference submission is missing: {reference}")
    label_column = args.label_column
    if label_column == "auto":
        if reference.is_file():
            with reference.open(newline="", encoding="utf-8-sig") as stream:
                label_column = prediction_column(csv.DictReader(stream).fieldnames)
        else:
            label_column = "predicted_label"
    try:
        import numpy as np
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"Missing {exc.name!r} in {sys.executable}; install numpy and torch in this environment") from exc
    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # A library caller may have initialized inter-op execution already.
        pass
    verification_start = time.perf_counter()
    hashes = verify_manifest(bundle, "frozen.sha256.json")
    verify_manifest(participants, "checksums.json")
    verification_seconds = time.perf_counter() - verification_start
    setup_start = time.perf_counter()
    runtime = load_frozen_runtime(bundle)
    frozen_model, cfg = load_cpu_model(runtime, bundle, torch)
    model = (make_cpu_executor(frozen_model, cfg["selected_candidate"], torch)
             if args.implementation == "packed-real" else frozen_model)
    model_setup_seconds = time.perf_counter() - setup_start
    index_start = time.perf_counter()
    root, rows, template_ids = runtime.data.read_blind_index(participants)
    rows.sort(key=lambda row: (row["file"], row["row"]))
    dataset = runtime.data.BlindDataset(root, rows)
    index_seconds = time.perf_counter() - index_start
    print(f"Frozen model: {bundle / 'bank.pt'}", flush=True)
    print(f"CPU ONLY; threads={torch.get_num_threads()}, batch size={args.batch_size}; signals={len(rows):,}", flush=True)
    print(f"Submission columns: sample_id,{label_column}", flush=True)
    if args.check_only:
        print("All checks passed; no predictions generated.", flush=True)
        return 0
    execution_start = time.perf_counter()
    equivalence = (verify_cpu_executor(frozen_model, model, dataset, cfg["selected_candidate"], torch, np)
                   if args.implementation == "packed-real" else None)
    execution_check_seconds = time.perf_counter() - execution_start
    if equivalence is not None:
        print(f"Packed-real execution verified against frozen scores on {equivalence['samples']} signals", flush=True)
    del frozen_model
    hardware = runtime.data.hardware_info(torch.device("cpu"))
    hardware.update(logical_cpus=os.cpu_count(), torch_threads=torch.get_num_threads(),
                    torch_interop_threads=torch.get_num_interop_threads(), accelerators_used=[])
    if platform.system() == "Darwin":
        brand = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True)
        if brand.returncode == 0:
            hardware["cpu_model"] = brand.stdout.strip()
    benchmark, measurements = benchmark_single_signals(model, dataset, rows, cfg["selected_candidate"],
                                                       args.benchmark_samples, args.warmup, args.target_ms, torch, np)
    latency_path = output.with_name(output.name + ".latency.csv")
    latency_path.parent.mkdir(parents=True, exist_ok=True)
    with latency_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["sample_id", "end_to_end_ms", "model_with_fft_ms"])
        writer.writeheader()
        writer.writerows(measurements)
    stats = benchmark["end_to_end"]
    print(f"Single-signal latency INCLUDING FFT and I/Q loading/conversion: "
          f"mean={stats['mean_ms']:.3f} ms, median={stats['median_ms']:.3f} ms, "
          f"p95={stats['p95_ms']:.3f} ms, p99={stats['p99_ms']:.3f} ms", flush=True)
    report = dict(device="cpu", precision="float32", selected_candidate=cfg["selected_candidate"],
                  execution_implementation=args.implementation, execution_equivalence_check=equivalence,
                  execution_check_seconds=execution_check_seconds,
                  hardware=hardware, frozen_bundle=str(bundle), frozen_artifact_hashes=hashes,
                  inference_wrapper_sha256=sha256(Path(__file__)),
                  prediction_columns=["sample_id", label_column],
                  verification_seconds=verification_seconds, model_setup_seconds=model_setup_seconds,
                  model_setup_scope="Frozen runtime import, checkpoint load and one-time template-bank FFT preparation",
                  index_loading_seconds=index_seconds, single_signal_benchmark=benchmark,
                  latency_measurements=str(latency_path), reference_target_ms=args.target_ms,
                  target_interpretation="User-specified reference only; the official threshold, percentile, "
                                        "batch size and hardware requirements have not been supplied")
    if args.benchmark_only:
        report["total_seconds"] = time.perf_counter() - start
        report_path = output.with_name(output.name + ".benchmark.json")
    else:
        predictions, inference = infer_all(model, dataset, cfg["selected_candidate"], cfg,
                                           args.batch_size, args.workers, torch)
        serialization_start = time.perf_counter()
        by_id = write_submission(output, rows, template_ids, predictions, runtime.CLASSES, label_column)
        report.update(samples=len(rows), inference_seconds=inference["inference_seconds"],
                      seconds_per_pulse=inference["seconds_per_pulse"],
                      milliseconds_per_signal=inference["milliseconds_per_signal"],
                      batch_inference=inference, csv_writing_and_validation_seconds=time.perf_counter() - serialization_start,
                      predictions_path=str(output), predictions_sha256=sha256(output))
        if reference.is_file():
            report["reference_comparison"] = compare_reference(reference, by_id, runtime.CLASSES)
            print("Differences from reference submission:", report["reference_comparison"]["different_predictions"], flush=True)
        elif args.reference:
            raise FileNotFoundError(f"Requested reference submission is missing: {reference}")
        print(f"Full submission throughput: {inference['milliseconds_per_signal']:.3f} ms/signal "
              f"(batch size {args.batch_size}, includes FFT)", flush=True)
        print("CPU submission:", output, flush=True)
        report["total_seconds"] = time.perf_counter() - start
        report_path = output.with_name(output.name + ".inference.json")
    report["total_time_scope"] = "Entire main() call through output/hash/reference checks; excludes interpreter startup and final JSON write"
    write_json(report_path, report)
    print("Timing/provenance report:", report_path, flush=True)
    print(f"Timing comparison uses the configured {args.target_ms:g} ms reference; "
          "single-signal percentiles and batch throughput are reported separately.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, KeyError, IndexError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
