"""RadarML Lite Track 2: learned complex Fourier templates.
SPDX-License-Identifier: GPL-3.0-or-later
Uses training pulses only; no pretrained weights or external waveforms.
AI assistance: OpenAI Codex. Data helpers: radarml_track2_runtime.py.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from functools import lru_cache
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
import radarml_track2_runtime as data

CLASSES, ALL_SNRS = data.CLASSES, data.ALL_SNRS
CANDIDATES = tuple(f"{mode}:{pool}" for mode in
                   ("coherent", "coherent_clip", "unknown_phase", "free_amplitude")
                   for pool in ("max", "mixture"))
DEFAULT_CONFIG = dict(seed=42, train_pulses_per_file=80, template_snr=30,
                      selection_pulses_per_file=4, batch_size=128, workers=2,
                      candidates=list(CANDIDATES), window=[2400, 14640])


def prepare_development(root, manifest, kind="auto"):
    root = Path(root)
    if kind == "auto":
        compact = any(data.canonical_filename(p.name).endswith("_recorded.npy")
                      for p in root.rglob("*.npy"))
        expanded = any(data.canonical_filename(p.name).endswith("_single_channel.npy")
                       for p in root.rglob("*.npy"))
        if compact == expanded:
            raise ValueError("Choose a DATA_ROOT with exactly one development format, or set DATA_FORMAT")
        kind = "compact" if compact else "expanded"
    if kind == "expanded":
        rows, infos = data.prepare_development(root, manifest)
        for name, info in infos.items():
            info.update(kind=kind)
        return rows, infos
    if kind != "compact":
        raise ValueError("DATA_FORMAT must be auto, compact or expanded")
    rows = data.load_manifest(manifest)
    lookup = data.build_file_lookup(root, "compact")
    infos, used = {}, set()
    for i, name in enumerate(sorted({r["file"] for r in rows}), 1):
        path = data.resolve_file(name, lookup, "compact")
        sidecar, meta = data.resolve_metadata(name, path, lookup, "compact")
        if data.canonical_filename(data.expanded_name_from_source(meta["source_file"])) != data.canonical_filename(name.replace("_recorded", "_single_channel")):
            raise ValueError(f"Source identity mismatch: {sidecar}")
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
        if (arr.shape != (1, 100, 33600) or arr.dtype != np.int16 or
                meta.get("format") != "radarml-recorded-v1" or
                meta.get("source_pulse_indices") != list(range(100)) or
                meta.get("channel") != "ADC0" or meta.get("sample_rate_hz") != 120_000_000 or
                meta.get("output_shape") != list(arr.shape) or meta.get("dtype") != "int16"):
            raise ValueError(f"Invalid compact layout or metadata: {path}")
        if path in used or data.lookup_sha256(path) != meta.get("output_sha256"):
            raise ValueError(f"Duplicate or corrupted compact recording: {path}")
        used.add(path)
        infos[name] = dict(path=str(path), kind=kind, source_file=meta["source_file"],
                           label=CLASSES.index(data.family(name)), snr_to_axis={})
        if i % 100 == 0 or i == 660:
            print(f"Verified {i}/660 compact recordings", flush=True)
    return rows, infos


def add_supplied_noise(raw, snr, source_file, pulse, seed=42):
    """Same generator, power windows, seed convention and clipping as the ZIP's expander."""
    z = raw[::2].astype(np.float64) + 1j * raw[1::2].astype(np.float64)
    noise = np.mean(np.abs(z[:2400]) ** 2)
    signal = np.mean(np.abs(z[2520:14280]) ** 2) - noise
    if not np.isfinite(signal) or signal <= 0:
        raise ValueError("Nonpositive source signal power")
    sigma = np.sqrt(max(0., signal / (10. ** (snr / 10.)) - noise) / 2.)
    key = f"{seed}_{source_file}_{snr}_{pulse}"
    rng = np.random.default_rng(int(hashlib.md5(key.encode()).hexdigest()[:8], 16))
    z = z + rng.normal(0, sigma, z.size) + 1j * rng.normal(0, sigma, z.size)
    out = np.empty(raw.shape, dtype=np.float64)
    out[::2], out[1::2] = z.real, z.imag
    return np.clip(out, -32768, 32767).astype(np.int16)


@lru_cache(maxsize=8)
def mapped_array(path):
    return np.load(path, mmap_mode="r", allow_pickle=False)


class DevelopmentDataset(Dataset):
    def __init__(self, examples, infos, selection_ids=(), noise_seed=42):
        self.examples, self.infos = examples, infos
        self.selection_ids, self.noise_seed = set(selection_ids), noise_seed

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        name, pulse, snr, label = self.examples[index]
        info = self.infos[name]
        arr = mapped_array(info["path"])
        if info["kind"] == "expanded":
            raw = arr[0, info["snr_to_axis"][snr], pulse]
        else:
            raw = add_supplied_noise(arr[0, pulse], snr, info["source_file"], pulse, self.noise_seed)
        return (torch.from_numpy(data.interleaved_to_channels(raw)), label, snr,
                (name, pulse) in self.selection_ids)


def selected_identities(rows, count):
    grouped = defaultdict(list)
    for r in rows:
        if r["split"] == "validation":
            grouped[r["file"]].append(r["pulse_index"])
    return {(name, idx) for name, indices in grouped.items() for idx in sorted(indices)[:count]}


def fit_bank(rows, infos, output, cfg):
    """MLE waveform estimates, with optional phase alignment derived from training only."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    grouped = defaultdict(list)
    for r in rows:
        if r["split"] == "train":
            grouped[r["file"]].append(r["pulse_index"])
    if set(grouped) != set(infos):
        raise ValueError("Training recordings disagree with data lookup")
    templates, labels, coherence, fit_seconds = [], [], [], 0.
    start = time.perf_counter()
    for i, name in enumerate(sorted(grouped)):
        indices = sorted(grouped[name])[:cfg["train_pulses_per_file"]]
        info = infos[name]
        cache = output / f"template_{i:04d}.pt"
        identity = dict(file=name, indices=indices, template_snr=cfg["template_snr"],
                        window=cfg["window"], label=info["label"])
        if cache.exists():
            item = torch.load(cache, map_location="cpu", weights_only=True)
            if item["identity"] != identity:
                raise ValueError("Resume cache settings changed; use a new RUN_NAME")
        else:
            tick = time.perf_counter()
            arr = mapped_array(info["path"])
            if info["kind"] == "compact":
                raw = arr[0, indices]
            else:
                raw = arr[0, info["snr_to_axis"][cfg["template_snr"]], indices]
            z = (raw[:, ::2].astype(np.float32) + 1j * raw[:, 1::2].astype(np.float32)) / 32768.
            left, right = cfg["window"]
            z[:, :left], z[:, right:] = 0, 0
            ref = z[0]
            angles = np.angle(z @ ref.conj())
            lock = float(abs(np.exp(1j * angles).mean()))
            # Preserve observed phase whenever stable. When unstable, align training
            # pulses; validation can select the phase-invariant likelihood instead.
            mean = z.mean(0) if lock >= 0.98 else (z * np.exp(-1j * angles)[:, None]).mean(0)
            item = dict(identity=identity, template=torch.from_numpy(mean.astype(np.complex64)),
                        phase_lock=lock, fit_seconds=time.perf_counter() - tick)
            temporary = cache.with_suffix(".partial")
            torch.save(item, temporary)
            temporary.replace(cache)
        templates.append(item["template"])
        labels.append(info["label"])
        coherence.append(item["phase_lock"])
        fit_seconds += item["fit_seconds"]
        if i % 50 == 0 or i + 1 == len(grouped):
            print(f"Training templates: {i + 1}/{len(grouped)}", flush=True)
    bank = dict(templates=torch.stack(templates), labels=torch.tensor(labels),
                classes=list(CLASSES), track="B", model_family="complex_fft_template_bank")
    report = dict(source_pulses=len(grouped) * cfg["train_pulses_per_file"],
                  templates=len(grouped), fit_seconds=fit_seconds,
                  current_session_fit_seconds=time.perf_counter() - start,
                  min_phase_lock=min(coherence), median_phase_lock=float(np.median(coherence)),
                  phase_lock=coherence,
                  training_snr="Original recorded pulses" if next(iter(infos.values()))["kind"] == "compact"
                               else cfg["template_snr"],
                  fit_method="Mean waveform; training-derived phase alignment only if phase lock < 0.98")
    return bank, report


class FourierBank(nn.Module):
    def __init__(self, checkpoint):
        super().__init__()
        if (checkpoint["classes"] != list(CLASSES) or checkpoint["track"] != "B" or
                checkpoint["model_family"] != "complex_fft_template_bank"):
            raise ValueError("Unexpected bank format")
        templates, labels = checkpoint["templates"], checkpoint["labels"]
        if (templates.ndim != 2 or templates.shape[1] != 16800 or
                templates.dtype != torch.complex64 or labels.shape != templates.shape[:1] or
                set(labels.tolist()) != set(range(7)) or not torch.isfinite(templates).all()):
            raise ValueError("Invalid template bank")
        energy = templates.abs().square().sum(1)
        if (energy <= 0).any():
            raise ValueError("Empty training template")
        self.register_buffer("kernels", torch.fft.fft(templates, norm="ortho").conj().T.contiguous())
        self.register_buffer("energy", energy)
        self.register_buffer("labels", labels)
        self.groups = [torch.where(labels == c)[0].tolist() for c in range(7)]

    def components(self, signals):
        # Full complex Fourier coefficients retain coherent information discarded
        # by log-power spectrograms. Parseval gives the template inner product.
        z = torch.complex(signals[:, 0].float(), signals[:, 1].float())
        dot = torch.fft.fft(z, norm="ortho") @ self.kernels
        noise = z[:, :2400].abs().square().mean(1, keepdim=True).clamp_min(1e-10)
        clipped = ((signals <= -1.) | (signals >= 32767. / 32768.)).float().mean((1, 2))[:, None]
        alpha = (1. - clipped).clamp(0.1, 1.)
        return dot, noise, alpha

    def scores(self, signals, candidates):
        dot, noise, alpha = self.components(signals)
        modes = {c.split(":")[0] for c in candidates}
        raw = {}
        if "coherent" in modes:
            raw["coherent"] = (2. * dot.real - self.energy) / noise
        if "coherent_clip" in modes:
            raw["coherent_clip"] = (2. * alpha * dot.real - alpha.square() * self.energy) / noise
        if "unknown_phase" in modes:
            magnitude = 2. * dot.abs() / noise
            # Stable phase-marginalized known-amplitude Gaussian likelihood.
            raw["unknown_phase"] = (magnitude + torch.special.i0e(magnitude).clamp_min(1e-30).log()
                                     - self.energy / noise)
        if "free_amplitude" in modes:
            raw["free_amplitude"] = dot.abs().square() / (self.energy * noise)
        result = {}
        for candidate in candidates:
            mode, pool = candidate.split(":")
            if mode not in raw or pool not in {"max", "mixture"}:
                raise ValueError(f"Unknown candidate: {candidate}")
            values = raw[mode]
            if pool == "max":
                cls = [values[:, idx].amax(1) for idx in self.groups]
            else:
                # Uniform class prior and uniform template prior WITHIN a class.
                # This prevents classes with more configurations gaining extra mass.
                cls = [torch.logsumexp(values[:, idx], 1) - np.log(len(idx)) for idx in self.groups]
            result[candidate] = torch.stack(cls, 1)
        return result

    def forward(self, signals, candidate="coherent_clip:mixture"):
        return self.scores(signals, [candidate])[candidate]


@torch.inference_mode()
def evaluate_candidates(model, loader, device, candidates, snrs=ALL_SNRS, collect_audit=False):
    model.eval()
    matrices = {c: {s: np.zeros((7, 7), dtype=np.int64) for s in snrs} for c in candidates}
    audit = {c: {s: np.zeros((7, 7), dtype=np.int64) for s in snrs} for c in candidates}
    data.synchronize(device)
    start, seen = time.perf_counter(), 0
    for step, (signals, labels, batch_snrs, in_selection) in enumerate(loader, 1):
        scores = model.scores(signals.to(device, non_blocking=True), candidates)
        labels, batch_snrs = labels.numpy(), batch_snrs.numpy()
        for c, logits in scores.items():
            prediction = logits.argmax(1).cpu().numpy()
            for s in snrs:
                mask = batch_snrs == s
                np.add.at(matrices[c][s], (labels[mask], prediction[mask]), 1)
                if collect_audit:
                    holdout = mask & ~in_selection.numpy()
                    np.add.at(audit[c][s], (labels[holdout], prediction[holdout]), 1)
        seen += len(signals)
        if step == 1 or step % 100 == 0:
            print(f"Validation: {seen:,}/{len(loader.dataset):,}", flush=True)
    data.synchronize(device)
    elapsed = time.perf_counter() - start
    reports = {}
    for c in candidates:
        reports[c] = data.report_from_matrices(matrices[c])
        reports[c].update(candidate=c, inference_seconds=elapsed, seconds_per_pulse=elapsed / seen,
                          timing_scope="Loader, I/Q conversion, FFT, template scoring, class reduction and metric accumulation")
    audit_reports = {c: data.report_from_matrices(audit[c]) for c in candidates} if collect_audit else {}
    return reports, audit_reports


def verify_bundle(bundle):
    bundle = Path(bundle).resolve()
    hashes = json.loads((bundle / "frozen.sha256.json").read_text())
    required = {"bank.pt", "config.json", "radarml_track2_bank.py", "radarml_track2_runtime.py",
                "validation_report.json", "requirements.txt"}
    if not required.issubset(hashes):
        raise ValueError("Incomplete Fourier-bank bundle")
    for name, expected in hashes.items():
        if Path(name).name != name or data.lookup_sha256(bundle / name) != expected:
            raise ValueError(f"Frozen artifact changed: {name}")
    if (data.lookup_sha256(Path(__file__)) != hashes["radarml_track2_bank.py"] or
            data.lookup_sha256(Path(data.__file__)) != hashes["radarml_track2_runtime.py"]):
        raise ValueError("Run the code inside the frozen bundle")
    return hashes


def train_and_freeze(root, manifest, run, config=None, kind="auto", prepared=None):
    cfg = json.loads(json.dumps(DEFAULT_CONFIG if config is None else config))
    if (not 1 <= cfg["train_pulses_per_file"] <= 80 or
            not 1 <= cfg["selection_pulses_per_file"] < 20 or
            cfg["batch_size"] < 1 or cfg["workers"] < 0 or
            not cfg["candidates"] or not set(cfg["candidates"]).issubset(CANDIDATES) or
            not 0 <= cfg["window"][0] < cfg["window"][1] <= 16800):
        raise ValueError("Invalid training configuration")
    run, manifest = Path(run), Path(manifest)
    frozen = run / "frozen"
    if (frozen / "frozen.sha256.json").exists():
        verify_bundle(frozen)
        print("Already frozen; reusing completed run:", frozen, flush=True)
        return json.loads((frozen / "validation_report.json").read_text())
    run.mkdir(parents=True, exist_ok=True)
    state = dict(config=cfg, manifest_sha256=data.lookup_sha256(manifest), data_format=kind)
    state_path = run / "run_settings.json"
    if state_path.exists() and json.loads(state_path.read_text()) != state:
        raise ValueError("Run settings changed; choose a new RUN_NAME")
    data.write_json(state_path, state)
    data.seed_everything(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tick = time.perf_counter()
    rows, infos = prepared if prepared is not None else prepare_development(root, manifest, kind)
    setup_seconds = time.perf_counter() - tick
    # Assert separation before ANY prototype estimation.
    training_ids = {(r["file"], r["pulse_index"]) for r in rows if r["split"] == "train"}
    validation_ids = {(r["file"], r["pulse_index"]) for r in rows if r["split"] == "validation"}
    if training_ids & validation_ids:
        raise ValueError("Source-pulse leakage")
    # Bind resumable training caches to their dataset identity (not Kaggle mount paths).
    identity = {}
    for name, info in infos.items():
        meta_path = Path(info["path"]).with_suffix(".json")
        # resolve_metadata has already checked source/array pairings in prepare_development;
        # the array hash also handles Kaggle's renamed FSK files correctly.
        identity[name] = data.lookup_sha256(Path(info["path"]))
    identity_path = run / "training_data_hashes.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("Training dataset changed since the cached fit")
    data.write_json(identity_path, identity)
    bank, fit_report = fit_bank(rows, infos, run / "training_cache", cfg)
    model = FourierBank(bank).to(device).eval()
    selection_ids = selected_identities(rows, cfg["selection_pulses_per_file"])
    examples = data.build_examples(rows, infos, "validation", cfg["selection_pulses_per_file"], ALL_SNRS)
    loader = data.make_loader(DevelopmentDataset(examples, infos), cfg["batch_size"], cfg["workers"], cfg["seed"])
    reports, _ = evaluate_candidates(model, loader, device, cfg["candidates"])
    best = cfg["candidates"][0]
    for candidate in cfg["candidates"][1:]:
        score, old = reports[candidate], reports[best]
        delta = score["mean_balanced_accuracy"] - old["mean_balanced_accuracy"]
        if delta > 1e-6 or (abs(delta) <= 1e-6 and score["lowest_three_mean_balanced_accuracy"] > old["lowest_three_mean_balanced_accuracy"] + 1e-6):
            best = candidate
    data.write_json(run / "selection_candidates.json", reports)
    print("Selected on validation only:", best, reports[best]["mean_balanced_accuracy"], flush=True)
    examples = data.build_examples(rows, infos, "validation", 20, ALL_SNRS)
    loader = data.make_loader(DevelopmentDataset(examples, infos, selection_ids), cfg["batch_size"], cfg["workers"], cfg["seed"])
    full, audit = evaluate_candidates(model, loader, device, [best], collect_audit=True)
    report = full[best]
    report.update(full_published_validation=True, pulses_per_recording=20,
                  selection_pulses_per_recording=cfg["selection_pulses_per_file"])
    audit[best].update(candidate=best, excludes_selection_source_pulses=True)
    cfg.update(track="B", track_number=2, model_family="complex_fft_template_bank", classes=list(CLASSES),
               selected_candidate=best, inference_precision="float32", inference_augmentation="none",
               manifest_sha256=data.MANIFEST_SHA256, hardware=data.hardware_info(device),
               actual_data_format=next(iter(infos.values()))["kind"], setup_seconds=setup_seconds,
               model_parameters=int(bank["templates"].numel() * 2),
               external_training_data=False, pretrained=False)
    frozen.mkdir(parents=True, exist_ok=True)
    torch.save(bank, frozen / "bank.pt")
    cfg["model_bytes"] = (frozen / "bank.pt").stat().st_size
    data.write_json(frozen / "config.json", cfg)
    data.write_json(frozen / "fit_report.json", fit_report)
    data.write_json(frozen / "selection_candidates.json", reports)
    data.write_json(frozen / "selection_validation.json", reports[best])
    data.write_json(frozen / "validation_report.json", report)
    data.write_json(frozen / "audit_validation.json", audit[best])
    shutil.copy2(manifest, frozen / "split_manifest.csv")
    shutil.copy2(identity_path, frozen / "training_data_hashes.json")
    for source in (Path(__file__), Path(data.__file__)):
        target = frozen / source.name
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
    (frozen / "requirements.txt").write_text(f"numpy=={np.__version__}\ntorch=={str(torch.__version__).split('+')[0]}\n")
    (frozen / "inference_command.txt").write_text(
        "python radarml_track2_bank.py infer --bundle . --blind-root /path/to/participants --output /path/to/submission.csv\n")
    summary = f"""RadarML Lite Track 2 / B: learned complex Fourier template classifier

Training: {fit_report['source_pulses']:,} published training pulses only, seed {cfg['seed']}.
Template source: {fit_report['training_snr']}. {fit_report['templates']} complex mean waveforms,
with phase alignment only for training sets whose phase lock is below 0.98.
Fixed acquisition window: samples {cfg['window']}. No filename-derived cropping.
Inference: unitary complex FFT, training-template inner products, per-pulse noise
normalization; selected likelihood/class reduction: {best}.
The coherent_clip score attenuates the expected template by 1 minus the observed
per-pulse clipping fraction. Mixture pooling uses equal class priors.
The eight alternatives are fixed analytic scoring rules, not eight neural networks.
Only validation chooses the scoring rule; learned template parameters use train only.

Selection: first {cfg['selection_pulses_per_file']} published validation indices per recording.
Full validation ({report['samples']:,}): {report['mean_balanced_accuracy']:.8f} mean balanced accuracy.
Separate validation audit excluding selection pulses: {audit[best]['mean_balanced_accuracy']:.8f}.
Fit time: {fit_report['fit_seconds']:.2f} s, excluding dataset verification and validation.
Validation inference including FFT: {report['seconds_per_pulse']:.6f} s/pulse.
Model size: {cfg['model_bytes']:,} bytes. Hardware and versions: config.json.

Limitations: specifically evaluates known configurations with repeatable acquisition
phase and timing. Accuracy on new configurations, phase rotations, time offsets or
hardware is not established. Clipping correction is an approximation. Validation
is development evidence, not a blind score. No 0.995 leaderboard guarantee is made.
This is a new experiment; if the event's model freeze has passed, organizer approval
is needed to replace the previously frozen entry. No blind signals enter this notebook's training.

Attribution: Ritchie, White and Hosford (2025), RadarML, https://doi.org/10.5522/04/30752767.v1.
Organizer helpers adapted under GPL-3.0. AI coding assistance: OpenAI Codex.
Method reference: https://www.mathworks.com/help/signal/ug/radar-pulse-compression.html
Implementation reference: https://docs.pytorch.org/docs/stable/generated/torch.fft.fft.html
"""
    (frozen / "method_summary.txt").write_text(summary, encoding="utf-8")
    hashes = {p.name: data.lookup_sha256(p) for p in sorted(frozen.iterdir()) if p.is_file() and p.name != "frozen.sha256.json"}
    data.write_json(frozen / "frozen.sha256.json", hashes)
    print("FROZEN BUNDLE:", frozen, "| full validation:", report["mean_balanced_accuracy"], flush=True)
    return report


def predict_blind(bundle, blind_root, output, batch_size=128, workers=2):
    bundle, output = Path(bundle).resolve(), Path(output).resolve()
    hashes = verify_bundle(bundle)
    if output.is_relative_to(bundle):
        raise ValueError("Keep generated predictions outside the frozen bundle")
    cfg = json.loads((bundle / "config.json").read_text())
    if cfg["selected_candidate"] not in CANDIDATES or cfg["inference_precision"] != "float32":
        raise ValueError("Invalid frozen inference settings")
    data.seed_everything(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(bundle / "bank.pt", map_location="cpu", weights_only=True)
    model = FourierBank(checkpoint).to(device).eval()
    del checkpoint
    root, rows, template_ids = data.read_blind_index(blind_root)
    rows.sort(key=lambda r: (r["file"], r["row"]))
    loader = data.make_loader(data.BlindDataset(root, rows), batch_size, workers, cfg["seed"])
    predictions = []
    data.synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode():
        for step, signals in enumerate(loader, 1):
            logits = model(signals.to(device, non_blocking=True), cfg["selected_candidate"])
            predictions.extend(logits.argmax(1).cpu().tolist())
            if step == 1 or step % 100 == 0:
                print(f"Blind inference: {len(predictions):,}/{len(rows):,}", flush=True)
    data.synchronize(device)
    elapsed = time.perf_counter() - start
    if len(predictions) != len(rows):
        raise RuntimeError("Incomplete inference")
    by_id = {r["sample_id"]: CLASSES[p] for r, p in zip(rows, predictions)}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".partial")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample_id", "predicted_label"])
        writer.writerows((sid, by_id[sid]) for sid in template_ids)
    with temporary.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        saved = list(reader)
        if (reader.fieldnames != ["sample_id", "predicted_label"] or
                [r["sample_id"] for r in saved] != template_ids or
                any(r["predicted_label"] not in CLASSES for r in saved)):
            raise ValueError("Serialized submission is invalid")
    temporary.replace(output)
    report = dict(samples=len(rows), inference_seconds=elapsed, seconds_per_pulse=elapsed / len(rows),
                  precision="float32", selected_candidate=cfg["selected_candidate"],
                  timing_scope="Cold loader startup, disk reads, transfers, FFT, template likelihood and prediction collection",
                  hardware=data.hardware_info(device), frozen_artifact_hashes=hashes,
                  predictions_sha256=data.lookup_sha256(output), batch_size=batch_size, workers=workers)
    data.write_json(output.with_name(output.name + ".inference.json"), report)
    print("Submission:", output, flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    infer = commands.add_parser("infer")
    infer.add_argument("--bundle", required=True)
    infer.add_argument("--blind-root", required=True)
    infer.add_argument("--output", required=True)
    infer.add_argument("--batch-size", type=int, default=128)
    infer.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0:
        parser.error("batch-size must be positive and workers nonnegative")
    predict_blind(args.bundle, args.blind_root, args.output, args.batch_size, args.workers)


if __name__ == "__main__":
    main()
