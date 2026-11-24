"""Development-only trainer for the compact complex-STFT neural classifier.
SPDX-License-Identifier: GPL-3.0-or-later. AI coding assistance: OpenAI Codex.
This module and its training cache are NOT required or exported for inference.
"""
from collections import defaultdict
import copy
import csv
from functools import lru_cache
import json
import math
from pathlib import Path
import shutil
import time
import zipfile

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import radarml_track2_runtime as helpers
import radarml_track2_neural_infer as infer

DEFAULT_CONFIG = dict(seed=42, epochs=80, samples_per_epoch=105600, batch_size=256, workers=2,
                      learning_rate=.001, weight_decay=.001, label_smoothing=.005,
                      clean_loss_weight=.25, feature_consistency_weight=2.,
                      ema_decay=.999, warmup_epochs=2, patience=18, max_run_hours=10.5,
                      train_pulses_per_file=80, selection_pulses_per_file=4,
                      train_snrs=list(helpers.ALL_SNRS), hard_snr_weight=2.,
                      model_kwargs=dict(infer.MODEL_KWARGS))


def find_data_root(explicit=None, kind="auto", search_root=Path("/kaggle/input")):
    root = Path(explicit if explicit is not None else search_root).expanduser().resolve()
    if kind not in {"auto", "compact", "expanded"} or not root.is_dir():
        raise ValueError(f"Set an existing DATA_ROOT and valid DATA_FORMAT; searched {root}")
    counts = defaultdict(int)
    for path in root.rglob("*.npy"):
        name = helpers.canonical_filename(path.name)
        fmt = ("compact" if name.endswith("_recorded.npy") else
               "expanded" if name.endswith("_single_channel.npy") else None)
        if fmt is None or (kind != "auto" and kind != fmt):
            continue
        for parent in path.parents:
            if not parent.is_relative_to(root):
                break
            counts[(parent, fmt)] += 1
    complete = [(p, f) for (p, f), count in counts.items() if count == 660]
    complete = [(p, f) for p, f in complete if not any(q != p and q.is_relative_to(p) and g == f for q, g in complete)]
    if len(complete) != 1:
        choices = "\n".join(f"{p} ({f})" for p, f in complete)
        folders = "\n".join(str(p) for p in sorted(root.iterdir()) if p.is_dir())
        raise ValueError(f"Expected one complete 660-recording development dataset under {root}; found {len(complete)}. "
                         f"Set DATA_ROOT/DATA_FORMAT explicitly if needed.\nCandidates:\n{choices}\nFolders:\n{folders}")
    return complete[0][0]


def prepare(root, manifest, kind):
    root = Path(root)
    if kind == "auto":
        compact = any(helpers.canonical_filename(p.name).endswith("_recorded.npy") for p in root.rglob("*.npy"))
        expanded = any(helpers.canonical_filename(p.name).endswith("_single_channel.npy") for p in root.rglob("*.npy"))
        if compact == expanded:
            raise ValueError("Set DATA_FORMAT to choose one development format")
        kind = "compact" if compact else "expanded"
    if kind == "expanded":
        rows, infos = helpers.prepare_development(root, manifest)
        for info in infos.values():
            info["kind"] = kind
        return rows, infos
    if kind != "compact":
        raise ValueError("DATA_FORMAT must be auto, compact or expanded")
    rows = helpers.load_manifest(manifest)
    lookup = helpers.build_file_lookup(root, "compact")
    infos, used = {}, set()
    for index, name in enumerate(sorted({r["file"] for r in rows}), 1):
        path = helpers.resolve_file(name, lookup, "compact")
        sidecar, meta = helpers.resolve_metadata(name, path, lookup, "compact")
        expected = name.replace("_recorded.npy", "_single_channel.npy")
        if helpers.canonical_filename(helpers.expanded_name_from_source(meta["source_file"])) != helpers.canonical_filename(expected):
            raise ValueError(f"Wrong source identity: {sidecar}")
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
        if (arr.shape != (1, 100, 33600) or arr.dtype != np.int16 or
                meta.get("format") != "radarml-recorded-v1" or meta.get("source_pulse_indices") != list(range(100)) or
                meta.get("sample_rate_hz") != 120000000 or meta.get("channel") != "ADC0" or
                path in used or helpers.lookup_sha256(path) != meta.get("output_sha256")):
            raise ValueError(f"Invalid or corrupted compact recording: {path}")
        used.add(path)
        infos[name] = dict(path=str(path), label=infer.CLASSES.index(helpers.family(name)), kind=kind,
                           source_file=meta["source_file"], snr_to_axis={})
        if index % 100 == 0 or index == 660:
            print(f"Verified compact data: {index}/660", flush=True)
    return rows, infos


def atomic_save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    torch.save(value, temporary)
    temporary.replace(path)


def build_training_cache(rows, infos, directory, cfg, manifest_hash):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    groups = defaultdict(list)
    validation_ids = set()
    for row in rows:
        if row["split"] == "train":
            groups[row["file"]].append(row["pulse_index"])
        else:
            validation_ids.add((row["file"], row["pulse_index"]))
    selected = [[name, sorted(indices)[:cfg["train_pulses_per_file"]], infos[name]["label"]]
                for name, indices in sorted(groups.items())]
    if any((name, idx) in validation_ids for name, indices, _ in selected for idx in indices):
        raise ValueError("Validation source pulse entered training cache")
    if set(groups) != set(infos):
        raise ValueError("Training file identities do not match the supplied data")
    count = sum(len(indices) for _, indices, _ in selected)
    identity = dict(version=1, manifest_sha256=manifest_hash, source_format=next(iter(infos.values()))["kind"],
                    base_snr=30, records=selected, count=count)
    identity_path = directory / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("Training cache identity changed; use a new RUN_NAME")
    helpers.write_json(identity_path, identity)
    raw_path, labels_path = directory / "train_iq.npy", directory / "labels.npy"
    if raw_path.exists() and labels_path.exists():
        cache = np.load(raw_path, mmap_mode="r", allow_pickle=False)
        labels = np.load(labels_path, allow_pickle=False)
        if cache.shape != (count, 33600) or cache.dtype != np.int16 or labels.shape != (count,):
            raise ValueError("Invalid completed training cache")
        print(f"Reusing training-only cache: {count:,} source pulses", flush=True)
        return raw_path, labels_path
    temporary = directory / "train_iq.partial.npy"
    progress_path = directory / "progress.json"
    completed = json.loads(progress_path.read_text())["completed"] if progress_path.exists() and temporary.exists() else 0
    cache = (np.load(temporary, mmap_mode="r+", allow_pickle=False) if completed else
             np.lib.format.open_memmap(temporary, mode="w+", dtype=np.int16, shape=(count, 33600)))
    labels, offset = np.empty(count, np.int64), 0
    for index, (name, indices, label) in enumerate(selected):
        stop = offset + len(indices)
        labels[offset:stop] = label
        if index >= completed:
            info = infos[name]
            arr = np.load(info["path"], mmap_mode="r", allow_pickle=False)
            cache[offset:stop] = (arr[0, indices] if info["kind"] == "compact" else
                                  arr[0, info["snr_to_axis"][30], indices])
            cache.flush()
            helpers.write_json(progress_path, dict(completed=index + 1))
        offset = stop
        if index % 50 == 0 or index + 1 == len(selected):
            print(f"Training-only cache: {index + 1}/{len(selected)} recordings", flush=True)
    cache.flush()
    del cache
    temporary.replace(raw_path)
    np.save(labels_path, labels, allow_pickle=False)
    print(f"Cache size: {raw_path.stat().st_size / 1e9:.2f} GB (development only; never exported)", flush=True)
    return raw_path, labels_path


class CachedTraining(Dataset):
    def __init__(self, path, labels):
        self.path, self.labels = str(path), np.load(labels, allow_pickle=False)
        self.array = None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        if self.array is None:
            self.array = np.load(self.path, mmap_mode="r", allow_pickle=False)
        raw = self.array[index]
        iq = np.stack((raw[::2], raw[1::2])).astype(np.float32) / 32768.
        return torch.from_numpy(iq), int(self.labels[index])


class NoiseAugment(nn.Module):
    """Noise/clipping derived only from each training pulse, across every requested SNR."""
    def __init__(self, snrs, hard_weight=2.):
        super().__init__()
        self.register_buffer("snrs", torch.tensor(snrs, dtype=torch.float32))
        self.register_buffer("probabilities", torch.tensor([hard_weight if s <= -20 else 1. for s in snrs]))

    def forward(self, base, easy=False):
        with torch.autocast(device_type=base.device.type, enabled=False):
            base = base.float()
            power = base.square().sum(1)
            noise = power[:, :2400].mean(1)
            signal = (power[:, 2520:14280].mean(1) - noise).clamp_min(1e-12)
            probabilities = self.probabilities.clone()
            if easy:
                probabilities[self.snrs < -15] = 0
            choice = torch.multinomial(probabilities, len(base), replacement=True)
            snr = self.snrs[choice]
            added = (signal / torch.pow(10., snr / 10.) - noise).clamp_min(0.)
            noisy = base + torch.randn_like(base) * (added / 2.).sqrt()[:, None, None]
            # Match the ADC cap and int16 truncation convention, after independent
            # augmentation noise. Requested SNR stays a target, not a model input.
            noisy = torch.trunc(noisy.clamp(-1., 32767. / 32768.) * 32768.) / 32768.
            return noisy, snr


@lru_cache(maxsize=8)
def mapped(path):
    return np.load(path, mmap_mode="r", allow_pickle=False)


def supplied_noise(raw, snr, source, pulse):
    # Validation-only reconstruction, bitwise equivalent to the supplied compact
    # expander (seed 42), not the independently generated training augmentation.
    import hashlib
    z = raw[::2].astype(np.float64) + 1j * raw[1::2].astype(np.float64)
    noise = np.mean(abs(z[:2400]) ** 2)
    signal = np.mean(abs(z[2520:14280]) ** 2) - noise
    if signal <= 0:
        raise ValueError("Nonpositive source signal power")
    sigma = np.sqrt(max(0., signal / 10. ** (snr / 10.) - noise) / 2.)
    seed = int(hashlib.md5(f"42_{source}_{snr}_{pulse}".encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    z = z + rng.normal(0, sigma, len(z)) + 1j * rng.normal(0, sigma, len(z))
    out = np.empty(raw.shape, np.float64)
    out[::2], out[1::2] = z.real, z.imag
    return np.clip(out, -32768, 32767).astype(np.int16)


class Validation(Dataset):
    def __init__(self, examples, infos, selection_ids=()):
        self.examples, self.infos, self.selection_ids = examples, infos, set(selection_ids)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        name, pulse, snr, label = self.examples[index]
        info = self.infos[name]
        arr = mapped(info["path"])
        raw = (arr[0, info["snr_to_axis"][snr], pulse] if info["kind"] == "expanded" else
               supplied_noise(arr[0, pulse], snr, info["source_file"], pulse))
        return torch.from_numpy(helpers.interleaved_to_channels(raw)), label, snr, (name, pulse) in self.selection_ids


def loader(dataset, batch_size, workers, seed, sampler=None, cuda=False):
    options = dict(batch_size=batch_size, num_workers=workers, shuffle=False, sampler=sampler,
                   pin_memory=cuda, generator=torch.Generator().manual_seed(seed), worker_init_fn=helpers.seed_worker)
    if workers:
        options.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(dataset, **options)


@torch.inference_mode()
def evaluate(models, data_loader, device, audit=False):
    matrices = {name: {s: np.zeros((7, 7), np.int64) for s in helpers.ALL_SNRS} for name in models}
    holdouts = {name: {s: np.zeros((7, 7), np.int64) for s in helpers.ALL_SNRS} for name in models}
    for model in models.values():
        model.eval()
    infer.synchronize(device)
    start, seen = time.perf_counter(), 0
    for step, (signals, labels, snrs, selected) in enumerate(data_loader, 1):
        signals = signals.to(device, non_blocking=True)
        truth, snrs = labels.numpy(), snrs.numpy()
        for name, model in models.items():
            predictions = model(signals).argmax(1).cpu().numpy()
            for snr in helpers.ALL_SNRS:
                mask = snrs == snr
                np.add.at(matrices[name][snr], (truth[mask], predictions[mask]), 1)
                if audit:
                    other = mask & ~selected.numpy()
                    np.add.at(holdouts[name][snr], (truth[other], predictions[other]), 1)
        seen += len(signals)
        if step == 1 or step % 100 == 0:
            print(f"Validation: {seen:,}/{len(data_loader.dataset):,}", flush=True)
    infer.synchronize(device)
    seconds = time.perf_counter() - start
    reports = {name: helpers.report_from_matrices(value) for name, value in matrices.items()}
    for report in reports.values():
        report.update(inference_seconds=seconds, seconds_per_pulse=seconds / seen,
                      timing_scope="Data loading, transfer, complete STFT, neural inference and metrics; FP32")
    return reports, ({name: helpers.report_from_matrices(value) for name, value in holdouts.items()} if audit else {})


def cpu_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def checkpoint_for_export(state, cfg, epoch, variant):
    return dict(state_dict=state, model_kwargs=cfg["model_kwargs"], classes=list(infer.CLASSES), track="B",
                model_family="compact_complex_stft_neural", selected_epoch=epoch, selected_variant=variant,
                seed=cfg["seed"], sample_rate_hz=120000000)


def train(root, manifest, run_dir, config=None, kind="auto", prepared=None):
    cfg = json.loads(json.dumps(DEFAULT_CONFIG if config is None else config))
    if (not 1 <= cfg["train_pulses_per_file"] <= 80 or not 1 <= cfg["selection_pulses_per_file"] < 20 or
            cfg["epochs"] < 1 or cfg["samples_per_epoch"] < 7 or cfg["batch_size"] < 1 or cfg["workers"] < 0 or
            set(cfg["train_snrs"]) != set(helpers.ALL_SNRS) or cfg["max_run_hours"] <= 0):
        raise ValueError("Invalid training settings; all 13 training SNRs must remain available")
    run = Path(run_dir)
    run.mkdir(parents=True, exist_ok=True)
    config_path = run / "run_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != cfg:
        raise ValueError("Configuration changed; choose a new RUN_NAME")
    helpers.write_json(config_path, cfg)
    export = run / "export"
    if (export / "complete.json").exists():
        old = json.loads((export / "complete.json").read_text())
        if infer.sha256(export / "model.pt") != old["model_sha256"]:
            raise ValueError("Saved model changed after export")
        print("Completed model already exported:", export, flush=True)
        return json.loads((export / "validation_report.json").read_text())
    start = time.perf_counter()
    helpers.seed_everything(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = helpers.find_manifest(root, manifest)
    rows, infos = prepared if prepared is not None else prepare(root, manifest, kind)
    cache_raw, cache_labels = build_training_cache(rows, infos, run / "training_cache", cfg, helpers.lookup_sha256(manifest))
    training = CachedTraining(cache_raw, cache_labels)
    labels = training.labels
    counts = np.bincount(labels, minlength=7)
    if (counts == 0).any():
        raise ValueError("Every class must be present in training")
    grouped = defaultdict(list)
    for row in rows:
        if row["split"] == "validation":
            grouped[row["file"]].append(row["pulse_index"])
    selection_ids = {(name, idx) for name, values in grouped.items()
                     for idx in sorted(values)[:cfg["selection_pulses_per_file"]]}
    selection_examples = helpers.build_examples(rows, infos, "validation", cfg["selection_pulses_per_file"], helpers.ALL_SNRS)
    select_loader = loader(Validation(selection_examples, infos), cfg["batch_size"], cfg["workers"], cfg["seed"], cuda=device.type == "cuda")
    model = infer.CompactSTFTClassifier(**cfg["model_kwargs"]).to(device)
    ema = copy.deepcopy(model).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    augment = NoiseAugment(cfg["train_snrs"], cfg["hard_snr_weight"]).to(device)
    history, epoch_start, updates, best_score, best_low, stale, fit_seconds = [], 1, 0, -1., -1., 0, 0.
    resume_path = run / "resume.pt"
    if resume_path.exists():
        saved = torch.load(resume_path, map_location="cpu", weights_only=True)
        if saved["config"] != cfg:
            raise ValueError("Resume configuration differs")
        model.load_state_dict(saved["model"])
        ema.load_state_dict(saved["ema"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        epoch_start, updates = saved["next_epoch"], saved["updates"]
        history, best_score, best_low, stale, fit_seconds = saved["history"], saved["best_score"], saved["best_low"], saved["stale"], saved["fit_seconds"]
        torch.set_rng_state(saved["rng_state"])
        if device.type == "cuda" and saved["cuda_rng"]:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        print("Resuming at epoch", epoch_start, flush=True)
    print("Trainable parameters:", sum(p.numel() for p in model.parameters()), flush=True)
    print("Training source pulses:", len(training), "| SNRs:", cfg["train_snrs"], flush=True)
    setup_seconds = time.perf_counter() - start
    stop_reason = "epochs_completed"
    for epoch in range(epoch_start, cfg["epochs"] + 1):
        if time.perf_counter() - start >= cfg["max_run_hours"] * 3600:
            stop_reason = "wall_time_budget"
            break
        sampler = WeightedRandomSampler(torch.as_tensor(1. / counts[labels], dtype=torch.double),
                                        cfg["samples_per_epoch"], replacement=True,
                                        generator=torch.Generator().manual_seed(cfg["seed"] + epoch))
        train_loader = loader(training, cfg["batch_size"], cfg["workers"], cfg["seed"] + epoch,
                              sampler, device.type == "cuda")
        model.train()
        tick, seen, loss_total, batches = time.perf_counter(), 0, 0., 0
        snr_counts = {s: 0 for s in cfg["train_snrs"]}
        for step, (base, target) in enumerate(train_loader, 1):
            base, target = base.to(device, non_blocking=True), target.to(device, non_blocking=True)
            noisy, requested_snr = augment(base, easy=epoch <= cfg["warmup_epochs"])
            progress = (epoch - 1 + (step - 1) / len(train_loader)) / cfg["epochs"]
            ramp = min(1., (epoch - 1 + step / len(train_loader)) / max(1, cfg["warmup_epochs"]))
            lr = cfg["learning_rate"] * ramp * (.05 + .95 * (1. + math.cos(math.pi * progress)) / 2.)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                logits, features = model(noisy, return_embedding=True)
                clean_logits, clean_features = model(base, return_embedding=True)
                loss = F.cross_entropy(logits, target, label_smoothing=cfg["label_smoothing"])
                loss = loss + cfg["clean_loss_weight"] * F.cross_entropy(clean_logits, target)
                consistency = F.mse_loss(F.normalize(features[:, :-4].float(), dim=1),
                                         F.normalize(clean_features[:, :-4].detach().float(), dim=1))
                loss = loss + cfg["feature_consistency_weight"] * consistency
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss; checkpoint from the preceding epoch remains intact")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            scaler.step(optimizer)
            scaler.update()
            updates += 1
            decay = min(cfg["ema_decay"], 1. - 1. / (updates + 1.))
            with torch.no_grad():
                averaged_state = ema.state_dict()
                for key, value in model.state_dict().items():
                    average = averaged_state[key]
                    if value.is_floating_point():
                        average.mul_(decay).add_(value, alpha=1. - decay)
                    else:
                        average.copy_(value)
            seen += len(base)
            loss_total += float(loss.detach()) * len(base)
            batches += 1
            values, frequencies = torch.unique(requested_snr, return_counts=True)
            for value, frequency in zip(values.cpu().tolist(), frequencies.cpu().tolist()):
                snr_counts[int(value)] += frequency
            if step == 1 or step % 100 == 0:
                print(f"Epoch {epoch}/{cfg['epochs']} batch {step}/{len(train_loader)} loss {loss_total / seen:.5f}", flush=True)
            if time.perf_counter() - start >= cfg["max_run_hours"] * 3600:
                stop_reason = "wall_time_budget"
                break
        infer.synchronize(device)
        fit_seconds += time.perf_counter() - tick
        # Release loader workers before running validation. Cache remains memory-mapped.
        del train_loader
        reports, _ = evaluate({"raw": model, "ema": ema}, select_loader, device)
        better_this_epoch = False
        for variant in ("raw", "ema"):
            report = reports[variant]
            score, low = report["mean_balanced_accuracy"], report["lowest_three_mean_balanced_accuracy"]
            better = score > best_score + 1e-6 or (abs(score - best_score) <= 1e-6 and low > best_low + 1e-6)
            if better:
                best_score, best_low = score, low
                best = ema if variant == "ema" else model
                atomic_save(run / "best.pt", checkpoint_for_export(cpu_state(best), cfg, epoch, variant))
                helpers.write_json(run / "best_selection_validation.json", report)
                better_this_epoch = True
        stale = 0 if better_this_epoch else stale + 1
        record = dict(epoch=epoch, train_loss=loss_total / seen, draws=seen, batches=batches,
                      requested_snr_counts={str(k): v for k, v in snr_counts.items()}, learning_rate=lr,
                      raw_selection_score=reports["raw"]["mean_balanced_accuracy"],
                      ema_selection_score=reports["ema"]["mean_balanced_accuracy"], best_score=best_score)
        history.append(record)
        helpers.write_json(run / "history.json", history)
        atomic_save(resume_path, dict(model=cpu_state(model), ema=cpu_state(ema), optimizer=optimizer.state_dict(),
                                      scaler=scaler.state_dict(), config=cfg, next_epoch=epoch + 1, updates=updates,
                                      history=history, best_score=best_score, best_low=best_low, stale=stale,
                                      fit_seconds=fit_seconds, rng_state=torch.get_rng_state(),
                                      cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else []))
        print(json.dumps(record), flush=True)
        if stop_reason == "wall_time_budget":
            break
        if epoch > cfg["warmup_epochs"] and stale >= cfg["patience"]:
            stop_reason = "early_stopping"
            break
    if not (run / "best.pt").exists():
        raise RuntimeError("No epoch completed; rerun with more wall-time budget or prepare/cache data first")
    selected, selected_checkpoint = infer.load_model(run / "best.pt", device)
    full_examples = helpers.build_examples(rows, infos, "validation", 20, helpers.ALL_SNRS)
    full_loader = loader(Validation(full_examples, infos, selection_ids), cfg["batch_size"], cfg["workers"], cfg["seed"], cuda=device.type == "cuda")
    full, audit = evaluate({"selected": selected}, full_loader, device, audit=True)
    report, audit_report = full["selected"], audit["selected"]
    report.update(full_published_validation=True, selected_epoch=selected_checkpoint["selected_epoch"],
                  selected_variant=selected_checkpoint["selected_variant"])
    audit_report.update(excludes_selection_source_pulses=True)
    export.mkdir(parents=True, exist_ok=True)
    shutil.copy2(run / "best.pt", export / "model.pt")
    shutil.copy2(Path(infer.__file__), export / "radarml_track2_neural_infer.py")
    helpers.write_json(export / "validation_report.json", report)
    helpers.write_json(export / "audit_validation.json", audit_report)
    helpers.write_json(export / "training_config.json", cfg)
    shutil.copy2(run / "history.json", export / "history.json")
    shutil.copy2(run / "best_selection_validation.json", export / "selection_validation.json")
    (export / "requirements.txt").write_text(f"numpy=={np.__version__}\ntorch=={str(torch.__version__).split('+')[0]}\n")
    summary = dict(model_family="compact_complex_stft_neural", classes=list(infer.CLASSES), track="B",
                   parameters=sum(p.numel() for p in selected.parameters()), model_bytes=(export / "model.pt").stat().st_size,
                   training_source_pulses=len(training), eligible_training_snrs=cfg["train_snrs"],
                   training_draws=sum(row["draws"] for row in history), epochs_completed=len(history),
                   fit_seconds=fit_seconds, setup_seconds_this_session=setup_seconds, stop_reason=stop_reason,
                   hardware=infer.hardware(device), manifest_sha256=helpers.lookup_sha256(manifest),
                   inference_inputs="model.pt and anonymous participant package only", external_data=False, pretrained=False,
                   notes="No training waveforms, template bank, teacher or saved predictions are used at inference. "
                         "All 13 SNRs are available in training after the two warm-up epochs. "
                         "The 0.996 goal has not been guaranteed. Full/audit scores are measured after training.")
    helpers.write_json(export / "training_summary.json", summary)
    (export / "method_summary.txt").write_text(
        "Track 2: compact complex-STFT neural classifier, trained from random initialization.\n"
        "Explicit two-sided complex STFT: 256 frequency bins x 66 time frames, 256-sample rectangular windows, "
        "non-overlapping blocks and right zero padding. Phase-sensitive learned projection plus a small power-STFT CNN.\n"
        "Gradient training: class-balanced source-pulse sampling, all 13 noise levels after warm-up, "
        "paired clean/noisy cross-entropy and feature consistency; raw/EMA checkpoint selection on validation only.\n"
        "Training cache contains training identities only and is excluded from the export. No validation pulse fits weights.\n"
        f"Full validation balanced accuracy: {report['mean_balanced_accuracy']:.8f}; separate audit: {audit_report['mean_balanced_accuracy']:.8f}.\n"
        "Training, hardware and SNR details: training_summary.json and training_config.json.\n"
        "Inference needs model.pt, radarml_track2_neural_infer.py and the blind package only.\n"
        "Limitations: validation covers known waveform configurations; blind accuracy, unseen-configuration "
        "generalization and official runtime eligibility are not guaranteed.\n"
        "Attribution: Ritchie, White and Hosford (2025), RadarML, https://doi.org/10.5522/04/30752767.v1; "
        "organizer GPL-3.0 data helpers; AI coding assistance: OpenAI Codex.\n"
        "API references: https://docs.pytorch.org/docs/stable/generated/torch.functional.stft.html and "
        "https://docs.pytorch.org/docs/stable/notes/amp_examples.html\n", encoding="utf-8")
    helpers.write_json(export / "complete.json", dict(model_sha256=infer.sha256(export / "model.pt")))
    print("EXPORTED MODEL:", export / "model.pt", flush=True)
    print("FULL VALIDATION:", report["mean_balanced_accuracy"], "| AUDIT:", audit_report["mean_balanced_accuracy"], flush=True)
    return report


def package(run):
    """Package inference artifacts and resume weights; NEVER package the training cache."""
    run = Path(run)
    model_zip = run.with_name(run.name + "_model.zip")
    with zipfile.ZipFile(model_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted((run / "export").iterdir()):
            if path.is_file():
                archive.write(path, path.name)
    resume_zip = run.with_name(run.name + "_resume.zip")
    with zipfile.ZipFile(resume_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ("resume.pt", "best.pt", "run_config.json", "history.json", "best_selection_validation.json"):
            if (run / name).exists():
                archive.write(run / name, name)
    return model_zip, resume_zip
