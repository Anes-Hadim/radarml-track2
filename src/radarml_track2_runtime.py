"""RadarML Lite Track 2 / Track B: from-scratch STFT classification.
SPDX-License-Identifier: GPL-3.0-or-later
Lookup code adapted from the participant's synapseon.ipynb and the organizer's
GPL-3.0 utilities. RadarML attribution: Ritchie, White and Hosford (2025).
AI coding assistance: OpenAI Codex. No pretrained weights or external data.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

CLASSES = ("barker", "cw", "fmcw", "fsk", "hyp", "nlfm", "quad")
ALL_SNRS = (30, 25, 20, 15, 10, 5, 0, -5, -10, -15, -20, -25, -30)
SAMPLE_RATE_HZ = 120_000_000
MANIFEST_SHA256 = "4ccc4937f4ba0e9993837bb26a5ba4463f1a880b689a753ad3a0e39b76ebafc7"

# %% File identity and published split
def family(filename):
    parts = Path(filename).stem.lower().split("_")
    if len(parts) < 3:
        raise ValueError(f"Unrecognized waveform filename: {filename}")
    token = parts[1]
    if token.startswith("barker"):
        return "barker"
    if re.fullmatch(r"fsk(?:[#-]?[0-3])?", token):
        return "fsk"
    if token == "fsk" and len(parts) > 2 and parts[2] in {"0", "1", "2", "3"}:
        return "fsk"
    if token in CLASSES:
        return token
    raise ValueError(f"Unrecognized modulation in {filename}: {token}")

def canonical_filename(filename):
    # Kaggle can append " (19)" before an extension. Preserve frequency signs.
    filename = re.sub(r" \(\d+\)(?=\.(?:npy|json)$)", "", filename)
    return re.sub(r"(?<=_)fsk[#_-]?([0-3])(?=_)", r"fsk\1", filename, flags=re.IGNORECASE)

def expanded_name_from_source(source_file):
    """Recreate the organizer's expanded name from JSON's original source_file."""
    source = Path(source_file).name
    if source.endswith("_recorded.npy"):
        return source.replace("_recorded.npy", "_single_channel.npy")
    if source.endswith("_single_channel.npy"):
        return source
    parts = Path(source).stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"Unexpected source_file in JSON sidecar: {source}")
    try:
        parts[-1] = "-".join(f'{240 - int(piece.removesuffix("Mhz"))}Mhz'
                             for piece in parts[-1].split("-"))
    except ValueError as exc:
        raise ValueError(f"Unexpected source frequency in JSON sidecar: {source}") from exc
    return "_".join(parts) + "_single_channel.npy"

def build_file_lookup(root, kind):
    suffix = "_single_channel.npy" if kind == "expanded" else "_recorded.npy"
    files = [p for p in root.rglob("*.npy") if canonical_filename(p.name).endswith(suffix)]
    if not files:
        raise FileNotFoundError(f"No {suffix} files in {root}")
    file_set = set(files)
    arrays, metadata = defaultdict(set), defaultdict(list)
    for path in files:
        arrays[canonical_filename(path.name)].add(path)
    for sidecar in root.rglob("*.json"):
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        if not isinstance(meta, dict) or not meta.get("source_file"):
            continue
        alias = expanded_name_from_source(meta["source_file"])
        if kind == "compact":
            alias = alias.replace("_single_channel.npy", "_recorded.npy")
        key = canonical_filename(alias)
        metadata[key].append((sidecar, meta))
        paired_array = sidecar.with_suffix(".npy")
        if paired_array in file_set:
            arrays[key].add(paired_array)
    hash_to_paths, path_to_hash = defaultdict(list), {}
    if kind == "expanded":
        fsk_paths = [p for p in files if family(canonical_filename(p.name)) == "fsk"]
        print(f"Hashing {len(fsk_paths)} uploaded FSK arrays to recover their original identities...")
        for count, path in enumerate(fsk_paths, 1):
            digest = lookup_sha256(path)
            hash_to_paths[digest].append(path)
            path_to_hash[path] = digest
            if count % 25 == 0 or count == len(fsk_paths):
                print(f"Hashed {count}/{len(fsk_paths)} FSK arrays")
        for key, entries in metadata.items():
            if "_fsk" in key.lower():
                for _, meta in entries:
                    for path in hash_to_paths.get(meta.get("output_sha256"), []):
                        arrays[key].add(path)
    return {"arrays": arrays, "metadata": metadata,
            "hash_to_paths": hash_to_paths, "path_to_hash": path_to_hash}

def lookup_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def resolve_file(manifest_name, lookup, kind):
    desired = (manifest_name.replace("_recorded.npy", "_single_channel.npy")
               if kind == "expanded" else manifest_name)
    key = canonical_filename(desired)
    if kind == "expanded" and family(manifest_name) == "fsk":
        expected = {meta.get("output_sha256") for _, meta in lookup["metadata"].get(key, [])}
        digests_found = {digest for digest in expected if digest in lookup["hash_to_paths"]}
        if len(digests_found) == 1:
            digest = next(iter(digests_found))
            return sorted(lookup["hash_to_paths"][digest])[0]
        raise ValueError(f"FSK array for {manifest_name} cannot be matched by SHA256. "
                         f"Matching hashes found: {len(digests_found)}. "
                         "The Kaggle dataset may be missing an NPY file or its JSON may be stale.")
    matches = list(lookup["arrays"].get(key, []))
    paired = {sidecar.with_suffix(".npy") for sidecar, _ in lookup["metadata"].get(key, [])}
    if paired:
        matches = [path for path in matches if path in paired]
    if matches:
        shallowest = min(len(path.parts) for path in matches)
        matches = [path for path in matches if len(path.parts) == shallowest]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        expected_hashes = {meta.get("output_sha256") for _, meta in lookup["metadata"].get(key, [])}
        matching = [(path, digest) for path in matches
                    if (digest := lookup_sha256(path)) in expected_hashes]
        if matching and len({digest for _, digest in matching}) == 1:
            return sorted(path for path, _ in matching)[0]
    examples = [p.name for paths in lookup["arrays"].values() for p in paths
                if "fsk" in p.name.lower()][:5]
    raise ValueError(f"Could not uniquely match {manifest_name}: {len(matches)} candidate arrays. "
                     f"Example Kaggle FSK names: {examples}. Check array uploads and JSON checksums.")

def resolve_metadata(manifest_name, path, lookup, kind):
    desired = (manifest_name.replace("_recorded.npy", "_single_channel.npy")
               if kind == "expanded" else manifest_name)
    candidates = lookup["metadata"].get(canonical_filename(desired), [])
    if kind == "expanded" and family(manifest_name) == "fsk":
        digest = lookup["path_to_hash"].get(path)
        matched = [(sidecar, meta) for sidecar, meta in candidates
                   if meta.get("output_sha256") == digest]
        if matched:
            return sorted(matched, key=lambda pair: str(pair[0]))[0]
        raise ValueError(f"FSK metadata checksum missing for {manifest_name}")
    exact = [(sidecar, meta) for sidecar, meta in candidates if sidecar == path.with_suffix(".json")]
    if len(exact) == 1:
        return exact[0]
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        digest = lookup_sha256(path)
        matched = [(sidecar, meta) for sidecar, meta in candidates
                   if meta.get("output_sha256") == digest]
        if matched:
            return sorted(matched, key=lambda pair: str(pair[0]))[0]
    raise ValueError(f"Could not pair {path.name} with a JSON sidecar for {manifest_name}")

def load_manifest(path):
    path = Path(path)
    if lookup_sha256(path) != MANIFEST_SHA256:
        raise ValueError("Manifest checksum differs from the published release. Use the original CSV, not a regenerated split.")
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["file", "pulse_index", "split"]:
            raise ValueError("Expected manifest columns: file,pulse_index,split")
        rows = list(reader)
    if not rows:
        raise ValueError("Empty manifest")
    identities = set()
    per_file = Counter()
    for row in rows:
        if row["split"] not in {"train", "validation"}:
            raise ValueError(f"Unexpected split {row['split']}")
        pulse = int(row["pulse_index"])
        if not 0 <= pulse < 100:
            raise ValueError(f"Pulse index outside 0..99: {pulse}")
        identity = (row["file"], pulse)
        if identity in identities:
            raise ValueError(f"Duplicate source pulse: {identity}")
        identities.add(identity)
        per_file[(row["file"], row["split"])] += 1
        row["pulse_index"] = pulse
    for name in {r["file"] for r in rows}:
        if per_file[(name, "train")] != 80 or per_file[(name, "validation")] != 20:
            raise ValueError(f"Manifest is not 80/20 for {name}")
    if len(rows) != 66_000 or len(per_file) != 1_320:
        raise ValueError("Expected the complete published 660-recording, 66,000-pulse manifest")
    print(f"Manifest: {len(rows):,} source pulses across {len(per_file)//2} recordings")
    return rows


# %% Expanded data: no expansion or synthetic-noise generation
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def find_manifest(root, explicit=None):
    if explicit:
        path = Path(explicit)
        if path.is_dir():
            candidates = sorted(path.rglob("split_manifest.csv"))
        else:
            candidates = [path] if path.is_file() else []
    else:
        candidates = [p for p in (Path(root) / "split_manifest.csv",
                                  Path(root).parent / "split_manifest.csv") if p.is_file()]
        if not candidates and Path("/kaggle/input").is_dir():
            candidates = sorted(Path("/kaggle/input").rglob("split_manifest.csv"))
    # Multiple copies are fine only when all have the published identity.
    valid = [p for p in candidates if lookup_sha256(p) == MANIFEST_SHA256]
    if not valid:
        raise FileNotFoundError("Attach the published split_manifest.csv (repository hackathon/ copy); set MANIFEST_PATH to it.")
    return valid[0]


def prepare_development(root, manifest_path):
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Expanded DATA_ROOT does not exist: {root}. Update the Kaggle path in the settings cell.")
    rows = load_manifest(manifest_path)
    lookup = build_file_lookup(root, "expanded")
    infos, paths_used = {}, set()
    for name in sorted({r["file"] for r in rows}):
        path = resolve_file(name, lookup, "expanded")
        sidecar, meta = resolve_metadata(name, path, lookup, "expanded")
        # Verify that the source identity really maps to this published recording.
        expected_name = name.replace("_recorded.npy", "_single_channel.npy")
        if canonical_filename(expanded_name_from_source(meta["source_file"])) != canonical_filename(expected_name):
            raise ValueError(f"Wrong source identity in {sidecar}")
        snrs = tuple(meta["snr_db"])
        if len(snrs) != 13 or set(snrs) != set(ALL_SNRS):
            raise ValueError(f"Expected all 13 requested SNRs in {sidecar}; found {snrs}")
        if meta.get("source_pulse_indices") != list(range(100)):
            raise ValueError(f"Pulse-axis mapping is not the organizer's 0..99 mapping: {sidecar}")
        if meta.get("sample_rate_hz") != SAMPLE_RATE_HZ or meta.get("channel") != "ADC0":
            raise ValueError(f"Unexpected receiver/sample rate: {sidecar}")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.shape != (1, 13, 100, 33600) or array.dtype != np.int16:
            raise ValueError(f"Expected expanded int16 (1,13,100,33600): {path}, {array.shape}, {array.dtype}")
        if meta.get("output_shape") != list(array.shape) or meta.get("dtype") != "int16":
            raise ValueError(f"NPY/JSON layout disagreement: {path}")
        del array
        if path in paths_used:
            raise ValueError(f"Two source identities resolve to the same NPY: {path}")
        paths_used.add(path)
        infos[name] = {"path": str(path), "label": CLASSES.index(family(name)),
                       "snr_to_axis": {int(s): i for i, s in enumerate(snrs)}}
    print(f"Validated {len(infos)} expanded arrays; source pulses retain their published split.", flush=True)
    return rows, infos


def build_examples(rows, infos, split, pulse_limit, snrs):
    if split not in {"train", "validation"}:
        raise ValueError("Only the published train and validation splits are used here")
    limit = 80 if split == "train" else 20
    if not 1 <= pulse_limit <= limit or not snrs or not set(snrs).issubset(ALL_SNRS):
        raise ValueError("Invalid source-pulse limit or requested SNR subset")
    grouped = defaultdict(list)
    for row in rows:
        if row["split"] == split:
            grouped[row["file"]].append(row["pulse_index"])
    # SNR first improves sequential validation reads; selection uses original indices.
    return [(name, pulse, int(snr), infos[name]["label"])
            for name in sorted(grouped) for snr in snrs
            for pulse in sorted(grouped[name])[:pulse_limit]]


def interleaved_to_channels(row):
    if row.shape != (33600,) or row.dtype != np.int16:
        raise ValueError(f"Expected one interleaved int16 pulse: {row.shape}, {row.dtype}")
    return np.stack((row[0::2], row[1::2])).astype(np.float32) / 32768.0


class DevelopmentDataset(Dataset):
    def __init__(self, examples, infos):
        self.examples, self.infos = examples, infos

    @staticmethod
    @lru_cache(maxsize=24)
    def _array(path):
        return np.load(path, mmap_mode="r", allow_pickle=False)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        name, pulse, snr, label = self.examples[index]
        info = self.infos[name]
        row = self._array(info["path"])[0, info["snr_to_axis"][snr], pulse]
        # Label and requested SNR are returned for supervision/metrics only.
        return torch.from_numpy(interleaved_to_channels(row)), label, snr


def make_loader(dataset, batch_size, workers, seed, sampler=None):
    options = dict(batch_size=batch_size, num_workers=workers,
                   pin_memory=torch.cuda.is_available(), shuffle=False,
                   generator=torch.Generator().manual_seed(seed), worker_init_fn=seed_worker)
    if workers:
        options.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(dataset, sampler=sampler, **options)


# %% Track B representation and model, initialized from scratch
class TwoResolutionSTFT(nn.Module):
    def __init__(self, n_ffts=(256, 1024), hop_length=64, image_size=(256, 256)):
        super().__init__()
        self.n_ffts, self.hop_length, self.image_size = tuple(n_ffts), hop_length, tuple(image_size)
        for n_fft in self.n_ffts:
            self.register_buffer(f"window_{n_fft}", torch.hann_window(n_fft), persistent=False)

    def forward(self, x):
        # FFT remains float32/complex64 even during mixed-precision training.
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            rms = x.square().sum(1).mean(1, keepdim=True).sqrt().clamp_min(1e-6)
            iq = torch.complex(x[:, 0], x[:, 1]) / rms
            views = []
            for n_fft in self.n_ffts:
                window = getattr(self, f"window_{n_fft}")
                z = torch.stft(iq, n_fft=n_fft, hop_length=self.hop_length,
                               win_length=n_fft, window=window, center=True,
                               pad_mode="constant", onesided=False, return_complex=True)
                power = torch.fft.fftshift(z.abs().square(), dim=1) / window.square().sum()
                # Fixed acquisition window, never filename pulse duration or SNR label.
                # Use only frames fully inside the 0..20 us noise interval.
                first = math.ceil((n_fft / 2) / self.hop_length)
                stop = math.floor((2400 - n_fft / 2) / self.hop_length) + 1
                noise_floor = power[:, :, first:stop].mean((1, 2), keepdim=True).clamp_min(1e-8)
                view = torch.log1p(power / noise_floor).unsqueeze(1)
                # Averaging preserves narrow-band energy without taking only the largest bin.
                views.append(F.adaptive_avg_pool2d(view, self.image_size))
            return torch.cat(views, dim=1)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.GroupNorm(8, out_channels), nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.GroupNorm(8, out_channels))
        self.skip = (nn.Identity() if in_channels == out_channels and stride == 1 else
                     nn.Conv2d(in_channels, out_channels, 1, stride, bias=False))

    def forward(self, x):
        return F.silu(self.main(x) + self.skip(x))


class Track2Model(nn.Module):
    def __init__(self, n_ffts=(256, 1024), hop_length=64, image_size=(256, 256), dropout=0.2):
        super().__init__()
        self.representation = TwoResolutionSTFT(n_ffts, hop_length, image_size)
        self.features = nn.Sequential(
            nn.Conv2d(len(n_ffts), 24, 5, 2, 2, bias=False),
            nn.GroupNorm(8, 24), nn.SiLU(inplace=True),
            ResidualBlock(24, 48, 2), ResidualBlock(48, 48),
            ResidualBlock(48, 96, 2), ResidualBlock(96, 96),
            ResidualBlock(96, 128, 2), ResidualBlock(128, 128))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
                                  nn.Dropout(dropout), nn.Linear(128 * 4 * 4, 256),
                                  nn.SiLU(), nn.Dropout(dropout), nn.Linear(256, len(CLASSES)))

    def forward(self, iq):
        return self.head(self.features(self.representation(iq)))


# %% Organizer-compatible metric and measured evaluation
def report_from_matrices(matrices):
    per_snr = []
    for snr in ALL_SNRS:
        if snr not in matrices:
            continue
        matrix = matrices[snr]
        totals = matrix.sum(1)
        if (totals == 0).any():
            raise ValueError(f"Requested SNR {snr} lacks one or more classes")
        per_snr.append({"requested_snr_db": int(snr),
                        "balanced_accuracy": float(np.mean(matrix.diagonal() / totals)),
                        "accuracy": float(matrix.trace() / matrix.sum()),
                        "class_recall": (matrix.diagonal() / totals).tolist(),
                        "confusion_matrix": matrix.tolist()})
    low = [p["balanced_accuracy"] for p in per_snr if p["requested_snr_db"] in (-20, -25, -30)]
    return {"classes": list(CLASSES), "samples": int(sum(m.sum() for m in matrices.values())),
            "mean_balanced_accuracy": float(np.mean([p["balanced_accuracy"] for p in per_snr])),
            "lowest_three_mean_balanced_accuracy": float(np.mean(low)) if len(low) == 3 else None,
            "per_snr": per_snr}


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def evaluate(model, loader, device, snrs=ALL_SNRS):
    model.eval()
    matrices = {int(s): np.zeros((7, 7), dtype=np.int64) for s in snrs}
    synchronize(device)
    start = time.perf_counter()
    for signals, labels, batch_snrs in loader:
        prediction = model(signals.to(device, non_blocking=True)).argmax(1).cpu().numpy()
        labels, batch_snrs = labels.numpy(), batch_snrs.numpy()
        for snr in snrs:
            mask = batch_snrs == snr
            np.add.at(matrices[int(snr)], (labels[mask], prediction[mask]), 1)
    synchronize(device)
    seconds = time.perf_counter() - start
    report = report_from_matrices(matrices)
    report.update(inference_seconds=seconds, seconds_per_pulse=seconds / report["samples"],
                  timing_scope="DataLoader, disk reads, transfer, RMS/noise normalization, both STFTs, CNN, metric accumulation; FP32")
    return report


def hardware_info(device):
    return {"device": str(device), "cpu": platform.processor(), "platform": platform.platform(),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "gpu_memory_bytes": torch.cuda.get_device_properties(device).total_memory if device.type == "cuda" else None,
            "python": sys.version, "numpy": str(np.__version__), "torch": str(torch.__version__),
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version()}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


# %% Training, checkpoint selection and reproducible freeze bundle
DEFAULT_CONFIG = {
    "seed": 42, "train_pulses_per_file": 80, "train_snrs": list(ALL_SNRS),
    "selection_pulses_per_file": 4, "batch_size": 64, "workers": 2,
    "epochs": 20, "samples_per_epoch": 105600, "patience": 6,
    "learning_rate": 0.0006, "weight_decay": 0.01, "label_smoothing": 0.02,
    "train_amp": True,
    "model_kwargs": {"n_ffts": [256, 1024], "hop_length": 64,
                     "image_size": [256, 256], "dropout": 0.2},
}


def train_and_freeze(data_root, manifest_path, output, config=None, prepared=None):
    cfg = json.loads(json.dumps(DEFAULT_CONFIG if config is None else config))
    output = Path(output)
    # Never overwrite frozen artifacts during a rerun or a new experiment.
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output is not empty: {output}. Choose a new RUN_NAME for a new experiment.")
    output.mkdir(parents=True, exist_ok=True)
    if not 1 <= cfg["selection_pulses_per_file"] <= 20:
        raise ValueError("Selection validation uses 1..20 published validation pulses per recording")
    if cfg["epochs"] < 1 or cfg["samples_per_epoch"] < 7:
        raise ValueError("Need at least one epoch and seven samples per epoch")
    seed_everything(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setup_start = time.perf_counter()
    manifest_path = find_manifest(data_root, manifest_path)
    if prepared is None:
        rows, infos = prepare_development(data_root, manifest_path)
        preparation_seconds = 0.0
    else:
        rows, infos, preparation_seconds = prepared
    train_ex = build_examples(rows, infos, "train", cfg["train_pulses_per_file"], cfg["train_snrs"])
    select_ex = build_examples(rows, infos, "validation", cfg["selection_pulses_per_file"], ALL_SNRS)
    full_ex = build_examples(rows, infos, "validation", 20, ALL_SNRS)
    train_ids = {(e[0], e[1]) for e in train_ex}
    assert not train_ids.intersection((e[0], e[1]) for e in full_ex), "Source-pulse leakage"
    labels = np.array([e[3] for e in train_ex], dtype=np.int64)
    counts = np.bincount(labels, minlength=7)
    if (counts == 0).any():
        raise ValueError("All seven classes must be present in training")
    sampler = WeightedRandomSampler(torch.as_tensor(1.0 / counts[labels], dtype=torch.double),
                                    cfg["samples_per_epoch"], replacement=True,
                                    generator=torch.Generator().manual_seed(cfg["seed"]))
    train_loader = make_loader(DevelopmentDataset(train_ex, infos), cfg["batch_size"], cfg["workers"], cfg["seed"], sampler)
    select_loader = make_loader(DevelopmentDataset(select_ex, infos), cfg["batch_size"], cfg["workers"], cfg["seed"])
    print("Eligible training examples:", len(train_ex), "| sampled per epoch:", cfg["samples_per_epoch"], flush=True)
    print("Eligible training class counts:", dict(zip(CLASSES, counts.tolist())), flush=True)
    print("Selection validation:", len(select_ex), "| final full validation:", len(full_ex), flush=True)
    model = Track2Model(**cfg["model_kwargs"]).to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    use_amp = bool(cfg["train_amp"] and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    best_score, best_low, best_epoch, stale = -1.0, -1.0, 0, 0
    history, fit_seconds, selection_seconds = [], 0.0, 0.0
    setup_seconds = time.perf_counter() - setup_start + preparation_seconds
    synchronize(device)
    start = time.perf_counter()
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        epoch_start = time.perf_counter()
        total_loss, seen = 0.0, 0
        for step, (signals, targets, _) in enumerate(train_loader, 1):
            signals, targets = signals.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = criterion(model(signals), targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach()) * len(targets)
            seen += len(targets)
            if step == 1 or step % 100 == 0:
                print(f"Epoch {epoch}/{cfg['epochs']}, batch {step}/{len(train_loader)}, loss {total_loss/seen:.4f}", flush=True)
        synchronize(device)
        fit_seconds += time.perf_counter() - epoch_start
        report = evaluate(model, select_loader, device)
        selection_seconds += report["inference_seconds"]
        score, low = report["mean_balanced_accuracy"], report["lowest_three_mean_balanced_accuracy"]
        record = {"epoch": epoch, "train_loss": total_loss / seen,
                  "learning_rate": optimizer.param_groups[0]["lr"],
                  "selection_mean_balanced_accuracy": score, "selection_lowest_three": low}
        history.append(record)
        print(json.dumps(record), flush=True)
        better = score > best_score + 1e-6 or (abs(score - best_score) <= 1e-6 and low > best_low + 1e-6)
        if better:
            best_score, best_low, best_epoch, stale = score, low, epoch, 0
            torch.save({"state_dict": model.state_dict(), "classes": list(CLASSES),
                        "track": "B", "model_kwargs": cfg["model_kwargs"]}, output / "model.pt")
            report.update(selected_epoch=epoch, pulses_per_recording=cfg["selection_pulses_per_file"],
                          full_published_validation=cfg["selection_pulses_per_file"] == 20)
            write_json(output / "selection_validation.json", report)
        else:
            stale += 1
        write_json(output / "training_history.json", history)
        scheduler.step()
        if stale >= cfg["patience"]:
            print("Early stopping on validation only.", flush=True)
            break
    synchronize(device)
    development_seconds = time.perf_counter() - start
    # Evaluate exactly the saved best checkpoint, never the final in-memory epoch.
    checkpoint = torch.load(output / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    del checkpoint, optimizer, train_loader, select_loader
    full_loader = make_loader(DevelopmentDataset(full_ex, infos), cfg["batch_size"], cfg["workers"], cfg["seed"])
    full_report = evaluate(model, full_loader, device)
    assert full_report["samples"] == 171600
    full_report.update(selected_epoch=best_epoch, pulses_per_recording=20, full_published_validation=True)
    write_json(output / "validation_report.json", full_report)
    cfg.update(track="B", track_number=2, classes=list(CLASSES), sample_rate_hz=SAMPLE_RATE_HZ,
               manifest_sha256=MANIFEST_SHA256, train_source_pulses=len(train_ids),
               eligible_training_examples=len(train_ex), eligible_class_counts=dict(zip(CLASSES, counts.tolist())),
               subset_rule="First N ascending published pulse indices per recording, separately within each split",
               sampler="Inverse-class-count weighted replacement sampling; every selected SNR has equal sampling probability within each class",
               optimizer="AdamW", schedule="CosineAnnealingLR by epoch", gradient_clip_norm=5.0,
               selection_rule="Mean balanced accuracy across all 13 SNRs on selection validation; tolerance 1e-6, lowest-three-SNR tie-break",
               selected_epoch=best_epoch, epochs_completed=len(history), training_draws=len(history)*cfg["samples_per_epoch"],
               inference_precision="float32", inference_augmentation="none", training_augmentation="none beyond supplied SNR variants",
               model_parameters=parameter_count, model_bytes=(output / "model.pt").stat().st_size,
               setup_seconds=setup_seconds, fit_seconds=fit_seconds, selection_validation_seconds=selection_seconds,
               development_seconds=development_seconds, hardware=hardware_info(device),
               preprocessing="Complex RMS per pulse; centered two-sided Hann STFT with zero padding; log1p(power / per-pulse noise-window mean power); frequency shift; adaptive mean pooling",
               pretrained=False, external_training_data=False)
    write_json(output / "config.json", cfg)
    shutil.copyfile(Path(__file__), output / "radarml_track2_runtime.py")
    shutil.copyfile(manifest_path, output / "split_manifest.csv")
    # Public version pins accept the CUDA build already installed on Kaggle.
    # Exact build, CUDA/cuDNN and hardware are retained in config.json.
    torch_public_version = str(torch.__version__).split("+", 1)[0]
    (output / "requirements.txt").write_text(
        f"# Exact CUDA/build details: config.json; use the original Kaggle environment.\n"
        f"numpy=={np.__version__}\ntorch=={torch_public_version}\n", encoding="utf-8")
    (output / "inference_command.txt").write_text(
        "python radarml_track2_runtime.py infer --bundle . --blind-root /path/to/blind_test_participants --output /path/to/predictions.csv\n", encoding="utf-8")
    summary = f"""RadarML Lite — Track 2 / B (time-frequency)

Method: two-sided Hann STFT at {cfg['model_kwargs']['n_ffts']} samples, hop
{cfg['model_kwargs']['hop_length']}; per-pulse RMS/noise-window normalization,
log power and adaptive pooling; a residual 2D CNN with GroupNorm, trained
from scratch with class-balanced replacement sampling, AdamW and cosine LR.
No filename, ID, metadata, pulse index or requested-SNR label is a model input.

Published split: {MANIFEST_SHA256}. Source-pulse training subset:
{len(train_ids):,} ({cfg['train_pulses_per_file']} / 80 per recording), SNRs
{cfg['train_snrs']}. All variants retain the source pulse's split. Seed:
{cfg['seed']}. Draws per epoch: {cfg['samples_per_epoch']:,}; this sampling
budget does not guarantee every eligible variant is visited each epoch.
Training draws actually used: {cfg['training_draws']:,}.

Checkpoint selected on {cfg['selection_pulses_per_file']} validation pulses
per recording at all 13 SNRs; selected epoch: {best_epoch}. After selection,
the saved checkpoint was evaluated on all 171,600 published validation
examples: mean balanced accuracy {full_report['mean_balanced_accuracy']:.6f};
lowest-three-SNR mean {full_report['lowest_three_mean_balanced_accuracy']:.6f}.
See validation_report.json for per-SNR scores and all confusion matrices.
Validation pulses were never fitted or added to final training.

Hardware/dependency versions: config.json and requirements.txt. Model:
{parameter_count:,} parameters, {cfg['model_bytes']:,} bytes. Fit-only time:
{fit_seconds:.1f} s; fit plus selection/checkpoint overhead:
{development_seconds:.1f} s; dataset setup: {setup_seconds:.1f} s.
Full-validation inference: {full_report['seconds_per_pulse']:.6f} s/pulse,
including loading, transfers, normalization, STFTs and CNN. Blind-run timing
is recorded separately in predictions.csv.inference.json after official release.

Limitations: validation and blind test use known waveform configurations;
results do not establish generalization to other receivers or configurations.
Requested SNR differs from achieved SNR, and int16 clipping affects low SNRs.
Pooling trades spectral/time detail for speed. No blind score is known here.

Attribution: Ritchie, White and Hosford (2025), RadarML v1,
https://doi.org/10.5522/04/30752767.v1; organizer Radar_ML_lite code and rules,
GPL-3.0; file-identity lookup adapted from the participant's synapseon.ipynb.
AI assistance: OpenAI Codex helped inspect documentation and implement/debug
the notebook and runtime. No external learned weights or training data used.

Before submitting: identify your team/name and organizer-issued pipeline
identifier in the submission message; keep the hashed bundle unchanged.
Send this frozen bundle at the announced freeze deadline; after blind release
run the inference command, then add predictions.csv and its inference report.
"""
    (output / "method_summary.txt").write_text(summary, encoding="utf-8")
    hashes = {p.name: lookup_sha256(p) for p in sorted(output.iterdir()) if p.is_file()}
    write_json(output / "frozen.sha256.json", hashes)
    print("Full validation mean balanced accuracy:", full_report["mean_balanced_accuracy"], flush=True)
    print("Frozen artifacts:", output, flush=True)
    return full_report


# %% Blind inference: frozen code/config/weights, anonymous shard loader
def verify_bundle(bundle):
    bundle = Path(bundle)
    hashes = json.loads((bundle / "frozen.sha256.json").read_text(encoding="utf-8"))
    required = {"model.pt", "config.json", "radarml_track2_runtime.py", "requirements.txt"}
    if not required.issubset(hashes):
        raise ValueError("Incomplete freeze manifest")
    for name, expected in hashes.items():
        if Path(name).name != name or lookup_sha256(bundle / name) != expected:
            raise ValueError(f"Frozen artifact changed or invalid: {name}")
    # The function actually executing must also be the frozen runtime.
    if lookup_sha256(Path(__file__)) != hashes["radarml_track2_runtime.py"]:
        raise ValueError("Execute the runtime inside the frozen bundle")
    return hashes


def read_blind_index(blind_root, expected_count=171600):
    root = Path(blind_root).resolve()
    if not (root / "index.csv").is_file() and (root / "test" / "index.csv").is_file():
        root = root / "test"
    with (root / "index.csv").open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not {"sample_id", "file", "row"}.issubset(reader.fieldnames or []):
            raise ValueError("Anonymous index must provide sample_id,file,row")
        rows = list(reader)
    template = root / "submission_template.csv"
    if not template.is_file():
        template = root.parent / "submission_template.csv"
    with template.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["sample_id", "predicted_label"]:
            raise ValueError("Unexpected submission template columns")
        ids = [r["sample_id"] for r in reader]
    index_ids = [r["sample_id"] for r in rows]
    if (len(rows) != expected_count or any(not i for i in index_ids) or
            len(set(index_ids)) != len(index_ids) or len(set(ids)) != len(ids) or set(index_ids) != set(ids)):
        raise ValueError("Blind index/template must contain every expected anonymous ID once")
    locations = set()
    for row in rows:
        relative = Path(row["file"])
        path = (root / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(root):
            raise ValueError("Shard path escapes the blind directory")
        row["row"] = int(row["row"])
        if row["row"] < 0 or (row["file"], row["row"]) in locations:
            raise ValueError("Duplicate or invalid blind shard row")
        locations.add((row["file"], row["row"]))
    return root, rows, ids


class BlindDataset(Dataset):
    def __init__(self, root, rows):
        self.root, self.rows = Path(root), rows

    @staticmethod
    @lru_cache(maxsize=8)
    def _array(path):
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.ndim != 2 or array.shape[1] != 33600 or array.dtype != np.int16:
            raise ValueError(f"Expected int16 blind shard (N,33600): {path}")
        return array

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        info = self.rows[index]
        row = self._array(str(self.root / info["file"]))[info["row"]]
        return torch.from_numpy(interleaved_to_channels(row))


def predict_blind(bundle, blind_root, output, batch_size=64, workers=2):
    bundle, output = Path(bundle), Path(output)
    hashes = verify_bundle(bundle)
    cfg = json.loads((bundle / "config.json").read_text(encoding="utf-8"))
    if cfg["track"] != "B" or cfg["classes"] != list(CLASSES) or cfg["inference_precision"] != "float32":
        raise ValueError("Unexpected frozen pipeline settings")
    seed_everything(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(bundle / "model.pt", map_location="cpu", weights_only=True)
    if (checkpoint["track"] != "B" or checkpoint["classes"] != list(CLASSES) or
            checkpoint["model_kwargs"] != cfg["model_kwargs"]):
        raise ValueError("Checkpoint disagrees with frozen configuration")
    model = Track2Model(**cfg["model_kwargs"]).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    del checkpoint
    root, rows, template_ids = read_blind_index(blind_root)
    # Storage-only ordering; IDs are used solely to put outputs into the template.
    rows.sort(key=lambda r: (r["file"], r["row"]))
    loader = make_loader(BlindDataset(root, rows), batch_size, workers, cfg["seed"])
    synchronize(device)
    start = time.perf_counter()
    predictions = []
    with torch.inference_mode():
        for step, signals in enumerate(loader, 1):
            predictions.extend(model(signals.to(device, non_blocking=True)).argmax(1).cpu().tolist())
            if step == 1 or step % 100 == 0:
                print(f"Blind inference: {len(predictions):,}/{len(rows):,}", flush=True)
    synchronize(device)
    elapsed = time.perf_counter() - start
    if len(predictions) != len(rows):
        raise RuntimeError("Incomplete blind inference")
    by_id = {r["sample_id"]: CLASSES[p] for r, p in zip(rows, predictions)}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".partial")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("sample_id", "predicted_label"))
        writer.writerows((sid, by_id[sid]) for sid in template_ids)
    # Check the actual serialized file before publishing it as predictions.csv.
    with temporary.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == ["sample_id", "predicted_label"]
        saved = list(reader)
    assert [r["sample_id"] for r in saved] == template_ids
    assert all(r["predicted_label"] in CLASSES for r in saved)
    temporary.replace(output)
    report = {"samples": len(rows), "inference_seconds": elapsed, "seconds_per_pulse": elapsed / len(rows),
              "timing_scope": "Cold DataLoader startup, shard loading, transfers, normalization, both STFTs, CNN and prediction collection; excludes model/index loading and CSV writing",
              "precision": "float32", "hardware": hardware_info(device),
              "frozen_artifact_hashes": hashes, "predictions_sha256": lookup_sha256(output),
              "batch_size": batch_size, "workers": workers}
    write_json(output.with_name(output.name + ".inference.json"), report)
    print("Predictions:", output, "| seconds/pulse:", report["seconds_per_pulse"], flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    infer = commands.add_parser("infer", help="Inference after official blind-test release only")
    infer.add_argument("--bundle", required=True)
    infer.add_argument("--blind-root", required=True)
    infer.add_argument("--output", required=True)
    infer.add_argument("--batch-size", type=int, default=64)
    infer.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    predict_blind(args.bundle, args.blind_root, args.output, args.batch_size, args.workers)


if __name__ == "__main__":
    main()
