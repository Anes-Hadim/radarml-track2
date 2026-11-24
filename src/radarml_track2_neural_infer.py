"""Compact Track 2 neural inference: complex STFT -> neural network -> seven labels.
SPDX-License-Identifier: GPL-3.0-or-later. AI coding assistance: OpenAI Codex.
At inference only this code, model.pt and the blind participant package are needed.
No training examples, waveform-template bank, teacher or previous predictions.
"""
import argparse
import csv
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

CLASSES = ("barker", "cw", "fmcw", "fsk", "hyp", "nlfm", "quad")
MODEL_KWARGS = dict(n_fft=256, projection_rank=64, hidden=128, dropout=.1)


class ComplexSTFT(nn.Module):
    """Explicit frequency x time grid, preserving phase and all 16,800 input samples."""
    def __init__(self, n_fft=256):
        super().__init__()
        self.n_fft = n_fft
        self.frames = math.ceil(16800 / n_fft)
        self.register_buffer("window", torch.ones(n_fft), persistent=False)

    def forward(self, iq):
        with torch.autocast(device_type=iq.device.type, enabled=False):
            iq = iq.float()
            power = iq.square().sum(1)
            rms2 = power.mean(1, keepdim=True).clamp_min(1e-10)
            z = torch.complex(iq[:, 0], iq[:, 1]) / rms2.sqrt()
            z = F.pad(z, (0, self.frames * self.n_fft - z.shape[-1]))
            stft = torch.stft(z, n_fft=self.n_fft, hop_length=self.n_fft,
                              win_length=self.n_fft, window=self.window, normalized=True,
                              center=False, onesided=False, return_complex=True)
            stft = torch.fft.fftshift(stft, dim=1)
            # Non-overlapping rectangular STFT blocks form a unitary block transform:
            # the two-channel complex view retains information and an explicit time axis.
            phase = torch.stack((stft.real, stft.imag), 1)
            noise = power[:, :2400].mean(1, keepdim=True).clamp_min(1e-10)
            relative_noise = (noise / rms2).clamp_min(1e-6)
            image = torch.log1p(stft.abs().square() / relative_noise[:, :, None]).unsqueeze(1)
            active = power[:, 2520:14280].mean(1, keepdim=True)
            tail = power[:, 14640:].mean(1, keepdim=True)
            clipped = ((iq <= -1.) | (iq >= 32767. / 32768.)).float().mean((1, 2))[:, None]
            stats = torch.cat((relative_noise.log(), torch.log1p(active / noise),
                               torch.log1p(tail / noise), clipped), 1)
            return phase, image, stats


class SeparableBlock(nn.Module):
    def __init__(self, incoming, outgoing, stride=(2, 2)):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(incoming, incoming, 3, stride, 1, groups=incoming, bias=False),
            nn.GroupNorm(4, incoming), nn.SiLU(),
            nn.Conv2d(incoming, outgoing, 1, bias=False),
            nn.GroupNorm(4, outgoing), nn.SiLU())

    def forward(self, x):
        return self.layers(x)


class CompactSTFTClassifier(nn.Module):
    """Learned low-rank phase features + small power-image CNN; no configuration search."""
    def __init__(self, n_fft=256, projection_rank=64, hidden=128, dropout=.1):
        super().__init__()
        self.representation = ComplexSTFT(n_fft)
        features = 2 * n_fft * self.representation.frames
        if projection_rank < 8 or projection_rank % 2:
            raise ValueError("projection_rank must be an even number >= 8")
        global_rank = projection_rank // 2
        self.phase_global = nn.Sequential(nn.Flatten(), nn.Linear(features, global_rank),
                                           nn.LayerNorm(global_rank), nn.SiLU())
        local_features = 8 * math.ceil(n_fft / 4) * math.ceil(self.representation.frames / 2)
        self.phase_local = nn.Sequential(
            nn.Conv2d(2, 8, 3, (2, 1), 1, bias=False), nn.GroupNorm(4, 8), nn.SiLU(),
            SeparableBlock(8, 8, (2, 2)), nn.Flatten(), nn.Linear(local_features, projection_rank),
            nn.LayerNorm(projection_rank), nn.SiLU())
        self.phase_merge = nn.Sequential(nn.Linear(global_rank + projection_rank, hidden),
                                          nn.LayerNorm(hidden), nn.SiLU())
        self.power_branch = nn.Sequential(
            nn.Conv2d(1, 12, (5, 3), (2, 1), (2, 1), bias=False),
            nn.GroupNorm(4, 12), nn.SiLU(),
            SeparableBlock(12, 24), SeparableBlock(24, 40), SeparableBlock(40, 64),
            nn.AdaptiveAvgPool2d((8, 4)), nn.Flatten(), nn.Linear(64 * 8 * 4, hidden), nn.SiLU())
        self.head = nn.Sequential(nn.Linear(2 * hidden + 4, hidden), nn.SiLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, len(CLASSES)))

    def forward(self, iq, return_embedding=False):
        phase, power, stats = self.representation(iq)
        phase_features = self.phase_merge(torch.cat((self.phase_global(phase), self.phase_local(phase)), 1))
        embedding = torch.cat((phase_features, self.power_branch(power), stats), 1)
        logits = self.head(embedding)
        return (logits, embedding) if return_embedding else logits


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_model(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (checkpoint.get("model_family") != "compact_complex_stft_neural" or
            checkpoint.get("classes") != list(CLASSES) or checkpoint.get("track") != "B"):
        raise ValueError("This script requires the compact STFT neural model, not a template bank")
    model = CompactSTFTClassifier(**checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval(), checkpoint


def find_participants(explicit=None, search_root=Path("/kaggle/input")):
    root = Path(explicit if explicit is not None else search_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Participant folder does not exist: {root}")
    if (root / "index.csv").is_file() and (root.parent / "submission_template.csv").is_file():
        return root.parent
    candidates = {p.parent for p in root.rglob("submission_template.csv")
                  if (p.parent / "test/index.csv").is_file() or (p.parent / "index.csv").is_file()}
    if len(candidates) != 1:
        raise ValueError(f"Expected one participant package under {root}; found {len(candidates)}. Set BLIND_ROOT explicitly.")
    return candidates.pop()


def read_index(participants, expected_count=171600):
    root = Path(participants).resolve()
    if (root / "test/index.csv").is_file():
        root = root / "test"
    with (root / "index.csv").open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["sample_id", "file", "row"]:
            raise ValueError("Expected anonymous index columns: sample_id,file,row")
        rows = list(reader)
    template = root / "submission_template.csv"
    if not template.exists():
        template = root.parent / "submission_template.csv"
    with template.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames not in (["sample_id", "predicted_label"], ["sample_id", "label"]):
            raise ValueError("Unexpected submission template columns")
        ids = [r["sample_id"] for r in reader]
    index_ids = [r["sample_id"] for r in rows]
    if (len(rows) != expected_count or len(set(index_ids)) != len(rows) or
            len(set(ids)) != len(ids) or set(ids) != set(index_ids) or any(not sid for sid in ids)):
        raise ValueError("Index/template IDs must match and contain every expected sample once")
    locations = set()
    for row in rows:
        path = (root / row["file"]).resolve()
        if not path.is_relative_to(root) or Path(row["file"]).is_absolute():
            raise ValueError("Shard path escapes the participant folder")
        row["row"] = int(row["row"])
        if row["row"] < 0 or (row["file"], row["row"]) in locations:
            raise ValueError("Invalid/duplicate shard row")
        locations.add((row["file"], row["row"]))
    rows.sort(key=lambda r: (r["file"], r["row"]))
    return root, rows, ids


class BlindSignals(Dataset):
    def __init__(self, root, rows):
        self.root, self.rows = Path(root), rows

    @staticmethod
    @lru_cache(maxsize=8)
    def array(path):
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        if value.ndim != 2 or value.shape[1] != 33600 or value.dtype != np.int16:
            raise ValueError(f"Expected int16 shard (N,33600): {path}")
        return value

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        info = self.rows[index]
        raw = self.array(str(self.root / info["file"]))[info["row"]]
        iq = np.stack((raw[::2], raw[1::2])).astype(np.float32) / 32768.
        return torch.from_numpy(iq)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def hardware(device):
    return dict(device=str(device), cpu=platform.processor(), logical_cpus=os.cpu_count(),
                threads=torch.get_num_threads(), platform=platform.platform(), python=sys.version,
                torch=str(torch.__version__), numpy=str(np.__version__),
                gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None)


def latency_stats(times):
    a = np.asarray(times)
    return dict(samples=len(a), mean_ms=float(a.mean()), median_ms=float(np.median(a)),
                p95_ms=float(np.percentile(a, 95)), p99_ms=float(np.percentile(a, 99)),
                max_ms=float(a.max()), fraction_at_or_below_10ms=float(np.mean(a <= 10.)))


@torch.inference_mode()
def predict(model_path, participants, output, device_name="cpu", batch_size=128,
            workers=0, threads=4, benchmark_samples=1000, warmup=20, label_column="label",
            expected_count=171600):
    start = time.perf_counter()
    model_path, output = Path(model_path).resolve(), Path(output).resolve()
    participants = find_participants(participants)
    if output.is_relative_to(participants) or output == model_path:
        raise ValueError("Choose an output outside the model and blind input files")
    if label_column not in {"label", "predicted_label"}:
        raise ValueError("Invalid label column")
    if min(batch_size, threads, benchmark_samples) < 1 or min(workers, warmup) < 0:
        raise ValueError("Invalid inference settings")
    torch.set_num_threads(threads)
    device = torch.device(device_name)
    if device.type not in {"cpu", "cuda"} or (device.type == "cuda" and not torch.cuda.is_available()):
        raise ValueError("Choose an available cpu or cuda device")
    tick = time.perf_counter()
    model, checkpoint = load_model(model_path, device)
    synchronize(device)
    setup_seconds = time.perf_counter() - tick
    root, rows, template_ids = read_index(participants, expected_count)
    dataset = BlindSignals(root, rows)
    indices = np.linspace(0, len(rows) - 1, min(benchmark_samples, len(rows)), dtype=np.int64)
    for i in range(warmup):
        signal = dataset[int(indices[i % len(indices)])][None].to(device)
        model(signal).argmax(1).item()
    synchronize(device)
    latencies = []
    for index in indices:
        tick = time.perf_counter()
        signal = dataset[int(index)][None].to(device)
        model(signal).argmax(1).item()
        synchronize(device)
        latencies.append((time.perf_counter() - tick) * 1000.)
    single = latency_stats(latencies)
    print("Single-signal latency, including I/Q conversion and STFT:", single, flush=True)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers, shuffle=False,
                        pin_memory=device.type == "cuda")
    predictions = []
    synchronize(device)
    tick = time.perf_counter()
    for step, signals in enumerate(loader, 1):
        predictions.extend(model(signals.to(device, non_blocking=True)).argmax(1).cpu().tolist())
        if step == 1 or step % 100 == 0:
            print(f"Inference: {len(predictions):,}/{len(rows):,}", flush=True)
    synchronize(device)
    seconds = time.perf_counter() - tick
    if len(predictions) != len(rows):
        raise RuntimeError("Incomplete predictions")
    by_id = {r["sample_id"]: CLASSES[p] for r, p in zip(rows, predictions)}
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
                any(r[label_column] not in CLASSES for r in saved)):
            raise ValueError("Invalid serialized submission")
    temporary.replace(output)
    report = dict(samples=len(rows), model_bytes=model_path.stat().st_size,
                  model_parameters=sum(p.numel() for p in model.parameters()), model_setup_seconds=setup_seconds,
                  inference_seconds=seconds, milliseconds_per_signal_throughput=1000. * seconds / len(rows),
                  single_signal_latency=single, hardware=hardware(device), precision="float32",
                  timing_scope="Single-signal and batch timing include loading/conversion, transfer if applicable, "
                               "full complex STFT, both neural branches and class prediction; exclude one-time model loading. "
                               "OS cache state is uncontrolled. Single-signal warm-up is excluded.",
                  model_sha256=sha256(model_path), inference_code_sha256=sha256(Path(__file__)),
                  predictions_sha256=sha256(output), columns=["sample_id", label_column],
                  batch_size=batch_size, workers=workers, warmup_signals=warmup,
                  total_seconds=time.perf_counter() - start)
    write_json(output.with_name(output.name + ".inference.json"), report)
    print("Submission:", output, flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--participants", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--benchmark-samples", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--label-column", choices=["label", "predicted_label"], default="label")
    args = parser.parse_args()
    predict(args.model, args.participants, args.output, args.device, args.batch_size,
            args.workers, args.threads, args.benchmark_samples, args.warmup, args.label_column)


if __name__ == "__main__":
    main()
