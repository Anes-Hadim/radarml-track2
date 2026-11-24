"""CPU inference for the trained RadarML Track 2 complex-STFT neural classifier.

Required inputs: --model /path/model.pt --participants /path/participants
Outputs: submission.csv, submission.csv.latency.csv, submission.csv.inference.json
SPDX-License-Identifier: GPL-3.0-or-later.
"""
import argparse
import csv
from functools import lru_cache
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

CLASSES = ("barker", "cw", "fmcw", "fsk", "hyp", "nlfm", "quad")


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

    def forward(self, iq):
        phase, power, stats = self.representation(iq)
        phase_features = self.phase_merge(torch.cat((self.phase_global(phase), self.phase_local(phase)), 1))
        embedding = torch.cat((phase_features, self.power_branch(power), stats), 1)
        return self.head(embedding)


def load_model(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (checkpoint.get("model_family") != "compact_complex_stft_neural" or
            checkpoint.get("classes") != list(CLASSES) or checkpoint.get("track") != "B"):
        raise ValueError("Expected the compact STFT neural model.pt")
    model = CompactSTFTClassifier(**checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.float().eval()


def read_index(folder):
    """Use anonymous IDs only for bookkeeping; never pass them to the network."""
    supplied = Path(folder).expanduser().resolve()
    candidates = (supplied, supplied / "test", supplied / "participants",
                  supplied / "participants/test")
    roots = [p for p in candidates if (p / "index.csv").is_file()]
    if len(roots) != 1:
        raise ValueError("Point --participants to the blind participants folder or its test subfolder")
    root = roots[0]
    with (root / "index.csv").open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["sample_id", "file", "row"]:
            raise ValueError("Expected index.csv columns sample_id,file,row")
        rows = list(reader)
    ids = [r["sample_id"] for r in rows]
    if not ids or any(not sid for sid in ids) or len(ids) != len(set(ids)):
        raise ValueError("Index must contain nonempty, unique sample IDs")
    locations = set()
    for row in rows:
        path = (root / row["file"]).resolve()
        if not path.is_relative_to(root) or Path(row["file"]).is_absolute():
            raise ValueError("Shard path escapes the blind data folder")
        row["row"] = int(row["row"])
        location = (str(path), row["row"])
        if row["row"] < 0 or location in locations:
            raise ValueError("Invalid or duplicated shard row")
        locations.add(location)
        row["path"] = path
    # Preserve the official order when the blind package supplies a template.
    # Only its sample IDs are read; its label column never controls output naming.
    for template in (root / "submission_template.csv", root.parent / "submission_template.csv"):
        if template.is_file():
            with template.open(newline="", encoding="utf-8-sig") as stream:
                reader = csv.DictReader(stream)
                if not reader.fieldnames or "sample_id" not in reader.fieldnames:
                    raise ValueError("Submission template must contain sample_id")
                ordered_ids = [r["sample_id"] for r in reader]
            if len(ordered_ids) != len(ids) or set(ordered_ids) != set(ids):
                raise ValueError("Template and index sample IDs do not match")
            ids = ordered_ids
            break
    rows.sort(key=lambda r: (str(r["path"]), r["row"]))
    return supplied, rows, ids


@lru_cache(maxsize=8)
def open_shard(path):
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.ndim != 2 or array.shape[1] != 33600 or array.dtype != np.int16:
        raise ValueError(f"Expected an int16 shard of shape (N,33600): {path}")
    return array


def load_signal(row):
    raw = open_shard(str(row["path"]))[row["row"]]
    iq = np.stack((raw[::2], raw[1::2])).astype(np.float32) / 32768.
    return torch.from_numpy(iq)[None]


def write_csv(path, columns, rows):
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(rows)
    temporary.replace(path)


@torch.inference_mode()
def predict(model_path, participants, output="submission.csv", threads=4):
    """Measure every actual prediction separately, including I/Q loading and STFT."""
    if threads < 1:
        raise ValueError("--threads must be at least 1")
    start = time.perf_counter()
    model_path = Path(model_path).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    latency_path = output.with_name(output.name + ".latency.csv")
    report_path = output.with_name(output.name + ".inference.json")
    supplied, rows, ordered_ids = read_index(participants)
    protected_inputs = {model_path, *(row["path"] for row in rows)}
    for target in (output, latency_path, report_path):
        if target in protected_inputs or target.is_relative_to(supplied):
            raise ValueError("Write output outside the blind folder and model file")
    torch.set_num_threads(threads)
    tick = time.perf_counter()
    model = load_model(model_path)
    setup_seconds = time.perf_counter() - tick
    # Warm-up is excluded. All submitted signals are then processed again and timed.
    warmup_signals = 20
    for i in range(warmup_signals):
        model(load_signal(rows[i % len(rows)])).argmax(1).item()
    print(f"CPU only; threads={threads}; batch size=1; signals={len(rows):,}", flush=True)
    print("Timing every signal, including I/Q loading/conversion, STFT and class selection", flush=True)
    predictions, elapsed = {}, {}
    tick_all = time.perf_counter()
    for i, row in enumerate(rows, 1):
        tick = time.perf_counter()
        label = CLASSES[model(load_signal(row)).argmax(1).item()]
        milliseconds = (time.perf_counter() - tick) * 1000.
        sid = row["sample_id"]
        predictions[sid], elapsed[sid] = label, milliseconds
        if i == 1 or i % 1000 == 0 or i == len(rows):
            average_ms = 1000. * (time.perf_counter() - tick_all) / i
            print(f"Inference: {i:,}/{len(rows):,}; elapsed average {average_ms:.3f} ms/signal", flush=True)
    loop_seconds = time.perf_counter() - tick_all
    if set(predictions) != set(ordered_ids):
        raise RuntimeError("Incomplete predictions; submission was not written")
    times = np.array([elapsed[sid] for sid in ordered_ids], dtype=np.float64)
    stats = dict(samples=len(times), mean_ms=float(times.mean()), median_ms=float(np.median(times)),
                 p95_ms=float(np.percentile(times, 95)), p99_ms=float(np.percentile(times, 99)),
                 max_ms=float(times.max()), fraction_at_or_below_10ms=float(np.mean(times <= 10.)))
    output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(output, ["sample_id", "label"], ((sid, predictions[sid]) for sid in ordered_ids))
    write_csv(latency_path, ["sample_id", "inference_ms"],
              ((sid, f"{elapsed[sid]:.6f}") for sid in ordered_ids))
    report = dict(samples=len(rows), columns=["sample_id", "label"], batch_size=1,
                  model_bytes=model_path.stat().st_size,
                  model_parameters=sum(p.numel() for p in model.parameters()),
                  model_setup_seconds=setup_seconds, warmup_signals_excluded=warmup_signals,
                  single_signal_latency=stats, inference_loop_seconds=loop_seconds,
                  elapsed_average_ms_per_signal=1000. * loop_seconds / len(rows),
                  hardware=dict(device="cpu", cpu=platform.processor(), machine=platform.machine(),
                                logical_cpus=os.cpu_count(), threads=torch.get_num_threads(),
                                platform=platform.platform(), python=sys.version.split()[0],
                                torch=str(torch.__version__), numpy=str(np.__version__)),
                  precision="float32", latency_file=str(latency_path),
                  timing_scope="One signal per forward pass. Each inference_ms includes memory-mapped shard access, "
                               "int16 I/Q loading/conversion, the complete complex STFT, neural network and class selection. "
                               "It excludes one-time index/weight loading, 20 warm-up passes, CSV writing and progress logging. "
                               "OS cache state is uncontrolled; this is not a guaranteed cold-disk benchmark.",
                  total_seconds=time.perf_counter() - start)
    temporary = report_path.with_name(report_path.name + ".partial")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(report_path)
    print(f"Single-signal CPU latency: mean={stats['mean_ms']:.3f} ms, "
          f"median={stats['median_ms']:.3f} ms, p95={stats['p95_ms']:.3f} ms, "
          f"p99={stats['p99_ms']:.3f} ms", flush=True)
    print(f"Submission: {output}\nPer-signal timings: {latency_path}\nTiming summary: {report_path}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Path to the trained neural model.pt")
    parser.add_argument("--participants", type=Path, required=True,
                        help="Blind participants folder or its test subfolder")
    parser.add_argument("--output", type=Path, default=Path("submission.csv"),
                        help="Output CSV (default: submission.csv in the current directory)")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads (default: 4)")
    args = parser.parse_args()
    predict(args.model, args.participants, args.output, args.threads)


if __name__ == "__main__":
    main()


