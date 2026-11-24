"""Final CPU submission: real weights, isolated two-path CLI and per-signal timings."""
import csv
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "results-2/track2_neural_v1/export/model.pt"
SCRIPT = ROOT / "track2/neural_submission.py"


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class SubmissionChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.final = module("final_neural", SCRIPT)
        cls.original = module("original_neural", ROOT / "track2/radarml_track2_neural_infer.py")

    def test_real_checkpoint_logits_match_original_implementation(self):
        final = self.final.load_model(MODEL)
        original, _ = self.original.load_model(MODEL, torch.device("cpu"))
        shard = np.load(ROOT / "status/participants/test/signals_0000.npy", mmap_mode="r")
        raw = shard[np.arange(0, 256, 8)]
        signals = torch.from_numpy(np.stack((raw[:, ::2], raw[:, 1::2]), 1).astype(np.float32) / 32768.)
        with torch.inference_mode():
            expected = original(signals)
            actual = torch.cat([final(signal[None]) for signal in signals])
        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-5)
        self.assertEqual(actual.argmax(1).tolist(), expected.argmax(1).tolist())

    def test_isolated_cli_requires_only_two_paths_and_writes_each_signal_time(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            shutil.copy2(SCRIPT, root / "infer.py")
            shutil.copy2(MODEL, root / "model.pt")
            blind = root / "participants"
            (blind / "test").mkdir(parents=True)
            shard = np.load(ROOT / "status/participants/test/signals_0000.npy", mmap_mode="r")
            raw = np.array(shard[:7])
            np.save(blind / "test/signals.npy", raw)
            with (blind / "test/index.csv").open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["sample_id", "file", "row"])
                writer.writerows((f"sample-{i}", "signals.npy", i) for i in (4, 2, 6, 0, 3, 5, 1))
            ids = [f"sample-{i}" for i in reversed(range(7))]
            with (blind / "submission_template.csv").open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["sample_id", "s"])
                writer.writerows((sid, "") for sid in ids)
            result = subprocess.run(
                [sys.executable, "infer.py", "--model", "model.pt", "--participants", "participants"],
                cwd=root, env=dict(os.environ, PYTHONPATH=""), capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            with (root / "submission.csv").open() as stream:
                reader = csv.DictReader(stream)
                predictions = list(reader)
                self.assertEqual(reader.fieldnames, ["sample_id", "label"])
            with (root / "submission.csv.latency.csv").open() as stream:
                timings = list(csv.DictReader(stream))
            self.assertEqual([r["sample_id"] for r in predictions], ids)
            self.assertEqual([r["sample_id"] for r in timings], ids)
            self.assertTrue(all(float(r["inference_ms"]) > 0 for r in timings))
            report = json.loads((root / "submission.csv.inference.json").read_text())
            self.assertEqual(report["samples"], 7)
            self.assertEqual(report["batch_size"], 1)
            self.assertEqual(report["single_signal_latency"]["samples"], 7)
            self.assertEqual(report["hardware"]["device"], "cpu")
            self.assertAlmostEqual(report["single_signal_latency"]["mean_ms"],
                                   np.mean([float(r["inference_ms"]) for r in timings]), places=5)
            final = self.final.load_model(MODEL)
            signals = torch.from_numpy(np.stack((raw[:, ::2], raw[:, 1::2]), 1).astype(np.float32) / 32768.)
            with torch.inference_mode():
                expected = final(signals).argmax(1).tolist()
            self.assertEqual([r["label"] for r in predictions],
                             [self.final.CLASSES[expected[int(sid.split("-")[1])]] for sid in ids])
            # Direct test-subfolder paths work too, with the same official ordering.
            _, _, direct_ids = self.final.read_index(blind / "test")
            self.assertEqual(direct_ids, ids)
            with self.assertRaisesRegex(ValueError, "outside"):
                self.final.predict(MODEL, blind, blind / "submission.csv")
            self.assertFalse((blind / "submission.csv").exists())


if __name__ == "__main__":
    unittest.main()
