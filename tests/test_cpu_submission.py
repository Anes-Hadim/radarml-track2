"""CPU submission timing, reference agreement and output integrity checks."""
import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("cpu_submission", ROOT / "status/generate_submission.py")
cpu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cpu)


class CPUInferenceChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.saved_path = list(sys.path)
        cls.runtime_names = ("radarml_track2_runtime", "radarml_track2_bank")
        cls.saved_modules = {name: sys.modules.pop(name) for name in cls.runtime_names if name in sys.modules}
        cls.bundle = cpu.DEFAULT_BUNDLE.resolve()
        cpu.verify_manifest(cls.bundle, "frozen.sha256.json")
        cls.runtime = cpu.load_frozen_runtime(cls.bundle)

    @classmethod
    def tearDownClass(cls):
        for name in cls.runtime_names:
            sys.modules.pop(name, None)
        sys.modules.update(cls.saved_modules)
        sys.path[:] = cls.saved_path

    def test_latency_statistics_report_tail_and_reference_separately(self):
        stats = cpu.latency_statistics([1., 2., 3., 40.], 10., np)
        self.assertEqual(stats["samples"], 4)
        self.assertEqual(stats["mean_ms"], 11.5)
        self.assertEqual(stats["median_ms"], 2.5)
        self.assertEqual(stats["fraction_at_or_below_target"], .75)
        self.assertFalse(stats["mean_at_or_below_target"])
        self.assertFalse(stats["p95_at_or_below_target"])
        with self.assertRaises(ValueError):
            cpu.latency_statistics([], 10., np)

    def test_packed_real_execution_matches_complex_fft_likelihoods(self):
        frozen, cfg = cpu.load_cpu_model(self.runtime, self.bundle, torch)
        torch.manual_seed(42)
        signals = torch.randn(8, 2, 16800) * .01
        signals[0, 0, :100] = -1.
        signals[0, 1, :100] = 32767. / 32768.
        with torch.inference_mode():
            for candidate in ("coherent:max", "coherent:mixture", "coherent_clip:max", "coherent_clip:mixture"):
                packed = cpu.make_cpu_executor(frozen, candidate, torch)
                expected, actual = frozen(signals, candidate), packed(signals, candidate)
                row_scale = expected.abs().amax(1, keepdim=True).clamp_min(1.)
                self.assertLess(float(((actual - expected).abs() / row_scale).max()), 2e-5)
                self.assertEqual(expected.argmax(1).tolist(), actual.argmax(1).tolist())

    def test_complete_cpu_script_including_single_signal_fft_and_csv(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            participants = root / "participants"
            shard_root = participants / "test"
            shard_root.mkdir(parents=True)
            checkpoint = torch.load(self.bundle / "bank.pt", map_location="cpu", weights_only=True)
            indices = [int(torch.where(checkpoint["labels"] == c)[0][0]) for c in range(7)]
            templates = checkpoint["templates"][indices].numpy()
            raw = np.empty((7, 33600), dtype=np.int16)
            raw[:, ::2] = np.clip(templates.real * 32768, -32768, 32767).astype(np.int16)
            raw[:, 1::2] = np.clip(templates.imag * 32768, -32768, 32767).astype(np.int16)
            np.save(shard_root / "fixture.npy", raw)
            with (shard_root / "index.csv").open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["sample_id", "file", "row"])
                writer.writerows((f"fixture-{i}", "fixture.npy", i) for i in [4, 2, 6, 0, 3, 5, 1])
            template_ids = [f"fixture-{i}" for i in reversed(range(7))]
            with (participants / "submission_template.csv").open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["sample_id", "predicted_label"])
                writer.writerows((sid, "") for sid in template_ids)
            reference = root / "reference.csv"
            with reference.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["sample_id", "label"])
                writer.writerows((f"fixture-{i}", self.runtime.CLASSES[i]) for i in range(7))
            checksums = {p.relative_to(participants).as_posix(): cpu.sha256(p)
                         for p in participants.rglob("*") if p.is_file()}
            (participants / "checksums.json").write_text(json.dumps(checksums))
            output = root / "cpu.csv"
            original = self.runtime.data.read_blind_index
            args = ["--bundle", str(self.bundle), "--participants", str(participants),
                    "--output", str(output), "--reference", str(reference), "--threads", "2",
                    "--batch-size", "3", "--benchmark-samples", "7", "--warmup", "2"]
            with patch.object(self.runtime.data, "read_blind_index",
                              side_effect=lambda p: original(p, expected_count=7)):
                self.assertEqual(cpu.main(args), 0)
            with output.open() as stream:
                reader = csv.DictReader(stream)
                self.assertEqual(reader.fieldnames, ["sample_id", "label"])
                saved = list(reader)
            self.assertEqual([r["sample_id"] for r in saved], template_ids)
            report = json.loads(output.with_name(output.name + ".inference.json").read_text())
            self.assertEqual(report["device"], "cpu")
            self.assertEqual(report["hardware"]["device"], "cpu")
            self.assertEqual(report["hardware"]["accelerators_used"], [])
            self.assertEqual(report["samples"], 7)
            self.assertEqual(report["reference_comparison"]["different_predictions"], 0)
            self.assertEqual(report["predictions_sha256"], cpu.sha256(output))
            self.assertEqual(report["prediction_columns"], ["sample_id", "label"])
            benchmark = report["single_signal_benchmark"]
            self.assertEqual(benchmark["batch_size"], 1)
            self.assertEqual(benchmark["end_to_end"]["samples"], 7)
            self.assertEqual(benchmark["warmup_signals_excluded"], 2)
            self.assertGreaterEqual(benchmark["end_to_end"]["mean_ms"], benchmark["model_with_fft"]["mean_ms"])
            self.assertEqual(report["batch_inference"]["batch_size"], 3)
            self.assertIn("input FFT", report["batch_inference"]["timing_scope"])
            self.assertAlmostEqual(report["milliseconds_per_signal"], report["inference_seconds"] / 7 * 1000)
            self.assertTrue(output.with_name(output.name + ".latency.csv").is_file())
            self.assertFalse(output.with_name(output.name + ".partial").exists())
            # Reference comparison must not silently accept missing/extra IDs.
            reference.write_text("sample_id,predicted_label\nwrong,cw\n")
            with self.assertRaisesRegex(ValueError, "mismatched IDs"):
                cpu.compare_reference(reference, {r["sample_id"]: r["label"] for r in saved}, self.runtime.CLASSES)

    def test_incomplete_predictions_do_not_publish_a_csv(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "bad.csv"
            with self.assertRaisesRegex(ValueError, "Incomplete predictions"):
                cpu.write_submission(output, [{"sample_id": "one"}], ["one"], [], self.runtime.CLASSES)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
