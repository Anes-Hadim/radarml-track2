"""Neural Track 2: phase-preserving STFT, real training/resume, isolated inference."""
import ast
import copy
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "track2"))
import radarml_track2_neural_infer as inference
import radarml_track2_neural_train as training


class NeuralChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_stft_preserves_energy_phase_and_batch_independence(self):
        torch.manual_seed(42)
        signals = torch.randn(3, 2, 16800)
        representation = inference.ComplexSTFT()
        complex_view, image, stats = representation(signals)
        self.assertEqual(tuple(complex_view.shape), (3, 2, 256, 66))
        self.assertEqual(tuple(image.shape), (3, 1, 256, 66))
        self.assertEqual(tuple(stats.shape), (3, 4))
        energy = complex_view.square().sum((1, 2, 3))
        torch.testing.assert_close(energy, torch.full_like(energy, 16800.), rtol=2e-6, atol=.005)
        z = torch.complex(signals[:, 0], signals[:, 1])
        rotated = z * torch.exp(torch.tensor(.7j))
        other = representation(torch.stack((rotated.real, rotated.imag), 1))[0]
        self.assertFalse(torch.allclose(complex_view, other))
        model = inference.CompactSTFTClassifier().eval()
        with torch.inference_mode():
            together = model(signals)
            separate = torch.cat([model(signal[None]) for signal in signals])
        torch.testing.assert_close(together, separate, rtol=3e-5, atol=3e-6)
        self.assertLess(sum(p.numel() * p.element_size() for p in model.parameters()), 11_000_000)
        self.assertEqual(tuple(together.shape), (3, 7))
        # Exercise gradients and an optimizer step on the production-size architecture.
        model.train()
        before = model.phase_global[1].weight.detach().clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        loss = torch.nn.functional.cross_entropy(model(signals), torch.tensor([0, 1, 2]))
        loss.backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
        optimizer.step()
        self.assertFalse(torch.equal(before, model.phase_global[1].weight))

    def test_noise_augmentation_all_snrs_clips_and_preserves_int16_grid(self):
        torch.manual_seed(4)
        base = torch.randn(512, 2, 16800) * .0001
        t = torch.arange(16800)
        base[:, 0, 2400:14400] += .015 * torch.cos(.1 * t[2400:14400])
        base[:, 1, 2400:14400] += .015 * torch.sin(.1 * t[2400:14400])
        augment = training.NoiseAugment(training.helpers.ALL_SNRS)
        noisy, snrs = augment(base)
        self.assertEqual(set(snrs.int().tolist()), set(training.helpers.ALL_SNRS))
        self.assertGreaterEqual(float(noisy.min()), -1.)
        self.assertLessEqual(float(noisy.max()), 32767. / 32768.)
        self.assertTrue(torch.equal(noisy * 32768., torch.trunc(noisy * 32768.)))
        _, easy_snrs = augment(base[:64], easy=True)
        self.assertGreaterEqual(float(easy_snrs.min()), -15.)
        self.assertFalse(torch.equal(noisy, base))

    def test_compact_validation_noise_matches_supplied_expander(self):
        sys.path.insert(0, str(ROOT / "recorded_dataset"))
        from make_dataset_hackathon import degrade
        path = next((ROOT / "recorded_dataset").glob("*_recorded.npy"))
        raw = np.load(path, mmap_mode="r")[0, 0]
        source = json.loads(path.with_suffix(".json").read_text())["source_file"]
        for snr in (30, -25, -30):
            seed = int(hashlib.md5(f"42_{source}_{snr}_0".encode()).hexdigest()[:8], 16)
            expected = degrade(raw, snr, np.random.default_rng(seed))[0]
            np.testing.assert_array_equal(training.supplied_noise(raw, snr, source, 0), expected)

    def test_real_training_interruption_resume_export_and_isolated_inference(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows, infos = [], {}
            rng = np.random.default_rng(3)
            for c, label in enumerate(inference.CLASSES):
                name = f"fixture_{label}_recorded.npy"
                phase = np.arange(16800) * (.05 + .02 * c)
                wave = .02 * np.exp(1j * phase)
                wave[:2400] = wave[14640:] = 0
                raw = np.empty((1, 4, 33600), np.int16)
                for pulse in range(4):
                    z = wave + (rng.normal(size=16800) + 1j * rng.normal(size=16800)) * .0001
                    raw[0, pulse, ::2] = (z.real * 32768).astype(np.int16)
                    raw[0, pulse, 1::2] = (z.imag * 32768).astype(np.int16)
                    rows.append(dict(file=name, pulse_index=pulse, split="train" if pulse < 2 else "validation"))
                path = root / name
                np.save(path, raw)
                infos[name] = dict(path=str(path), kind="compact", label=c, source_file=name, snr_to_axis={})
            cfg = copy.deepcopy(training.DEFAULT_CONFIG)
            cfg.update(epochs=2, samples_per_epoch=7, batch_size=7, workers=0, max_run_hours=.1,
                       train_pulses_per_file=2, selection_pulses_per_file=1, warmup_epochs=0)
            cfg["model_kwargs"].update(projection_rank=8, hidden=16, dropout=0.)
            run = root / "run"
            actual_evaluate = training.evaluate
            calls = []

            def interrupt_second_epoch(*args, **kwargs):
                calls.append(1)
                if len(calls) == 2:
                    raise KeyboardInterrupt()
                return actual_evaluate(*args, **kwargs)

            with patch.object(training, "evaluate", side_effect=interrupt_second_epoch):
                with self.assertRaises(KeyboardInterrupt):
                    training.train(root, ROOT / "hackathon/split_manifest.csv", run, cfg,
                                   kind="compact", prepared=(rows, infos))
            resume = torch.load(run / "resume.pt", map_location="cpu", weights_only=True)
            self.assertEqual(resume["next_epoch"], 2)
            raw_cache = np.load(run / "training_cache/train_iq.npy", mmap_mode="r")
            self.assertEqual(raw_cache.shape, (14, 33600))
            for i, name in enumerate(sorted(infos)):
                source = np.load(infos[name]["path"])
                np.testing.assert_array_equal(raw_cache[i * 2:i * 2 + 2], source[0, :2])
            report = training.train(root, ROOT / "hackathon/split_manifest.csv", run, cfg,
                                    kind="compact", prepared=(rows, infos))
            self.assertEqual(report["samples"], 182)
            history = json.loads((run / "history.json").read_text())
            self.assertEqual([r["epoch"] for r in history], [1, 2])
            self.assertEqual(sum(row["draws"] for row in history), 14)
            audit = json.loads((run / "export/audit_validation.json").read_text())
            self.assertEqual(audit["samples"], 91)
            model_zip, resume_zip = training.package(run)
            with zipfile.ZipFile(model_zip) as archive:
                names = archive.namelist()
                self.assertIn("model.pt", names)
                self.assertIn("radarml_track2_neural_infer.py", names)
                self.assertFalse(any("cache" in n or n == "resume.pt" for n in names))
            with zipfile.ZipFile(resume_zip) as archive:
                self.assertIn("resume.pt", archive.namelist())
                self.assertFalse(any("cache" in n for n in archive.namelist()))
            saved = torch.load(run / "export/model.pt", map_location="cpu", weights_only=True)
            self.assertNotIn("templates", saved)
            self.assertNotIn("optimizer", saved)
            self.assertTrue(all("template" not in key for key in saved["state_dict"]))
            # Save only inference code + model + seven anonymous fixture signals into
            # another directory, then execute with no repo/training modules on sys.path.
            isolated = root / "isolated"
            isolated.mkdir()
            for name in ("model.pt", "radarml_track2_neural_infer.py"):
                shutil.copy2(run / "export" / name, isolated / name)
            blind = isolated / "participants"
            (blind / "test").mkdir(parents=True)
            raw = np.stack([np.load(infos[f"fixture_{label}_recorded.npy"]["path"])[0, 2]
                            for label in inference.CLASSES])
            np.save(blind / "test/signals.npy", raw)
            with (blind / "test/index.csv").open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["sample_id", "file", "row"])
                writer.writerows((f"anon-{i}", "signals.npy", i) for i in (4, 2, 6, 0, 3, 5, 1))
            ids = [f"anon-{i}" for i in reversed(range(7))]
            with (blind / "submission_template.csv").open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["sample_id", "predicted_label"])
                writer.writerows((sid, "") for sid in ids)
            env = dict(os.environ, PYTHONPATH="")
            result = subprocess.run([sys.executable, "-c",
                "import radarml_track2_neural_infer as m; "
                "m.predict('model.pt','participants','submission.csv',batch_size=3,workers=0,threads=2,"
                "benchmark_samples=7,warmup=1,expected_count=7)"], cwd=isolated, env=env,
                capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            with (isolated / "submission.csv").open() as stream:
                reader = csv.DictReader(stream)
                predictions = list(reader)
                self.assertEqual(reader.fieldnames, ["sample_id", "label"])
            self.assertEqual([r["sample_id"] for r in predictions], ids)
            timing = json.loads((isolated / "submission.csv.inference.json").read_text())
            self.assertEqual(timing["samples"], 7)
            self.assertEqual(timing["hardware"]["device"], "cpu")
            self.assertEqual(timing["single_signal_latency"]["samples"], 7)
            self.assertIn("full complex STFT", timing["timing_scope"])
            model, _ = inference.load_model(isolated / "model.pt", torch.device("cpu"))
            signals = torch.from_numpy(np.stack([np.stack((r[::2], r[1::2])).astype(np.float32) / 32768. for r in raw]))
            with torch.inference_mode():
                expected = model(signals).argmax(1).tolist()
            self.assertEqual([r["label"] for r in predictions],
                             [inference.CLASSES[expected[int(sid.split('-')[1])]] for sid in ids])

    def test_notebook_valid_compiles_and_embeds_exact_sources(self):
        import nbformat
        from IPython.core.inputtransformer2 import TransformerManager
        notebook = nbformat.read(ROOT / "status/radar-track2-neural.ipynb", as_version=4)
        nbformat.validate(notebook)
        transform = TransformerManager()
        embedded = {}
        for i, cell in enumerate(notebook.cells):
            if cell.cell_type != "code":
                continue
            compile(transform.transform_cell(cell.source), f"cell_{i}", "exec")
            if cell.source.startswith("%%writefile "):
                line, source = cell.source.split("\n", 1)
                embedded[line.split()[1]] = source
        for name in ("radarml_track2_runtime.py", "radarml_track2_neural_infer.py", "radarml_track2_neural_train.py"):
            self.assertEqual(ast.dump(ast.parse(embedded[name])), ast.dump(ast.parse((ROOT / "track2" / name).read_text())))
        self.assertNotIn("radarml_track2_bank.py", embedded)


if __name__ == "__main__":
    unittest.main()
