"""Coherent Fourier classifier: mathematical, split, export and notebook checks."""
import ast
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "track2"))
sys.path.insert(0, str(ROOT / "recorded_dataset"))
import radarml_track2_bank as bank
import radarml_track2_runtime as data
from make_dataset_hackathon import degrade


class BankChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        rng = np.random.default_rng(42)
        templates = (rng.normal(size=(7, 16800)) + 1j * rng.normal(size=(7, 16800))).astype(np.complex64) * .01
        templates[:, :2400] = 0
        cls.checkpoint = dict(templates=torch.from_numpy(templates), labels=torch.arange(7),
                              classes=list(data.CLASSES), track="B", model_family="complex_fft_template_bank")

    def test_expander_noise_is_bitwise_identical(self):
        path = next((ROOT / "recorded_dataset").glob("*_recorded.npy"))
        raw = np.load(path, mmap_mode="r")[0, 0]
        source = json.loads(path.with_suffix(".json").read_text())["source_file"]
        for snr in (30, 0, -25, -30):
            seed = int(hashlib.md5(f"42_{source}_{snr}_0".encode()).hexdigest()[:8], 16)
            expected, _, _ = degrade(raw, snr, np.random.default_rng(seed))
            np.testing.assert_array_equal(bank.add_supplied_noise(raw, snr, source, 0), expected)

    def test_complex_fft_matches_direct_inner_product_and_likelihood(self):
        model = bank.FourierBank(self.checkpoint).eval()
        torch.manual_seed(4)
        signals = torch.randn(3, 2, 16800) * .02
        signals[0, 0, 100] = -1.
        signals[0, 1, 200] = 32767. / 32768.
        dot, noise, alpha = model.components(signals)
        z = torch.complex(signals[:, 0], signals[:, 1])
        expected = z @ self.checkpoint["templates"].conj().T
        torch.testing.assert_close(dot, expected, atol=2e-6, rtol=3e-5)
        likelihood = (2 * expected.real - self.checkpoint["templates"].abs().square().sum(1)) / noise
        torch.testing.assert_close(model(signals, "coherent:max"), likelihood, atol=.01, rtol=3e-5)
        self.assertLess(float(alpha[0]), 1.)
        for scores in model.scores(signals, bank.CANDIDATES).values():
            self.assertEqual(tuple(scores.shape), (3, 7))
            self.assertTrue(torch.isfinite(scores).all())
        pure = torch.stack([self.checkpoint["templates"].real, self.checkpoint["templates"].imag], 1)
        for candidate in bank.CANDIDATES:
            self.assertEqual(model(pure, candidate).argmax(1).tolist(), list(range(7)))

    def test_phase_invariant_rule_and_mixture_class_prior(self):
        model = bank.FourierBank(self.checkpoint)
        torch.manual_seed(9)
        z = torch.complex(torch.randn(2, 16800), torch.randn(2, 16800)) * .02
        to_iq = lambda x: torch.stack((x.real, x.imag), 1)
        a = model(to_iq(z), "unknown_phase:mixture")
        rotated = z * torch.exp(torch.tensor(.7j))
        torch.testing.assert_close(a, model(to_iq(rotated), "unknown_phase:mixture"), atol=.02, rtol=3e-5)
        duplicated = dict(self.checkpoint)
        duplicated["templates"] = torch.cat((duplicated["templates"], duplicated["templates"][3:4]), 0)
        duplicated["labels"] = torch.cat((duplicated["labels"], torch.tensor([3])), 0)
        other = bank.FourierBank(duplicated)
        torch.testing.assert_close(a, other(to_iq(z), "unknown_phase:mixture"), atol=2e-5, rtol=2e-5)

    def test_train_resume_freeze_and_template_ordered_inference(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            infos, rows = {}, []
            rng = np.random.default_rng(5)
            for c, label in enumerate(data.CLASSES):
                name = f"fixture_{label}_recorded.npy"
                # Distinct fixed signals, realistic noise-only leading acquisition window.
                time = np.arange(16800, dtype=np.float32)
                waveform = .03 * np.exp(2j * np.pi * (.03 + c * .04) * time)
                waveform[:2400] = waveform[14640:] = 0
                raw = np.empty((1, 4, 33600), np.int16)
                for pulse in range(4):
                    z = waveform + (rng.normal(size=16800) + 1j*rng.normal(size=16800)) * .0001
                    raw[0, pulse, ::2] = (z.real * 32768).astype(np.int16)
                    raw[0, pulse, 1::2] = (z.imag * 32768).astype(np.int16)
                    rows.append(dict(file=name, pulse_index=pulse, split="train" if pulse < 2 else "validation"))
                path = root / name
                np.save(path, raw)
                infos[name] = dict(path=str(path), kind="compact", label=c, source_file=name, snr_to_axis={})
            cfg = dict(bank.DEFAULT_CONFIG, train_pulses_per_file=2, selection_pulses_per_file=1,
                       batch_size=7, workers=0)
            run = root / "run"
            report = bank.train_and_freeze(root, ROOT / "hackathon/split_manifest.csv", run, cfg,
                                           kind="compact", prepared=(rows, infos))
            self.assertEqual(report["samples"], 182)
            frozen = run / "frozen"
            self.assertEqual(json.loads((frozen / "audit_validation.json").read_text())["samples"], 91)
            hashes = bank.verify_bundle(frozen)
            self.assertIn("training_data_hashes.json", hashes)
            # Cached templates must resume without reading source samples again.
            with patch.object(bank, "mapped_array", side_effect=AssertionError("Source reread on resume")):
                restored, _ = bank.fit_bank(rows, infos, run / "training_cache", cfg)
            checkpoint = torch.load(frozen / "bank.pt", map_location="cpu", weights_only=True)
            torch.testing.assert_close(restored["templates"], checkpoint["templates"])
            self.assertEqual(bank.train_and_freeze(root, ROOT / "hackathon/split_manifest.csv", run, cfg), report)
            # Anonymous inference fixture deliberately changes index and template order.
            blind = root / "blind"
            blind.mkdir()
            raw = np.stack([np.load(infos[f"fixture_{label}_recorded.npy"]["path"])[0, 2]
                            for label in data.CLASSES])
            np.save(blind / "signals.npy", raw)
            with (blind / "index.csv").open("w", newline="") as f:
                w = csv.writer(f); w.writerow(["sample_id", "file", "row"])
                w.writerows((f"anon-{i}", "signals.npy", i) for i in [4, 2, 6, 0, 3, 5, 1])
            ids = [f"anon-{i}" for i in reversed(range(7))]
            with (blind / "submission_template.csv").open("w", newline="") as f:
                w = csv.writer(f); w.writerow(["sample_id", "predicted_label"])
                w.writerows((sid, "") for sid in ids)
            original = data.read_blind_index
            with patch.object(data, "read_blind_index", side_effect=lambda p: original(p, expected_count=7)):
                inference = bank.predict_blind(frozen, blind, root / "submission.csv", batch_size=3, workers=0)
            self.assertEqual(inference["samples"], 7)
            with (root / "submission.csv").open() as f:
                reader = csv.DictReader(f); saved = list(reader)
                self.assertEqual(reader.fieldnames, ["sample_id", "predicted_label"])
            self.assertEqual([r["sample_id"] for r in saved], ids)
            self.assertEqual([r["predicted_label"] for r in saved], list(reversed(data.CLASSES)))
            # Execute the actual notebook's final cell in train-and-submit mode.
            # Its subprocess handoff runs inference here with only fixture size patched.
            import nbformat
            notebook = nbformat.read(ROOT / "status/radar-track2-strong.ipynb", as_version=4)
            cell = next(c.source for c in notebook.cells if c.cell_type == "code"
                        and "def find_participants_root(" in c.source)
            output = root / "from_notebook.csv"
            namespace = dict(Path=Path, hashlib=hashlib, json=json, subprocess=subprocess, sys=sys,
                             GENERATE_SUBMISSION=True, TRAIN_MODEL=True, RUN_DIR=run,
                             FROZEN_BUNDLE=None, BLIND_ROOT=blind, PREDICTIONS_PATH=output,
                             BATCH_SIZE=3, WORKERS=0, WORK_ROOT=root)

            def execute_frozen_command(command, check):
                self.assertTrue(check)
                self.assertEqual(Path(command[1]), frozen.resolve() / "radarml_track2_bank.py")
                options = {command[i]: command[i + 1] for i in range(3, len(command), 2)}
                with patch.object(data, "read_blind_index", side_effect=lambda p: original(p, expected_count=7)):
                    bank.predict_blind(options["--bundle"], options["--blind-root"], options["--output"],
                                       int(options["--batch-size"]), int(options["--workers"]))

            with patch.object(subprocess, "run", side_effect=execute_frozen_command) as runner, \
                    patch("IPython.display.display"), patch("builtins.print"):
                exec(compile(cell, "notebook_submission_cell", "exec"), namespace)
            runner.assert_called_once()
            with output.open() as stream:
                self.assertEqual(list(csv.DictReader(stream)), saved)
            self.assertTrue(output.with_name(output.name + ".inference.json").is_file())
            (frozen / "config.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "Frozen artifact changed"):
                bank.verify_bundle(frozen)

    def test_notebook_finds_nested_participants_and_rejects_ambiguity(self):
        import nbformat
        notebook = nbformat.read(ROOT / "status/radar-track2-strong.ipynb", as_version=4)
        cell = next(c.source for c in notebook.cells if c.cell_type == "code"
                    and "def find_participants_root(" in c.source)
        function = next(node for node in ast.parse(cell).body if isinstance(node, ast.FunctionDef))
        namespace = {"Path": Path}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "participant_finder", "exec"), namespace)
        find = namespace["find_participants_root"]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaisesRegex(FileNotFoundError, "No participant package"):
                find(search_root=root)
            package = root / "kaggle-dataset" / "participants"
            (package / "test").mkdir(parents=True)
            (package / "test/index.csv").write_text("sample_id,file,row\n")
            (package / "submission_template.csv").write_text("sample_id,predicted_label\n")
            self.assertEqual(find(search_root=root), package.resolve())
            self.assertEqual(find(explicit=package.parent), package.resolve())
            self.assertEqual(find(explicit=package / "test"), package.resolve())
            other = root / "another-dataset"
            other.mkdir()
            (other / "index.csv").write_text("sample_id,file,row\n")
            (other / "submission_template.csv").write_text("sample_id,predicted_label\n")
            with self.assertRaisesRegex(ValueError, "Multiple participant packages"):
                find(search_root=root)
            self.assertEqual(find(explicit=package), package.resolve())

    def test_notebook_detects_development_data_in_renamed_kaggle_folders(self):
        import nbformat
        notebook = nbformat.read(ROOT / "status/radar-track2-strong.ipynb", as_version=4)
        cell = next(c.source for c in notebook.cells if c.cell_type == "code"
                    and "def find_development_root(" in c.source)
        function = next(node for node in ast.parse(cell).body if isinstance(node, ast.FunctionDef))
        namespace = {"Path": Path, "helpers": data}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "development_finder", "exec"), namespace)
        find = namespace["find_development_root"]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaisesRegex(FileNotFoundError, "No complete development dataset"):
                find(search_root=root)
            expanded = root / "datasets/hadimanes/radar-ml/expanded (2)/expanded"
            expanded.mkdir(parents=True)
            for i in range(660):
                (expanded / f"fixture_{i:04d}_single_channel.npy").touch()
            participants = root / "participants/test"
            participants.mkdir(parents=True)
            (participants / "signals_0000.npy").touch()
            self.assertEqual(find(search_root=root), expanded.resolve())
            self.assertEqual(find(explicit=expanded.parent), expanded.resolve())
            compact = root / "another-dataset/recorded_dataset"
            compact.mkdir(parents=True)
            for i in range(660):
                (compact / f"fixture_{i:04d}_recorded.npy").touch()
            with self.assertRaisesRegex(ValueError, "Multiple development datasets"):
                find(search_root=root)
            self.assertEqual(find(kind="expanded", search_root=root), expanded.resolve())
            self.assertEqual(find(kind="compact", search_root=root), compact.resolve())
            with self.assertRaisesRegex(FileNotFoundError, "does not exist"):
                find(explicit=root / "wrong-folder")

    def test_notebook_valid_compiles_and_embeds_exact_sources(self):
        import nbformat
        from IPython.core.inputtransformer2 import TransformerManager
        notebook = nbformat.read(ROOT / "status/radar-track2-strong.ipynb", as_version=4)
        nbformat.validate(notebook)
        transformer = TransformerManager()
        embedded = {}
        for i, cell in enumerate(notebook.cells):
            if cell.cell_type != "code":
                continue
            compile(transformer.transform_cell(cell.source), f"notebook_cell_{i}", "exec")
            if cell.source.startswith("%%writefile "):
                line, source = cell.source.split("\n", 1)
                embedded[line.split()[1]] = source
        for name in ("radarml_track2_bank.py", "radarml_track2_runtime.py"):
            self.assertEqual(ast.dump(ast.parse(embedded[name])),
                             ast.dump(ast.parse((ROOT / "track2" / name).read_text())))


if __name__ == "__main__":
    unittest.main()
