"""Track B checks using development pulses and anonymous local fixtures only."""
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
sys.path.insert(0, str(ROOT / "track2"))
import radarml_track2_runtime as rt

spec = importlib.util.spec_from_file_location("organizer_score", ROOT / "hackathon/score_submission.py")
organizer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(organizer)


class Track2Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        rt.seed_everything(42)
        cls.rows = rt.load_manifest(ROOT / "hackathon/split_manifest.csv")
        signals, raw_rows = [], []
        for label in rt.CLASSES:
            row = next(r for r in cls.rows if r["split"] == "train" and rt.family(r["file"]) == label)
            array = np.load(ROOT / "recorded_dataset" / row["file"], mmap_mode="r")
            raw = array[0, row["pulse_index"]].copy()
            signals.append(torch.from_numpy(rt.interleaved_to_channels(raw)))
            raw_rows.append(raw)
        cls.signals, cls.raw_rows = torch.stack(signals), np.stack(raw_rows)

    def test_published_manifest_and_source_mapping(self):
        counts = {}
        for row in self.rows:
            counts.setdefault(row["file"], {"train": 0, "validation": 0})[row["split"]] += 1
        self.assertEqual(len(counts), 660)
        for name, count in counts.items():
            self.assertEqual(count, {"train": 80, "validation": 20})
            meta = json.loads((ROOT / "recorded_dataset" / name).with_suffix(".json").read_text())
            self.assertEqual(rt.expanded_name_from_source(meta["source_file"]),
                             name.replace("_recorded.npy", "_single_channel.npy"))
        fake_infos = {name: {"label": rt.CLASSES.index(rt.family(name))} for name in counts}
        train = rt.build_examples(self.rows, fake_infos, "train", 80, rt.ALL_SNRS)
        val = rt.build_examples(self.rows, fake_infos, "validation", 20, rt.ALL_SNRS)
        self.assertEqual(len(train), 686400)
        self.assertEqual(len(val), 171600)
        self.assertFalse({e[:2] for e in train} & {e[:2] for e in val})

    def test_two_sided_stft_and_actual_training_step(self):
        model = rt.Track2Model()
        view = model.representation(self.signals)
        self.assertEqual(tuple(view.shape), (7, 2, 256, 256))
        self.assertTrue(torch.isfinite(view).all())
        # Positive/negative baseband tones must appear on their respective halves.
        time = torch.arange(16800) / rt.SAMPLE_RATE_HZ
        tones = torch.stack([torch.stack([torch.cos(2*torch.pi*f*time), torch.sin(2*torch.pi*f*time)])
                             for f in (-15e6, 15e6)])
        image = model.representation(tones)
        peak = image[:, 0].mean(-1).argmax(-1)
        self.assertLess(int(peak[0]), 128)
        self.assertGreater(int(peak[1]), 128)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        before = model.head[-1].weight.detach().clone()
        logits = model(self.signals)
        loss = torch.nn.functional.cross_entropy(logits, torch.arange(7))
        loss.backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
        optimizer.step()
        self.assertFalse(torch.equal(before, model.head[-1].weight))
        model.eval()
        with torch.inference_mode():
            one_at_a_time = torch.cat([model(x[None]) for x in self.signals])
            batched = model(self.signals)
        torch.testing.assert_close(one_at_a_time, batched, atol=2e-6, rtol=2e-5)

    def test_metric_matches_organizer_across_all_snrs(self):
        matrices = {s: np.zeros((7, 7), dtype=np.int64) for s in rt.ALL_SNRS}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with (root / "key.csv").open("w", newline="") as a, (root / "pred.csv").open("w", newline="") as b:
                key, pred = csv.writer(a), csv.writer(b)
                key.writerow(["sample_id", "split", "label", "requested_snr_db"])
                pred.writerow(["sample_id", "predicted_label"])
                n = 0
                for j, snr in enumerate(rt.ALL_SNRS):
                    # Deliberately unequal class counts and nontrivial errors.
                    for cls in range(7):
                        for k in range(cls + 1):
                            guess = cls if (j+k) % 3 else (cls+1) % 7
                            key.writerow([str(n), "test", rt.CLASSES[cls], snr])
                            pred.writerow([str(n), rt.CLASSES[guess]])
                            matrices[snr][cls, guess] += 1
                            n += 1
            actual = rt.report_from_matrices(matrices)
            expected = organizer.score(root / "key.csv", root / "pred.csv")
            self.assertEqual(actual["mean_balanced_accuracy"], expected["mean_balanced_accuracy"])
            for a, b in zip(actual["per_snr"], expected["per_snr"]):
                for field in ("requested_snr_db", "balanced_accuracy", "accuracy", "confusion_matrix"):
                    self.assertEqual(a[field], b[field])

    def test_fsk_collision_recovery_uses_hash_not_upload_suffix(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            names = ["10us_fsk#0_10Mhz_5Mhz--5Mhz_recorded.npy", "10us_fsk#1_10Mhz_5Mhz--5Mhz_recorded.npy"]
            for index, name in enumerate(names):
                # Intentionally swap numeric upload suffixes and destroy subtype identity.
                array_path = root / f"10us_fsk0_10Mhz_5Mhz--5Mhz_single_channel ({1-index}).npy"
                np.save(array_path, np.full((1, 1), index, dtype=np.int16))
                meta = {"source_file": name, "output_sha256": rt.lookup_sha256(array_path)}
                (root / f"10us_fsk0_10Mhz_5Mhz--5Mhz_single_channel ({index}).json").write_text(json.dumps(meta))
            lookup = rt.build_file_lookup(root, "expanded")
            for index, name in enumerate(names):
                resolved = rt.resolve_file(name, lookup, "expanded")
                _, meta = rt.resolve_metadata(name, resolved, lookup, "expanded")
                self.assertEqual(int(np.load(resolved)[0, 0]), index)
                self.assertEqual(meta["source_file"], name)

    def test_frozen_blind_inference_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bundle, blind = root / "bundle", root / "blind"
            bundle.mkdir()
            blind.mkdir()
            model = rt.Track2Model()
            cfg = dict(rt.DEFAULT_CONFIG, track="B", classes=list(rt.CLASSES), inference_precision="float32")
            rt.write_json(bundle / "config.json", cfg)
            torch.save({"state_dict": model.state_dict(), "classes": list(rt.CLASSES),
                        "track": "B", "model_kwargs": cfg["model_kwargs"]}, bundle / "model.pt")
            (bundle / "radarml_track2_runtime.py").write_bytes(Path(rt.__file__).read_bytes())
            (bundle / "requirements.txt").write_text("numpy\ntorch\n")
            hashes = {p.name: rt.lookup_sha256(p) for p in bundle.iterdir()}
            rt.write_json(bundle / "frozen.sha256.json", hashes)
            np.save(blind / "anonymous.npy", self.raw_rows)
            with (blind / "index.csv").open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["sample_id", "file", "row"])
                writer.writerows((f"anon-{i}", "anonymous.npy", i) for i in (4, 2, 6, 0, 3, 5, 1))
            template_ids = [f"anon-{i}" for i in reversed(range(7))]
            with (blind / "submission_template.csv").open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["sample_id", "predicted_label"])
                writer.writerows((sid, "") for sid in template_ids)
            # Only fixture size is reduced; production requires exactly 171,600.
            original_reader = rt.read_blind_index
            with patch.object(rt, "read_blind_index", side_effect=lambda p: original_reader(p, expected_count=7)):
                output = root / "predictions.csv"
                report = rt.predict_blind(bundle, blind, output, batch_size=3, workers=0)
                self.assertEqual(report["samples"], 7)
            with output.open() as stream:
                reader = csv.DictReader(stream)
                self.assertEqual(reader.fieldnames, ["sample_id", "predicted_label"])
                saved = list(reader)
            self.assertEqual([r["sample_id"] for r in saved], template_ids)
            self.assertTrue(all(r["predicted_label"] in rt.CLASSES for r in saved))
            with torch.inference_mode():
                model.eval()
                expected = model(self.signals).argmax(1).tolist()
            self.assertEqual([r["predicted_label"] for r in saved],
                             [rt.CLASSES[expected[int(sid.split("-")[1])]] for sid in template_ids])
            (bundle / "config.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "Frozen artifact changed"):
                rt.verify_bundle(bundle)
            # Fail closed on duplicate IDs and path traversal in the actual CSV loader.
            (blind / "index.csv").write_text("sample_id,file,row\nanon-0,../outside.npy,0\n")
            (blind / "submission_template.csv").write_text("sample_id,predicted_label\nanon-0,\n")
            with self.assertRaisesRegex(ValueError, "escapes"):
                original_reader(blind, expected_count=1)

    def test_notebook_is_valid_and_embedded_runtime_matches(self):
        import ast
        import nbformat
        from IPython.core.inputtransformer2 import TransformerManager
        notebook = nbformat.read(ROOT / "radarml_track2_kaggle.ipynb", as_version=4)
        nbformat.validate(notebook)
        transformer = TransformerManager()
        runtime_parts = []
        for i, cell in enumerate(notebook.cells):
            if cell.cell_type != "code":
                continue
            compile(transformer.transform_cell(cell.source), f"notebook_cell_{i}", "exec")
            if cell.source.startswith("%%writefile"):
                runtime_parts.append(cell.source.split("\n", 1)[1])
        embedded = ast.dump(ast.parse("\n".join(runtime_parts)), include_attributes=False)
        reviewable = ast.dump(ast.parse(Path(rt.__file__).read_text()), include_attributes=False)
        self.assertEqual(embedded, reviewable)

    def test_training_orchestration_creates_complete_immutable_bundle(self):
        # Exercise a real optimizer step and the complete writer/loader workflow.
        # The large data iterator and validation summaries are fixture stand-ins;
        # this test is not a competition training run or measured validation score.
        fixture_signals = self.signals

        class FixtureDataset(torch.utils.data.Dataset):
            def __init__(self, examples, infos):
                self.examples = examples

            def __len__(self):
                return len(self.examples)

            def __getitem__(self, index):
                _, _, snr, label = self.examples[index]
                return fixture_signals[label], label, snr

        def fixture_examples(rows, infos, split, limit, snrs):
            return [(f"fixture-{c}", 0 if split == "train" else 1, s, c)
                    for s in snrs for c in range(7)]

        validation_class_counts = [1440, 240, 1920, 3840, 1920, 1920, 1920]
        metric = rt.report_from_matrices({s: np.diag(validation_class_counts) for s in rt.ALL_SNRS})
        metric.update(inference_seconds=0.1, seconds_per_pulse=0.1/171600)
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "new-run"
            cfg = dict(rt.DEFAULT_CONFIG, epochs=1, samples_per_epoch=7, workers=0, batch_size=7)
            with patch.object(rt, "build_examples", side_effect=fixture_examples), \
                 patch.object(rt, "DevelopmentDataset", FixtureDataset), \
                 patch.object(rt, "evaluate", side_effect=lambda *a, **kw: dict(metric)):
                rt.train_and_freeze(ROOT / "recorded_dataset", ROOT / "hackathon/split_manifest.csv",
                                    output, cfg, prepared=(self.rows, {}, 0.0))
            hashes = rt.verify_bundle(output)
            self.assertTrue({"model.pt", "config.json", "radarml_track2_runtime.py",
                             "requirements.txt", "validation_report.json", "selection_validation.json",
                             "training_history.json", "method_summary.txt", "inference_command.txt",
                             "split_manifest.csv"}.issubset(hashes))
            checkpoint = torch.load(output / "model.pt", map_location="cpu", weights_only=True)
            model = rt.Track2Model(**checkpoint["model_kwargs"])
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            with self.assertRaisesRegex(FileExistsError, "not empty"):
                rt.train_and_freeze(ROOT / "recorded_dataset", ROOT / "hackathon/split_manifest.csv", output, cfg)


if __name__ == "__main__":
    unittest.main()
