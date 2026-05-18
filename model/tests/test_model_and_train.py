from __future__ import annotations

import json
import math
import tempfile
import unittest
import wave
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from procmusic_model.config import DatasetConfig, ExperimentConfig, ModelConfig, TrainingConfig
from procmusic_model.data import MusicSeparationDataset, collate_separation_batch
from procmusic_model.losses import SeparationLoss
from procmusic_model.losses.separation import _stop_loss
from procmusic_model.systems import OpenSetSeparator
from procmusic_model.training.train import train
from procmusic_model.training.train import _predicted_source_count


SAMPLE_RATE = 44_100


class ModelAndTrainTest(unittest.TestCase):
    def test_stop_loss_ignores_logits_after_true_stop(self) -> None:
        logits = torch.tensor([[0.0, 0.0, 20.0]])
        source_count = torch.tensor([1])
        expected = torch.nn.functional.binary_cross_entropy_with_logits(
            logits[:, :2],
            torch.tensor([[0.0, 1.0]]),
        )
        self.assertTrue(torch.allclose(_stop_loss(logits, source_count), expected))

    def test_predicted_source_count_uses_max_steps_when_no_stop(self) -> None:
        stop_prob = torch.tensor(
            [
                [0.1, 0.2, 0.3],
                [0.1, 0.7, 0.8],
            ]
        )
        predicted = _predicted_source_count(stop_prob, threshold=0.5)
        self.assertTrue(torch.equal(predicted, torch.tensor([2, 1])))

    def test_model_forward_backward(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_project(root, "project_000000", source_count=2)
            dataset = MusicSeparationDataset(root, sample_rate=SAMPLE_RATE, segment_seconds=0.25, random_crop=False)
            batch = next(iter(DataLoader(dataset, batch_size=1, collate_fn=collate_separation_batch)))
            model_config = ModelConfig(sample_rate=SAMPLE_RATE, n_fft=128, hop_length=32, win_length=128, hidden_dim=16, encoder_channels=8, max_steps=2, refine_channels=8)
            model = OpenSetSeparator(model_config, channels=batch.mixture.shape[1])
            criterion = SeparationLoss(ExperimentConfig().loss, model_config)
            output = model.forward_train(batch)
            self.assertEqual(output.estimated_sources.shape[1], model_config.max_steps)
            self.assertEqual(output.stop_logits.shape[1], model_config.max_steps + 1)
            loss = criterion(output, batch).total
            loss.backward()
            self.assertTrue(torch.isfinite(loss))

    def test_separate_stops_before_first_prediction(self) -> None:
        model_config = ModelConfig(sample_rate=SAMPLE_RATE, n_fft=128, hop_length=32, win_length=128, hidden_dim=16, encoder_channels=8, max_steps=2, refine_channels=8)
        model = OpenSetSeparator(model_config)
        model.stop_predictor.forward = lambda summary: summary.new_full((summary.shape[0],), 100.0)
        model.decoder = _FailingModule()
        model.refiner = _FailingModule()
        mixture = torch.zeros(1, 1, 4096)

        result = model.separate(mixture)

        self.assertEqual(result["sources"].shape, (1, 0, 1, 4096))
        self.assertEqual(result["stop_logits"].shape, (1, 1))
        self.assertEqual(result["residual"].shape, (1, 1, 4096))
        self.assertTrue(torch.equal(result["residual"], mixture))
        self.assertTrue(torch.equal(result["predicted_source_count"], torch.tensor([0])))

    def test_separate_residual_tracks_actual_predictions(self) -> None:
        model_config = ModelConfig(sample_rate=SAMPLE_RATE, n_fft=128, hop_length=32, win_length=128, hidden_dim=16, encoder_channels=8, max_steps=3, refine_channels=8)
        model = OpenSetSeparator(model_config)
        calls = {"count": 0}

        def stop_after_one(summary):
            value = -100.0 if calls["count"] == 0 else 100.0
            calls["count"] += 1
            return summary.new_full((summary.shape[0],), value)

        model.stop_predictor.forward = stop_after_one
        model.refiner = _ConstantSource(0.25)
        mixture = torch.zeros(1, 1, 4096)

        result = model.separate(mixture)

        self.assertEqual(result["sources"].shape[1], 1)
        self.assertEqual(result["stop_logits"].shape, (1, 2))
        self.assertTrue(torch.allclose(result["residual"], torch.full_like(mixture, -0.25)))
        self.assertTrue(torch.equal(result["predicted_source_count"], torch.tensor([1])))

    def test_separate_rejects_batched_inference(self) -> None:
        model_config = ModelConfig(sample_rate=SAMPLE_RATE, n_fft=128, hop_length=32, win_length=128, hidden_dim=16, encoder_channels=8, max_steps=2, refine_channels=8)
        model = OpenSetSeparator(model_config)
        with self.assertRaises(ValueError):
            model.separate(torch.zeros(2, 1, 4096))

    def test_train_loop_smoke_writes_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as data_tmp, tempfile.TemporaryDirectory() as run_tmp:
            root = Path(data_tmp)
            _write_project(root, "project_000000", source_count=2)
            config = ExperimentConfig(
                dataset=DatasetConfig(root=str(root), sample_rate=SAMPLE_RATE, segment_seconds=0.25, random_crop=False),
                model=ModelConfig(sample_rate=SAMPLE_RATE, n_fft=128, hop_length=32, win_length=128, hidden_dim=16, encoder_channels=8, max_steps=2, refine_channels=8),
                training=TrainingConfig(output_dir=run_tmp, batch_size=1, epochs=1, max_steps=2, checkpoint_every=1, amp=False),
            )
            train(config, device="cpu")
            self.assertTrue((Path(run_tmp) / "checkpoints" / "last.pt").exists())
            self.assertTrue(any((Path(run_tmp) / "tb").glob("events.out.tfevents.*")) or (Path(run_tmp) / "scalars.jsonl").exists())


def _write_project(root: Path, project_id: str, source_count: int, samples: int = 16_384) -> None:
    project_dir = root / project_id
    stems_dir = project_dir / "stems"
    stems_dir.mkdir(parents=True)
    sources = []
    for index in range(source_count):
        frequency = 220.0 + index * 110.0
        audio = [0.1 * math.sin(2.0 * math.pi * frequency * sample / SAMPLE_RATE) for sample in range(samples)]
        sources.append(audio)
        _write_wav(stems_dir / f"{index:03d}_source.wav", audio)
    mixture = [sum(source[sample] for source in sources) for sample in range(samples)]
    _write_wav(project_dir / "mixture.wav", mixture)
    metadata = {
        "project_id": project_id,
        "source_count": source_count,
        "tracks": [
            {"track_id": f"trk_{index:03d}", "instrument_category": "synth", "role": "melody"}
            for index in range(source_count)
        ],
        "render": {
            "status": "rendered",
            "mixture_path": f"{project_id}/mixture.wav",
            "stems": [
                {"track_id": f"trk_{index:03d}", "stem_path": f"{project_id}/stems/{index:03d}_source.wav"}
                for index in range(source_count)
            ],
        },
    }
    (project_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    with (root / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"project_id": project_id, "seed": 1, "status": "rendered"}) + "\n")


def _write_wav(path: Path, audio: list[float]) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        pcm = bytearray()
        for sample in audio:
            value = max(-1.0, min(1.0, sample))
            pcm.extend(int(round(value * 32767.0)).to_bytes(2, "little", signed=True))
        handle.writeframes(bytes(pcm))


class _FailingModule(torch.nn.Module):
    def forward(self, *args, **kwargs):
        raise AssertionError("module should not run after pre-step stop")


class _ConstantSource(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def forward(self, rough_source, residual, mixture):
        return mixture.new_full(mixture.shape, self.value)


if __name__ == "__main__":
    unittest.main()
