from __future__ import annotations

import json
import math
import tempfile
import unittest
import wave
from pathlib import Path

from procmusic_model.data import MusicSeparationDataset, collate_separation_batch


SAMPLE_RATE = 44_100


class DatasetTest(unittest.TestCase):
    def test_dataset_reads_rendered_manifest_and_ignores_ok(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_project(root, "project_000000", source_count=2)
            _write_project(root, "project_000001", source_count=1)
            _write_manifest(
                root,
                [
                    {"project_id": "project_000000", "seed": 1, "status": "rendered"},
                    {"project_id": "project_000000", "seed": 1, "status": "rendered"},
                    {"project_id": "project_000001", "seed": 2, "status": "ok"},
                ],
            )

            dataset = MusicSeparationDataset(root, sample_rate=SAMPLE_RATE, segment_seconds=0.05, random_crop=False)
            example = dataset[0]

            self.assertEqual(len(dataset), 1)
            self.assertEqual(example.metadata["project_id"], "project_000000")
            self.assertEqual(example.mixture.ndim, 2)
            self.assertEqual(example.sources.ndim, 3)
            self.assertEqual(example.source_count, example.metadata["source_count"])

    def test_malformed_manifest_falls_back_to_project_scan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_project(root, "project_000000", source_count=1)
            (root / "manifest.jsonl").write_text("not json\n", encoding="utf-8")

            dataset = MusicSeparationDataset(root, sample_rate=SAMPLE_RATE, segment_seconds=0.05, random_crop=False)

            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset[0].metadata["project_id"], "project_000000")

    def test_manifest_with_only_ok_records_is_not_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_project(root, "project_000000", source_count=1)
            _write_manifest(root, [{"project_id": "project_000000", "seed": 1, "status": "ok"}])

            with self.assertRaises(FileNotFoundError):
                MusicSeparationDataset(root, sample_rate=SAMPLE_RATE, segment_seconds=0.05, random_crop=False)

    def test_collate_pads_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_project(root, "project_000000", source_count=2)
            _write_project(root, "project_000001", source_count=1)
            _write_manifest(
                root,
                [
                    {"project_id": "project_000000", "seed": 1, "status": "rendered"},
                    {"project_id": "project_000001", "seed": 2, "status": "rendered"},
                ],
            )
            dataset = MusicSeparationDataset(root, sample_rate=SAMPLE_RATE, segment_seconds=0.05, random_crop=False)
            batch = collate_separation_batch([dataset[0], dataset[1]])

            self.assertEqual(batch.mixture.shape[0], 2)
            self.assertEqual(batch.sources.shape[0], 2)
            self.assertEqual(batch.source_mask.tolist(), [[True, True], [True, False]])


def _write_manifest(root: Path, records: list[dict]) -> None:
    with (root / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def _write_project(root: Path, project_id: str, source_count: int, samples: int = 4096) -> None:
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


if __name__ == "__main__":
    unittest.main()
