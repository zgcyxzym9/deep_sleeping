import json
from pathlib import Path

from procmusic_dataset.config import BatchConfig, GenerationConfig
from procmusic_dataset.pipeline import DatasetPipeline
from procmusic_dataset.renderers.reference import ReferenceRenderer


def test_pipeline_writes_artifacts(tmp_path: Path):
    config = BatchConfig(
        output_dir=tmp_path,
        count=1,
        seed=77,
        generation=GenerationConfig(min_tracks=2, max_tracks=2, min_bars=2, max_bars=2, sample_rate=8000),
    )
    DatasetPipeline(config, ReferenceRenderer(sample_rate=8000)).run()

    project_dir = tmp_path / "project_000000"
    metadata_path = project_dir / "metadata.json"

    assert metadata_path.exists()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["seed"] == 77
    assert metadata["source_count"] == 2
    assert (project_dir / "mixture.wav").exists()
    assert (project_dir / "midi" / "project.mid").exists()
    assert len(list((project_dir / "stems").glob("*.wav"))) == 2


def test_pipeline_resume_skips_existing(tmp_path: Path):
    config = BatchConfig(
        output_dir=tmp_path,
        count=1,
        seed=77,
        generation=GenerationConfig(min_tracks=1, max_tracks=1, min_bars=1, max_bars=1, sample_rate=8000),
    )
    pipeline = DatasetPipeline(config, ReferenceRenderer(sample_rate=8000))
    pipeline.run()
    metadata_path = tmp_path / "project_000000" / "metadata.json"
    before = metadata_path.stat().st_mtime_ns

    pipeline.run()

    assert metadata_path.stat().st_mtime_ns == before


def test_pipeline_overwrite_removes_stale_artifacts(tmp_path: Path):
    first = BatchConfig(
        output_dir=tmp_path,
        count=1,
        seed=77,
        generation=GenerationConfig(min_tracks=3, max_tracks=3, min_bars=1, max_bars=1, sample_rate=8000),
    )
    DatasetPipeline(first, ReferenceRenderer(sample_rate=8000)).run()

    second = BatchConfig(
        output_dir=tmp_path,
        count=1,
        seed=78,
        generation=GenerationConfig(min_tracks=1, max_tracks=1, min_bars=1, max_bars=1, sample_rate=8000),
        overwrite=True,
    )
    DatasetPipeline(second, ReferenceRenderer(sample_rate=8000)).run()

    project_dir = tmp_path / "project_000000"
    metadata = json.loads((project_dir / "metadata.json").read_text(encoding="utf-8"))
    assert len(list((project_dir / "stems").glob("*.wav"))) == metadata["source_count"] == 1
    assert sorted(path.name for path in (project_dir / "midi").glob("*.mid")) == ["project.mid"]
