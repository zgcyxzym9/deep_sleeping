from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path

from .config import BatchConfig
from .generator import ProjectGenerator
from .models import ProjectSpec, project_metadata
from .renderers.base import Renderer


LOGGER = logging.getLogger("procmusic_dataset")


class DatasetPipeline:
    def __init__(self, config: BatchConfig, renderer: Renderer) -> None:
        config.validate()
        self.config = config
        self.renderer = renderer
        self.generator = ProjectGenerator(config.generation)

    def run(self) -> None:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self.config.output_dir / "manifest.jsonl"
        for index in range(self.config.count):
            seed = self.config.seed + index
            project = self.generator.generate(index, seed)
            project_dir = self.config.output_dir / project.project_id
            metadata_path = project_dir / "metadata.json"
            if metadata_path.exists() and not self.config.overwrite:
                LOGGER.info("skip existing project %s", project.project_id)
                continue
            try:
                started = time.perf_counter()
                result = self._render_one(project, project_dir)
                self._append_manifest(manifest_path, {"project_id": project.project_id, "status": result.status, "seed": seed})
                LOGGER.info(
                    "%s %s -> %s (%.2fs)",
                    result.status,
                    project.project_id,
                    result.mixture_path,
                    time.perf_counter() - started,
                )
            except Exception as exc:
                self._append_manifest(
                    manifest_path,
                    {"project_id": project.project_id, "status": "failed", "seed": seed, "error": repr(exc)},
                )
                LOGGER.exception("failed project %s", project.project_id)

    def _render_one(self, project: ProjectSpec, project_dir: Path):
        if self.config.overwrite:
            _clean_project_artifacts(project_dir)
        project_dir.mkdir(parents=True, exist_ok=True)
        render = self.renderer.render(project, project_dir)
        metadata = project_metadata(project, render)
        _write_json_atomic(project_dir / "metadata.json", metadata)
        return render

    def _append_manifest(self, path: Path, record: dict) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _write_json_atomic(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    tmp.replace(path)


def _clean_project_artifacts(project_dir: Path) -> None:
    for dirname in ("stems", "midi"):
        path = project_dir / dirname
        if path.exists():
            shutil.rmtree(path)
    for filename in ("metadata.json", "metadata.json.tmp", "mixture.wav", "mixture_preview.mp3", "render_plan.json"):
        path = project_dir / filename
        if path.exists():
            path.unlink()
