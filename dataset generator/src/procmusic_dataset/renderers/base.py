from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from procmusic_dataset.models import ProjectSpec, RenderResult


class Renderer(ABC):
    @abstractmethod
    def render(self, project: ProjectSpec, output_dir: Path) -> RenderResult:
        """Render a project into output_dir and return artifact paths plus measurements."""
