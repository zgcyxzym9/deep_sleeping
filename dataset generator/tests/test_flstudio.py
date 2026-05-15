from pathlib import Path

from procmusic_dataset.config import GenerationConfig
from procmusic_dataset.daw.flstudio import FLPresetCatalog, FLStudioConfig, FLStudioRenderer
from procmusic_dataset.generator import ProjectGenerator


def test_flstudio_catalog_scans_channel_presets(tmp_path: Path):
    root = tmp_path / "fl"
    preset_dir = root / "Data" / "Patches" / "Channel presets" / "3x Osc"
    plugin_preset_dir = root / "Data" / "Patches" / "Plugin presets" / "Generators" / "Sytrus" / "Bass"
    preset_dir.mkdir(parents=True)
    plugin_preset_dir.mkdir(parents=True)
    (root / "FL64.exe").write_bytes(b"")
    (preset_dir / "Bassline.fst").write_bytes(b"")
    (plugin_preset_dir / "Deep bass.fst").write_bytes(b"")

    catalog = FLPresetCatalog.scan(FLStudioConfig(root=root))

    assert len(catalog.presets) == 2
    assert catalog.presets[0].plugin == "3x Osc"
    assert {preset.plugin for preset in catalog.presets} == {"3x Osc", "Sytrus"}


def test_flstudio_renderer_writes_render_plan(tmp_path: Path):
    root = tmp_path / "fl"
    preset_dir = root / "Data" / "Patches" / "Channel presets" / "3x Osc"
    plugin_preset_dir = root / "Data" / "Patches" / "Plugin presets"
    preset_dir.mkdir(parents=True)
    plugin_preset_dir.mkdir(parents=True)
    (root / "FL64.exe").write_bytes(b"")
    (preset_dir / "Bassline.fst").write_bytes(b"")

    project = ProjectGenerator(GenerationConfig(min_tracks=1, max_tracks=1, min_bars=1, max_bars=1)).generate(0, 11)
    result = FLStudioRenderer(FLStudioConfig(root=root)).render(project, tmp_path / "project")

    assert result.status == "prepared"
    assert result.render_plan_path is not None
    assert Path(result.render_plan_path).exists()
    assert (tmp_path / "project" / "flstudio_midi" / "project.mid").exists()
