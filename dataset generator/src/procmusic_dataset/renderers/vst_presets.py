from __future__ import annotations

import random
import time
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from procmusic_dataset.models import ProjectSpec, TrackSpec, normalize_path


LOGGER = logging.getLogger("procmusic_dataset")
FINGERPRINT_PARAMETER_LIMIT = 128
_PLUGIN_PARAMETER_CACHE: dict[str, list[dict[str, Any]]] = {}


@dataclass(frozen=True)
class VSTPreset:
    name: str
    path: Path | None
    category: str | None = None


def select_plugin_sound(plugin_path: Path, processor: object, project: ProjectSpec, track: TrackSpec, attempt: int) -> dict:
    started = time.perf_counter()
    rng = random.Random(f"{project.seed}:{track.track_id}:surge-sound:{attempt}")
    presets = plugin_presets(plugin_path, processor)
    parameters = plugin_parameters(processor, plugin_path)
    preset_result = try_select_preset(processor, track, presets, rng, parameters)
    if preset_result["patch_selection_status"] == "preset_or_program_selected":
        preset_result["parameter_randomization"] = []
        preset_result["available_parameter_count"] = len(parameters)
        LOGGER.debug(
            "selected plugin sound track=%s status=%s presets=%d parameters=%d seconds=%.3f",
            track.name,
            preset_result["patch_selection_status"],
            len(presets),
            len(parameters),
            time.perf_counter() - started,
        )
        return preset_result

    randomized = randomize_plugin_parameters(processor, parameters, rng)
    status = "parameter_randomized" if randomized else preset_result["patch_selection_status"]
    LOGGER.debug(
        "selected plugin sound track=%s status=%s presets=%d parameters=%d randomized=%d seconds=%.3f",
        track.name,
        status,
        len(presets),
        len(parameters),
        len(randomized),
        time.perf_counter() - started,
    )
    return {
        **preset_result,
        "patch_selection_status": status,
        "available_parameter_count": len(parameters),
        "parameter_randomization": randomized,
    }


def plugin_presets(plugin_path: Path, processor: object) -> list[VSTPreset]:
    filesystem_presets = filesystem_plugin_presets(plugin_path)
    if filesystem_presets:
        return filesystem_presets
    return [VSTPreset(name, None) for name in plugin_preset_names(processor)]


def filesystem_plugin_presets(plugin_path: Path) -> list[VSTPreset]:
    return list(_filesystem_plugin_presets_cached(str(plugin_path.resolve())))


@lru_cache(maxsize=16)
def _filesystem_plugin_presets_cached(plugin_path: str) -> tuple[VSTPreset, ...]:
    if "surge" not in str(plugin_path).lower():
        return ()
    roots = [
        Path("C:/ProgramData/Surge XT/patches_factory"),
        Path("C:/ProgramData/Surge XT/patches_3rdparty"),
        Path.home() / "Documents" / "Surge XT" / "patches_user",
    ]
    presets: list[VSTPreset] = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.suffix.lower() not in {".fxp", ".vstpreset"}:
                continue
            try:
                category = path.relative_to(root).parts[0] if len(path.relative_to(root).parts) > 1 else None
            except ValueError:
                category = None
            presets.append(VSTPreset(path.stem, path, category))
    return tuple(presets)


def plugin_preset_names(processor: object) -> list[str]:
    for method_name in ("get_preset_names", "get_presets", "get_program_names", "get_programs"):
        method = getattr(processor, method_name, None)
        if not callable(method):
            continue
        try:
            values = method()
        except Exception:
            continue
        names = [stringify_plugin_value(value) for value in as_list(values)]
        names = [name for name in names if name]
        if names:
            return names
    return []


def try_select_preset(
    processor: object,
    track: TrackSpec,
    presets: list[VSTPreset],
    rng: random.Random,
    parameters: list[dict[str, Any]] | None = None,
) -> dict:
    result = {
        "preset_or_program": None,
        "preset_path": None,
        "preset_category": None,
        "patch_selection_status": "default_or_unavailable",
        "available_preset_count": len(presets),
    }
    if not presets:
        return result

    candidates = rank_presets(track, presets, rng)
    for selected_index, preset in candidates[:10]:
        loaded, verified = load_preset(processor, selected_index, preset, parameters)
        if loaded and verified:
            return {
                **result,
                "preset_or_program": preset.name,
                "preset_path": normalize_path(preset.path) if preset.path else None,
                "preset_category": preset.category,
                "patch_selection_status": "preset_or_program_selected",
            }
        if loaded:
            result["patch_selection_status"] = "preset_loaded_but_unverified"
    selected = candidates[0][1]
    return {
        **result,
        "preset_or_program": selected.name,
        "preset_path": normalize_path(selected.path) if selected.path else None,
        "preset_category": selected.category,
        "patch_selection_status": result["patch_selection_status"]
        if result["patch_selection_status"] != "default_or_unavailable"
        else "preset_or_program_unavailable",
    }


def load_preset(
    processor: object,
    selected_index: int,
    preset: VSTPreset,
    parameters: list[dict[str, Any]] | None = None,
) -> tuple[bool, bool]:
    before = parameter_fingerprint(processor, parameters)
    if preset.path is not None:
        method_name = "load_vst3_preset" if preset.path.suffix.lower() == ".vstpreset" else "load_preset"
        method = getattr(processor, method_name, None)
        if callable(method):
            try:
                method(str(preset.path.resolve()))
                return True, preset_changed_parameters(processor, before, parameters)
            except Exception:
                return False, False

    for method_name in ("set_preset", "set_program", "load_program"):
        method = getattr(processor, method_name, None)
        if not callable(method):
            continue
        for value in (preset.name, selected_index):
            try:
                method(value)
                return True, preset_changed_parameters(processor, before, parameters)
            except Exception:
                continue
    return False, False


def parameter_fingerprint(processor: object, parameters: list[dict[str, Any]] | None = None) -> dict[int, float]:
    values: dict[int, float] = {}
    get_parameter = getattr(processor, "get_parameter", None)
    if not callable(get_parameter):
        return values
    for parameter in fingerprint_parameters(parameters or plugin_parameters(processor)):
        index = parameter.get("index")
        if not isinstance(index, int):
            continue
        try:
            values[index] = round(float(get_parameter(index)), 7)
        except Exception:
            continue
    return values


def preset_changed_parameters(
    processor: object, before: dict[int, float], parameters: list[dict[str, Any]] | None = None
) -> bool:
    if not before:
        return True
    after = parameter_fingerprint(processor, parameters)
    if not after:
        return True
    comparable = set(before) & set(after)
    return any(abs(before[index] - after[index]) > 1e-6 for index in comparable)


def fingerprint_parameters(parameters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(parameters) <= FINGERPRINT_PARAMETER_LIMIT:
        return parameters
    ranked = rank_parameters_for_randomization(parameters, random.Random(0))
    selected = {int(parameter["index"]) for parameter in ranked[: FINGERPRINT_PARAMETER_LIMIT // 2] if isinstance(parameter.get("index"), int)}
    step = max(1, len(parameters) // (FINGERPRINT_PARAMETER_LIMIT - len(selected)))
    for parameter in parameters[::step]:
        index = parameter.get("index")
        if isinstance(index, int):
            selected.add(index)
        if len(selected) >= FINGERPRINT_PARAMETER_LIMIT:
            break
    return [parameter for parameter in parameters if parameter.get("index") in selected]


def rank_presets(track: TrackSpec, presets: list[VSTPreset], rng: random.Random) -> list[tuple[int, VSTPreset]]:
    role_tokens = {
        "bass": ("bass", "basses", "sub", "acid", "fm"),
        "harmony": ("pad", "pads", "keys", "keyboards", "string", "strings", "poly", "chord", "chords", "organ", "epiano"),
        "melody": ("lead", "leads", "pluck", "plucks", "bell", "bells", "arp", "arps", "seq", "mono"),
    }
    tokens = role_tokens.get(track.role, ())
    scored = []
    for index, preset in enumerate(presets):
        text = f"{preset.category or ''} {preset.name}".lower().replace(" ", "_")
        score = sum(1 for token in tokens if token in text)
        scored.append((score, rng.random(), index, preset))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [(index, preset) for _, _, index, preset in scored]


def plugin_parameters(processor: object, plugin_path: Path | None = None) -> list[dict[str, Any]]:
    cache_key = str(plugin_path.resolve()) if plugin_path is not None else None
    if cache_key and cache_key in _PLUGIN_PARAMETER_CACHE:
        return [dict(parameter) for parameter in _PLUGIN_PARAMETER_CACHE[cache_key]]

    parameters = _read_plugin_parameters(processor)
    if cache_key and parameters:
        _PLUGIN_PARAMETER_CACHE[cache_key] = [dict(parameter) for parameter in parameters]
    return parameters


def _read_plugin_parameters(processor: object) -> list[dict[str, Any]]:
    for method_name in ("get_plugin_parameters_description", "get_parameters_description"):
        method = getattr(processor, method_name, None)
        if not callable(method):
            continue
        try:
            return [normalize_parameter_description(index, value) for index, value in enumerate(as_list(method()))]
        except Exception:
            continue

    names = call_noarg(processor, "get_parameter_names")
    if names is not None:
        return [{"index": index, "name": stringify_plugin_value(name)} for index, name in enumerate(as_list(names))]

    count = call_noarg(processor, "get_num_parameters")
    if isinstance(count, int):
        parameters = []
        for index in range(max(0, count)):
            name = None
            method = getattr(processor, "get_parameter_name", None)
            if callable(method):
                try:
                    name = method(index)
                except Exception:
                    name = None
            parameters.append({"index": index, "name": stringify_plugin_value(name) or f"parameter_{index}"})
        return parameters

    return []


def randomize_plugin_parameters(processor: object, parameters: list[dict[str, Any]], rng: random.Random) -> list[dict]:
    randomized = []
    for parameter in rank_parameters_for_randomization(parameters, rng):
        value = random_parameter_value(parameter.get("name", ""), rng)
        if set_plugin_parameter(processor, parameter, value):
            randomized.append({"name": parameter.get("name"), "index": parameter.get("index"), "value": value})
        if len(randomized) >= 28:
            break
    return randomized


def rank_parameters_for_randomization(parameters: list[dict[str, Any]], rng: random.Random) -> list[dict[str, Any]]:
    scored = []
    for parameter in parameters:
        name = stringify_plugin_value(parameter.get("name"))
        score = timbre_parameter_score(name)
        if score <= 0:
            continue
        scored.append((score, rng.random(), parameter))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [parameter for _, _, parameter in scored]


def timbre_parameter_score(name: str) -> int:
    text = name.lower().replace(" ", "_")
    if not is_safe_timbre_parameter(text):
        return 0
    score = 1
    primary = (
        "a_osc_1_type",
        "a_osc_1_shape",
        "a_osc_1_width",
        "a_osc_1_sub_mix",
        "a_osc_2_type",
        "a_osc_2_shape",
        "a_osc_2_width",
        "a_osc_2_sub_mix",
        "a_filter_1_cutoff",
        "a_filter_1_resonance",
        "a_filter_1_type",
        "a_amp_eg_attack",
        "a_amp_eg_decay",
        "a_amp_eg_sustain",
        "a_amp_eg_release",
        "a_waveshaper_type",
        "a_waveshaper_drive",
        "a_chorus",
        "a_delay",
        "a_reverb",
    )
    if any(token in text for token in primary):
        score += 20
    if text.startswith("a_"):
        score += 5
    if "osc" in text or "filter" in text or "waveshaper" in text:
        score += 4
    if "lfo" in text or "scene_lfo" in text:
        score -= 3
    return score


def random_parameter_value(name: object, rng: random.Random) -> float:
    text = stringify_plugin_value(name).lower().replace(" ", "_")
    if "cutoff" in text:
        return round(rng.uniform(0.35, 0.9), 6)
    if "resonance" in text or "reson" in text:
        return round(rng.uniform(0.05, 0.65), 6)
    if "attack" in text:
        return round(rng.uniform(0.0, 0.45), 6)
    if "decay" in text:
        return round(rng.uniform(0.05, 0.75), 6)
    if "sustain" in text:
        return round(rng.uniform(0.25, 0.95), 6)
    if "release" in text:
        return round(rng.uniform(0.05, 0.7), 6)
    if "pitch" in text:
        return round(rng.uniform(0.42, 0.58), 6)
    if "type" in text or "route" in text:
        return round(rng.choice([0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]), 6)
    return round(rng.uniform(0.08, 0.92), 6)


def set_plugin_parameter(processor: object, parameter: dict[str, Any], value: float) -> bool:
    index = parameter.get("index")
    name = parameter.get("name")
    for method_name in ("set_parameter", "set_parameter_value", "set_param"):
        method = getattr(processor, method_name, None)
        if not callable(method):
            continue
        for key in (index, name):
            if key is None:
                continue
            try:
                method(key, value)
                return True
            except Exception:
                continue
    return False


def normalize_parameter_description(index: int, value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        result = dict(value)
        result.setdefault("index", index)
        result["name"] = stringify_plugin_value(result.get("name") or result.get("label") or result.get("id"))
        return result
    name = getattr(value, "name", None) or getattr(value, "label", None) or getattr(value, "id", None)
    return {"index": index, "name": stringify_plugin_value(name or value)}


def is_safe_timbre_parameter(name: object) -> bool:
    text = stringify_plugin_value(name).lower().replace(" ", "_")
    if not text:
        return False
    blocked = ("volume", "level", "gain", "output", "master", "bypass", "mute", "solo", "panic", "polyphony")
    if any(token in text for token in blocked):
        return False
    allowed = (
        "osc",
        "wave",
        "shape",
        "width",
        "sync",
        "fm",
        "filter",
        "cutoff",
        "reson",
        "attack",
        "decay",
        "sustain",
        "release",
        "env",
        "lfo",
        "mod",
        "drive",
        "dist",
        "chorus",
        "delay",
        "reverb",
        "unison",
        "detune",
    )
    return any(token in text for token in allowed)


def call_noarg(target: object, method_name: str) -> object | None:
    method = getattr(target, method_name, None)
    if not callable(method):
        return None
    try:
        return method()
    except Exception:
        return None


def as_list(values: object) -> list:
    if values is None:
        return []
    if hasattr(values, "tolist"):
        values = values.tolist()
    if isinstance(values, (list, tuple)):
        return list(values)
    return []


def stringify_plugin_value(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()
