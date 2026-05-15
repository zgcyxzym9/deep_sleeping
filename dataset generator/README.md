# Procedural Music Dataset Generator

This repository contains a modular Python pipeline for generating synthetic multitrack music datasets for open-set music source separation research.

The current implementation is intentionally DAW-independent at the core:

- deterministic project generation from seeds
- procedural MIDI composition
- isolated stems and mixture WAV rendering via a lightweight reference renderer
- metadata JSON export
- resumable batch generation
- explicit renderer abstraction for future FL Studio automation

The reference renderer is not intended to replace FL Studio. It exists so the data model, generation logic, batch orchestration, metadata, and tests can be verified without a GUI DAW.

## Layout

```text
src/procmusic_dataset/
  cli.py                 CLI entry point
  config.py              batch and generation config
  generator.py           deterministic project generator
  models.py              metadata/data schema dataclasses
  midi.py                minimal Standard MIDI File writer
  pipeline.py            batch rendering, resume, logging
  renderers/
    base.py              renderer interface
    reference.py         built-in WAV/MIDI reference renderer
  daw/
    flstudio.py          FL Studio adapter placeholder
tests/
```

Generated datasets use this shape:

```text
dataset/
  manifest.jsonl
  project_000000/
    metadata.json
    mixture.wav
    midi/
      project.mid
    stems/
      000_pad.wav
      001_drum_kit.wav
```

## Quick Start

```powershell
$env:PYTHONPATH = "src"
python -m procmusic_dataset.cli --out .\generated --count 4 --seed 1234
```

Or after editable installation:

```powershell
pip install -e .
procmusic-generate --out .\generated --count 4 --seed 1234
```

## FL Studio Preparation

If FL Studio is installed at `D:\fl`, the pipeline can select installed FL Studio presets and prepare per-track MIDI plus a render plan:

```powershell
$env:PYTHONPATH = "src"
python -m procmusic_dataset.cli --renderer flstudio-plan --fl-root D:\fl --out .\generated_fl_plan --count 1 --seed 2026
```

This writes `flstudio_render_plan.json` with selected `.fst` preset paths, per-track MIDI files, expected stem paths, and manual/automation steps. It does not yet claim that audio has been exported by FL Studio; metadata marks this state as `prepared`.

## Scripted VST Rendering

For automated dataset generation with real plugin audio, use the VST renderer. It does not create `.flp` files or drive the FL Studio UI. Pitched tracks are rendered through a scriptable VST backend, while drum tracks use a deterministic internal drum fallback so every stem and mixture is written under the dataset directory.

Install the optional VST dependencies first:

```powershell
pip install -e .[vst]
```

Then render a small dataset:

```powershell
$env:PYTHONPATH = "src"
python -m procmusic_dataset.cli --renderer vst --vst-plugin "D:\fl\Orchestral VSTi v1.03\Orchestral.dll" --out .\generated_vst --count 1 --seed 2026
```

The training artifacts remain WAV files:

```text
dataset/
  project_000000/
    mixture.wav
    midi/
      project.mid
      000_synth_lead.mid
    stems/
      000_synth_lead.wav
```

FLEX is not used by this renderer because this installation has not exposed a verified standalone VST/Python-hostable FLEX entry point. The FL Studio GUI export probe remains an experimental diagnostic tool only.

## Architecture

The pipeline is split into five layers:

1. `ProjectGenerator` creates reproducible abstract multitrack projects.
2. `Renderer` implementations turn projects into audio/MIDI artifacts.
3. `DatasetPipeline` handles batch execution, resume, atomic metadata writes, and failure logging.
4. metadata dataclasses define a JSON-compatible schema for research use.
5. DAW adapters can be added without changing generation or metadata code.

## Metadata

Each project exports:

- seed, BPM, key, scale, arrangement, source count
- per-track instrument category, patch, synth parameters, effects, MIDI notes
- automation descriptors and sidechain usage
- rendered artifacts and approximate loudness/energy

## FL Studio Integration Plan

The `FLStudioRenderer` is intentionally a stub until automation is verified on the target machine. A production adapter should:

- launch FL Studio in an isolated worker process
- create or load a template project
- instantiate plugins/presets from an allowlist
- import generated MIDI per track
- assign mixer routes and effect chains
- export mixture and soloed stems
- write crash diagnostics and return a structured render result

The core pipeline already treats a DAW renderer as replaceable, so FL Studio-specific code should stay inside `src/procmusic_dataset/daw/`.

## Tests

```powershell
pytest
```
