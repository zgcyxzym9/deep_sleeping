# deep_sleeping

Research code for procedural music dataset generation and open-set music source separation.

The repository is split into two Python subprojects:

- `dataset generator/`: generates deterministic synthetic multitrack music projects, metadata, MIDI, stems, and mixtures through reference, FL Studio plan, or VST rendering paths.
- `model/`: contains the source separation model, training scripts, evaluation scripts, tools, configs, and tests.

Generated datasets, training runs, cached Python files, reports, CSV outputs, and large archives are intentionally excluded from version control.

## Repository Layout

```text
dataset generator/
  pyproject.toml
  README.md
  src/procmusic_dataset/
  tests/
model/
  pyproject.toml
  configs/
  src/procmusic_model/
  tests/
```

## Dataset Generator

Install from the dataset generator directory:

```powershell
cd "dataset generator"
pip install -e .
```

Run the reference generator:

```powershell
procmusic-generate --out .\generated --count 4 --seed 1234
```

For details about FL Studio plan generation and VST rendering, see `dataset generator/README.md`.

## Model

Install from the model directory:

```powershell
cd model
pip install -e .
```

Train with the default config:

```powershell
procmusic-train --config configs/default.json
```

Available command line entry points include:

- `procmusic-train`
- `procmusic-train-discriminator`
- `procmusic-eval`
- `procmusic-eval-discriminator`
- `procmusic-inspect-dataset`
- `procmusic-overfit-one-batch`
- `procmusic-run-separation`
- `procmusic-render-debug-spectrogram`

## Tests

Run tests inside each subproject:

```powershell
cd "dataset generator"
pytest

cd ..\model
pytest
```
