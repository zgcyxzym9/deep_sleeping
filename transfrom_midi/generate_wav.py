from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import pretty_midi


DEFAULT_INPUT_DIR = r"C:\Users\fison\Desktop\class\DL\myproject\guitar\outputs"


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Render MIDI files into per-instrument WAV files."
	)
	parser.add_argument(
		"--input-dir",
		default=DEFAULT_INPUT_DIR,
		help="Directory containing .mid files.",
	)
	parser.add_argument(
		"--output-dir",
		default=None,
		help="Directory to write .wav files. Defaults to <input-dir>/wav.",
	)
	parser.add_argument(
		"--soundfont",
		default=r"C:\Users\fison\Desktop\class\DL\myproject\guitar\GeneralUser-GS.sf2",
		help="Path to a .sf2 soundfont file.",
	)
	parser.add_argument(
		"--instrument",
		default="all",
		help=(
			"Comma-separated General MIDI instrument names to render (default: all). "
			"Use 'drums' for percussion."
		),
	)
	parser.add_argument(
		"--target-instrument",
		default="",
		help=(
			"Force all tracks into a single instrument name (e.g. 'acoustic_guitar_steel'). "
			"Use 'drums' for percussion."
		),
	)
	return parser.parse_args()


def sanitize_name(value: str) -> str:
	return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)


def normalize_instrument_name(value: str) -> str:
	return sanitize_name(value.strip().lower())


def parse_instrument_filter(value: str) -> set[str]:
	if not value:
		return set()
	if value.strip().lower() == "all":
		return set()
	return {
		normalize_instrument_name(part)
		for part in value.split(",")
		if part.strip()
	}


def instrument_label(inst: pretty_midi.Instrument) -> str:
	if inst.is_drum:
		return "drums"
	name = inst.name.strip() or pretty_midi.program_to_instrument_name(inst.program)
	return sanitize_name(name.lower())


def build_instrument_program_map() -> dict[str, int]:
	return {
		sanitize_name(pretty_midi.program_to_instrument_name(program).lower()): program
		for program in range(128)
	}


def resolve_target_instrument(name: str, program_map: dict[str, int]) -> tuple[int, bool]:
	normalized = normalize_instrument_name(name)
	if normalized == "drums":
		return 0, True
	if normalized not in program_map:
		available = ", ".join(sorted(program_map.keys()))
		raise ValueError(
			f"Unknown instrument '{name}'. Available names include: {available}"
		)
	return program_map[normalized], False


def render_target_instrument(
	midi_path: Path,
	output_dir: Path,
	soundfont: Path,
	fluidsynth_path: str,
	target_instrument: str,
	program_map: dict[str, int],
) -> None:
	midi = pretty_midi.PrettyMIDI(str(midi_path))
	if not midi.instruments:
		return

	program, is_drum = resolve_target_instrument(target_instrument, program_map)
	single = pretty_midi.PrettyMIDI(initial_tempo=midi.estimate_tempo())
	merged = pretty_midi.Instrument(program=program, is_drum=is_drum)
	for inst in midi.instruments:
		merged.notes.extend(inst.notes)
	merged.notes.sort(key=lambda note: (note.start, note.pitch, note.end))
	single.instruments.append(merged)

	label = normalize_instrument_name(target_instrument)
	wav_name = f"{midi_path.stem}_{label}.wav"
	wav_path = output_dir / wav_name

	with tempfile.TemporaryDirectory() as tmpdir:
		tmp_midi = Path(tmpdir) / "one.mid"
		single.write(str(tmp_midi))
		subprocess.run(
			[
				fluidsynth_path,
				"-ni",
				"-F",
				str(wav_path),
				"-r",
				"44100",
				str(soundfont),
				str(tmp_midi),
			],
			check=True,
		)


def render_instruments(
	midi_path: Path,
	output_dir: Path,
	soundfont: Path,
	fluidsynth_path: str,
	allowed_instruments: set[str],
) -> None:
	midi = pretty_midi.PrettyMIDI(str(midi_path))
	if not midi.instruments:
		return

	for index, inst in enumerate(midi.instruments, start=1):
		single = pretty_midi.PrettyMIDI(initial_tempo=midi.estimate_tempo())
		single.instruments.append(inst)
		label = instrument_label(inst)
		if allowed_instruments and label not in allowed_instruments:
			continue
		wav_name = f"{midi_path.stem}_{index:02d}_{label}.wav"
		wav_path = output_dir / wav_name

		with tempfile.TemporaryDirectory() as tmpdir:
			tmp_midi = Path(tmpdir) / "one.mid"
			single.write(str(tmp_midi))
			subprocess.run(
				[
					fluidsynth_path,
					"-ni",
					"-F",
					str(wav_path),
					"-r",
					"44100",
					str(soundfont),
					str(tmp_midi),
				],
				check=True,
			)


def main() -> None:
	args = parse_args()
	input_dir = Path(args.input_dir)
	output_dir = Path(args.output_dir) if args.output_dir else input_dir / "wav"
	soundfont = Path(args.soundfont)
	allowed_instruments = parse_instrument_filter(args.instrument)
	target_instrument = args.target_instrument.strip()

	if not input_dir.exists():
		raise FileNotFoundError(f"Input dir not found: {input_dir}")
	if not soundfont.exists():
		raise FileNotFoundError(f"SoundFont not found: {soundfont}")
	fluidsynth_path = shutil.which("fluidsynth")
	if not fluidsynth_path:
		raise FileNotFoundError(
			"fluidsynth not found on PATH. Install FluidSynth and reopen your terminal."
		)
	program_map = build_instrument_program_map()

	output_dir.mkdir(parents=True, exist_ok=True)

	for midi_path in input_dir.glob("*.mid"):
		if target_instrument:
			render_target_instrument(
				midi_path,
				output_dir,
				soundfont,
				fluidsynth_path,
				target_instrument,
				program_map,
			)
		else:
			render_instruments(
				midi_path,
				output_dir,
				soundfont,
				fluidsynth_path,
				allowed_instruments,
			)


if __name__ == "__main__":
	main()
