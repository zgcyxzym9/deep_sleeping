"""Snap MIDI note boundaries to the nearest note-value grid."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pretty_midi


VALID_DENOMINATORS = (1, 2, 4, 8, 16, 32)
DEFAULT_MIN_NOTE_DENOMINATOR = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Snap MIDI note start/end times to whole, half, quarter, eighth, "
            "sixteenth, or thirty-second note grids."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input MIDI file.",
    )
    parser.add_argument(
        "--min-note-denominator",
        type=int,
        choices=VALID_DENOMINATORS,
        default=DEFAULT_MIN_NOTE_DENOMINATOR,
        help=(
            "Smallest note grid used to divide each beat. "
            "1=whole, 2=half, 4=quarter, 8=eighth, 16=sixteenth, 32=thirty-second."
        ),
    )
    return parser.parse_args()


def load_midi(input_path: Path) -> pretty_midi.PrettyMIDI:
    if not input_path.exists():
        raise FileNotFoundError(f"Input MIDI file does not exist: {input_path}")
    return pretty_midi.PrettyMIDI(str(input_path))


def save_midi(midi: pretty_midi.PrettyMIDI, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(output_path))


def note_grid_step_ticks(resolution: int, min_denominator: int) -> float:
    return (4.0 * resolution) / min_denominator


def snap_tick_to_grid(original_tick: float, step_ticks: float) -> float:
    if original_tick <= 0:
        return 0.0
    snapped_units = math.floor((original_tick / step_ticks) + 0.5)
    return float(snapped_units * step_ticks)


def fractional_tick_to_time(midi: pretty_midi.PrettyMIDI, tick_value: float) -> float:
    if tick_value <= 0:
        return 0.0

    if float(tick_value).is_integer():
        return float(midi.tick_to_time(int(tick_value)))

    lower_tick = int(tick_value)
    upper_tick = lower_tick + 1
    lower_time = float(midi.tick_to_time(lower_tick))
    upper_time = float(midi.tick_to_time(upper_tick))
    fraction = tick_value - lower_tick
    return lower_time + ((upper_time - lower_time) * fraction)


def quantize_midi_notes(
    midi: pretty_midi.PrettyMIDI,
    min_note_denominator: int,
) -> tuple[int, int, int]:
    grid_step_ticks = note_grid_step_ticks(midi.resolution, min_note_denominator)
    notes_processed = 0
    notes_changed = 0
    notes_dropped = 0

    for instrument in midi.instruments:
        quantized_notes: list[pretty_midi.Note] = []
        for note in instrument.notes:
            original_start_tick = float(midi.time_to_tick(note.start))
            original_end_tick = float(midi.time_to_tick(note.end))

            quantized_start_tick = snap_tick_to_grid(
                original_start_tick,
                grid_step_ticks,
            )
            quantized_end_tick = snap_tick_to_grid(
                original_end_tick,
                grid_step_ticks,
            )

            notes_processed += 1
            note_changed = (
                quantized_start_tick != original_start_tick
                or quantized_end_tick != original_end_tick
            )

            if quantized_start_tick == quantized_end_tick:
                notes_dropped += 1
                notes_changed += 1
                continue

            note.start = fractional_tick_to_time(midi, quantized_start_tick)
            note.end = fractional_tick_to_time(midi, quantized_end_tick)
            quantized_notes.append(note)

            if note_changed:
                notes_changed += 1

        quantized_notes.sort(key=lambda note: (note.start, note.pitch, note.end))
        instrument.notes = quantized_notes

    return notes_processed, notes_changed, notes_dropped


def build_output_path(input_path: Path) -> Path:
    return input_path.parent / f"{input_path.stem}-edit.mid"


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_path = build_output_path(input_path)
    midi = load_midi(input_path)
    notes_processed, notes_changed, notes_dropped = quantize_midi_notes(
        midi,
        min_note_denominator=args.min_note_denominator,
    )
    save_midi(midi, output_path)

    result = {
        "input_path": str(input_path.resolve()),
        "output_path": str(output_path.resolve()),
        "min_note_denominator": args.min_note_denominator,
        "notes_processed": notes_processed,
        "notes_changed": notes_changed,
        "notes_dropped": notes_dropped,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
