"""Guitar audio to MIDI transcription powered by Spotify Basic Pitch."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torchaudio

try:
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import predict as basic_pitch_predict
    import pretty_midi
except ImportError as exc:  # pragma: no cover - optional dependency at runtime
    ICASSP_2022_MODEL_PATH = None
    basic_pitch_predict = None
    pretty_midi = None
    BASIC_PITCH_IMPORT_ERROR: Exception | None = exc
else:
    BASIC_PITCH_IMPORT_ERROR = None


GUITAR_MIN_MIDI = 40
GUITAR_MAX_MIDI = 88
GUITAR_MIN_FREQUENCY_HZ = 440.0 * (2.0 ** ((GUITAR_MIN_MIDI - 69) / 12.0))
GUITAR_MAX_FREQUENCY_HZ = 440.0 * (2.0 ** ((GUITAR_MAX_MIDI - 69) / 12.0))
DEFAULT_FRAME_THRESHOLD = 0.5
DEFAULT_MINIMUM_NOTE_LENGTH_MS = 100
DEFAULT_ONSET_THRESHOLD = 0.7
DEFAULT_BPM = 120.0


@dataclass(frozen=True)
class AudioMetadata:
    sample_rate: int | None
    duration_seconds: float | None


@dataclass(frozen=True)
class NoteEvent:
    pitch: int
    start_seconds: float
    end_seconds: float
    velocity: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe a guitar recording to a MIDI file with Basic Pitch."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input guitar recording.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output .mid path. Defaults to the input stem with a .mid extension.",
    )
    parser.add_argument(
        "--min-note-duration",
        type=float,
        default=DEFAULT_MINIMUM_NOTE_LENGTH_MS / 1000.0,
        help="Shortest note duration to keep in the MIDI output, in seconds.",
    )
    parser.add_argument(
        "--onset-threshold",
        type=float,
        default=DEFAULT_ONSET_THRESHOLD,
        help="Basic Pitch onset activation threshold.",
    )
    parser.add_argument(
        "--frame-threshold",
        type=float,
        default=DEFAULT_FRAME_THRESHOLD,
        help="Basic Pitch frame activation threshold.",
    )
    parser.add_argument(
        "--minimum-frequency",
        type=float,
        default=GUITAR_MIN_FREQUENCY_HZ,
        help="Minimum note frequency to keep, in Hz.",
    )
    parser.add_argument(
        "--maximum-frequency",
        type=float,
        default=GUITAR_MAX_FREQUENCY_HZ,
        help="Maximum note frequency to keep, in Hz.",
    )
    parser.add_argument(
        "--multiple-pitch-bends",
        action="store_true",
        help="Allow overlapping notes to keep independent pitch bends.",
    )
    parser.add_argument(
        "--bpm",
        type=float,
        default=DEFAULT_BPM,
        help="Tempo metadata stored in the output MIDI file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input audio file does not exist: {input_path}")

    output_path = Path(args.output) if args.output else input_path.with_suffix(".mid")
    metadata = probe_audio_metadata(input_path)
    notes, midi_data = transcribe_guitar_to_midi(
        input_path=input_path,
        minimum_note_duration_seconds=args.min_note_duration,
        onset_threshold=args.onset_threshold,
        frame_threshold=args.frame_threshold,
        minimum_frequency_hz=args.minimum_frequency,
        maximum_frequency_hz=args.maximum_frequency,
        multiple_pitch_bends=args.multiple_pitch_bends,
        melodia_trick=False,
        bpm=args.bpm,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    midi_data.write(str(output_path))

    result = {
        "backend": "basic-pitch",
        "input_path": str(input_path.resolve()),
        "output_path": str(output_path.resolve()),
        "sample_rate": metadata.sample_rate,
        "duration_seconds": (
            round(metadata.duration_seconds, 3)
            if metadata.duration_seconds is not None
            else None
        ),
        "notes_detected": len(notes),
        "notes_preview": [
            {
                "pitch": note.pitch,
                "start_seconds": round(note.start_seconds, 3),
                "end_seconds": round(note.end_seconds, 3),
                "velocity": note.velocity,
            }
            for note in notes[:12]
        ],
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


def transcribe_guitar_to_midi(
    input_path: str | Path,
    minimum_note_duration_seconds: float,
    onset_threshold: float,
    frame_threshold: float,
    minimum_frequency_hz: float,
    maximum_frequency_hz: float,
    multiple_pitch_bends: bool,
    melodia_trick: bool,
    bpm: float,
) -> tuple[list[NoteEvent], Any]:
    if basic_pitch_predict is None or ICASSP_2022_MODEL_PATH is None:
        raise RuntimeError(
            "Basic Pitch is not available. Install it with `pip install basic-pitch`."
        ) from BASIC_PITCH_IMPORT_ERROR

    _model_output, midi_data, note_events = basic_pitch_predict(
        audio_path=str(input_path),
        model_or_model_path=ICASSP_2022_MODEL_PATH,
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
        minimum_note_length=minimum_note_duration_seconds * 1000.0,
        minimum_frequency=minimum_frequency_hz,
        maximum_frequency=maximum_frequency_hz,
        multiple_pitch_bends=multiple_pitch_bends,
        melodia_trick=melodia_trick,
        midi_tempo=bpm,
    )
    tag_midi_as_guitar(midi_data)
    return basic_pitch_note_events_to_notes(note_events), midi_data


def probe_audio_metadata(input_path: Path) -> AudioMetadata:
    try:
        info = torchaudio.info(str(input_path))
    except Exception:
        return AudioMetadata(sample_rate=None, duration_seconds=None)

    sample_rate = int(info.sample_rate) if info.sample_rate > 0 else None
    if sample_rate is None or info.num_frames <= 0:
        duration_seconds = None
    else:
        duration_seconds = float(info.num_frames / sample_rate)
    return AudioMetadata(sample_rate=sample_rate, duration_seconds=duration_seconds)


def basic_pitch_note_events_to_notes(
    note_events: list[tuple[float, float, int, float, list[int] | None]],
) -> list[NoteEvent]:
    notes: list[NoteEvent] = []
    for start_seconds, end_seconds, pitch_midi, amplitude, _pitch_bends in note_events:
        pitch = int(round(pitch_midi))
        if pitch < GUITAR_MIN_MIDI or pitch > GUITAR_MAX_MIDI:
            continue

        velocity = int(round(max(1.0, min(127.0, amplitude * 127.0))))
        notes.append(
            NoteEvent(
                pitch=pitch,
                start_seconds=float(start_seconds),
                end_seconds=float(max(end_seconds, start_seconds)),
                velocity=velocity,
            )
        )

    notes.sort(key=lambda note: (note.start_seconds, note.pitch, note.end_seconds))
    return notes


def tag_midi_as_guitar(midi_data: Any) -> None:
    if pretty_midi is None:
        return

    guitar_program = pretty_midi.instrument_name_to_program("Acoustic Guitar (steel)")
    for instrument in midi_data.instruments:
        instrument.program = guitar_program
        instrument.name = "Guitar"


if __name__ == "__main__":
    main()
