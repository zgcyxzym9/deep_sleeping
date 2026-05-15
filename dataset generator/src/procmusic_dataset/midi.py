from __future__ import annotations

from pathlib import Path

from .models import ProjectSpec, TrackSpec


TICKS_PER_BEAT = 480


def write_midi(project: ProjectSpec, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tracks = [_tempo_track(project.bpm)] + [_instrument_track(track.midi_channel, track.notes) for track in project.tracks]
    header = b"MThd" + (6).to_bytes(4, "big") + (1).to_bytes(2, "big") + len(tracks).to_bytes(2, "big") + TICKS_PER_BEAT.to_bytes(2, "big")
    body = b"".join(b"MTrk" + len(track).to_bytes(4, "big") + track for track in tracks)
    path.write_bytes(header + body)


def write_track_midi(project: ProjectSpec, track: TrackSpec, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tracks = [_tempo_track(project.bpm), _instrument_track(0, track.notes)]
    header = b"MThd" + (6).to_bytes(4, "big") + (1).to_bytes(2, "big") + len(tracks).to_bytes(2, "big") + TICKS_PER_BEAT.to_bytes(2, "big")
    body = b"".join(b"MTrk" + len(track_data).to_bytes(4, "big") + track_data for track_data in tracks)
    path.write_bytes(header + body)


def _tempo_track(bpm: int) -> bytes:
    mpqn = int(60_000_000 / bpm)
    return _varlen(0) + b"\xff\x51\x03" + mpqn.to_bytes(3, "big") + _varlen(0) + b"\xff\x2f\x00"


def _instrument_track(channel: int, notes: list) -> bytes:
    events: list[tuple[int, int, bytes]] = []
    for note in notes:
        start = int(round(note.start_beat * TICKS_PER_BEAT))
        end = int(round((note.start_beat + note.duration_beats) * TICKS_PER_BEAT))
        pitch = int(note.pitch)
        velocity = int(note.velocity)
        events.append((start, 1, bytes([0x90 | channel, pitch, velocity])))
        events.append((max(start + 1, end), 0, bytes([0x80 | channel, pitch, 0])))
    events.sort()

    current = 0
    data = bytearray()
    for tick, _order, payload in events:
        data += _varlen(tick - current)
        data += payload
        current = tick
    data += _varlen(0) + b"\xff\x2f\x00"
    return bytes(data)


def _varlen(value: int) -> bytes:
    if value < 0:
        raise ValueError("variable-length MIDI value cannot be negative")
    buffer = value & 0x7F
    value >>= 7
    bytes_out = [buffer]
    while value:
        bytes_out.insert(0, (value & 0x7F) | 0x80)
        value >>= 7
    return bytes(bytes_out)
