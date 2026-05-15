from procmusic_dataset.config import GenerationConfig
from procmusic_dataset.generator import ProjectGenerator
from procmusic_dataset.models import to_json_dict


def test_generator_is_deterministic():
    config = GenerationConfig(min_tracks=3, max_tracks=3, min_bars=4, max_bars=4)
    generator = ProjectGenerator(config)

    first = generator.generate(0, 123)
    second = generator.generate(0, 123)

    assert to_json_dict(first) == to_json_dict(second)
    assert first.source_count == 3
    assert all(track.notes for track in first.tracks)


def test_generator_validates_empty_instrument_pool():
    config = GenerationConfig(allow_drums=False, allow_synths=False, allow_acoustic=False)
    generator = ProjectGenerator(config)

    try:
        generator.generate(0, 1)
    except ValueError as exc:
        assert "instrument constraints" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_generator_does_not_repeat_instrument_categories():
    config = GenerationConfig(min_tracks=10, max_tracks=10, min_bars=1, max_bars=1)
    project = ProjectGenerator(config).generate(0, 2026)

    categories = [track.instrument_category for track in project.tracks]
    assert len(categories) == len(set(categories))
    assert categories.count("drum_kit") <= 1


def test_generator_keeps_role_pitch_ranges_controlled():
    config = GenerationConfig(min_tracks=7, max_tracks=7, min_bars=2, max_bars=2)
    project = ProjectGenerator(config).generate(0, 2031)
    ranges = {
        "bass": (34, 55, 12),
        "harmony": (45, 76, 16),
        "melody": (50, 84, 18),
    }

    for track in project.tracks:
        if track.role not in ranges:
            continue
        low, high, max_span = ranges[track.role]
        pitches = [note.pitch for note in track.notes]
        assert min(pitches) >= low
        assert max(pitches) <= high
        assert max(pitches) - min(pitches) <= max_span
