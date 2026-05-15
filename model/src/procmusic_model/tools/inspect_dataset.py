from __future__ import annotations

import argparse
from collections import Counter

from procmusic_model.config import add_common_args, apply_overrides, load_config
from procmusic_model.data import MusicSeparationDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args()
    config = apply_overrides(load_config(args.config), args)
    dataset = MusicSeparationDataset(config.dataset.root, config.dataset.sample_rate, config.dataset.segment_seconds, False, config.dataset.mono)
    categories = Counter()
    durations = []
    for example in dataset:
        durations.append(example.mixture.shape[-1] / config.dataset.sample_rate)
        categories.update(track["instrument_category"] for track in example.metadata["tracks"])
    print(f"projects: {len(dataset)}")
    print(f"segment_seconds: min={min(durations):.2f} max={max(durations):.2f}")
    print(f"source_count: {[dataset[idx].source_count for idx in range(len(dataset))]}")
    print(f"categories: {dict(categories)}")


if __name__ == "__main__":
    main()
