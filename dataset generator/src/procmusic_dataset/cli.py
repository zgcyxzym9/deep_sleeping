from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .config import BatchConfig, GenerationConfig
from .daw.flstudio import FLStudioConfig, FLStudioRenderer
from .pipeline import DatasetPipeline
from .renderers.reference import ReferenceRenderer
from .renderers.vst import DEFAULT_VST_PLUGIN, VSTRenderer, VSTRendererConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate procedural multitrack music datasets.")
    parser.add_argument("--out", required=True, type=Path, help="Output dataset directory.")
    parser.add_argument("--count", type=int, default=1, help="Number of projects to generate.")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed.")
    parser.add_argument("--min-tracks", type=int, default=3)
    parser.add_argument("--max-tracks", type=int, default=8)
    parser.add_argument("--min-bars", type=int, default=8)
    parser.add_argument("--max-bars", type=int, default=24)
    parser.add_argument("--sample-rate", type=int, default=44_100)
    parser.add_argument("--renderer", choices=["reference", "flstudio-plan", "vst"], default="reference")
    parser.add_argument("--fl-root", type=Path, default=Path("D:/fl"))
    parser.add_argument("--fl-exe", type=Path, default=None)
    parser.add_argument("--vst-plugin", type=Path, default=DEFAULT_VST_PLUGIN)
    parser.add_argument("--vst-sample-rate", type=int, default=44_100)
    parser.add_argument("--write-preview-mp3", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")
    generation = GenerationConfig(
        min_tracks=args.min_tracks,
        max_tracks=args.max_tracks,
        min_bars=args.min_bars,
        max_bars=args.max_bars,
        sample_rate=args.sample_rate,
    )
    config = BatchConfig(output_dir=args.out, count=args.count, seed=args.seed, generation=generation, overwrite=args.overwrite)
    if args.renderer == "flstudio-plan":
        renderer = FLStudioRenderer(FLStudioConfig(root=args.fl_root, executable=args.fl_exe))
    elif args.renderer == "vst":
        renderer = VSTRenderer(
            VSTRendererConfig(
                plugin_path=args.vst_plugin,
                sample_rate=args.vst_sample_rate,
                write_preview_mp3=args.write_preview_mp3,
            )
        )
    else:
        renderer = ReferenceRenderer(sample_rate=args.sample_rate)
    DatasetPipeline(config, renderer).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
