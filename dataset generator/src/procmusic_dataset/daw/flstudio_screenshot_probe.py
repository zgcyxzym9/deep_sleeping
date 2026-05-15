from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

from PIL import ImageGrab

from procmusic_dataset.daw.flstudio_probe import close_windows
from procmusic_dataset.daw.flstudio_export_probe import _wait_for_main_window


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Open FL Studio project and capture a desktop screenshot for UI calibration.")
    parser.add_argument("--exe", type=Path, default=Path("D:/fl/FL64.exe"))
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--settle-seconds", type=float, default=10.0)
    parser.add_argument("--close", action="store_true")
    args = parser.parse_args(argv)

    process = subprocess.Popen([str(args.exe), str(args.project)])
    _wait_for_main_window(process.pid, args.timeout)
    time.sleep(args.settle_seconds)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    ImageGrab.grab().save(args.out)
    if args.close:
        close_windows(process.pid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
