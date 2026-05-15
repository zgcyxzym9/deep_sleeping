from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import win32con
import win32gui
import win32process


@dataclass(frozen=True)
class WindowInfo:
    hwnd: int
    title: str
    class_name: str
    visible: bool


@dataclass(frozen=True)
class ProbeResult:
    executable: str
    project_path: str
    process_id: int
    windows: list[WindowInfo]
    load_timeout_seconds: float
    detected_main_window: bool


def probe_flstudio(executable: Path, project_path: Path, timeout_seconds: float = 45.0) -> ProbeResult:
    if not executable.exists():
        raise FileNotFoundError(executable)
    if not project_path.exists():
        raise FileNotFoundError(project_path)

    process = subprocess.Popen([str(executable), str(project_path)])
    deadline = time.monotonic() + timeout_seconds
    windows: list[WindowInfo] = []
    detected = False
    while time.monotonic() < deadline:
        windows = _windows_for_pid(process.pid)
        detected = any(_looks_like_fl_window(window) for window in windows)
        if detected:
            break
        if process.poll() is not None:
            break
        time.sleep(0.5)

    return ProbeResult(
        executable=executable.as_posix(),
        project_path=project_path.as_posix(),
        process_id=process.pid,
        windows=windows,
        load_timeout_seconds=timeout_seconds,
        detected_main_window=detected,
    )


def close_windows(process_id: int) -> None:
    for window in _windows_for_pid(process_id):
        if win32gui.IsWindow(window.hwnd):
            try:
                win32gui.PostMessage(window.hwnd, win32con.WM_CLOSE, 0, 0)
            except Exception:
                continue


def _windows_for_pid(process_id: int) -> list[WindowInfo]:
    result: list[WindowInfo] = []

    def callback(hwnd: int, _extra: object) -> bool:
        _thread_id, pid = win32process.GetWindowThreadProcessId(hwnd)
        if pid == process_id:
            title = win32gui.GetWindowText(hwnd)
            class_name = win32gui.GetClassName(hwnd)
            result.append(WindowInfo(hwnd, title, class_name, win32gui.IsWindowVisible(hwnd)))
        return True

    win32gui.EnumWindows(callback, None)
    return result


def _looks_like_fl_window(window: WindowInfo) -> bool:
    text = f"{window.title} {window.class_name}".lower()
    return window.visible and ("fl studio" in text or "tfldesktop" in text or "tfldisplay" in text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Probe whether FL Studio can load a project file.")
    parser.add_argument("--exe", type=Path, default=Path("D:/fl/FL64.exe"))
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--close", action="store_true", help="Close detected FL windows after probing.")
    args = parser.parse_args(argv)

    result = probe_flstudio(args.exe, args.project, args.timeout)
    payload = asdict(result)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    if args.close:
        close_windows(result.process_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
