from __future__ import annotations

import argparse
import ctypes
import json
import subprocess
import time
from pathlib import Path

import win32con
import win32com.client
import win32gui
import win32process
from PIL import ImageGrab


def export_existing_project_to_wav(
    executable: Path,
    project_path: Path,
    output_wav: Path,
    load_timeout_seconds: float = 45.0,
    render_timeout_seconds: float = 180.0,
    trace_path: Path | None = None,
    settle_seconds: float = 20.0,
    hotkey_method: str = "wscript",
    hotkey_key_delay_seconds: float = 0.05,
    hotkey: str = "ctrl+shift+r",
    render_screenshot_path: Path | None = None,
) -> bool:
    trace: list[dict] = []
    if not executable.exists():
        raise FileNotFoundError(executable)
    if not project_path.exists():
        raise FileNotFoundError(project_path)
    output_wav = output_wav.resolve()
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    if output_wav.exists():
        output_wav.unlink()
    default_output = project_path.with_suffix(output_wav.suffix).resolve()
    default_output_before = _file_signature(default_output)

    process = subprocess.Popen([str(executable), str(project_path)])
    started_at = time.time()
    main_hwnd = _wait_for_main_window(process.pid, load_timeout_seconds)
    _trace(trace, "after_load_wait", process.pid)
    if main_hwnd is None:
        _write_trace(trace_path, trace)
        _terminate(process)
        return False

    time.sleep(settle_seconds)
    _trace(trace, "after_settle", process.pid)
    _activate_window(main_hwnd)
    time.sleep(1.0)

    _send_export_hotkey(main_hwnd, hotkey_method, hotkey_key_delay_seconds, hotkey)
    save_hwnd = _wait_for_any_dialog(process.pid, timeout_seconds=20.0)
    _trace(trace, "after_export_hotkey", process.pid)
    if save_hwnd is None:
        _write_trace(trace_path, trace)
        _terminate(process)
        return False

    _set_save_dialog_filename(save_hwnd, output_wav)
    time.sleep(0.3)
    _confirm_dialog(save_hwnd)

    render_hwnd = _wait_for_render_window(process.pid, timeout_seconds=20.0)
    _trace(trace, "after_save_path_enter", process.pid)
    if render_hwnd is not None:
        _activate_window(render_hwnd)
        _capture_window_screenshot(render_hwnd, render_screenshot_path)
        time.sleep(0.5)
        _click_render_start_button(render_hwnd)

    deadline = time.monotonic() + render_timeout_seconds
    while time.monotonic() < deadline:
        if output_wav.exists() and output_wav.stat().st_size > 44:
            _trace(trace, "output_detected", process.pid)
            _write_trace(trace_path, trace)
            _close_process_windows(process.pid)
            return True
        if (
            default_output != output_wav
            and _render_window_is_closed(process.pid)
            and _is_new_or_changed_file(default_output, default_output_before, started_at)
        ):
            default_output.replace(output_wav)
            _trace(trace, "default_output_moved", process.pid)
            _write_trace(trace_path, trace)
            _close_process_windows(process.pid)
            return True
        time.sleep(1.0)

    _trace(trace, "render_timeout", process.pid)
    _write_trace(trace_path, trace)
    _terminate(process)
    return False


def _wait_for_main_window(process_id: int, timeout_seconds: float) -> int | None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        for hwnd, title, class_name, visible in _windows_for_pid(process_id):
            text = f"{title} {class_name}".lower()
            if visible and ("fl studio" in text or class_name == "TFruityLoopsMainForm"):
                return hwnd
        time.sleep(0.5)
    return None


def _wait_for_any_dialog(process_id: int, timeout_seconds: float) -> int | None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        for hwnd, title, class_name, visible in _windows_for_pid(process_id):
            if visible and class_name == "#32770":
                return hwnd
            lowered = title.lower()
            if visible and any(token in lowered for token in ("save", "export", "保存", "另存")):
                return hwnd
        time.sleep(0.25)
    return None


def _wait_for_render_window(process_id: int, timeout_seconds: float) -> int | None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        for hwnd, title, class_name, visible in _windows_for_pid(process_id):
            lowered = title.lower()
            if visible and any(token in lowered for token in ("render", "export", "渲染", "导出")):
                return hwnd
        time.sleep(0.25)
    return None


def _render_window_is_closed(process_id: int) -> bool:
    return _wait_for_render_window(process_id, timeout_seconds=0.1) is None


def _windows_for_pid(process_id: int) -> list[tuple[int, str, str, bool]]:
    result: list[tuple[int, str, str, bool]] = []

    def callback(hwnd: int, _extra: object) -> bool:
        _thread_id, pid = win32process.GetWindowThreadProcessId(hwnd)
        if pid == process_id:
            result.append((hwnd, win32gui.GetWindowText(hwnd), win32gui.GetClassName(hwnd), bool(win32gui.IsWindowVisible(hwnd))))
        return True

    win32gui.EnumWindows(callback, None)
    return result


def _child_windows(hwnd: int) -> list[tuple[int, str, str, bool]]:
    result: list[tuple[int, str, str, bool]] = []

    def callback(child: int, _extra: object) -> bool:
        result.append((child, win32gui.GetWindowText(child), win32gui.GetClassName(child), bool(win32gui.IsWindowVisible(child))))
        return True

    win32gui.EnumChildWindows(hwnd, callback, None)
    return result


def _set_save_dialog_filename(dialog_hwnd: int, output_wav: Path) -> None:
    _activate_window(dialog_hwnd)
    filename_control = _find_likely_filename_control(dialog_hwnd)
    if filename_control is not None:
        win32gui.SetFocus(filename_control)
        time.sleep(0.1)
        _send_hotkey_sequence([win32con.VK_CONTROL, ord("A")])
        _send_text(str(output_wav))
    else:
        _send_text(str(output_wav))


def _confirm_dialog(dialog_hwnd: int) -> None:
    _activate_window(dialog_hwnd)
    buttons = _child_windows(dialog_hwnd)
    primary_buttons = [
        child for child in buttons if child[2] == "Button" and _is_primary_dialog_button(child[1])
    ]
    if primary_buttons:
        win32gui.SendMessage(primary_buttons[0][0], win32con.BM_CLICK, 0, 0)
        return
    _send_key(win32con.VK_RETURN)


def _find_likely_filename_control(dialog_hwnd: int) -> int | None:
    left, top, right, bottom = win32gui.GetWindowRect(dialog_hwnd)
    mid_y = top + ((bottom - top) * 0.45)
    controls: list[tuple[int, int, int]] = []
    for child_hwnd, _title, class_name, visible in _child_windows(dialog_hwnd):
        if not visible or class_name not in {"Edit", "ComboBox"}:
            continue
        child_left, child_top, child_right, child_bottom = win32gui.GetWindowRect(child_hwnd)
        if child_bottom <= mid_y or child_right - child_left < 80:
            continue
        class_rank = 0 if class_name == "Edit" else 1
        controls.append((child_top, class_rank, child_hwnd))
    if controls:
        controls.sort(reverse=True)
        return controls[0][2]

    edit_controls = [child for child in _child_windows(dialog_hwnd) if child[2] == "Edit" and child[3]]
    if edit_controls:
        return edit_controls[0][0]
    return None


def _is_primary_dialog_button(title: str) -> bool:
    normalized = title.replace("&", "").lower()
    return any(token in normalized for token in ("保存", "save", "打开", "open"))


def _activate_window(hwnd: int) -> None:
    try:
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        _force_foreground_window(hwnd)
    except Exception:
        try:
            win32gui.BringWindowToTop(hwnd)
        except Exception:
            pass


class _KeyboardInput(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.c_ushort),
        ("wScan", ctypes.c_ushort),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class _InputUnion(ctypes.Union):
    _fields_ = [("ki", _KeyboardInput)]


class _Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong), ("union", _InputUnion)]


INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004


def _send_key(vk: int) -> None:
    _send_keyboard_input(vk, 0, 0)
    _send_keyboard_input(vk, 0, KEYEVENTF_KEYUP)


def _click_render_start_button(render_hwnd: int) -> None:
    left, top, right, bottom = win32gui.GetWindowRect(render_hwnd)
    x = right - 62
    y = bottom - 34
    ctypes.windll.user32.SetCursorPos(x, y)
    ctypes.windll.user32.mouse_event(0x0002, 0, 0, 0, 0)
    time.sleep(0.05)
    ctypes.windll.user32.mouse_event(0x0004, 0, 0, 0, 0)


def _send_export_hotkey(hwnd: int, method: str, key_delay_seconds: float, hotkey: str) -> None:
    if method == "wscript":
        _send_hotkey_wscript(hwnd, hotkey)
    elif method == "postmessage":
        _post_hotkey(hwnd, hotkey)
    elif method == "sendinput":
        _send_hotkey_sequence(_parse_hotkey(hotkey), key_delay_seconds)
    else:
        raise ValueError(f"unknown hotkey method: {method}")


def _send_hotkey_wscript(hwnd: int, hotkey: str) -> None:
    numlock_was_on = bool(ctypes.windll.user32.GetKeyState(win32con.VK_NUMLOCK) & 1)
    title = win32gui.GetWindowText(hwnd)
    shell = win32com.client.Dispatch("WScript.Shell")
    if title:
        shell.AppActivate(title)
    time.sleep(0.5)
    shell.SendKeys(_wscript_hotkey(hotkey))
    time.sleep(0.5)
    numlock_is_on = bool(ctypes.windll.user32.GetKeyState(win32con.VK_NUMLOCK) & 1)
    if numlock_is_on != numlock_was_on:
        _send_key(win32con.VK_NUMLOCK)


def _post_hotkey(hwnd: int, hotkey: str) -> None:
    keys = _parse_hotkey(hotkey)
    for key in keys:
        win32gui.PostMessage(hwnd, win32con.WM_KEYDOWN, key, 0)
    for key in reversed(keys):
        win32gui.PostMessage(hwnd, win32con.WM_KEYUP, key, 0)


def _send_hotkey_sequence(keys: list[int], key_delay_seconds: float = 0.05) -> None:
    time.sleep(0.1)
    for key in keys:
        _send_keyboard_input(key, 0, 0)
        time.sleep(key_delay_seconds)
    for key in reversed(keys):
        _send_keyboard_input(key, 0, KEYEVENTF_KEYUP)
        time.sleep(0.05)


def _parse_hotkey(hotkey: str) -> list[int]:
    mapping = {
        "ctrl": win32con.VK_CONTROL,
        "control": win32con.VK_CONTROL,
        "shift": win32con.VK_SHIFT,
        "alt": win32con.VK_MENU,
        "r": ord("R"),
    }
    keys = []
    for token in hotkey.lower().split("+"):
        token = token.strip()
        if token not in mapping:
            raise ValueError(f"unsupported hotkey token: {token}")
        keys.append(mapping[token])
    return keys


def _wscript_hotkey(hotkey: str) -> str:
    result = ""
    key_token = ""
    for token in hotkey.lower().split("+"):
        token = token.strip()
        if token in {"ctrl", "control"}:
            result += "^"
        elif token == "shift":
            result += "+"
        elif token == "alt":
            result += "%"
        else:
            key_token = token
    return result + key_token


def _send_text(text: str) -> None:
    for char in text:
        code = ord(char)
        _send_keyboard_input(0, code, KEYEVENTF_UNICODE)
        _send_keyboard_input(0, code, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP)


def _send_keyboard_input(vk: int, scan: int, flags: int) -> None:
    extra = ctypes.c_ulong(0)
    keyboard = _KeyboardInput(vk, scan, flags, 0, ctypes.pointer(extra))
    item = _Input(INPUT_KEYBOARD, _InputUnion(ki=keyboard))
    ctypes.windll.user32.SendInput(1, ctypes.pointer(item), ctypes.sizeof(item))


def _file_signature(path: Path) -> tuple[int, int] | None:
    if not path.exists():
        return None
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def _is_new_or_changed_file(path: Path, before: tuple[int, int] | None, started_at: float) -> bool:
    if not path.exists() or path.stat().st_size <= 44:
        return False
    stat = path.stat()
    after = (stat.st_size, stat.st_mtime_ns)
    return after != before and stat.st_mtime >= started_at


def _force_foreground_window(hwnd: int) -> None:
    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
    foreground = user32.GetForegroundWindow()
    current_thread = kernel32.GetCurrentThreadId()
    target_thread = user32.GetWindowThreadProcessId(hwnd, None)
    foreground_thread = user32.GetWindowThreadProcessId(foreground, None) if foreground else 0

    if foreground_thread:
        user32.AttachThreadInput(current_thread, foreground_thread, True)
    user32.AttachThreadInput(current_thread, target_thread, True)
    try:
        user32.ShowWindow(hwnd, win32con.SW_RESTORE)
        user32.BringWindowToTop(hwnd)
        user32.SetForegroundWindow(hwnd)
        user32.SetFocus(hwnd)
    finally:
        user32.AttachThreadInput(current_thread, target_thread, False)
        if foreground_thread:
            user32.AttachThreadInput(current_thread, foreground_thread, False)


def _close_process_windows(process_id: int) -> None:
    for hwnd, _title, _class_name, _visible in _windows_for_pid(process_id):
        if win32gui.IsWindow(hwnd):
            try:
                win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)
            except Exception:
                continue


def _terminate(process: subprocess.Popen) -> None:
    _close_process_windows(process.pid)
    try:
        process.terminate()
    except Exception:
        pass


def _trace(trace: list[dict], label: str, process_id: int) -> None:
    trace.append(
        {
            "label": label,
            "windows": [
                {
                    "hwnd": hwnd,
                    "title": title,
                    "class_name": class_name,
                    "visible": visible,
                    "children": [
                        {"hwnd": child_hwnd, "title": child_title, "class_name": child_class, "visible": child_visible}
                        for child_hwnd, child_title, child_class, child_visible in _child_windows(hwnd)
                    ]
                    if visible and class_name in {"#32770", "TWAVRenderForm"}
                    else [],
                }
                for hwnd, title, class_name, visible in _windows_for_pid(process_id)
            ],
        }
    )


def _write_trace(path: Path | None, trace: list[dict]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(trace, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Experimental FL Studio GUI export probe.")
    parser.add_argument("--exe", type=Path, default=Path("D:/fl/FL64.exe"))
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--load-timeout", type=float, default=45.0)
    parser.add_argument("--render-timeout", type=float, default=180.0)
    parser.add_argument("--trace", type=Path, default=None)
    parser.add_argument("--settle-seconds", type=float, default=20.0)
    parser.add_argument("--hotkey-method", choices=["wscript", "sendinput", "postmessage"], default="wscript")
    parser.add_argument("--hotkey-key-delay", type=float, default=0.05)
    parser.add_argument("--hotkey", default="ctrl+shift+r")
    parser.add_argument("--render-screenshot", type=Path, default=None)
    args = parser.parse_args(argv)

    ok = export_existing_project_to_wav(
        args.exe,
        args.project,
        args.out,
        args.load_timeout,
        args.render_timeout,
        args.trace,
        args.settle_seconds,
        args.hotkey_method,
        args.hotkey_key_delay,
        args.hotkey,
        args.render_screenshot,
    )
    print("ok" if ok else "failed")
    return 0 if ok else 2


def _capture_window_screenshot(hwnd: int, path: Path | None) -> None:
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        ImageGrab.grab().save(path)
    except Exception:
        return


if __name__ == "__main__":
    raise SystemExit(main())
