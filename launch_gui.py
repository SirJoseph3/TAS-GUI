import os
import subprocess
import sys


def _message_box(title: str, text: str) -> None:
    """Best-effort Windows error dialog (falls back to stderr)."""
    try:
        import ctypes

        ctypes.windll.user32.MessageBoxW(None, text, title, 0x10)  # MB_ICONERROR
    except Exception:
        try:
            sys.stderr.write(f"{title}: {text}\n")
        except Exception:
            pass


def main() -> int:
    # When frozen (PyInstaller), sys.executable points to the .exe in the install folder.
    # When running as a script, __file__ is inside the repo root.
    if getattr(sys, "frozen", False):
        root = os.path.dirname(sys.executable)
    else:
        root = os.path.dirname(os.path.abspath(__file__))

    gui_script = os.path.join(root, "gui_app.py")
    if not os.path.exists(gui_script):
        _message_box(
            "TAS GUI",
            f"gui_app.py not found next to {os.path.basename(sys.executable)}\n\nExpected: {gui_script}",
        )
        return 2

    pythonw = os.path.join(root, "pythonw.exe")
    python = os.path.join(root, "python.exe")

    if os.path.exists(pythonw):
        interpreter = pythonw
    elif os.path.exists(python):
        interpreter = python
    else:
        _message_box(
            "TAS GUI",
            f"pythonw.exe/python.exe not found in {root}\n\nThis launcher must live in the TAS folder.",
        )
        return 2

    env = os.environ.copy()

    # Run from the TAS root so relative paths (weights/, ffmpeg/, presets/) resolve correctly.
    try:
        subprocess.Popen([interpreter, gui_script], cwd=root, env=env)
    except Exception as e:
        _message_box("TAS GUI", f"Failed to launch GUI: {e}")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
