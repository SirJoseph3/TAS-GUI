# Packaging / EXE build notes (Windows)

This repo is already a **portable** distribution: it ships a bundled Python (`python.exe` / `pythonw.exe`) and all runtime dependencies.

Because the app ships **large models/weights** (and often CUDA/NVDEC dependencies), the folder can easily be **6GB+**.

## Key point: EXE does not reduce size
Freezing to an `.exe` (PyInstaller/Nuitka/etc.) typically **does not shrink** the distribution. It often **increases** size (duplicate DLLs / bundled Python).

If you need a smaller download, the only real lever is to **separate assets**:

- Ship a smaller base package (app + minimal deps)
- Download models/weights on first run **or** provide a separate “weights pack”

## Recommended approach for this project: Launcher EXE
`gui_app.py` launches `main.py` as a subprocess. Fully freezing the GUI into a standalone EXE requires additional work (packaging backend too).

So the simplest/most robust approach is:

- Keep the current folder structure (portable)
- Build a **small launcher EXE** (`TAS-GUI.exe`) that simply runs:
  - `pythonw.exe gui_app.py`

### Build steps
1) Run:

```powershell
.\build_gui_launcher.ps1
```

2) Result:
- `TAS-GUI.exe` will be created in the repo root.

### Distribution
- Distribute `TAS-GUI.exe` **together with** the existing folder contents (`pythonw.exe`, `Lib/`, `weights/`, `ffmpeg/`, etc.)
- For download size, compress the folder using 7-Zip (`.7z`) or create an installer (Inno Setup / NSIS).
