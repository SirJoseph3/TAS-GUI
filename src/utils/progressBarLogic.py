from rich.progress import (
    Progress,
    TimeElapsedColumn,
    TimeRemainingColumn,
    BarColumn,
    TextColumn,
)
import src.constants as cs
from rich.progress import ProgressColumn
from time import time
from src.utils.aeComms import progressState

import os
import logging
import psutil


class SpeedColumn(ProgressColumn):
    """Displays the current download speed in MB/s."""

    def render(self, task):
        elapsed = task.elapsed or 0
        speed = task.completed / elapsed if elapsed > 0 else 0
        return f"Speed: [magenta]{speed:.2f} MB/s[/magenta]"


class FPSColumn(ProgressColumn):
    def __init__(self):
        super().__init__()
        self.startTime = None

    def render(self, task):
        if self.startTime is None:
            self.startTime = time()
        elapsed = time() - self.startTime
        fps = task.completed / elapsed if elapsed > 0 else 0
        return f"FPS: [magenta]{fps:.2f}[/magenta]"


class MemoryColumn(ProgressColumn):
    def __init__(self, totalFrames: int):
        super().__init__()
        self.advanceCount = 0
        self.updateInterval = max(1, totalFrames // 1000)
        self.cachedMem = 0
        self.process = psutil.Process(os.getpid())
        self.lastUpdate = 0

    def render(self, task):
        self.advanceCount += 1
        if self.advanceCount - self.lastUpdate >= self.updateInterval:
            self.lastUpdate = self.advanceCount
            try:
                mem = self.process.memory_info().rss / (1024 * 1024)
                if mem > self.cachedMem:
                    self.cachedMem = mem
                else:
                    self.cachedMem = (self.cachedMem + mem) / 2
            except psutil.NoSuchProcess:
                self.process = psutil.Process(os.getpid())
        return f"Mem: [yellow]{self.cachedMem:.1f}MB[/yellow]"


class ProgressBarLogic:
    def __init__(
        self,
        totalFrames: int,
        title: str = "Processing",
    ):
        """
        Initializes the progress bar for the given range of frames.

        Args:
            totalFrames (int): The total number of frames to process"""
        self.totalFrames = totalFrames
        self.completed = 0

    def __enter__(self):
        self.startTime = time()
        self.guiProgress = os.environ.get("TAS_GUI_PROGRESS") == "1"
        
        if cs.ADOBE:
            self.advanceCount = 0
            # More frequent updates - every 0.5% or at least every 10 frames
            self.updateInterval = max(10, self.totalFrames // 200)
            logging.info(f"Update interval: {self.updateInterval} frames")

        else:
            if self.guiProgress:
                # In GUI mode we don't render Rich's progress UI. We only emit parseable [PROGRESS] lines.
                self.progress = None
                self.task = None
                # Avoid flooding stdout; emitting every frame can become a bottleneck.
                self._lastGuiEmitTime = 0.0
            else:
                self.progress = Progress(
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    "•",
                    TextColumn("Elapsed Time:"),
                    TimeElapsedColumn(),
                    "•",
                    TextColumn("ETA:"),
                    TimeRemainingColumn(),
                    "•",
                    FPSColumn(),
                    "•",
                    MemoryColumn(self.totalFrames),
                    "•",
                    TextColumn("Frames: [green]{task.completed}/{task.total}[/green]"),
                )
                self.task = self.progress.add_task("Processing:", total=self.totalFrames)
                self.progress.start()

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if cs.ADOBE:
            currentTime = time()
            elapsedTime = currentTime - self.startTime
            fps = self.completed / elapsedTime if elapsedTime > 0 else 0

            progressState.update(
                {
                    "currentFrame": self.completed,
                    "totalFrames": self.totalFrames,
                    "fps": round(fps, 2),
                    "eta": 0.0,
                    "elapsedTime": elapsedTime,
                    "status": "Finishing...",
                }
            )
        else:
            if not self.guiProgress and self.progress:
                self.progress.stop()

    def advance(self, advance=1):
        if cs.ADOBE:
            self.completed += advance
            self.advanceCount += advance

            framesSinceLastUpdate = self.completed % self.updateInterval
            shouldUpdate = (
                framesSinceLastUpdate < advance or self.completed >= self.totalFrames
            )

            if shouldUpdate:
                currentTime = time()
                elapsedTime = currentTime - self.startTime
                fps = self.completed / elapsedTime if elapsedTime > 0 else 0

                if fps > 0 and self.completed < self.totalFrames:
                    remainingFrames = self.totalFrames - self.completed
                    eta = remainingFrames / fps
                else:
                    eta = 0

                progressState.update(
                    {
                        "currentFrame": self.completed,
                        "totalFrames": self.totalFrames,
                        "fps": round(fps, 2),
                        "eta": eta,
                        "elapsedTime": elapsedTime,
                        "status": "Processing...",
                    }
                )

        else:
            self.completed += advance

            if not self.guiProgress:
                self.progress.update(self.task, advance=advance)

            # Emit parseable progress lines:
            # - CLI: keep it throttled (every ~1%) to avoid noisy logs
            # - GUI: emit throttled with \r overwrite for smooth progress (avoid stdout bottlenecks)
            should_emit = False
            end_char = "\n"

            if self.guiProgress:
                end_char = "\n" if self.completed >= self.totalFrames else "\r"
                if end_char == "\n":
                    should_emit = True
                else:
                    # Update UI at ~4Hz; this is smooth enough and avoids per-frame flush cost.
                    now = time()
                    last_emit = getattr(self, "_lastGuiEmitTime", 0.0)
                    if now - last_emit >= 0.25:
                        self._lastGuiEmitTime = now
                        should_emit = True
            else:
                update_interval = max(1, self.totalFrames // 100)
                if self.completed % update_interval == 0 or self.completed >= self.totalFrames:
                    should_emit = True

            if should_emit:
                currentTime = time()
                elapsedTime = currentTime - self.startTime if hasattr(self, 'startTime') else 0
                fps = self.completed / elapsedTime if elapsedTime > 0 else 0
                percent = int((self.completed / self.totalFrames) * 100) if self.totalFrames > 0 else 0
                
                hours = int(elapsedTime // 3600)
                minutes = int((elapsedTime % 3600) // 60)
                seconds = int(elapsedTime % 60)
                elapsed_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                
                if fps > 0 and self.completed < self.totalFrames:
                    remainingFrames = self.totalFrames - self.completed
                    eta_seconds = remainingFrames / fps
                    eta_hours = int(eta_seconds // 3600)
                    eta_minutes = int((eta_seconds % 3600) // 60)
                    eta_secs = int(eta_seconds % 60)
                    eta_str = f"{eta_hours:02d}:{eta_minutes:02d}:{eta_secs:02d}"
                else:
                    eta_str = "00:00:00"
                
                import sys
                sys.stdout.write(
                    f"[PROGRESS] {percent}% | Frames: {self.completed}/{self.totalFrames} | FPS: {fps:.2f} | Elapsed: {elapsed_str} | ETA: {eta_str}{end_char}"
                )
                sys.stdout.flush()

    def __call__(self, advance=1):
        self.advance(advance)


def _emit_gui_download_progress(state, end_char: str):
    import sys

    # When total is unknown (0), fall back to indeterminate download reporting.
    total = int(getattr(state, "totalData", 0) or 0)
    completed = int(getattr(state, "completed", 0) or 0)
    indeterminate = bool(getattr(state, "_downloadIndeterminate", False)) or total <= 0

    if indeterminate:
        sys.stdout.write(f"[DOWNLOAD] Downloading... | Data: {completed} MB{end_char}")
        sys.stdout.flush()
        return

    total = max(1, total)
    completed = min(completed, total)
    percent = int((completed / total) * 100)

    # Avoid writing the exact same percent repeatedly in tight loops.
    if percent == getattr(state, "_lastPercent", -1) and end_char == "\r":
        return

    state._lastPercent = percent
    sys.stdout.write(f"[DOWNLOAD] {percent}% | Data: {completed}/{total} MB{end_char}")
    sys.stdout.flush()

    def _emit_gui_progress(self, end_char: str):
        import sys

        # When total is unknown (0), fall back to indeterminate download reporting.
        if self._downloadIndeterminate:
            sys.stdout.write(
                f"[DOWNLOAD] Downloading... | Data: {int(self.completed)} MB{end_char}"
            )
            sys.stdout.flush()
            return

        total = max(1, int(self.totalData))
        completed = min(int(self.completed), total)
        percent = int((completed / total) * 100)

        # Avoid writing the exact same percent repeatedly in tight loops.
        if percent == self._lastPercent and end_char == "\r":
            return

        self._lastPercent = percent
        sys.stdout.write(f"[DOWNLOAD] {percent}% | Data: {completed}/{total} MB{end_char}")
        sys.stdout.flush()

    def updateTotal(self, newTotal: int):
        """
        Updates the total value of the progress bar.

        Args:
            newTotal (int): The new total value
        """
        self.totalFrames = newTotal
        if not cs.ADOBE and not getattr(self, "guiProgress", False):
            self.progress.update(self.task, total=newTotal)


class ProgressBarDownloadLogic:
    def __init__(self, totalData: int, title: str):
        """
        Initializes the progress bar for the given range of data.

        Args:
            totalData (int): The total amount of data to process
            title (str): The title of the progress bar
        """
        # Existing callers pass (totalMB + 1). Clamp for robustness when content-length is unknown.
        self.totalData = max(0, int(totalData) - 1)
        self.title = title
        self.guiProgress = False
        self.completed = 0
        self.startTime = None
        self._lastPercent = -1
        self._downloadIndeterminate = False

    def __enter__(self):
        self.startTime = time()
        self.guiProgress = os.environ.get("TAS_GUI_PROGRESS") == "1"

        if self.guiProgress:
            self.completed = 0
            self._lastPercent = -1
            self._downloadIndeterminate = self.totalData <= 0
            _emit_gui_download_progress(self, end_char="\r")
            return self

        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            "•",
            TextColumn("Elapsed Time:"),
            TimeElapsedColumn(),
            "•",
            TextColumn("ETA:"),
            TimeRemainingColumn(),
            "•",
            SpeedColumn(),
            "•",
            TextColumn("Data: [cyan]{task.completed}/{task.total} MB[/cyan]"),
        )
        self.task = self.progress.add_task(self.title, total=self.totalData)
        self.progress.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.guiProgress:
            # Ensure the last \r update is finalized with a newline.
            _emit_gui_download_progress(self, end_char="\n")
            return

        self.progress.stop()

    def setTotal(self, newTotal: int):
        """
        Updates the total value of the progress bar.

        Args:
            newTotal (int): The new total value"""
        self.totalData = max(0, int(newTotal))

        if self.guiProgress:
            self._downloadIndeterminate = self.totalData <= 0
            _emit_gui_download_progress(self, end_char="\r")
            return

        self.progress.update(self.task, total=newTotal)

    def advance(self, advance=1):
        if self.guiProgress:
            self.completed += advance
            self._downloadIndeterminate = self.totalData <= 0

            end_char = (
                "\n"
                if (not self._downloadIndeterminate and self.completed >= self.totalData)
                else "\r"
            )
            _emit_gui_download_progress(self, end_char=end_char)
            return

        task = self.progress.tasks[self.task]
        if task.start_time is None:
            task.start_time = time()
        self.progress.update(self.task, advance=advance)

    def __call__(self, advance=1):
        self.advance(advance)
