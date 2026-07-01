"""Background daemon thread that periodically samples host-level resource metrics."""

from __future__ import annotations

import atexit
import logging
import os
import shutil
import signal
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    import psutil

    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


def _directory_size_bytes(path: Path) -> int:
    """Return the recursive size of files under ``path``."""
    total = 0
    stack = [path]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    try:
                        if entry.is_symlink():
                            continue
                        if entry.is_file(follow_symlinks=False):
                            total += entry.stat(follow_symlinks=False).st_size
                        elif entry.is_dir(follow_symlinks=False):
                            stack.append(Path(entry.path))
                    except OSError:
                        continue
        except OSError:
            continue
    return total


class ResourceMonitor:
    """Background daemon thread that periodically samples host-level resource metrics.

    Tracks peak and average values for RAM, CPU, process RSS, and disk usage
    of watched directories. Registers atexit and SIGTERM handlers to ensure
    a summary is printed even on crash or SLURM preemption.
    """

    def __init__(
        self,
        sample_interval: float = 5.0,
        watch_dirs: list[str | Path] | None = None,
    ) -> None:
        self.sample_interval = sample_interval
        self._watch_dirs = [Path(d) for d in (watch_dirs or [])]

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = False
        self._stopped = False
        self._cached_summary: dict[str, Any] | None = None

        # Counters (protected by _lock).
        self._count = 0
        self._start_time = 0.0
        self._peak_ram = 0
        self._sum_ram = 0
        self._peak_cpu = 0.0
        self._sum_cpu = 0.0
        self._peak_rss = 0
        self._sum_rss = 0
        self._peak_disk: dict[str, int] = {}

        self._prev_sigterm_handler: Any = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background sampling thread."""
        if not _HAS_PSUTIL:
            logger.warning(
                "psutil is not installed; ResourceMonitor will not collect metrics"
            )
            return

        if self._started:
            return

        self._started = True
        self._start_time = time.monotonic()

        # Prime cpu_percent so the first real call returns a meaningful value.
        psutil.cpu_percent()

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

        atexit.register(self._atexit_handler)

        self._prev_sigterm_handler = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, self._sigterm_handler)

    def stop(self) -> dict[str, Any]:
        """Stop sampling and return the summary dict."""
        if self._stopped:
            return self._cached_summary or {}

        if not self._started:
            return {}

        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

        self._stopped = True
        self._cached_summary = self.summary()
        return self._cached_summary

    def summary(self) -> dict[str, Any]:
        """Return current peak/avg stats without stopping."""
        if not _HAS_PSUTIL or not self._started:
            return {}

        with self._lock:
            count = self._count
            if count == 0:
                return {}

            return {
                "ram_peak_mb": round(self._peak_ram / 1024 / 1024, 1),
                "ram_avg_mb": round(self._sum_ram / count / 1024 / 1024, 1),
                "cpu_peak_pct": round(self._peak_cpu, 1),
                "cpu_avg_pct": round(self._sum_cpu / count, 1),
                "process_rss_peak_mb": round(self._peak_rss / 1024 / 1024, 1),
                "process_rss_avg_mb": round(self._sum_rss / count / 1024 / 1024, 1),
                "disk_used_peak_mb": {
                    d: round(v / 1024 / 1024, 1)
                    for d, v in self._peak_disk.items()
                },
                "samples": count,
                "duration_sec": round(time.monotonic() - self._start_time, 1),
            }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """Loop executed by the daemon thread."""
        while not self._stop_event.is_set():
            self._sample()
            self._stop_event.wait(self.sample_interval)

    def _sample(self) -> None:
        """Collect one round of metrics."""
        rss = psutil.Process().memory_info().rss
        ram_used = psutil.virtual_memory().used
        cpu = psutil.cpu_percent(interval=None)

        disk_readings: dict[str, int] = {}
        for d in self._watch_dirs:
            try:
                if d.exists():
                    disk_readings[str(d)] = _directory_size_bytes(d)
            except OSError:
                pass

        with self._lock:
            self._count += 1

            self._sum_rss += rss
            if rss > self._peak_rss:
                self._peak_rss = rss

            self._sum_ram += ram_used
            if ram_used > self._peak_ram:
                self._peak_ram = ram_used

            self._sum_cpu += cpu
            if cpu > self._peak_cpu:
                self._peak_cpu = cpu

            for path_str, used in disk_readings.items():
                prev = self._peak_disk.get(path_str)
                if prev is None or used > prev:
                    self._peak_disk[path_str] = used

    def _log_summary(self, s: dict[str, Any]) -> None:
        """Log the summary dict in a human-readable format."""
        if not s:
            return

        lines = [
            f"Resource usage summary ({s['samples']} samples over {s['duration_sec']}s):",
            f"  RAM: peak={s['ram_peak_mb']} MB, avg={s['ram_avg_mb']} MB",
            f"  CPU: peak={s['cpu_peak_pct']}%, avg={s['cpu_avg_pct']}%",
            f"  Process RSS: peak={s['process_rss_peak_mb']} MB, avg={s['process_rss_avg_mb']} MB",
        ]
        for path_str, peak_mb in s.get("disk_used_peak_mb", {}).items():
            lines.append(f"  Disk: {path_str}={peak_mb} MB peak")

        logger.info("\n".join(lines))

    def _atexit_handler(self) -> None:
        """Called on normal interpreter exit."""
        if self._stopped:
            return
        s = self.stop()
        try:
            self._log_summary(s)
        except (ValueError, OSError):
            # Logging may fail if stderr is already closed during shutdown.
            pass

    def _sigterm_handler(self, sig: int, frame: Any) -> None:
        """Called on SIGTERM (e.g. SLURM preemption)."""
        s = self.stop()
        self._log_summary(s)

        prev = self._prev_sigterm_handler
        if callable(prev):
            prev(sig, frame)
        else:
            raise SystemExit(1)
