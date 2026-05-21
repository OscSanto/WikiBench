"""System metrics for wikibench — RAM, CPU, Raspberry Pi temperature."""
from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass, field


@dataclass
class SystemSnapshot:
    ram_mb:  float = 0.0
    cpu_pct: float = 0.0
    temp_c:  float | None = None


def _ram_mb() -> float:
    import psutil
    return psutil.virtual_memory().used / (1024 * 1024)


def _pi_temp() -> float | None:
    try:
        out = subprocess.check_output(
            ["vcgencmd", "measure_temp"], text=True, timeout=2
        )
        # "temp=42.8'C"
        return float(out.strip().removeprefix("temp=").removesuffix("'C"))
    except Exception:
        return None


def snapshot_before() -> SystemSnapshot:
    import psutil
    psutil.cpu_percent(interval=None)  # prime the counter; first call is always 0
    return SystemSnapshot(ram_mb=_ram_mb(), cpu_pct=0.0, temp_c=_pi_temp())


def snapshot_after(before: SystemSnapshot) -> SystemSnapshot:
    import psutil
    return SystemSnapshot(
        ram_mb  = _ram_mb(),
        cpu_pct = psutil.cpu_percent(interval=None),
        temp_c  = _pi_temp(),
    )


# ── Background CPU monitor (averages over the LLM call) ───────────────────────

class CpuMonitor:
    """Poll CPU usage in a background thread; call .stop() to get the mean."""

    def __init__(self, interval: float = 0.5):
        self._interval = interval
        self._samples: list[float] = []
        self._stop_evt = threading.Event()
        self._thread   = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        import psutil
        while not self._stop_evt.wait(self._interval):
            self._samples.append(psutil.cpu_percent(interval=None))

    def stop(self) -> float | None:
        self._stop_evt.set()
        self._thread.join(timeout=5)
        return sum(self._samples) / len(self._samples) if self._samples else None
