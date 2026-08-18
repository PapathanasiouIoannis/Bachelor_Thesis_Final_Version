"""Process-isolated log-path and merge helpers for pair experiments."""

from __future__ import annotations

import os
from pathlib import Path


def worker_log_path(run_log_path: Path) -> Path:
    """Return a process-isolated temporary log path for one worker."""

    return run_log_path.with_name(
        f"{run_log_path.stem}.worker-{os.getpid()}{run_log_path.suffix}"
    )


def merge_worker_logs(run_log_path: Path) -> None:
    """Merge process-isolated worker logs into the run's canonical log."""

    pattern = f"{run_log_path.stem}.worker-*{run_log_path.suffix}"
    worker_logs = sorted(run_log_path.parent.glob(pattern))
    if not worker_logs:
        return
    run_log_path.parent.mkdir(parents=True, exist_ok=True)
    with run_log_path.open("a", encoding="utf-8") as destination:
        for worker_log in worker_logs:
            destination.write(worker_log.read_text(encoding="utf-8"))
    for worker_log in worker_logs:
        try:
            worker_log.unlink()
        except PermissionError:
            # An unexpectedly failed worker may still own its handler. The log
            # has already been merged and remains inside this isolated run.
            pass


__all__ = ["merge_worker_logs", "worker_log_path"]
