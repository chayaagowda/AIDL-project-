"""
Simple step-level logger that writes to stdout and optionally to a log file.
"""

import os
import time


class Logger:
    """Lightweight logger for training loops."""

    def __init__(self, log_dir: str, name: str, log_every_n_steps: int = 50):
        os.makedirs(log_dir, exist_ok=True)
        self.log_every_n_steps = log_every_n_steps
        self._step = 0
        self._epoch = 0
        log_path = os.path.join(log_dir, f"{name}.log")
        self._file = open(log_path, "a", buffering=1)  # line-buffered
        self._log(f"=== New run started at {time.strftime('%Y-%m-%d %H:%M:%S')} ===")

    # ------------------------------------------------------------------
    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch
        self._log(f"\n--- Epoch {epoch} ---")

    def log_step(self, loss: float, **kwargs) -> None:
        self._step += 1
        if self._step % self.log_every_n_steps == 0:
            extras = "  ".join(f"{k}={v:.4f}" for k, v in kwargs.items())
            msg = f"[Epoch {self._epoch} | Step {self._step}]  loss={loss:.4f}  {extras}"
            self._log(msg)

    def log_metrics(self, metrics: dict, split: str = "val") -> None:
        parts = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items() if k != "confusion_matrix")
        msg = f"[{split.upper()} @ Epoch {self._epoch}]  {parts}"
        self._log(msg)

    def close(self) -> None:
        self._file.close()

    # ------------------------------------------------------------------
    def _log(self, msg: str) -> None:
        print(msg)
        self._file.write(msg + "\n")
