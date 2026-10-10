"""Capture d'écran d'une région avec mss."""
from __future__ import annotations

import numpy as np


class ScreenCapturer:
    def __init__(self, monitor: int, region: dict):
        import mss  # import tardif : permet d'utiliser le reste sans écran

        self._sct = mss.mss()
        monitors = self._sct.monitors
        if monitor < 0 or monitor >= len(monitors):
            raise ValueError(f"Écran {monitor} inexistant (disponibles : 1..{len(monitors) - 1})")
        mon = monitors[monitor]
        self.bbox = {
            "left": mon["left"] + int(region["x"]),
            "top": mon["top"] + int(region["y"]),
            "width": int(region["width"]),
            "height": int(region["height"]),
        }

    def grab(self) -> np.ndarray:
        """Renvoie la région en BGR (uint8)."""
        shot = self._sct.grab(self.bbox)
        return np.ascontiguousarray(np.asarray(shot)[:, :, :3])

    def close(self) -> None:
        self._sct.close()


def grab_full_monitor(monitor: int) -> np.ndarray:
    import mss

    with mss.mss() as sct:
        shot = sct.grab(sct.monitors[monitor])
        return np.ascontiguousarray(np.asarray(shot)[:, :, :3])
