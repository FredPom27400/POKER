"""Journal des décisions : capture PNG + réponse, dans logs/AAAA-MM-JJ/."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


class DecisionLogger:
    def __init__(self, directory: Path):
        self.dir = Path(directory)

    def save_image(self, table: np.ndarray, prefix: str) -> Path:
        """Capture seule (diagnostic), ex. logs/AAAA-MM-JJ/masquee_HHMMSS.png."""
        now = datetime.now()
        day = self.dir / now.strftime("%Y-%m-%d")
        day.mkdir(parents=True, exist_ok=True)
        img = day / f"{prefix}_{now.strftime('%H%M%S')}.png"
        cv2.imwrite(str(img), table)
        return img

    def log(self, table: np.ndarray, advice_text: str, ok: bool, model: str = "", error: str = "") -> Path:
        now = datetime.now()
        day = self.dir / now.strftime("%Y-%m-%d")
        day.mkdir(parents=True, exist_ok=True)
        stem = now.strftime("%H%M%S_%f")[:-3]
        img = day / f"{stem}.png"
        cv2.imwrite(str(img), table)
        entry = {"time": now.isoformat(timespec="seconds"), "image": img.name, "ok": ok,
                 "model": model, "response": advice_text, "error": error}
        (day / f"{stem}.txt").write_text(
            f"{entry['time']}\n{advice_text or 'ERREUR : ' + error}\n", encoding="utf-8")
        with open(day / "decisions.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return img
