#!/usr/bin/env python3
"""Teste la détection du tour sur des captures de référence (par défaut ./samples/).

    python test_detection.py                 # toutes les images de samples/
    python test_detection.py samples/x.png --debug

Les images peuvent être des captures plein écran (la région de la table du
config.yaml est alors découpée) ou des captures de la table seule.
Si le nom de fichier contient un indice, le résultat est vérifié :
  tour  : « turn », « tour », « montour », « action »
  attente : « wait », « attente », « presel », « preselection »
Avec --debug, une image annotée est écrite dans samples/_debug/.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from poker_assistant.config import DEFAULT_CONFIG_PATH, load_config
from poker_assistant.detection import TurnDetector, annotate

TURN_HINTS = ("turn", "tour", "action")
WAIT_HINTS = ("wait", "attente", "presel")
EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def expected_from_name(name: str) -> bool | None:
    n = name.lower()
    if any(h in n for h in WAIT_HINTS):
        return False
    if any(h in n for h in TURN_HINTS):
        return True
    return None


def to_table(img, region: dict):
    """Découpe la table si l'image est une capture plein écran qui la contient."""
    h, w = img.shape[:2]
    x, y, rw, rh = region["x"], region["y"], region["width"], region["height"]
    if (w, h) != (rw, rh) and x + rw <= w and y + rh <= h and (w > rw or h > rh):
        return img[y:y + rh, x:x + rw]
    return img


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", default=["samples"])
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    ap.add_argument("--table-only", action="store_true", help="ne pas découper : images = table seule")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    det = TurnDetector(cfg["detection"], cfg["action_bar"])
    files = []
    for p in map(Path, args.paths):
        files += sorted(f for f in p.rglob("*") if f.suffix.lower() in EXTS and "_debug" not in f.parts) \
            if p.is_dir() else [p]
    if not files:
        print("Aucune image trouvée. Dépose tes captures dans samples/ "
              "(ex. montour_01.png, attente_01.png).")
        return 1

    checked = errors = 0
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            print(f"[illisible] {f}")
            continue
        table = img if args.table_only else to_table(img, cfg["capture"]["region"])
        d = det.detect(table)
        exp = expected_from_name(f.name)
        status = ""
        if exp is not None:
            checked += 1
            if exp != d.is_turn:
                errors += 1
                status = "  <-- ERREUR"
            else:
                status = "  ok"
        print(f"{f.name:40s} {d.summary()}{status}")
        if args.debug:
            out = f.parent / "_debug"
            out.mkdir(exist_ok=True)
            cv2.imwrite(str(out / f.name), annotate(table, cfg["action_bar"], d))

    if checked:
        print(f"\n{checked - errors}/{checked} correctes")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
