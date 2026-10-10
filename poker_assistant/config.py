"""Chargement / sauvegarde de config.yaml avec valeurs par défaut."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ROOT / "config.yaml"

DEFAULTS: dict[str, Any] = {
    "claude": {
        "model": "claude-opus-5-5",
        "effort": "low",            # low = réponse rapide
        "max_tokens": 4000,
        "timeout": 30,              # secondes
        "max_retries": 2,           # retries automatiques du SDK (429, 5xx, réseau)
        "refusal_fallback": True,
        "max_image_width": 1600,
    },
    "capture": {
        "monitor": 1,               # index mss (1 = écran principal)
        "region": {"x": 0, "y": 0, "width": 1280, "height": 720},
        "interval": 0.5,
    },
    # Barre d'actions, en fractions (0-1) de la région de la table.
    "action_bar": {"x": 0.45, "y": 0.88, "width": 0.55, "height": 0.12},
    "detection": {
        "method": "hybrid",         # color | template | hybrid
        "red_ratio_min": 0.08,      # part minimale de pixels rouges dans la barre
        "min_buttons": 2,           # nb minimal de boutons rouges distincts
        "min_button_area": 0.01,    # aire min d'un bouton (fraction de la barre)
        "red_hue_low": 10,          # rouge = H <= low ou H >= high (OpenCV 0-179)
        "red_hue_high": 165,
        "red_sat_min": 100,
        "red_val_min": 80,
        "template_turn": "templates/turn_bar.png",
        "template_wait": "templates/wait_bar.png",
        "template_min_score": 0.35,
        "stable_frames": 2,         # frames consécutives pour valider un changement
        "fast_turn_end": True,      # une seule frame sans boutons suffit à finir le tour
    },
    "dedupe": {
        "change_ratio": 0.002,      # part de pixels changés = nouvelle situation
        "pixel_delta": 25,          # écart de gris pour qu'un pixel compte comme changé
        # Nouvelle décision sans que la barre disparaisse (adversaires en pré-action) :
        "recall_on_table_change": True,  # table changée puis stable -> nouvel appel
        "recall_on_layout": True,   # les boutons ont changé (CHECK/BET -> FOLD/CALL/RAISE)
        "recall_stable_frames": 2,  # frames stables exigées avant de rappeler
        "blink_frames": 2,          # barre absente <= N captures = même tour (pré-action)
        "max_calls_per_turn": 3,    # plafond d'appels pour un même tour continu
        "min_seconds_between_calls": 2.0,
        "max_retries_per_turn": 2,
        "retry_delay": 3.0,
    },
    "ui": {
        "enabled": True,
        "hotkey": "<ctrl>+<alt>+p",
        "geometry": "+20+20",
        "font_size": 12,
        "opacity": 0.93,
    },
    "logging": {"dir": "logs"},
}


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    data = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    return _merge(DEFAULTS, data)


def save_config(cfg: dict, path: str | Path | None = None) -> None:
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def resolve(path: str | Path) -> Path:
    """Chemins relatifs résolus depuis la racine du projet."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p
