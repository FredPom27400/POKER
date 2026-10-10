"""Raccourci clavier global (pynput) pour pause / reprise."""
from __future__ import annotations

from typing import Callable


def start_hotkey(combo: str, callback: Callable[[], object]):
    """Démarre l'écoute ; renvoie le listener ou None si indisponible
    (pynput absent, Wayland, permissions d'accessibilité macOS…)."""
    try:
        from pynput import keyboard

        listener = keyboard.GlobalHotKeys({combo: callback})
        listener.daemon = True
        listener.start()
        return listener
    except Exception as e:  # noqa: BLE001 - toute erreur = raccourci désactivé
        print(f"[raccourci] {combo} indisponible ({e}). Utilise le bouton Pause de la fenêtre.")
        return None
