"""Affichage console lisible : couleurs, séparateurs, conseil mis en valeur."""
from __future__ import annotations

import os
import re
import shutil
import sys
import textwrap
from datetime import datetime

try:  # couleurs ANSI sous Windows (invite de commandes classique)
    import colorama

    colorama.just_fix_windows_console()
except Exception:  # noqa: BLE001
    if os.name == "nt":
        os.system("")  # active le mode VT sur Windows 10+

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors="replace")  # pas de plantage si un caractère manque
    except Exception:  # noqa: BLE001
        pass

_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None

RESET, BOLD, DIM = "\033[0m", "\033[1m", "\033[2m"
RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, GREY = (
    "\033[91m", "\033[92m", "\033[93m", "\033[94m", "\033[95m", "\033[96m", "\033[90m")


def c(text: str, *styles: str) -> str:
    return f"{''.join(styles)}{text}{RESET}" if _COLOR and styles else text


def _width() -> int:
    return max(50, min(100, shutil.get_terminal_size((90, 20)).columns - 1))


def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _out(text: str = "") -> None:
    print(text, flush=True)


def _rule(label: str = "", color: str = GREY, char: str = "─") -> str:
    w = _width()
    if not label:
        return c(char * w, color)
    return c(char * 3 + " ", color) + label + " " + c(char * max(3, w - 5 - _visible_len(label)), color)


_ANSI = re.compile(r"\033\[[0-9;]*m")


def _visible_len(s: str) -> int:
    return len(_ANSI.sub("", s))


def _labelled(label: str, text: str, label_color: str, text_color: str = "") -> str:
    """« LABEL  texte » avec retour à la ligne aligné sous le texte."""
    lines = textwrap.wrap(text, width=_width() - 10) or [""]
    body = [c(l, text_color) if text_color else l for l in lines]
    first = f"  {c(label.ljust(6)[:6], label_color)}  {body[0]}"
    return "\n".join([first] + [" " * 10 + l for l in body[1:]])


def action_color(action: str) -> tuple[str, ...]:
    a = action.lower()
    if "all" in a and "in" in a or "tapis" in a or "shove" in a:
        return (BOLD, MAGENTA)
    if a.startswith(("raise", "relance", "bet", "mise")):
        return (BOLD, GREEN)
    if a.startswith(("call", "suivre", "paye")):
        return (BOLD, YELLOW)
    if a.startswith("check"):
        return (BOLD, CYAN)
    if a.startswith(("fold", "couch")):
        return (BOLD, RED)
    return (BOLD,)


HEX = {RED: "#ff6b6b", GREEN: "#5ee07a", YELLOW: "#ffd25e", CYAN: "#5ed8ff", MAGENTA: "#e07aff"}


def action_hex(action: str) -> str:
    """Couleur de l'action pour la fenêtre tkinter."""
    return next((HEX[s] for s in action_color(action) if s in HEX), "white")


def parse_fields(text: str) -> dict:
    """Sépare LECTURE, ACTION, RAISON et les remarques (incertitudes, notes…)."""
    f = {"sieges": "", "lecture": "", "action": "", "keys": "", "reason": "", "notes": []}
    for raw in text.splitlines():
        line = raw.strip().lstrip("-•* ").strip()
        if not line:
            continue
        m = re.match(r"(SI[EÈ]GES|LECTURE|ACTION|TOUCHES?|RAISON|REASON|NOTE|REMARQUE|INCERTAIN\w*)\s*:\s*(.*)", line, re.I)
        key = m.group(1).upper() if m else ""
        if key in ("SIEGES", "SIÈGES") and not f["sieges"]:
            f["sieges"] = m.group(2).strip()
        elif key == "LECTURE" and not f["lecture"]:
            f["lecture"] = m.group(2).strip()
        elif key == "ACTION" and not f["action"]:
            f["action"] = m.group(2).strip()
        elif key.startswith("TOUCHE") and not f["keys"]:
            f["keys"] = m.group(2).strip()
        elif key in ("RAISON", "REASON") and not f["reason"]:
            f["reason"] = m.group(2).strip()
        else:
            f["notes"].append(m.group(2).strip() if key in ("NOTE", "REMARQUE") or key.startswith("INCERTAIN")
                              else line)
    return f


def lecture_summary(lecture: str) -> str:
    """« K♦ 9♠ | préflop, pas de board | BTN | 1 adversaire (X) | 7,5 BB »
    -> « K♦ 9♠  ·  préflop  ·  BTN  ·  1 adversaire (X)  ·  à payer 7,5 BB »."""
    parts = [p.strip() for p in lecture.split("|")]
    if len(parts) >= 2 and re.search(r"pas de board|préflop|preflop|^-$|^aucun", parts[1], re.I):
        parts[1] = "préflop"
    if len(parts) >= 4 and re.fullmatch(r"\d+", parts[3]):
        parts[3] = f"{parts[3]} adv."
    if len(parts) >= 5:
        pay = parts[4]
        parts[4] = "rien à payer" if re.fullmatch(r"0(\s*BB)?|0 \(.*\)", pay) else f"à payer {pay}"
    return "  ·  ".join(p for p in parts if p)


def parse_advice(text: str) -> tuple[str, str, list[str]]:
    f = parse_fields(text)
    return f["action"], f["reason"], f["notes"]


# --- messages ---------------------------------------------------------------

def banner(region: dict, interval: float, model: str, speed: str, hotkey: str) -> None:
    _out()
    _out(_rule(c("POKER ASSISTANT", BOLD, CYAN), CYAN, "═"))
    _out(f"  {c('Zone', GREY)}     {region['width']}x{region['height']} à ({region['left']}, {region['top']})"
         f"   {c('Capture', GREY)} toutes les {interval} s")
    _out(f"  {c('Modèle', GREY)}   {model}{c('  (mode rapide)', GREEN) if speed == 'fast' else ''}")
    _out(f"  {c('Pause', GREY)}    {hotkey}      {c('Quitter', GREY)}  Ctrl+C ou fermer la fenêtre")
    _out(c("═" * _width(), CYAN))
    _out(c(f"[{_now()}] En attente de ton tour…", GREY))


def turn(reason: str, detail: str) -> None:
    _out()
    color = CYAN if reason == "nouveau tour" else YELLOW
    _out(_rule(c(f"{_now()}  ▶ MON TOUR", BOLD, color) + c(f"  · {reason}", color), color))
    _out(c(f"  {detail}", GREY))
    _out(c("  Analyse en cours…", GREY))


def advice(text: str, timing: str = "", model: str = "") -> None:
    f = parse_fields(text)
    action, reason, notes = f["action"], f["reason"], f["notes"]
    _out()
    if f["sieges"]:
        _out(_labelled("TABLE", f["sieges"], GREY, GREY))
    if f["lecture"]:
        _out(_labelled("LU", f["lecture"].replace("|", "·"), GREY, GREY))
    if action:
        keys = c(f"   ⌨ {f['keys']}", BOLD, CYAN) if f["keys"] else ""
        _out(f"  {c('ACTION', GREY)}  " + c(f"▌{action.upper()}", *action_color(action)) + keys)
    if reason:
        _out(_labelled("RAISON", reason, GREY))
    for n in notes:
        _out(_labelled("NOTE", n, YELLOW, YELLOW))
    if not action and not reason and not notes:
        _out(text)
    meta = [m for m in (timing, model) if m]
    if meta:
        _out(c(f"  temps {' · '.join(meta)}", GREY))
    _out(_rule())


def error(msg: str, retry: bool = False) -> None:
    _out()
    _out(c(f"  × ERREUR  {msg}", BOLD, RED) + (c("  (nouvel essai prévu)", YELLOW) if retry else ""))
    _out(_rule(color=RED))


def paused(state: bool) -> None:
    _out()
    if state:
        _out(c(f"[{_now()}] ‖ PAUSE", BOLD, YELLOW) + c("  (même raccourci pour reprendre)", GREY))
    else:
        _out(c(f"[{_now()}] ▶ REPRISE", BOLD, GREEN))


def info(msg: str) -> None:
    _out(c(f"[{_now()}] {msg}", GREY))


def warn(msg: str) -> None:
    _out(c(f"[{_now()}] ⚠ {msg}", YELLOW))
