#!/usr/bin/env python3
"""Calibration : sélection à la souris de la table et de la barre d'actions.

    python calibrate.py                    # capture l'écran 1 après 3 s
    python calibrate.py --monitor 2 --delay 5
    python calibrate.py --image capture.png  # depuis une capture plein écran existante
    python calibrate.py --template-only      # garde les régions, refait juste le template
    python calibrate.py --seats              # ajoute un thème : modèles du jeton D et du dos de carte

Dans chaque fenêtre : trace un rectangle, puis Entrée/Espace (c = annuler).
Le template sauvegardé dépend de l'état de la barre au moment de la capture :
mon tour (boutons rouges) -> templates/turn_bar.png,
présélection -> templates/wait_bar.png. Idéalement, calibre une fois dans chaque état.
"""
from __future__ import annotations

import argparse
import sys
import time

import cv2
import numpy as np

from poker_assistant.capture import grab_full_monitor
from poker_assistant.config import DEFAULT_CONFIG_PATH, load_config, resolve, save_config
from poker_assistant.detection import crop_bar, red_buttons

MAX_VIEW = (1600, 900)


def select_roi(title: str, img: np.ndarray) -> tuple[int, int, int, int]:
    """selectROI sur une version réduite si l'image dépasse l'écran ; renvoie des px d'origine."""
    h, w = img.shape[:2]
    scale = min(1.0, MAX_VIEW[0] / w, MAX_VIEW[1] / h)
    view = cv2.resize(img, (int(w * scale), int(h * scale))) if scale < 1 else img
    print(f"-> {title} (trace le rectangle puis Entrée)")
    x, y, rw, rh = cv2.selectROI(title, view, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(title)
    if rw == 0 or rh == 0:
        sys.exit("Sélection annulée.")
    return tuple(int(round(v / scale)) for v in (x, y, rw, rh))


def calibrate_seats(cfg: dict, table: np.ndarray) -> None:
    """Modèles du jeton D et du dos de carte (un par thème graphique, ils s'ajoutent)."""
    from poker_assistant.seats import SeatReader, find_players
    seats = cfg["seats"]
    players = find_players(table)
    print(f"Joueurs assis détectés sur cette capture : {len(players)}")
    x, y, rw, rh = select_roi("Jeton D - entoure le jeton du bouton (juste le rond)", table)
    seats["dealer_templates"] = _add(seats.get("dealer_templates"), "dealer", table[y:y + rh, x:x + rw])
    x, y, rw, rh = select_roi("Carte - entoure le symbole d'une carte cachee d'adversaire (sans le bord)", table)
    seats["cardback_templates"] = _add(seats.get("cardback_templates"), "cardback", table[y:y + rh, x:x + rw])
    if players:
        seats["ref_text_height"] = round(sorted(p[2] for p in players)[len(players) // 2], 1)
    seats["enabled"] = True
    facts = SeatReader(seats).read(table)
    print("Lecture sur cette capture : "
          + (facts.prompt() if facts else "impossible (jeton D ou joueurs introuvables), recommence"))


def _add(paths: list | None, kind: str, img: np.ndarray) -> list:
    """Enregistre un nouveau modèle sans écraser ceux des autres thèmes."""
    paths = list(paths or [])
    i = 1
    while f"templates/{kind}_{i}.png" in paths or resolve(f"templates/{kind}_{i}.png").exists():
        i += 1
    rel = f"templates/{kind}_{i}.png"
    _save(rel, img)
    return paths + [rel]


def _save(path: str, img: np.ndarray) -> None:
    p = resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p), img)


def bar_fractions(table_shape, rect) -> dict:
    h, w = table_shape[:2]
    x, y, bw, bh = rect
    return {"x": round(x / w, 4), "y": round(y / h, 4), "width": round(bw / w, 4), "height": round(bh / h, 4)}


def save_template(cfg: dict, bar: np.ndarray, label: str) -> str:
    key = "template_turn" if label == "turn" else "template_wait"
    path = resolve(cfg["detection"][key])
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), bar)
    return str(path)


def ask_label(ratio: float, n: int, cfg: dict) -> str:
    guess = "turn" if ratio >= cfg["detection"]["red_ratio_min"] and n >= cfg["detection"]["min_buttons"] else "wait"
    ans = input(f"Est-ce ton tour sur cette capture (boutons rouges) ? [o/n] (détecté : "
                f"{'oui' if guess == 'turn' else 'non'}) ").strip().lower()
    if not ans:
        return guess
    return "turn" if ans.startswith(("o", "y")) else "wait"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    ap.add_argument("--monitor", type=int, help="écran à capturer (1 = principal)")
    ap.add_argument("--delay", type=float, default=3.0, help="secondes avant la capture")
    ap.add_argument("--image", help="calibrer depuis une capture plein écran")
    ap.add_argument("--template-only", action="store_true", help="réutilise les régions existantes")
    ap.add_argument("--label", choices=["turn", "wait", "ask"], default="ask")
    ap.add_argument("--seats", action="store_true",
                    help="ajoute les modèles du jeton D et du dos de carte (un par thème de table)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.monitor is not None:
        cfg["capture"]["monitor"] = args.monitor

    if args.image:
        screen = cv2.imread(args.image)
        if screen is None:
            sys.exit(f"Image illisible : {args.image}")
    else:
        print(f"Capture de l'écran {cfg['capture']['monitor']} dans {args.delay:.0f} s : affiche ta table…")
        time.sleep(args.delay)
        screen = grab_full_monitor(cfg["capture"]["monitor"])

    if args.template_only or args.seats:
        r = cfg["capture"]["region"]
        table = screen[r["y"]:r["y"] + r["height"], r["x"]:r["x"] + r["width"]]
    else:
        x, y, w, h = select_roi("1/2 - Selectionne la TABLE", screen)
        cfg["capture"]["region"] = {"x": x, "y": y, "width": w, "height": h}
        table = screen[y:y + h, x:x + w]
        rect = select_roi("2/2 - Selectionne la BARRE D'ACTIONS", table)
        cfg["action_bar"] = bar_fractions(table.shape, rect)

    if args.seats:
        calibrate_seats(cfg, table)
        save_config(cfg, args.config)
        print(f"Configuration enregistrée dans {args.config}")
        return

    bar = crop_bar(table, cfg["action_bar"])
    ratio, boxes = red_buttons(bar, cfg["detection"])
    print(f"Barre : {bar.shape[1]}x{bar.shape[0]} px, pixels rouges {ratio:.1%}, boutons rouges {len(boxes)}")

    ref = resolve(cfg["detection"]["table_ref"])
    ref.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(ref), table)   # sert à repérer une table masquée par une autre fenêtre
    label = args.label if args.label != "ask" else ask_label(ratio, len(boxes), cfg)
    path = save_template(cfg, bar, label)
    print(f"Template '{label}' enregistré : {path}")
    if label == "turn" and ratio < cfg["detection"]["red_ratio_min"]:
        print("Attention : peu de rouge détecté alors que c'est ton tour. "
              "Vérifie la sélection de la barre ou baisse detection.red_ratio_min.")

    if not args.template_only and input("Enregistrer le jeton D et le dos de carte (position et adversaires lus sans IA) ? [O/n] ").strip().lower() in ("", "o", "oui", "y"):
        calibrate_seats(cfg, table)

    save_config(cfg, args.config)
    print(f"Configuration enregistrée dans {args.config}")


if __name__ == "__main__":
    main()
