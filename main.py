#!/usr/bin/env python3
"""Assistant poker : analyse la table quand c'est ton tour.

    python main.py                 # fenêtre + console
    python main.py --no-ui         # console seule
    python main.py --dry-run       # détection seule, sans appel API
    python main.py --image x.png   # envoie une capture de table à Claude et quitte
"""
from __future__ import annotations

import argparse
import os
import queue
import sys
import threading

import cv2

from poker_assistant.config import DEFAULT_CONFIG_PATH, load_config, resolve
from poker_assistant.engine import Engine, console
from poker_assistant.hotkeys import start_hotkey
from poker_assistant.journal import DecisionLogger


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    ap.add_argument("--no-ui", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="pas d'appel API")
    ap.add_argument("--image", help="analyser une capture et quitter")
    args = ap.parse_args()

    cfg = load_config(args.config)
    advisor = None
    if not args.dry_run:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("ANTHROPIC_API_KEY n'est pas définie.", file=sys.stderr)
            return 1
        from poker_assistant.claude_client import ClaudeAdvisor
        advisor = ClaudeAdvisor(cfg["claude"])
    logger = DecisionLogger(resolve(cfg["logging"]["dir"]))

    if args.image:
        table = cv2.imread(args.image)
        if table is None:
            print(f"Image illisible : {args.image}", file=sys.stderr)
            return 1
        if advisor is None:
            return 0
        a = advisor.ask(table)
        logger.log(table, a.text, a.ok, a.model, a.error)
        print(a.text if a.ok else f"ERREUR : {a.error}")
        return 0 if a.ok else 1

    from poker_assistant.capture import ScreenCapturer
    cap = ScreenCapturer(cfg["capture"]["monitor"], cfg["capture"]["region"])
    use_ui = cfg["ui"]["enabled"] and not args.no_ui
    events: queue.Queue = queue.Queue()
    engine = Engine(cfg, cap.grab, advisor.ask if advisor else None, logger, events if use_ui else None)
    hotkey = cfg["ui"]["hotkey"]
    start_hotkey(hotkey, engine.toggle_pause)
    console(f"Surveillance de {cap.bbox} toutes les {cfg['capture']['interval']} s "
            f"— modèle {cfg['claude']['model']} — pause : {hotkey}")

    try:
        if use_ui:
            try:
                from poker_assistant.ui import AdviceWindow
            except ImportError:
                console("tkinter indisponible : mode console.")
                use_ui = False
                engine.events = None
        if use_ui:
            worker = threading.Thread(target=engine.run, daemon=True)
            worker.start()
            AdviceWindow(cfg["ui"], events, engine.toggle_pause, engine.stop).run()
            engine.stop()
            worker.join(timeout=2)
        else:
            engine.run()
    except KeyboardInterrupt:
        engine.stop()
    finally:
        cap.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
