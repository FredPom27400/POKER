#!/usr/bin/env python3
"""Compare vitesse et conseils de plusieurs réglages sur tes vraies captures.

    python benchmark.py                       # les 3 dernières captures de logs/
    python benchmark.py logs/2026-10-10/205407_123.png autre.png
    python benchmark.py --last 5

Chaque capture est envoyée à chaque réglage ci-dessous : ça coûte quelques
centimes (à peu près le prix d'une main par réglage et par capture).
Compare ensuite les temps ET les conseils avant de changer config.yaml.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2

from poker_assistant import pretty
from poker_assistant.claude_client import ClaudeAdvisor
from poker_assistant.config import DEFAULT_CONFIG_PATH, load_config, resolve

CANDIDATES = [
    ("Opus 5.5 rapide", {"model": "claude-opus-5-5", "speed": "fast", "thinking": "auto", "effort": "low"}),
    ("Opus 5.5 normal", {"model": "claude-opus-5-5", "speed": "normal", "thinking": "auto", "effort": "low"}),
    ("Sonnet 5.5 réflexion", {"model": "claude-sonnet-5-5", "speed": "normal", "thinking": "auto", "effort": "low"}),
    ("Sonnet 5.5 sans réflexion", {"model": "claude-sonnet-5-5", "speed": "normal", "thinking": "off", "effort": "low"}),
    ("Haiku 5.5 sans réflexion", {"model": "claude-haiku-5-5", "speed": "normal", "thinking": "off", "effort": "low"}),
]


def last_captures(n: int) -> list[Path]:
    logs = resolve(load_config()["logging"]["dir"])
    files = sorted(logs.rglob("*.png"), key=lambda p: p.stat().st_mtime)
    return files[-n:]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="*")
    ap.add_argument("--last", type=int, default=3, help="nb de captures récentes de logs/ (défaut 3)")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY n'est pas définie.", file=sys.stderr)
        return 1
    images = [Path(p) for p in args.images] or last_captures(args.last)
    if not images:
        print("Aucune capture : joue quelques mains avec main.py (elles sont enregistrées dans logs/) "
              "ou donne des images en argument.", file=sys.stderr)
        return 1

    base = load_config(args.config)["claude"]
    advisors = [(name, ClaudeAdvisor(dict(base, **opts))) for name, opts in CANDIDATES]
    results: dict[str, list] = {name: [] for name, _ in CANDIDATES}

    for img_path in images:
        table = cv2.imread(str(img_path))
        if table is None:
            pretty.warn(f"Image illisible : {img_path}")
            continue
        print()
        print(pretty._rule(pretty.c(f"Capture {img_path.name}", pretty.BOLD, pretty.CYAN), pretty.CYAN, "═"))
        for name, adv in advisors:
            a = adv.ask(table)
            results[name].append(a)
            if a.ok:
                action, reason, _ = pretty.parse_advice(a.text)
                print(f"  {name:28s} {a.seconds:5.1f} s   "
                      + pretty.c(action or "?", *pretty.action_color(action)))
                print(pretty.c(f"  {'':28s}          {reason[:160]}", pretty.GREY))
            else:
                print(f"  {name:28s} " + pretty.c(f"ERREUR : {a.error}", pretty.RED))

    print()
    print(pretty._rule(pretty.c("Résumé (temps moyen)", pretty.BOLD), pretty.CYAN, "═"))
    for name, res in results.items():
        ok = [a for a in res if a.ok]
        if not ok:
            print(f"  {name:28s} aucune réponse")
            continue
        avg = sum(a.seconds for a in ok) / len(ok)
        think = sum(a.first_text for a in ok) / len(ok)
        toks = sum(a.out_tokens for a in ok) // len(ok)
        actions = " | ".join(pretty.parse_advice(a.text)[0] or "?" for a in ok)
        print(f"  {name:28s} {avg:5.1f} s  (1er mot {think:4.1f} s, {toks:4d} tokens)   {actions}")
    print()
    print("Pour adopter un réglage, recopie ses valeurs (model, speed, thinking) dans la section "
          "claude: de config.yaml.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
