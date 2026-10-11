#!/usr/bin/env python3
"""Mesure la qualité de lecture de la table sur des captures annotées.

    python eval_lecture.py                    # eval/cases.yaml, réglages de config.yaml
    python eval_lecture.py --model claude-opus-5-5 --effort medium

Pour chaque capture, compare la ligne LECTURE de Claude à la vérité terrain :
position, nombre d'adversaires encore en jeu, montant à payer. Affiche un score
par critère ; à relancer après chaque changement de prompt ou de modèle pour
vérifier que la lecture s'améliore. Coût : un appel API par capture.
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import cv2
import yaml

from poker_assistant import pretty
from poker_assistant.claude_client import ClaudeAdvisor
from poker_assistant.config import DEFAULT_CONFIG_PATH, load_config, resolve

POSITIONS = [("BTN", r"\bBTN\b|bouton|button"), ("SB", r"\bSB\b|petite blinde|small blind"),
             ("BB", r"\bBB\b(?!\s*\d)|grosse blinde|big blind"), ("CO", r"\bCO\b|cut ?off"),
             ("HJ", r"\bHJ\b|hijack"), ("UTG", r"\bUTG\b|under the gun"), ("MP", r"\bMP\b|middle")]


def parse_position(text: str) -> str:
    """Première position citée (« BB (bouton chez X) » -> BB)."""
    found = [(m.start(), name) for name, pat in POSITIONS if (m := re.search(pat, text, re.I))]
    return min(found)[1] if found else "?"


def first_number(text: str) -> float | None:
    m = re.search(r"\d+(?:[.,]\d+)?", text)
    return float(m.group().replace(",", ".")) if m else None


def parse_lecture(lecture: str) -> dict:
    parts = [p.strip() for p in lecture.split("|")]
    while len(parts) < 5:
        parts.append("")
    to_call = 0.0 if re.search(r"\b(rien|aucun|0)\b|check", parts[4], re.I) and first_number(parts[4]) in (None, 0) \
        else first_number(parts[4])
    return {"position": parse_position(parts[2]), "adversaires": first_number(parts[3]), "a_payer": to_call}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", default="eval/cases.yaml")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    ap.add_argument("--model")
    ap.add_argument("--effort")
    args = ap.parse_args()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY n'est pas définie.", file=sys.stderr)
        return 1

    with open(resolve(args.cases), encoding="utf-8") as f:
        spec = yaml.safe_load(f)
    claude = load_config(args.config)["claude"]
    if args.model:
        claude["model"] = args.model
    if args.effort:
        claude["effort"] = args.effort
    adv = ClaudeAdvisor(claude)
    folder = resolve(spec["dossier"])
    score = {"position": 0, "adversaires": 0, "a_payer": 0}
    n = 0
    print(f"Modèle {claude['model']} · effort {claude.get('effort')} · {len(spec['cas'])} captures\n")
    for case in spec["cas"]:
        img = cv2.imread(str(folder / f"{case['fichier']}.png"))
        if img is None:
            pretty.warn(f"Capture absente : {folder / case['fichier']}.png")
            continue
        a = adv.ask(img)
        if not a.ok:
            print(f"  {case['fichier']}  " + pretty.c(f"ERREUR {a.error}", pretty.RED))
            continue
        n += 1
        got = parse_lecture(pretty.parse_fields(a.text)["lecture"])
        ok = {"position": got["position"] in case["position"],
              "adversaires": got["adversaires"] == case["adversaires"],
              "a_payer": got["a_payer"] is not None and abs(got["a_payer"] - case["a_payer"]) < 0.01}
        for k, v in ok.items():
            score[k] += v
        cells = []
        for k, label in (("position", "pos"), ("adversaires", "adv"), ("a_payer", "payer")):
            want = "/".join(case[k]) if k == "position" else case[k]
            cell = f"{label} {got[k]}" + ("" if ok[k] else f" (attendu {want})")
            cells.append(pretty.c(cell, pretty.GREEN if ok[k] else pretty.RED))
        print(f"  {case['fichier']}  {case['cartes']:7s}  " + "  ".join(cells)
              + pretty.c(f"   → {pretty.parse_fields(a.text)['action']} ({a.seconds:.1f} s)", pretty.GREY))
    if n:
        print("\nScore : " + " · ".join(f"{k} {v}/{n}" for k, v in score.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
