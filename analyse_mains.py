"""Résumé texte des mains jouées, à partir de logs/ (sans les images).

Regroupe les conseils par main (mêmes cartes, moins de 3 min d'écart), repère les
mains premium (TT+, AK, AQs) et compte les tailles de mise conseillées après le flop.
Le rapport est écrit dans logs/rapport_mains.txt : petit, facile à envoyer.

    python analyse_mains.py            # ou double-clic sur analyser_mains.bat
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

from poker_assistant.config import load_config, resolve
from poker_assistant.pretty import parse_fields

RANKS = "23456789TJQKA"
CARD = re.compile(r"(10|[2-9TJQKA])\s*([♠♥♦♣shdc])", re.I)


def hole_cards(lecture: str) -> tuple[str, str] | None:
    cards = CARD.findall(lecture.split("|")[0])
    if len(cards) < 2:
        return None
    return tuple((("T" if r == "10" else r.upper()), s) for r, s in cards[:2])  # type: ignore[return-value]


def hand_name(cards) -> str:
    (r1, s1), (r2, s2) = sorted(cards, key=lambda c: -RANKS.index(c[0]))
    if r1 == r2:
        return r1 + r2
    return r1 + r2 + ("s" if s1 == s2 else "o")


def is_premium(name: str) -> bool:
    if len(name) == 2:
        return RANKS.index(name[0]) >= RANKS.index("T")
    return name[:2] == "AK" or name == "AQs"


def street(lecture: str) -> str:
    parts = [p.strip() for p in lecture.split("|")]
    if len(parts) < 2 or re.search(r"pas de board|préflop|preflop|^-$|^aucun", parts[1], re.I):
        return "préflop"
    n = len(CARD.findall(parts[1]))
    return {3: "flop", 4: "turn", 5: "river"}.get(n, "?")


def load(log_dir: Path) -> list[dict]:
    rows = []
    for f in sorted(log_dir.glob("*/decisions.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not e.get("ok"):
                continue
            fields = parse_fields(e.get("response", ""))
            cards = hole_cards(fields["lecture"])
            if cards:
                rows.append({"t": datetime.fromisoformat(e["time"]), "cards": cards, **fields})
    return rows


def group_hands(rows: list[dict]) -> list[list[dict]]:
    hands: list[list[dict]] = []
    for r in rows:
        last = hands[-1][-1] if hands else None
        if last and set(last["cards"]) == set(r["cards"]) and (r["t"] - last["t"]).total_seconds() < 180:
            hands[-1].append(r)
        else:
            hands.append([r])
    return hands


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", default=None, help="dossier des logs (défaut : celui de config.yaml)")
    args = ap.parse_args()
    log_dir = resolve(args.logs or load_config()["logging"]["dir"])
    hands = group_hands(load(log_dir))
    if not hands:
        print(f"Aucune décision trouvée dans {log_dir}")
        return 1

    out = [f"{len(hands)} mains analysées ({hands[0][0]['t']:%d/%m %H:%M} → {hands[-1][-1]['t']:%d/%m %H:%M})", ""]
    premium = [h for h in hands if is_premium(hand_name(h[0]["cards"]))]
    def bets_after_flop(h):
        return any(street(r["lecture"]) != "préflop" and re.match(r"(bet|raise|mise|relance)", r["action"], re.I)
                   for r in h)

    others = [h for h in hands if h not in premium and bets_after_flop(h)]
    out.append(f"MAINS PREMIUM (TT+, AK, AQs) : {len(premium)}   ·   autres mains avec mise après le flop : {len(others)}")
    for h in premium + others:
        out.append(f"\n{h[0]['t']:%d/%m %H:%M}  {hand_name(h[0]['cards'])}")
        for r in h:
            parts = [p.strip() for p in r["lecture"].split("|")]
            board = parts[1] if street(r["lecture"]) != "préflop" and len(parts) > 1 else ""
            extra = " | ".join(p for p in parts[2:5])
            out.append(f"  {street(r['lecture']):7} {board[:28]:28} {r['action'][:28]:28} {r['keys'][:16]:16} {extra[:70]}")

    sizes = Counter()
    for h in hands:
        for r in h:
            if street(r["lecture"]) != "préflop" and re.match(r"(bet|raise|mise|relance)", r["action"], re.I):
                k = r["keys"].split()[0] if r["keys"] else "?"
                sizes[k] += 1
    out.append("\nMISES ET RELANCES APRÈS LE FLOP, PAR TOUCHE (F4 = 35 %, F5 = 65 %, F6 = Pot, F7 = tapis) :")
    out += [f"  {k:8} {n}" for k, n in sizes.most_common()] or ["  aucune"]

    report = "\n".join(out)
    path = log_dir / "rapport_mains.txt"
    path.write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"\nRapport enregistré : {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
