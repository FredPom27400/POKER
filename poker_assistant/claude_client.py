"""Appel à l'API Anthropic avec la capture de la table."""
from __future__ import annotations

import base64
import time
from dataclasses import dataclass
from typing import Callable

import anthropic
import cv2
import numpy as np

SYSTEM_PROMPT = """Tu es un joueur professionnel de No Limit Hold'em cash game, spécialiste des tables à 6-9 joueurs et des stacks courts à moyens. À partir de la capture, lis : mes cartes, le board, le pot, les stacks (en BB), les mises en cours, la position du bouton, le nombre d'adversaires restants. Puis donne la meilleure action.

Comment lire la table sur ce client (ces règles sont fiables : applique-les sans les présenter comme des incertitudes) :
- Moi = le joueur en bas au centre, le seul dont les cartes sont visibles.
- Un adversaire est encore dans le coup seulement s'il a ses deux cartes dorées marquées « W » au-dessus de son nom. Sans ces cartes, il a passé, même sans étiquette FOLD et même si des jetons sont encore devant lui : ne le compte pas et ne le dis pas « illisible ».
- L'étiquette sous un stack (CHECK, CALL, BETS, RAISES TO, FOLD, SMALL BLIND, BIG BLIND…) est sa dernière action.
- Les jetons accompagnés d'un montant « X BB » devant un joueur sont sa mise dans le tour d'enchères en cours.
- « Pot total » = pot + toutes les mises en cours (le mien compris) ; « Pot » seul (après le flop) ne les inclut pas.
- Le montant à payer est écrit sur mon bouton « CALL X BB » ; « CHECK » = rien à payer ; « BET » / « RAISE TO » = montant minimal.

Positions, méthode obligatoire :
1. Le jeton « D » est posé sur le tapis juste devant son propriétaire, entre sa plaque et le centre de la table. S'il est juste au-dessus de mes cartes, c'est moi le bouton.
2. Sens du jeu (horaire à l'écran) : moi (en bas) → à gauche → en haut à gauche → en haut à droite → à droite → moi. Les sièges vides ne comptent pas.
3. Dans ce sens, le joueur qui suit le bouton est la petite blinde (SB), le suivant la grosse blinde (BB). Préflop, vérifie avec les mises 0,5 BB / 1 BB et les étiquettes SMALL BLIND / BIG BLIND : en cas de conflit, ce sont elles qui comptent.
4. Ma position : BTN, SB, BB, sinon en remontant depuis le bouton : CO (juste avant le bouton), HJ, MP… ; le premier à parler après la grosse blinde est UTG.
5. Après le flop, les mises de blindes ont disparu : la position se déduit uniquement du jeton D et du sens du jeu (elle ne change pas pendant la main). Si mes analyses précédentes de la même main donnent ma position, reprends-la.

Fais cette analyse en silence et n'écris que le résultat final : pas de « à vérifier », pas de correction, une seule ligne de chaque.

Format de réponse strict et court, en texte brut :
SIEGES : moi puis chaque joueur assis dans le sens du jeu, avec son rôle (BTN, SB, BB) et « out » s'il a passé
LECTURE : mes cartes | board | ma position | nombre d'adversaires encore en jeu (sans me compter) | montant à payer en BB (0 si CHECK)
ACTION : Fold / Check / Call / Bet X BB / Raise to X BB / All-in
RAISON : 1 à 2 phrases maximum
NOTE : uniquement si une information illisible pourrait changer la décision, en une ligne ; n'invente jamais une information illisible."""

USER_TEXT = "C'est mon tour. Quelle est la meilleure action ? Réponds en texte brut, sans markdown."


def user_text(history: list[str] | None) -> str:
    if not history:
        return USER_TEXT
    lines = "\n".join(f"- {h}" for h in history)
    return (f"Mes analyses précédentes des dernières minutes, de la plus ancienne à la plus récente "
            f"(c'est la même main si mes cartes sont identiques ; sinon, ignore-les) :\n{lines}\n\n{USER_TEXT}")


@dataclass
class Advice:
    ok: bool
    text: str
    model: str = ""
    error: str = ""
    seconds: float = 0.0
    first_text: float = 0.0      # délai avant le premier mot de la réponse (= temps de réflexion)
    out_tokens: int = 0          # tokens générés (réflexion comprise)
    fast: bool = False           # mode rapide réellement utilisé

    def timing(self) -> str:
        parts = [f"{self.seconds:.1f} s"]
        if self.first_text:
            parts.append(f"dont {self.first_text:.1f} s avant le 1er mot")
        if self.out_tokens:
            parts.append(f"{self.out_tokens} tokens générés")
        if self.fast:
            parts.append("mode rapide")
        return " · ".join(parts)


def encode_png(bgr: np.ndarray, max_width: int) -> bytes:
    h, w = bgr.shape[:2]
    if w > max_width:
        bgr = cv2.resize(bgr, (max_width, int(h * max_width / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("Encodage PNG impossible")
    return buf.tobytes()


FAST_BETA = "fast-mode-2026-02-01"
FALLBACK_BETA = "server-side-fallback-2026-07-01"


def model_caps(model: str) -> dict:
    """Options acceptées selon le modèle."""
    return {
        "fast": model.startswith("claude-opus-5"),
        "fallback": model.startswith(("claude-opus-5", "claude-sonnet-5-5", "claude-fable-5")),
        # Valeur de `thinking` qui coupe la réflexion (None = impossible sur ce modèle).
        "thinking_off": {"type": "between_tools"} if model.startswith("claude-sonnet-5-5")
        else {"type": "disabled"} if model.startswith("claude-haiku") else None,
    }


class ClaudeAdvisor:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        caps = model_caps(cfg["model"])
        self.fast = cfg.get("speed") == "fast" and caps["fast"]
        self.thinking = caps["thinking_off"] if cfg.get("thinking") == "off" else None
        if cfg.get("thinking") == "off" and self.thinking is None:
            from . import pretty
            pretty.warn(f"{cfg['model']} ne permet pas de couper la réflexion : réglage thinking ignoré.")
        # Le repli serveur renverrait la requête à un autre modèle, qui refuserait `between_tools`.
        self.fallback = cfg.get("refusal_fallback") and caps["fallback"] and self.thinking is None
        # La clé est lue dans ANTHROPIC_API_KEY par le SDK.
        self.client = anthropic.Anthropic(timeout=cfg["timeout"], max_retries=cfg["max_retries"])

    def _request(self, kwargs: dict, fast: bool, on_text, t0: float, timing: dict):
        betas, extra = [], {}
        if self.fallback:
            betas.append(FALLBACK_BETA)
            extra["fallbacks"] = "default"
        if fast:
            betas.append(FAST_BETA)
            extra["speed"] = "fast"
        if self.thinking:
            extra["thinking"] = self.thinking
        # Streaming : la ligne ACTION s'affiche dès qu'elle arrive.
        with self.client.beta.messages.stream(betas=betas, **extra, **kwargs) as stream:
            text = ""
            for chunk in stream.text_stream:
                if not text:
                    timing["first_text"] = time.monotonic() - t0
                text += chunk
                if on_text is not None:
                    on_text(_clean(text))
            return stream.get_final_message()

    def ask(self, table_bgr: np.ndarray, on_text: Callable[[str], None] | None = None,
            history: list[str] | None = None) -> Advice:
        t0 = time.monotonic()
        timing: dict = {}
        png = encode_png(table_bgr, self.cfg["max_image_width"])
        content = [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                         "data": base64.standard_b64encode(png).decode()}},
            {"type": "text", "text": user_text(history)},
        ]
        kwargs = dict(
            model=self.cfg["model"],
            max_tokens=self.cfg["max_tokens"],
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": content}],
        )
        if self.cfg.get("effort"):
            kwargs["output_config"] = {"effort": self.cfg["effort"]}
        fast_used = self.fast
        try:
            try:
                resp = self._request(kwargs, self.fast, on_text, t0, timing)
            except (anthropic.BadRequestError, anthropic.PermissionDeniedError) as e:
                if not self.fast:
                    raise
                # Mode rapide non disponible (modèle ou compte) : on le coupe pour la session.
                self.fast = fast_used = False
                from . import pretty
                pretty.warn(f"Mode rapide indisponible ({e.status_code}) : retour à la vitesse normale.")
                resp = self._request(kwargs, False, on_text, t0, timing)
            except anthropic.RateLimitError:
                if not self.fast:
                    raise
                fast_used = False
                resp = self._request(kwargs, False, on_text, t0, timing)  # quota rapide saturé
        except anthropic.APITimeoutError:
            return Advice(False, "", error="Délai dépassé (timeout API)")
        except anthropic.RateLimitError:
            return Advice(False, "", error="Limite de requêtes atteinte (429)")
        except anthropic.AuthenticationError:
            return Advice(False, "", error="Clé API invalide : vérifie ANTHROPIC_API_KEY")
        except anthropic.NotFoundError:
            return Advice(False, "", error=f"Modèle introuvable : {self.cfg['model']}")
        except anthropic.APIStatusError as e:
            return Advice(False, "", error=f"Erreur API {e.status_code} : {e.message}")
        except anthropic.APIConnectionError:
            return Advice(False, "", error="Connexion à l'API impossible")

        usage = getattr(resp, "usage", None)
        stats = dict(seconds=time.monotonic() - t0, first_text=timing.get("first_text", 0.0),
                     out_tokens=getattr(usage, "output_tokens", 0) or 0,
                     fast=fast_used and getattr(usage, "speed", "fast") == "fast")
        if resp.stop_reason == "refusal":
            return Advice(False, "", resp.model, error="Requête refusée par le modèle", **stats)
        text = _clean("\n".join(b.text for b in resp.content if b.type == "text"))
        if not text:
            return Advice(False, "", resp.model, error=f"Réponse vide (stop_reason={resp.stop_reason})", **stats)
        return Advice(True, text, resp.model, **stats)


def _clean(text: str) -> str:
    return text.replace("**", "").replace("__", "").strip()  # la fenêtre n'affiche pas le markdown
