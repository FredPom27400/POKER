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
Format de réponse strict et court :
- ACTION : Fold / Check / Call / Bet X BB / Raise to X BB / All-in
- RAISON : 1 à 2 phrases maximum
- Si une information est illisible ou incertaine, dis-le en une ligne au lieu d'inventer."""

USER_TEXT = "C'est mon tour. Quelle est la meilleure action ? Réponds en texte brut, sans markdown."


@dataclass
class Advice:
    ok: bool
    text: str
    model: str = ""
    error: str = ""
    seconds: float = 0.0


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


class ClaudeAdvisor:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.fast = cfg.get("speed") == "fast"
        # La clé est lue dans ANTHROPIC_API_KEY par le SDK.
        self.client = anthropic.Anthropic(timeout=cfg["timeout"], max_retries=cfg["max_retries"])

    def _request(self, kwargs: dict, fast: bool, on_text: Callable[[str], None] | None):
        betas, extra = [], {}
        if self.cfg.get("refusal_fallback"):
            betas.append(FALLBACK_BETA)
            extra["fallbacks"] = "default"
        if fast:
            betas.append(FAST_BETA)
            extra["speed"] = "fast"
        # Streaming : la ligne ACTION s'affiche dès qu'elle arrive.
        with self.client.beta.messages.stream(betas=betas, **extra, **kwargs) as stream:
            text = ""
            for chunk in stream.text_stream:
                text += chunk
                if on_text is not None:
                    on_text(_clean(text))
            return stream.get_final_message()

    def ask(self, table_bgr: np.ndarray, on_text: Callable[[str], None] | None = None) -> Advice:
        t0 = time.monotonic()
        png = encode_png(table_bgr, self.cfg["max_image_width"])
        content = [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                         "data": base64.standard_b64encode(png).decode()}},
            {"type": "text", "text": USER_TEXT},
        ]
        kwargs = dict(
            model=self.cfg["model"],
            max_tokens=self.cfg["max_tokens"],
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": content}],
        )
        if self.cfg.get("effort"):
            kwargs["output_config"] = {"effort": self.cfg["effort"]}
        try:
            try:
                resp = self._request(kwargs, self.fast, on_text)
            except (anthropic.BadRequestError, anthropic.PermissionDeniedError) as e:
                if not self.fast:
                    raise
                # Mode rapide non disponible (modèle ou compte) : on le coupe pour la session.
                self.fast = False
                from . import pretty
                pretty.warn(f"Mode rapide indisponible ({e.status_code}) : retour à la vitesse normale.")
                resp = self._request(kwargs, False, on_text)
            except anthropic.RateLimitError:
                if not self.fast:
                    raise
                resp = self._request(kwargs, False, on_text)   # quota rapide saturé : cet appel en normal
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

        seconds = time.monotonic() - t0
        if resp.stop_reason == "refusal":
            return Advice(False, "", resp.model, error="Requête refusée par le modèle", seconds=seconds)
        text = _clean("\n".join(b.text for b in resp.content if b.type == "text"))
        if not text:
            return Advice(False, "", resp.model, error=f"Réponse vide (stop_reason={resp.stop_reason})",
                          seconds=seconds)
        return Advice(True, text, resp.model, seconds=seconds)


def _clean(text: str) -> str:
    return text.replace("**", "").replace("__", "").strip()  # la fenêtre n'affiche pas le markdown
