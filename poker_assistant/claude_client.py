"""Appel à l'API Anthropic avec la capture de la table."""
from __future__ import annotations

import base64
from dataclasses import dataclass

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


def encode_png(bgr: np.ndarray, max_width: int) -> bytes:
    h, w = bgr.shape[:2]
    if w > max_width:
        bgr = cv2.resize(bgr, (max_width, int(h * max_width / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("Encodage PNG impossible")
    return buf.tobytes()


class ClaudeAdvisor:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        # La clé est lue dans ANTHROPIC_API_KEY par le SDK.
        self.client = anthropic.Anthropic(timeout=cfg["timeout"], max_retries=cfg["max_retries"])

    def ask(self, table_bgr: np.ndarray) -> Advice:
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
            if self.cfg.get("refusal_fallback"):
                resp = self.client.beta.messages.create(
                    betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs)
            else:
                resp = self.client.messages.create(**kwargs)
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

        if resp.stop_reason == "refusal":
            return Advice(False, "", resp.model, error="Requête refusée par le modèle")
        text = "\n".join(b.text for b in resp.content if b.type == "text").strip()
        text = text.replace("**", "").replace("__", "")  # la fenêtre n'affiche pas le markdown
        if not text:
            return Advice(False, "", resp.model, error=f"Réponse vide (stop_reason={resp.stop_reason})")
        return Advice(True, text, resp.model)
