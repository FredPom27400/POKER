"""Boucle principale : capture -> détection -> appel Claude (une fois par tour)."""
from __future__ import annotations

import inspect
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import numpy as np

from . import pretty
from .claude_client import Advice
from .detection import Debouncer, TableVisibility, TurnDetector
from .journal import DecisionLogger
from .seats import SeatReader
from .state import TurnTracker, signature


def console(msg: str) -> None:
    pretty.info(msg)


def describe(d) -> str:
    s = f"barre : {d.red_ratio:.0%} de rouge · {d.buttons} bouton{'s' if d.buttons > 1 else ''}"
    if d.turn_score is not None:
        s += f" · ressemblance tour {d.turn_score:.2f}"
    if d.wait_score is not None:
        s += f" / attente {d.wait_score:.2f}"
    return s


class Engine:
    def __init__(self, cfg: dict, grab: Callable[[], np.ndarray], ask: Callable[[np.ndarray], Advice] | None,
                 logger: DecisionLogger | None, events: "queue.Queue[tuple[str, str]] | None" = None):
        self.cfg = cfg
        self.grab = grab
        self.ask = ask                      # None = mode --dry-run (pas d'API)
        self.logger = logger
        self.events = events
        self.detector = TurnDetector(cfg["detection"], cfg["action_bar"])
        self.debouncer = Debouncer(cfg["detection"]["stable_frames"])
        self.visibility = TableVisibility(cfg["detection"].get("table_ref"),
                                          cfg["detection"].get("min_table_similarity", 0.0))
        self._hidden = 0                    # captures consécutives « table masquée »
        self.seats = SeatReader(cfg["seats"]) if cfg.get("seats", {}).get("enabled") else None
        self.tracker = TurnTracker(cfg["dedupe"])
        self.paused = threading.Event()
        self.stopped = threading.Event()
        self._results: "queue.Queue[tuple[np.ndarray, Advice]]" = queue.Queue()
        self._pool = ThreadPoolExecutor(max_workers=1)
        params = inspect.signature(ask).parameters if ask is not None else {}
        self._streams = "on_text" in params
        self._with_history = "history" in params
        self._with_facts = "facts" in params
        self._history: list[tuple[float, str]] = []   # (instant, résumé) des conseils récents

    # --- contrôle -------------------------------------------------------
    def toggle_pause(self) -> bool:
        if self.paused.is_set():
            self.paused.clear()
        else:
            self.paused.set()
        p = self.paused.is_set()
        pretty.paused(p)
        self._emit("paused", "1" if p else "0")
        return p

    def stop(self) -> None:
        self.stopped.set()

    def _emit(self, kind: str, msg: str) -> None:
        if self.events is not None:
            self.events.put((kind, msg))

    # --- boucle ---------------------------------------------------------
    def step(self) -> None:
        """Une itération (séparée de run() pour les tests)."""
        self._drain_results()
        table = self.grab()
        d = self.detector.detect(table)
        if d.is_turn and not self.visibility.visible(table):
            # Barre d'actions visible mais table cachée : analyser serait inutile.
            # Avertir seulement si ça dure (une animation peut tromper une capture).
            self._hidden += 1
            if self._hidden == 3:
                saved = ""
                if self.logger is not None:
                    saved = f" Capture : {self.logger.save_image(table, 'masquee')}"
                pretty.warn("Table masquée (une fenêtre est devant ?) : analyse suspendue "
                            f"jusqu'à ce qu'elle soit de nouveau visible. [{self.visibility.last}]{saved}")
                self._emit("error", "Table masquée : remets-la au premier plan")
            return
        if self._hidden >= 3:
            self._emit("waiting", "En attente de ton tour…")
        self._hidden = 0
        is_turn = self.debouncer.update(d.is_turn)
        if not d.is_turn and self.cfg["detection"]["fast_turn_end"]:
            # La barre peut ne disparaître qu'un instant si les adversaires ont
            # pré-cliqué : une seule frame sans boutons termine le tour.
            self.debouncer.force(False)
            is_turn = False
        was_turn = self.tracker.in_turn
        if self.tracker.should_call(is_turn, signature(table, self.cfg["action_bar"]), d.layout):
            facts = self.seats.read(table) if self.seats is not None and self.seats.ready else None
            detail = describe(d)
            if facts is not None:
                adv = f"{facts.opponents} adv." if facts.cards_seen else "cartes adverses non reconnues"
                detail += f" · lu sans IA : {facts.position}, {adv}, bouton {facts.button}"
            pretty.turn(self.tracker.reason, detail)
            self._emit("thinking", "Analyse en cours…")
            if self.ask is None:
                self._results.put((table, Advice(True, "(dry-run : pas d'appel API)")))
            else:
                self._pool.submit(self._call, table, facts.prompt() if facts is not None else None)
        elif was_turn and not is_turn:
            self._emit("status", "En attente de ton tour")
            if self.tracker.calls_this_turn:
                pretty.info("Fin du tour, en attente du prochain…")

    def _recent_history(self) -> list[str]:
        """Conseils des dernières minutes, pour que Claude suive la main en cours."""
        now, keep = time.monotonic(), self.cfg["claude"].get("history_seconds", 0)
        self._history = [(t, h) for t, h in self._history if now - t <= keep]
        return [f"il y a {now - t:.0f} s : {h}" for t, h in self._history]

    def _remember(self, text: str) -> None:
        f = pretty.parse_fields(text)
        if f["action"]:
            self._history.append((time.monotonic(), f"{f['lecture'] or '?'} → {f['action']}"))

    def _call(self, table: np.ndarray, facts: str | None = None) -> None:
        try:
            kwargs = {}
            if facts and self._with_facts:
                kwargs["facts"] = facts
            if self._streams:
                kwargs["on_text"] = lambda t: self._emit("advice", t)
            if self._with_history:
                kwargs["history"] = self._recent_history()
            advice = self.ask(table, **kwargs)
        except Exception as e:  # noqa: BLE001 - la boucle ne doit jamais planter
            advice = Advice(False, "", error=f"Erreur inattendue : {e}")
        self._results.put((table, advice))

    def _drain_results(self) -> None:
        while True:
            try:
                table, advice = self._results.get_nowait()
            except queue.Empty:
                return
            self.tracker.on_result(advice.ok)
            if advice.ok:
                pretty.advice(advice.text, advice.timing() if advice.seconds else "", advice.model)
                self._remember(advice.text)
                self._emit("advice", advice.text)
            else:
                retry = self.tracker.retry_at is not None
                pretty.error(advice.error, retry)
                self._emit("error", advice.error + (" (nouvel essai prévu)" if retry else ""))
            if self.logger is not None and self.ask is not None:
                try:
                    self.logger.log(table, advice.text, advice.ok, advice.model, advice.error)
                except OSError as e:
                    pretty.warn(f"Journal impossible : {e}")

    def run(self) -> None:
        interval = self.cfg["capture"]["interval"]
        self._emit("status", "Actif")
        while not self.stopped.is_set():
            t0 = time.monotonic()
            if not self.paused.is_set():
                try:
                    self.step()
                except Exception as e:  # noqa: BLE001
                    pretty.warn(f"Erreur de capture/détection : {e}")
            else:
                self._drain_results()
            self.stopped.wait(max(0.0, interval - (time.monotonic() - t0)))
        self._pool.shutdown(wait=False, cancel_futures=True)
