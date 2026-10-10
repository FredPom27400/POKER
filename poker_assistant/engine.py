"""Boucle principale : capture -> détection -> appel Claude (une fois par tour)."""
from __future__ import annotations

import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Callable

import numpy as np

from .claude_client import Advice
from .detection import Debouncer, TurnDetector
from .journal import DecisionLogger
from .state import TurnTracker, signature


def console(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


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
        self.tracker = TurnTracker(cfg["dedupe"])
        self.paused = threading.Event()
        self.stopped = threading.Event()
        self._results: "queue.Queue[tuple[np.ndarray, Advice]]" = queue.Queue()
        self._pool = ThreadPoolExecutor(max_workers=1)

    # --- contrôle -------------------------------------------------------
    def toggle_pause(self) -> bool:
        if self.paused.is_set():
            self.paused.clear()
        else:
            self.paused.set()
        p = self.paused.is_set()
        console("PAUSE" if p else "REPRISE")
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
        is_turn = self.debouncer.update(d.is_turn)
        if not d.is_turn and self.cfg["detection"]["fast_turn_end"]:
            # La barre peut ne disparaître qu'un instant si les adversaires ont
            # pré-cliqué : une seule frame sans boutons termine le tour.
            self.debouncer.force(False)
            is_turn = False
        was_turn = self.tracker.in_turn
        if self.tracker.should_call(is_turn, signature(table, self.cfg["action_bar"]), d.layout):
            console(f"Mon tour : {self.tracker.reason} ({d.summary()}) -> analyse…")
            self._emit("thinking", "Analyse en cours…")
            if self.ask is None:
                self._results.put((table, Advice(True, "(dry-run : pas d'appel API)")))
            else:
                self._pool.submit(self._call, table)
        elif was_turn and not is_turn:
            self._emit("status", "En attente de ton tour")

    def _call(self, table: np.ndarray) -> None:
        try:
            advice = self.ask(table)
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
                console("\n" + advice.text)
                self._emit("advice", advice.text)
            else:
                retry = " (nouvel essai prévu)" if self.tracker.retry_at is not None else ""
                console(f"ERREUR : {advice.error}{retry}")
                self._emit("error", f"{advice.error}{retry}")
            if self.logger is not None and self.ask is not None:
                try:
                    self.logger.log(table, advice.text, advice.ok, advice.model, advice.error)
                except OSError as e:
                    console(f"Journal impossible : {e}")

    def run(self) -> None:
        interval = self.cfg["capture"]["interval"]
        self._emit("status", "Actif")
        while not self.stopped.is_set():
            t0 = time.monotonic()
            if not self.paused.is_set():
                try:
                    self.step()
                except Exception as e:  # noqa: BLE001
                    console(f"Erreur de capture/détection : {e}")
            else:
                self._drain_results()
            self.stopped.wait(max(0.0, interval - (time.monotonic() - t0)))
        self._pool.shutdown(wait=False, cancel_futures=True)
