"""Petite fenêtre tkinter toujours au premier plan."""
from __future__ import annotations

import queue
import tkinter as tk
from typing import Callable

from .pretty import action_hex, lecture_summary, parse_fields

BG, PANEL, GREY, WHITE, YELLOW, RED = "#111", "#222", "#9a9a9a", "#f2f2f2", "#ffd25e", "#ff7070"


class AdviceWindow:
    """Action en gros et colorée, puis résumé de la lecture, raison et note."""

    def __init__(self, cfg: dict, events: "queue.Queue[tuple[str, str]]",
                 on_toggle_pause: Callable[[], bool], on_close: Callable[[], None]):
        self.events = events
        self.on_toggle_pause = on_toggle_pause
        self.on_close = on_close
        self.root = tk.Tk()
        self.root.title("Poker Assistant")
        self.root.configure(bg=BG)
        self.root.attributes("-topmost", True)
        try:
            self.root.attributes("-alpha", cfg["opacity"])
        except tk.TclError:
            pass
        self.root.geometry(cfg["geometry"])
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        fs = cfg["font_size"]
        self.wrap = cfg.get("width", 440) - 30

        top = tk.Frame(self.root, bg=PANEL)
        top.pack(fill="x")
        self.status = tk.Label(top, text="Démarrage…", fg=GREY, bg=PANEL, font=("Segoe UI", fs - 2), anchor="w")
        self.status.pack(side="left", fill="x", expand=True, padx=8, pady=4)
        self.btn = tk.Button(top, text="Pause", command=self._toggle, width=9)
        self.btn.pack(side="right", padx=6, pady=3)

        body = tk.Frame(self.root, bg=BG, padx=14, pady=10)
        body.pack(fill="both", expand=True)

        def label(font, fg, pady=(0, 0)):
            lab = tk.Label(body, text="", fg=fg, bg=BG, font=font, justify="left", anchor="w",
                           wraplength=self.wrap)
            lab.pack(fill="x", pady=pady)
            return lab

        self.action = label(("Segoe UI", fs + 12, "bold"), WHITE)
        self.info = label(("Segoe UI", fs - 1), GREY, (2, 8))
        self.reason = label(("Segoe UI", fs), WHITE)
        self.note = label(("Segoe UI", fs - 1, "italic"), YELLOW, (8, 0))
        self.action.config(text="En attente de ton tour…", font=("Segoe UI", fs + 2, "bold"), fg=GREY)
        self._big_font = ("Segoe UI", fs + 12, "bold")
        self._msg_font = ("Segoe UI", fs + 2, "bold")

        self.root.minsize(cfg.get("width", 440), 160)
        self.root.after(100, self._poll)

    def _toggle(self) -> None:
        paused = self.on_toggle_pause()
        self.btn.config(text="Reprendre" if paused else "Pause")

    def _close(self) -> None:
        self.on_close()
        self.root.destroy()

    def _message(self, text: str, color: str) -> None:
        """Message simple (attente, analyse, erreur) à la place du conseil."""
        self.action.config(text=text, font=self._msg_font, fg=color)
        for lab in (self.info, self.reason, self.note):
            lab.config(text="")

    def _show_advice(self, text: str) -> None:
        f = parse_fields(text)
        if not (f["action"] or f["lecture"] or f["reason"]):
            self._message(text, WHITE)     # réponse hors format : affichée telle quelle
            return
        self.action.config(text=(f["action"] or "…").upper(), font=self._big_font,
                           fg=action_hex(f["action"]) if f["action"] else GREY)
        self.info.config(text=lecture_summary(f["lecture"]) if f["lecture"] else "")
        self.reason.config(text=f["reason"])
        self.note.config(text="\n".join(f["notes"]))

    def _poll(self) -> None:
        try:
            while True:
                kind, msg = self.events.get_nowait()
                if kind == "advice":
                    self._show_advice(msg)
                elif kind == "error":
                    self._message(msg, RED)
                elif kind == "thinking":
                    self._message(msg, YELLOW)
                elif kind == "paused":
                    self.btn.config(text="Reprendre" if msg == "1" else "Pause")
                    self.status.config(text="En pause" if msg == "1" else "Actif")
                else:
                    self.status.config(text=msg)
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def run(self) -> None:
        self.root.mainloop()
