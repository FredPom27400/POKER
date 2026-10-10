"""Petite fenêtre tkinter toujours au premier plan."""
from __future__ import annotations

import queue
import tkinter as tk
from typing import Callable


class AdviceWindow:
    def __init__(self, cfg: dict, events: "queue.Queue[tuple[str, str]]",
                 on_toggle_pause: Callable[[], bool], on_close: Callable[[], None]):
        self.events = events
        self.on_toggle_pause = on_toggle_pause
        self.on_close = on_close
        self.root = tk.Tk()
        self.root.title("Poker Assistant")
        self.root.attributes("-topmost", True)
        try:
            self.root.attributes("-alpha", cfg["opacity"])
        except tk.TclError:
            pass
        self.root.geometry(cfg["geometry"])
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        fs = cfg["font_size"]

        top = tk.Frame(self.root, bg="#222")
        top.pack(fill="x")
        self.status = tk.Label(top, text="Démarrage…", fg="#bbb", bg="#222", font=("Helvetica", fs - 2), anchor="w")
        self.status.pack(side="left", fill="x", expand=True, padx=6, pady=3)
        self.btn = tk.Button(top, text="Pause", command=self._toggle, width=8)
        self.btn.pack(side="right", padx=4, pady=2)

        self.text = tk.Label(self.root, text="En attente de ton tour…", justify="left", anchor="nw",
                             wraplength=380, fg="white", bg="#111", font=("Helvetica", fs, "bold"),
                             padx=10, pady=8)
        self.text.pack(fill="both", expand=True)
        self.root.minsize(400, 120)
        self.root.after(100, self._poll)

    def _toggle(self) -> None:
        paused = self.on_toggle_pause()
        self.btn.config(text="Reprendre" if paused else "Pause")

    def _close(self) -> None:
        self.on_close()
        self.root.destroy()

    def _poll(self) -> None:
        try:
            while True:
                kind, msg = self.events.get_nowait()
                if kind == "advice":
                    self.text.config(text=msg, fg="white")
                elif kind == "error":
                    self.text.config(text=msg, fg="#ff7070")
                elif kind == "thinking":
                    self.text.config(text=msg, fg="#ffd060")
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
