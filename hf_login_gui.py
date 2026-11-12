#!/usr/bin/env python3
"""Petite interface Tk pour se connecter à HuggingFace et vérifier l'accès aux modèles pyannote.

Le token est enregistré par huggingface_hub (~/.cache/huggingface/token) : transcribe_diarize.py
le retrouve ensuite tout seul, sans --hf-token ni variable HF_TOKEN.
"""
from __future__ import annotations

import threading
import tkinter as tk
import webbrowser
from tkinter import messagebox, ttk

GATED_MODELS = [
    "pyannote/speaker-diarization-3.1",
    "pyannote/segmentation-3.0",
]
TOKENS_URL = "https://huggingface.co/settings/tokens"
SIGNUP_URL = "https://huggingface.co/join"


def check_model_access(repo: str, token: str) -> tuple[bool, str]:
    """True si les conditions du modèle ont été acceptées pour ce compte."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, GatedRepoError, HfHubHTTPError

    try:
        hf_hub_download(repo, "config.yaml", token=token)
        return True, "accès OK"
    except GatedRepoError:
        return False, "conditions non acceptées"
    except EntryNotFoundError:
        return True, "accès OK"  # le dépôt répond, simplement sans ce fichier
    except HfHubHTTPError as e:
        return False, f"erreur HTTP ({getattr(e.response, 'status_code', '?')})"
    except Exception as e:  # réseau, etc.
        return False, f"{type(e).__name__}: {e}"


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Connexion HuggingFace")
        self.resizable(False, False)
        pad = {"padx": 12, "pady": 6}

        ttk.Label(self, text="1. Créez un compte (gratuit) puis un token de type « Read » :",
                  wraplength=460).grid(row=0, column=0, columnspan=3, sticky="w", **pad)
        ttk.Button(self, text="Créer un compte", command=lambda: webbrowser.open(SIGNUP_URL)).grid(row=1, column=0, **pad)
        ttk.Button(self, text="Créer / voir mes tokens", command=lambda: webbrowser.open(TOKENS_URL)).grid(row=1, column=1, **pad)

        ttk.Label(self, text="2. Collez votre token (hf_…) :").grid(row=2, column=0, columnspan=3, sticky="w", **pad)
        self.token = tk.StringVar()
        self.entry = ttk.Entry(self, textvariable=self.token, show="•", width=52)
        self.entry.grid(row=3, column=0, columnspan=2, sticky="we", **pad)
        self.show = tk.BooleanVar(value=False)
        ttk.Checkbutton(self, text="Afficher", variable=self.show, command=self._toggle).grid(row=3, column=2, **pad)

        self.btn = ttk.Button(self, text="Se connecter et vérifier", command=self.on_login)
        self.btn.grid(row=4, column=0, columnspan=3, **pad)

        ttk.Label(self, text="3. Acceptez les conditions de chaque modèle (bouton « Ouvrir » si ✗) :",
                  wraplength=460).grid(row=5, column=0, columnspan=3, sticky="w", **pad)
        self.rows: dict[str, tuple[ttk.Label, ttk.Button]] = {}
        for i, repo in enumerate(GATED_MODELS):
            lbl = ttk.Label(self, text=f"?  {repo}")
            lbl.grid(row=6 + i, column=0, columnspan=2, sticky="w", padx=12)
            b = ttk.Button(self, text="Ouvrir", width=8,
                           command=lambda r=repo: webbrowser.open(f"https://huggingface.co/{r}"))
            b.grid(row=6 + i, column=2, padx=12, pady=2)
            self.rows[repo] = (lbl, b)

        self.status = tk.StringVar(value="Non connecté.")
        ttk.Label(self, textvariable=self.status, wraplength=460, foreground="#555").grid(
            row=9, column=0, columnspan=3, sticky="w", **pad)
        self.bind("<Return>", lambda _e: self.on_login())
        self.entry.focus()

    def _toggle(self) -> None:
        self.entry.config(show="" if self.show.get() else "•")

    def on_login(self) -> None:
        tok = self.token.get().strip()
        if not tok.startswith("hf_"):
            messagebox.showwarning("Token invalide", "Un token HuggingFace commence par « hf_ ».")
            return
        self.btn.state(["disabled"])
        self.status.set("Vérification en cours… (connexion Internet)")
        threading.Thread(target=self._worker, args=(tok,), daemon=True).start()

    def _worker(self, tok: str) -> None:
        try:
            from huggingface_hub import HfApi, login

            user = HfApi().whoami(token=tok)["name"]
            login(token=tok, add_to_git_credential=False)  # enregistre le token en local
            results = {r: check_model_access(r, tok) for r in GATED_MODELS}
            self.after(0, self._done, user, results, None)
        except Exception as e:
            self.after(0, self._done, None, {}, e)

    def _done(self, user, results, err) -> None:
        self.btn.state(["!disabled"])
        if err:
            self.status.set(f"Échec : {err}\nToken invalide, expiré, ou pas de connexion Internet ?")
            return
        for repo, (ok, msg) in results.items():
            lbl, _ = self.rows[repo]
            lbl.config(text=f"{'✓' if ok else '✗'}  {repo} — {msg}")
        if all(ok for ok, _ in results.values()):
            self.status.set(f"Connecté en tant que « {user} ». Token enregistré. Tout est prêt, "
                            "vous pouvez fermer cette fenêtre et lancer transcribe_diarize.py.")
        else:
            self.status.set(f"Connecté en tant que « {user} » (token enregistré), mais il reste des conditions "
                            "à accepter : ouvrez chaque modèle ✗, acceptez, puis recliquez sur « Se connecter et vérifier ».")


if __name__ == "__main__":
    App().mainloop()
