#!/usr/bin/env python3
"""Connexion HuggingFace en ligne de commande (fonctionne sous WSL, SSH, sans écran).

    python hf_login.py

Enregistre le token localement (~/.cache/huggingface/token) et vérifie l'accès aux modèles pyannote.
Ce module fournit aussi open_url() et check_model_access(), réutilisés par hf_login_gui.py.
"""
from __future__ import annotations

import getpass
import os
import shutil
import subprocess
import sys
import webbrowser

GATED_MODELS = [
    "pyannote/speaker-diarization-3.1",
    "pyannote/segmentation-3.0",
]
TOKENS_URL = "https://huggingface.co/settings/tokens"
SIGNUP_URL = "https://huggingface.co/join"


def is_wsl() -> bool:
    try:
        return "microsoft" in open("/proc/version").read().lower()
    except OSError:
        return False


def open_url(url: str) -> bool:
    """Ouvre le navigateur ; sous WSL, utilise le navigateur Windows. Renvoie False si impossible."""
    try:
        if is_wsl():
            if shutil.which("wslview"):
                subprocess.Popen(["wslview", url])
                return True
            if shutil.which("explorer.exe"):
                subprocess.Popen(["explorer.exe", url])  # renvoie parfois un code 1 : sans importance
                return True
            if shutil.which("cmd.exe"):
                subprocess.Popen(["cmd.exe", "/c", "start", "", url])
                return True
        return bool(webbrowser.open(url))
    except Exception:
        return False


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


def show_link(label: str, url: str) -> None:
    ok = open_url(url)
    print(f"  {label} : {url}" + ("" if ok else "   (copiez ce lien dans votre navigateur)"))


def main() -> None:
    from huggingface_hub import HfApi, login

    print("=== Connexion HuggingFace ===\n")
    print("1. Compte gratuit, puis token de type « Read » :")
    show_link("Créer un compte", SIGNUP_URL)
    show_link("Créer un token ", TOKENS_URL)

    token = os.environ.get("HF_TOKEN") or ""
    if token:
        print("\n(Token pris dans la variable HF_TOKEN.)")
    else:
        print("\n2. Collez le token (rien ne s'affiche pendant la saisie) puis Entrée :")
        token = getpass.getpass("   Token hf_… : ").strip()
    if not token.startswith("hf_"):
        sys.exit("Token invalide : il doit commencer par « hf_ ».")

    try:
        user = HfApi().whoami(token=token)["name"]
    except Exception as e:
        sys.exit(f"Échec de connexion : {e}\nToken invalide/expiré, ou pas de connexion Internet ?")
    login(token=token, add_to_git_credential=False)
    print(f"\n✓ Connecté en tant que « {user} » — token enregistré.\n")

    while True:
        print("3. Accès aux modèles pyannote :")
        results = {r: check_model_access(r, token) for r in GATED_MODELS}
        for repo, (ok, msg) in results.items():
            print(f"  {'✓' if ok else '✗'} {repo} — {msg}")
        missing = [r for r, (ok, _) in results.items() if not ok]
        if not missing:
            print("\nTout est prêt : vous pouvez lancer transcribe_diarize.py.")
            return
        print("\nAcceptez les conditions (bouton « Agree and access repository ») sur :")
        for r in missing:
            show_link(" ", f"https://huggingface.co/{r}")
        if input("\nAppuyez sur Entrée pour revérifier (q pour quitter) : ").strip().lower() == "q":
            return
        print()


if __name__ == "__main__":
    main()
