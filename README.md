# my-team-my-txt — Transcription locale avec identification des locuteurs

Transforme un enregistrement (MP4/M4A/MP3/WAV…) d'une discussion à plusieurs en texte, en indiquant **qui parle**.
Tout tourne en local, avec des projets libres :

| Étape | Outil |
|---|---|
| Conversion audio | [ffmpeg](https://ffmpeg.org) |
| Transcription | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (Whisper d'OpenAI, licence MIT) |
| Diarisation | [pyannote.audio](https://github.com/pyannote/pyannote-audio) 3.1 (MIT) |

Votre fichier (AAC 16 kHz, 44 kbit/s, stéréo) est parfaitement exploitable : 16 kHz est justement la fréquence native de Whisper et pyannote. Le script le convertit automatiquement en mono.

## Installation

Prérequis : Python 3.9+ et `ffmpeg` (`apt install ffmpeg`, `brew install ffmpeg` ou `choco install ffmpeg`).

```bash
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

### Accès au modèle de diarisation (une seule fois)

Les modèles pyannote sont gratuits mais « gated » : il faut accepter leurs conditions sur HuggingFace (connexion Internet uniquement pour ce téléchargement initial).

1. Créer un compte sur <https://huggingface.co>.
2. Accepter les conditions sur <https://hf.co/pyannote/speaker-diarization-3.1> et <https://hf.co/pyannote/segmentation-3.0>.
3. Créer un token de lecture : <https://hf.co/settings/tokens>.

Le modèle Whisper est téléchargé automatiquement au premier lancement. Ensuite, tout est en cache (`~/.cache/huggingface`) et le script fonctionne **hors ligne** (`HF_HUB_OFFLINE=1` possible).

## Utilisation

```bash
export HF_TOKEN=hf_xxxxxxxx          # 1er lancement seulement (Windows : set HF_TOKEN=...)
python transcribe_diarize.py discussion.mp4
```

Options utiles :

```bash
# Noms des 4 personnes (dans l'ordre où elles prennent la parole)
python transcribe_diarize.py discussion.mp4 --names Alice Bob Chloé David

# Nombre de locuteurs inconnu -> estimation automatique
python transcribe_diarize.py discussion.mp4 --num-speakers 0 --min-speakers 2 --max-speakers 6

# Choisir le modèle Whisper (tiny/base/small/medium/large-v3)
python transcribe_diarize.py discussion.mp4 --model medium
```

## Repli vers un service en ligne (optionnel)

Option `--backend` :

| Mode | Comportement |
|---|---|
| `local` (défaut) | Tout reste sur votre machine, hors ligne. |
| `online` | Envoie l'audio à un service cloud (transcription + diarisation). |
| `auto` | Essaie en local ; **si ça échoue** (mémoire insuffisante, token HuggingFace absent, erreur…), bascule sur le service en ligne. |

> ⚠️ Les modes `online` et `auto` envoient votre enregistrement à un tiers. Par sécurité, le script exige le drapeau explicite `--allow-online`.

Services pris en charge (clé API à créer chez le fournisseur ; offres d'essai/gratuites variables, à vérifier) :

```bash
export ASSEMBLYAI_API_KEY=xxxx     # --provider assemblyai (défaut)
export DEEPGRAM_API_KEY=xxxx       # --provider deepgram

python transcribe_diarize.py discussion.mp4 --backend auto --allow-online
python transcribe_diarize.py discussion.mp4 --backend online --provider deepgram --allow-online
```

L'audio est d'abord compressé en MP3 mono (~20 Mo/h) avant envoi. Les appels API (`AssemblyAI` v2, `Deepgram` /v1/listen) sont écrits d'après leur documentation mais **non testés** ici : si un fournisseur a changé son API, seule la fonction `online_*` correspondante est à ajuster.

## Machine modeste (ex. Core i3-10110U, 8 Go RAM, sans GPU)

- Sans GPU NVIDIA, le script choisit `--model small` en `int8` (~1 Go de RAM). `large-v3` est déconseillé (lent, ~3–4 Go de RAM). `medium` est un compromis si vous pouvez attendre.
- Ordre de grandeur sur un i3 2 cœurs : Whisper `small` ≈ 1–2× la durée de l'audio, diarisation pyannote ≈ 0,5–1×. Une réunion d'une heure peut donc prendre 1,5 à 3 h : lancez-la pendant la nuit, ou utilisez `--backend online`.
- Fermez les applications gourmandes pendant le traitement (8 Go partagés avec Windows).

## Résultats (dossier `sortie/`)

- `discussion.txt` — lisible :
  ```
  [00:00:03] Intervenant 1 : Bonjour à tous, on commence ?

  [00:00:06] Intervenant 2 : Oui, je suis prêt.
  ```
- `discussion.srt` — sous-titres (VLC, etc.)
- `discussion.json` — tours de parole + chaque mot avec horodatage et locuteur

## Conseils

- **GPU NVIDIA** : détecté automatiquement (`float16`, `large-v3`, très rapide).
- **Qualité** : le débit de 44 kbit/s est correct pour la parole. Fixer `--num-speakers 4` améliore sensiblement la diarisation (défaut du script).
- Les chevauchements de parole sont attribués à un seul locuteur (limite de l'approche mot-à-mot).
- Les « Intervenant N » sont numérotés par ordre d'apparition ; utilisez `--names` pour les renommer.
- Alternative 100 % sans compte HuggingFace : [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) propose des modèles de diarisation téléchargeables librement (intégration non incluse ici).

## Licence

Code du projet : MIT. Vérifiez les licences/conditions des modèles utilisés.
