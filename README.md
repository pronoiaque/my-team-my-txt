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

# Machine peu puissante : modèle plus léger
python transcribe_diarize.py discussion.mp4 --model small
```

## Résultats (dossier `sortie/`)

- `discussion.txt` — lisible :
  ```
  [00:00:03] Intervenant 1 : Bonjour à tous, on commence ?

  [00:00:06] Intervenant 2 : Oui, je suis prêt.
  ```
- `discussion.srt` — sous-titres (VLC, etc.)
- `discussion.json` — tours de parole + chaque mot avec horodatage et locuteur

## Conseils

- **GPU NVIDIA** : détecté automatiquement (`float16`, très rapide). Sur CPU, comptez grossièrement 1–3× la durée de l'audio avec `large-v3` ; `--model medium` ou `small` accélère nettement.
- **Qualité** : le débit de 44 kbit/s est correct pour la parole. Fixer `--num-speakers 4` améliore sensiblement la diarisation (défaut du script).
- Les chevauchements de parole sont attribués à un seul locuteur (limite de l'approche mot-à-mot).
- Les « Intervenant N » sont numérotés par ordre d'apparition ; utilisez `--names` pour les renommer.
- Alternative 100 % sans compte HuggingFace : [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) propose des modèles de diarisation téléchargeables librement (intégration non incluse ici).

## Licence

Code du projet : MIT. Vérifiez les licences/conditions des modèles utilisés.
