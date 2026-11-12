#!/usr/bin/env python3
"""Transcription + diarisation (qui parle) 100 % locale.

Pipeline :
  1. ffmpeg      : conversion de l'entrée (mp4/m4a/...) en WAV mono 16 kHz
  2. faster-whisper : transcription (avec horodatage des mots)
  3. pyannote.audio : diarisation (segmentation en locuteurs)
  4. fusion      : chaque mot est attribué au locuteur dont le segment le recouvre le plus

Sorties : <nom>.txt, <nom>.srt, <nom>.json dans le dossier --out.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


# --------------------------------------------------------------------------- #
# Utilitaires
# --------------------------------------------------------------------------- #
def to_wav(src: Path, dst: Path) -> None:
    """Convertit n'importe quel média en WAV mono 16 kHz (format attendu par les modèles)."""
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
           "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(dst)]
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        sys.exit("ffmpeg est introuvable : installez-le (apt install ffmpeg / brew install ffmpeg / choco install ffmpeg).")


def fmt_ts(t: float, sep: str = ".") -> str:
    h, rem = divmod(int(t * 1000), 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


@dataclass
class Word:
    start: float
    end: float
    text: str
    speaker: str = "INCONNU"


@dataclass
class Turn:
    speaker: str
    start: float
    end: float
    text: str


# --------------------------------------------------------------------------- #
# Étapes
# --------------------------------------------------------------------------- #
def transcribe(wav: Path, model_size: str, device: str, compute_type: str, language: str | None) -> list[Word]:
    from faster_whisper import WhisperModel

    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments, info = model.transcribe(
        str(wav),
        language=language,
        word_timestamps=True,
        vad_filter=True,               # ignore les silences -> moins d'hallucinations
        beam_size=5,
        condition_on_previous_text=False,
    )
    print(f"[whisper] langue : {info.language} (p={info.language_probability:.2f}), durée : {info.duration:.0f}s")
    words: list[Word] = []
    for seg in segments:
        for w in seg.words or []:
            words.append(Word(w.start, w.end, w.word))
        print(f"[whisper] {fmt_ts(seg.end)[:8]} / {info.duration:.0f}s", end="\r", flush=True)
    print()
    return words


def diarize(wav: Path, hf_token: str | None, device: str, num_speakers: int | None,
            min_speakers: int | None, max_speakers: int | None):
    import torch
    from pyannote.audio import Pipeline

    # hf_token n'est utile qu'au 1er téléchargement ; ensuite le cache local suffit (mode hors ligne).
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)
    if pipeline is None:
        sys.exit("Impossible de charger le modèle pyannote : token HF invalide ou conditions d'utilisation non acceptées (voir README).")
    pipeline.to(torch.device(device))

    kwargs = {}
    if num_speakers:
        kwargs["num_speakers"] = num_speakers
    else:
        if min_speakers:
            kwargs["min_speakers"] = min_speakers
        if max_speakers:
            kwargs["max_speakers"] = max_speakers

    # Charge l'audio en mémoire (évite les dépendances torchaudio/ffmpeg côté pyannote)
    import soundfile as sf
    data, sr = sf.read(str(wav), dtype="float32", always_2d=True)
    waveform = torch.from_numpy(data.T)
    result = pipeline({"waveform": waveform, "sample_rate": sr}, **kwargs)
    # pyannote >= 4 renvoie un objet avec .speaker_diarization ; 3.x renvoie l'Annotation directement
    annotation = getattr(result, "speaker_diarization", result)
    return annotation


def assign_speakers(words: list[Word], annotation) -> None:
    """Attribue à chaque mot le locuteur avec le plus grand recouvrement temporel."""
    tracks = [(t.start, t.end, spk) for t, _, spk in annotation.itertracks(yield_label=True)]
    for w in words:
        best, best_ov = None, 0.0
        for s, e, spk in tracks:
            ov = min(w.end, e) - max(w.start, s)
            if ov > best_ov:
                best, best_ov = spk, ov
        if best is None:  # mot dans un trou : prendre le locuteur le plus proche
            mid = (w.start + w.end) / 2
            best = min(tracks, key=lambda x: 0 if x[0] <= mid <= x[1] else min(abs(mid - x[0]), abs(mid - x[1])))[2]
        w.speaker = best


def build_turns(words: list[Word], max_gap: float = 1.5) -> list[Turn]:
    """Regroupe les mots consécutifs d'un même locuteur en « tours de parole »."""
    turns: list[Turn] = []
    for w in words:
        if turns and turns[-1].speaker == w.speaker and w.start - turns[-1].end <= max_gap:
            t = turns[-1]
            t.text += w.text
            t.end = w.end
        else:
            turns.append(Turn(w.speaker, w.start, w.end, w.text))
    for t in turns:
        t.text = t.text.strip()
    return turns


def rename_speakers(turns: list[Turn], words: list[Word], names: list[str] | None) -> dict[str, str]:
    """SPEAKER_03 -> « Intervenant 1 » (par ordre d'apparition) ou noms fournis via --names."""
    order: list[str] = []
    for t in turns:
        if t.speaker not in order:
            order.append(t.speaker)
    mapping = {}
    for i, spk in enumerate(order):
        mapping[spk] = names[i] if names and i < len(names) else f"Intervenant {i + 1}"
    for t in turns:
        t.speaker = mapping[t.speaker]
    for w in words:
        w.speaker = mapping.get(w.speaker, w.speaker)
    return mapping


# --------------------------------------------------------------------------- #
# Sorties
# --------------------------------------------------------------------------- #
def write_outputs(turns: list[Turn], words: list[Word], out_base: Path) -> None:
    with open(out_base.with_suffix(".txt"), "w", encoding="utf-8") as f:
        for t in turns:
            f.write(f"[{fmt_ts(t.start)[:8]}] {t.speaker} : {t.text}\n\n")

    with open(out_base.with_suffix(".srt"), "w", encoding="utf-8") as f:
        for i, t in enumerate(turns, 1):
            f.write(f"{i}\n{fmt_ts(t.start, ',')} --> {fmt_ts(t.end, ',')}\n{t.speaker}: {t.text}\n\n")

    with open(out_base.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump({
            "turns": [t.__dict__ for t in turns],
            "words": [w.__dict__ for w in words],
        }, f, ensure_ascii=False, indent=2)


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Transcription locale avec identification des locuteurs.")
    ap.add_argument("audio", type=Path, help="Fichier audio/vidéo (mp4, m4a, mp3, wav...)")
    ap.add_argument("-o", "--out", type=Path, default=Path("sortie"), help="Dossier de sortie (défaut : ./sortie)")
    ap.add_argument("-n", "--num-speakers", type=int, default=4, help="Nombre exact de locuteurs (défaut : 4 ; 0 = auto)")
    ap.add_argument("--min-speakers", type=int)
    ap.add_argument("--max-speakers", type=int)
    ap.add_argument("--names", nargs="+", help="Noms dans l'ordre d'apparition, ex : --names Alice Bob Chloé David")
    ap.add_argument("-m", "--model", default="large-v3",
                    help="Modèle Whisper : tiny/base/small/medium/large-v3/distil-large-v3 (défaut : large-v3)")
    ap.add_argument("-l", "--language", default="fr", help="Code langue (défaut : fr ; 'auto' pour détection)")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--compute-type", default=None, help="int8 (CPU, défaut) / float16 (GPU, défaut) / int8_float16...")
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"),
                    help="Token HuggingFace (1er lancement uniquement ; ou variable HF_TOKEN)")
    args = ap.parse_args()

    if not args.audio.exists():
        sys.exit(f"Fichier introuvable : {args.audio}")

    import torch
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    compute_type = args.compute_type or ("float16" if device == "cuda" else "int8")
    language = None if args.language == "auto" else args.language
    args.out.mkdir(parents=True, exist_ok=True)
    out_base = args.out / args.audio.stem

    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "audio.wav"
        print("[1/4] Conversion audio…")
        to_wav(args.audio, wav)

        print(f"[2/4] Transcription (modèle {args.model}, {device}/{compute_type})…")
        words = transcribe(wav, args.model, device, compute_type, language)

        print("[3/4] Diarisation (qui parle ?)…")
        annotation = diarize(wav, args.hf_token, device, args.num_speakers or None,
                             args.min_speakers, args.max_speakers)

    print("[4/4] Fusion et écriture…")
    assign_speakers(words, annotation)
    turns = build_turns(words)
    mapping = rename_speakers(turns, words, args.names)
    write_outputs(turns, words, out_base)

    print(f"\nLocuteurs détectés : {', '.join(mapping.values())}")
    print(f"Résultats : {out_base}.txt / .srt / .json")


if __name__ == "__main__":
    main()
