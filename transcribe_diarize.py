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

    model = WhisperModel(model_size, device=device, compute_type=compute_type, cpu_threads=os.cpu_count() or 4)
    # On charge nous-mêmes le WAV (16 kHz mono) : évite le décodeur PyAV de faster-whisper,
    # dont certaines versions sont incompatibles entre elles (erreur « metadata_errors »).
    import soundfile as sf
    audio, _sr = sf.read(str(wav), dtype="float32")
    segments, info = model.transcribe(
        audio,
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
    # pyannote 4.x utilise le modèle « community-1 » ; 3.x utilise « speaker-diarization-3.1 »
    from importlib.metadata import version
    major = int(version("pyannote.audio").split(".")[0])
    model_id = "pyannote/speaker-diarization-community-1" if major >= 4 else "pyannote/speaker-diarization-3.1"
    print(f"[pyannote] modèle : {model_id}")
    # pyannote 3.x : use_auth_token ; 4.x : token. hf_token=None -> token enregistré par hf_login.py / huggingface-cli login
    try:
        pipeline = Pipeline.from_pretrained(model_id, token=hf_token)
    except TypeError:
        pipeline = Pipeline.from_pretrained(model_id, use_auth_token=hf_token)
    if pipeline is None:
        raise RuntimeError("Impossible de charger le modèle pyannote : token HF absent/invalide ou conditions non acceptées (voir README).")
    pipeline.to(torch.device(device))

    # Petite carte graphique : lots réduits pour éviter « CUDA out of memory »
    if device == "cuda":
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if vram_gb < 6:
            for attr, val in (("segmentation_batch_size", 4), ("embedding_batch_size", 4)):
                try:
                    setattr(pipeline, attr, val)
                except Exception:
                    pass

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
    try:
        result = pipeline({"waveform": waveform, "sample_rate": sr}, **kwargs)
    except torch.cuda.OutOfMemoryError:
        # Carte saturée : on libère la VRAM et on refait la diarisation sur le processeur
        print("\n[!] Mémoire GPU insuffisante pour la diarisation -> repli sur le processeur (plus lent).")
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        pipeline.to(torch.device("cpu"))
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
# Backends EN LIGNE (repli) — l'audio est envoyé au fournisseur !
# --------------------------------------------------------------------------- #
def to_mp3(src: Path, dst: Path) -> None:
    """MP3 mono 16 kHz 48 kbit/s : ~20 Mo/h au lieu de ~115 Mo/h en WAV."""
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-vn", "-ac", "1",
                    "-ar", "16000", "-b:a", "48k", str(dst)], check=True)


def online_assemblyai(audio: Path, key: str, language: str | None, num_speakers: int | None) -> list[Word]:
    import time
    import requests

    h = {"authorization": key}
    base = "https://api.assemblyai.com/v2"
    print("[online] AssemblyAI : envoi du fichier…")
    with open(audio, "rb") as f:
        r = requests.post(f"{base}/upload", headers=h, data=f, timeout=600)
    r.raise_for_status()
    body = {"audio_url": r.json()["upload_url"], "speaker_labels": True, "punctuate": True, "format_text": True}
    if language:
        body["language_code"] = language
    else:
        body["language_detection"] = True
    if num_speakers:
        body["speakers_expected"] = num_speakers
    r = requests.post(f"{base}/transcript", headers=h, json=body, timeout=60)
    r.raise_for_status()
    tid = r.json()["id"]
    while True:
        j = requests.get(f"{base}/transcript/{tid}", headers=h, timeout=60).json()
        if j["status"] == "completed":
            break
        if j["status"] == "error":
            raise RuntimeError(f"AssemblyAI : {j.get('error')}")
        print(f"[online] statut : {j['status']}…", end="\r", flush=True)
        time.sleep(5)
    print()
    return [Word(w["start"] / 1000, w["end"] / 1000, " " + w["text"], f"SPEAKER_{w.get('speaker') or '?'}")
            for w in j["words"]]


def online_deepgram(audio: Path, key: str, language: str | None, num_speakers: int | None) -> list[Word]:
    import requests

    params = {"model": "nova-3", "diarize": "true", "punctuate": "true", "smart_format": "true"}
    params["language"] = language or "multi"
    print("[online] Deepgram : envoi du fichier…")
    with open(audio, "rb") as f:
        r = requests.post("https://api.deepgram.com/v1/listen", params=params, data=f, timeout=1800,
                          headers={"Authorization": f"Token {key}", "Content-Type": "audio/mpeg"})
    r.raise_for_status()
    ws = r.json()["results"]["channels"][0]["alternatives"][0]["words"]
    return [Word(w["start"], w["end"], " " + w.get("punctuated_word", w["word"]), f"SPEAKER_{w.get('speaker', '?')}")
            for w in ws]


ONLINE = {
    "assemblyai": ("ASSEMBLYAI_API_KEY", online_assemblyai),
    "deepgram": ("DEEPGRAM_API_KEY", online_deepgram),
}


def run_online(provider: str, audio: Path, language: str | None, num_speakers: int | None) -> list[Word]:
    env, fn = ONLINE[provider]
    key = os.environ.get(env)
    if not key:
        raise RuntimeError(f"Clé API manquante : définissez la variable d'environnement {env}.")
    return fn(audio, key, language, num_speakers)


def run_local(wav: Path, args, device: str, compute_type: str, language: str | None) -> list[Word]:
    # Cache : si la diarisation échoue, on ne refait pas la transcription (longue) au prochain lancement.
    st = args.audio.stat()
    key = {"model": args.model, "language": language, "size": st.st_size, "mtime": int(st.st_mtime)}
    cache = args.out / f"{args.audio.stem}.whisper.json"
    words = None
    if cache.exists() and not args.no_cache:
        try:
            data = json.loads(cache.read_text(encoding="utf-8"))
            if data.get("key") == key:
                words = [Word(w["start"], w["end"], w["text"]) for w in data["words"]]
                print(f"[whisper] transcription réutilisée depuis {cache} (--no-cache pour refaire)")
        except Exception:
            words = None
    if words is None:
        try:
            words = transcribe(wav, args.model, device, compute_type, language)
        except RuntimeError as e:
            if device != "cuda" or "out of memory" not in str(e).lower():
                raise
            print("\n[!] Mémoire GPU insuffisante pour Whisper -> repli sur le processeur (plus lent).")
            import gc
            gc.collect()
            words = transcribe(wav, args.model, "cpu", "int8", language)
        cache.write_text(json.dumps({"key": key, "words": [{"start": w.start, "end": w.end, "text": w.text} for w in words]},
                                    ensure_ascii=False), encoding="utf-8")
    print("[local] Diarisation (qui parle ?)…")
    annotation = diarize(wav, args.hf_token, device, args.num_speakers or None, args.min_speakers, args.max_speakers)
    assign_speakers(words, annotation)
    return words


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
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    ap = argparse.ArgumentParser(description="Transcription locale avec identification des locuteurs.")
    ap.add_argument("audio", type=Path, help="Fichier audio/vidéo (mp4, m4a, mp3, wav...)")
    ap.add_argument("-o", "--out", type=Path, default=Path("sortie"), help="Dossier de sortie (défaut : ./sortie)")
    ap.add_argument("-n", "--num-speakers", type=int, default=4, help="Nombre exact de locuteurs (défaut : 4 ; 0 = auto)")
    ap.add_argument("--min-speakers", type=int)
    ap.add_argument("--max-speakers", type=int)
    ap.add_argument("--names", nargs="+", help="Noms dans l'ordre d'apparition, ex : --names Alice Bob Chloé David")
    ap.add_argument("-m", "--model", default=None,
                    help="Modèle Whisper : tiny/base/small/medium/large-v3 (défaut : small sur CPU, large-v3 sur GPU)")
    ap.add_argument("--backend", default="local", choices=["local", "online", "auto"],
                    help="local (défaut, hors ligne) | online (service cloud) | auto (local, repli en ligne si échec)")
    ap.add_argument("--provider", default="assemblyai", choices=list(ONLINE),
                    help="Service en ligne utilisé pour online/auto (défaut : assemblyai)")
    ap.add_argument("--no-cache", action="store_true", help="Refait la transcription Whisper même si un cache existe")
    ap.add_argument("--allow-online", action="store_true",
                    help="Autorise explicitement l'envoi de l'audio à un service tiers (obligatoire pour online/auto)")
    ap.add_argument("-l", "--language", default="fr", help="Code langue (défaut : fr ; 'auto' pour détection)")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--compute-type", default=None, help="int8 (CPU, défaut) / float16 (GPU, défaut) / int8_float16...")
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"),
                    help="Token HuggingFace (1er lancement uniquement ; ou variable HF_TOKEN)")
    args = ap.parse_args()

    if args.hf_token:
        # Propague le token à tout le code HF (téléchargement du modèle Whisper inclus)
        os.environ["HF_TOKEN"] = args.hf_token

    if not args.audio.exists():
        sys.exit(f"Fichier introuvable : {args.audio}")

    if args.backend != "local" and not args.allow_online:
        sys.exit("Les modes online/auto envoient l'audio à un service tiers : ajoutez --allow-online pour confirmer.")

    language = None if args.language == "auto" else args.language
    args.out.mkdir(parents=True, exist_ok=True)
    out_base = args.out / args.audio.stem

    def local_params():
        import torch
        dev = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        vram_gb = 0.0
        if dev == "cuda":
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"[gpu] {torch.cuda.get_device_name(0)} — {vram_gb:.1f} Go de VRAM")
        if not args.model:
            # Choix du modèle selon la mémoire disponible (la diarisation pyannote partage aussi la carte)
            if dev == "cuda" and vram_gb >= 8:
                args.model = "large-v3"
            elif dev == "cuda" and vram_gb >= 5:
                args.model = "medium"
            else:
                args.model = "small"
        # "auto" : CTranslate2 choisit le meilleur type supporté par la carte (ex. float32 sur GTX 9xx, sans float16 rapide)
        return dev, args.compute_type or ("auto" if dev == "cuda" else "int8")

    with tempfile.TemporaryDirectory() as tmp:
        wav, mp3 = Path(tmp) / "audio.wav", Path(tmp) / "audio.mp3"
        print("[1/3] Conversion audio…")
        to_wav(args.audio, wav)

        words = None
        if args.backend in ("local", "auto"):
            try:
                device, compute_type = local_params()
                print(f"[2/3] Traitement LOCAL (Whisper {args.model}, {device}/{compute_type})…")
                words = run_local(wav, args, device, compute_type, language)
            except Exception as e:  # MemoryError, token absent, dépendance manquante...
                if args.backend == "local":
                    raise
                print(f"\n[!] Échec du traitement local ({type(e).__name__}: {e})\n    -> repli sur {args.provider} (en ligne).")
        if words is None:
            to_mp3(wav, mp3)
            print(f"[2/3] Traitement EN LIGNE ({args.provider})…")
            words = run_online(args.provider, mp3, language, args.num_speakers or None)

    print("[3/3] Mise en forme et écriture…")
    turns = build_turns(words)
    mapping = rename_speakers(turns, words, args.names)
    write_outputs(turns, words, out_base)

    print(f"\nLocuteurs détectés : {', '.join(mapping.values())}")
    print(f"Résultats : {out_base}.txt / .srt / .json")


if __name__ == "__main__":
    main()
