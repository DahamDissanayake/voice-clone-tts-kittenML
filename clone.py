"""Voice-cloning test for KittenTTS 2.

Usage:
    python clone.py my_voice.wav "Text you want spoken in your voice."
    python clone.py my_voice.wav "Text..." --start 0 --length 25 --out cloned.wav
"""
import argparse
import subprocess
import sys

# kittenml opens its voice JSON without an encoding, which breaks on Windows' cp1252 default.
if not sys.flags.utf8_mode:
    sys.exit(subprocess.call([sys.executable, "-X", "utf8", *sys.argv]))

import librosa
import soundfile as sf
from kittenml import KittenTTS

# KittenTTS 2 wants a 5-30 s single-speaker reference clip.
MAX_REF_SECONDS = 30

parser = argparse.ArgumentParser()
parser.add_argument("reference", help="recording of the voice to clone (wav/mp3/m4a/...)")
parser.add_argument("text", help="text to synthesize in the cloned voice")
parser.add_argument("--start", type=float, default=0.0, help="seconds into the recording to start the reference")
parser.add_argument("--length", type=float, default=25.0, help="reference length in seconds (5-30)")
parser.add_argument("--out", default="cloned.wav")
parser.add_argument("--small", action="store_true", help="use the smaller emb4 weights (506 MiB)")
parser.add_argument("--preset", default=None, help="e.g. 'expressive'")
parser.add_argument("--ref-text", default=None, help="exact words in the reference clip (skips Whisper)")
args = parser.parse_args()

length = max(5.0, min(args.length, MAX_REF_SECONDS))
audio, sr = librosa.load(args.reference, sr=None, mono=True, offset=args.start, duration=length)
audio, _ = librosa.effects.trim(audio, top_db=35)
ref_path = "reference_trimmed.wav"
sf.write(ref_path, audio, sr)
print(f"Reference: {len(audio) / sr:.1f}s from {args.reference} (saved {ref_path})")

model = KittenTTS("KittenML/kitten-tts-2", **({"weights": "emb4"} if args.small else {}))
kwargs = {"preset": args.preset} if args.preset else {}
if args.ref_text:
    kwargs["reference_text"] = args.ref_text
out = model.generate(args.text, reference=ref_path, **kwargs)
sf.write(args.out, out, model.sample_rate)
print(f"Wrote {args.out}")
