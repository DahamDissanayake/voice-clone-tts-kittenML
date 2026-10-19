"""Small local GUI for KittenTTS 2 voice cloning.

Run:  .venv\\Scripts\\python app.py   (opens http://127.0.0.1:7860)
"""
import os
import subprocess
import sys
import tempfile
import time

# kittenml opens its voice JSON without an encoding, which breaks on Windows' cp1252 default.
if __name__ == "__main__" and not sys.flags.utf8_mode:
    sys.exit(subprocess.call([sys.executable, "-X", "utf8", *sys.argv]))

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import gradio as gr
import librosa
import soundfile as sf
import torch
from kittenml import KittenTTS

MAX_REF_SECONDS = 30
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

_model = None


def get_model(progress):
    global _model
    if _model is None:
        progress(0.3, desc="Loading KittenTTS 2 (first run downloads ~950 MB)...")
        _model = KittenTTS("KittenML/kitten-tts-2")
    return _model


def process_voice(audio_path, start, length, progress=gr.Progress()):
    if not audio_path:
        raise gr.Error("Upload or record a voice sample first.")
    progress(0.05, desc="Reading audio...")
    length = max(5.0, min(float(length), MAX_REF_SECONDS))
    audio, sr = librosa.load(audio_path, sr=None, mono=True, offset=float(start), duration=length)
    audio, _ = librosa.effects.trim(audio, top_db=35)
    seconds = len(audio) / sr
    if seconds < 5:
        raise gr.Error(f"Only {seconds:.1f}s of speech found. The reference needs at least 5s.")

    progress(0.15, desc="Saving trimmed reference...")
    ref_path = os.path.join(tempfile.gettempdir(), f"kitten_ref_{int(time.time())}.wav")
    sf.write(ref_path, audio, sr)

    get_model(progress)
    progress(1.0, desc="Done")
    device = "GPU" if torch.cuda.is_available() else "CPU"
    return ref_path, ref_path, f"Voice ready: {seconds:.1f}s reference, running on {device}. Type some text and press Speak."


def speak(ref_path, ref_text, text, expressive, progress=gr.Progress()):
    if not ref_path:
        raise gr.Error("Process a voice first.")
    if not text or not text.strip():
        raise gr.Error("Type something to say.")
    model = get_model(progress)
    progress(0.5, desc="Generating speech..." if ref_text and ref_text.strip()
             else "Generating speech (transcribing reference with Whisper)...")
    kwargs = {"preset": "expressive"} if expressive else {}
    if ref_text and ref_text.strip():
        kwargs["reference_text"] = ref_text.strip()
    audio = model.generate(text.strip(), reference=ref_path, **kwargs)
    progress(0.95, desc="Saving...")
    out_path = os.path.join(OUT_DIR, f"cloned_{time.strftime('%Y%m%d_%H%M%S')}.wav")
    sf.write(out_path, audio, model.sample_rate)
    return out_path, f"Saved to {out_path}"


with gr.Blocks(title="Kitten Voice Clone") as demo:
    gr.Markdown("# Kitten Voice Clone\n1. Upload or record your voice  2. Process it  3. Type text and press Speak")
    ref_state = gr.State(None)

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 1. Voice sample")
            sample = gr.Audio(label="Upload or record (5-60s, one speaker)", sources=["upload", "microphone"], type="filepath")
            with gr.Row():
                start = gr.Slider(0, 60, value=0, step=1, label="Start at (s)")
                length = gr.Slider(5, MAX_REF_SECONDS, value=25, step=1, label="Use length (s)")
            ref_text = gr.Textbox(
                label="What is said in that clip (optional)", lines=3,
                placeholder="Paste the exact words spoken in the selected part. "
                            "Leave empty to auto-transcribe with Whisper (one-time ~3 GB download, slower).")
            process_btn = gr.Button("Process voice", variant="primary")
            ref_preview = gr.Audio(label="Reference the model will use", interactive=False)
            status = gr.Markdown()

        with gr.Column():
            gr.Markdown("### 2. Say something")
            text = gr.Textbox(label="Text", lines=5, placeholder="Type what you want your cloned voice to say...")
            expressive = gr.Checkbox(label="Expressive preset (more emotion, less stable)")
            speak_btn = gr.Button("Speak", variant="primary")
            result = gr.Audio(label="Cloned voice", interactive=False, autoplay=True)
            saved = gr.Markdown()

    process_btn.click(process_voice, [sample, start, length], [ref_state, ref_preview, status])
    speak_btn.click(speak, [ref_state, ref_text, text, expressive], [result, saved])
    text.submit(speak, [ref_state, ref_text, text, expressive], [result, saved])

if __name__ == "__main__":
    demo.queue().launch(inbrowser=True)
