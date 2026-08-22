"""
synthesize.py
================================================================
TEXT-TO-SPEECH INFERENCE — loads a checkpoint produced by afrolearner.py
and synthesizes audio (a .wav file) for arbitrary input text. This is the
piece that turns a trained VITS model into something an app can actually
play back; afrolearner.py itself only trains and never synthesizes.

USAGE
================================================================
    python synthesize.py --lang yor --text "Bawo ni o se wa" --out out.wav
    python synthesize.py --lang hau --text "Sannu" --checkpoint ./coqui_output/hau/some_run/best_model.pth

By default this looks for the language's checkpoint under
OUTPUT_ROOT/<lang>/ (same AFROLEARNER_OUTPUT_ROOT env var afrolearner.py
uses, so it finds whatever that script produced without extra setup):
  1. Prefers the most recently modified best_model*.pth anywhere under
     that directory (coqui-tts nests runs in a timestamped subfolder,
     e.g. OUTPUT_ROOT/yor/waxal_yor-<date>/best_model.pth).
  2. Falls back to the most recently modified checkpoint_*.pth if no
     best_model file exists yet (e.g. training was cut off before its
     first eval pass).
The matching config.json is expected in the same directory as whichever
checkpoint is selected — that's how coqui-tts lays out its output dirs.

Requires the same environment as afrolearner.py (coqui-tts installed,
transformers==4.57.1 — see that script's header for why).
================================================================
"""

import os
import argparse
import glob
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

OUTPUT_ROOT = os.environ.get("AFROLEARNER_OUTPUT_ROOT", "./coqui_output")

LANGUAGES = ["yor", "ibo", "hau", "ful", "pcm"]


def find_checkpoint(lang: str) -> tuple[str, str]:
    """
    Returns (checkpoint_path, config_path) for the given language, searching
    OUTPUT_ROOT/<lang>/ recursively. Raises FileNotFoundError with a clear
    message if nothing usable is found (e.g. that language hasn't been
    trained yet, or was trained to a different OUTPUT_ROOT).
    """
    lang_dir = os.path.join(OUTPUT_ROOT, lang)
    if not os.path.isdir(lang_dir):
        raise FileNotFoundError(
            f"No output directory for '{lang}' at {lang_dir}. "
            f"Train it first with afrolearner.py, or set AFROLEARNER_OUTPUT_ROOT "
            f"to wherever its checkpoints live."
        )

    def newest(pattern):
        matches = glob.glob(os.path.join(lang_dir, "**", pattern), recursive=True)
        return max(matches, key=os.path.getmtime) if matches else None

    checkpoint = newest("best_model*.pth") or newest("checkpoint_*.pth")
    if not checkpoint:
        raise FileNotFoundError(
            f"No best_model*.pth or checkpoint_*.pth found under {lang_dir}. "
            f"Training for '{lang}' may not have reached its first checkpoint yet."
        )

    config_path = os.path.join(os.path.dirname(checkpoint), "config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"Found checkpoint {checkpoint} but no config.json alongside it "
            f"(expected at {config_path})."
        )

    return checkpoint, config_path


def synthesize(lang: str, text: str, out_path: str,
                checkpoint: str = None, config: str = None) -> str:
    """
    Synthesizes `text` in the given language and writes it to `out_path`.
    Pass explicit checkpoint/config to bypass auto-discovery. Returns
    out_path on success.
    """
    import torch
    from TTS.utils.synthesizer import Synthesizer

    if not checkpoint or not config:
        found_ckpt, found_cfg = find_checkpoint(lang)
        checkpoint = checkpoint or found_ckpt
        config = config or found_cfg

    log.info(f"[{lang}] Loading checkpoint: {checkpoint}")
    log.info(f"[{lang}] Loading config:     {config}")

    synthesizer = Synthesizer(
        tts_checkpoint=checkpoint,
        tts_config_path=config,
        use_cuda=torch.cuda.is_available(),
    )

    wav = synthesizer.tts(text)

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    synthesizer.save_wav(wav, out_path)
    log.info(f"[{lang}] Wrote {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Synthesize speech from text using a trained afrolearner TTS checkpoint.")
    parser.add_argument("--lang", required=True, choices=LANGUAGES, help="Language code.")
    parser.add_argument("--text", required=True, help="Text to synthesize.")
    parser.add_argument("--out", default=None, help="Output .wav path (default: ./synth_<lang>.wav)")
    parser.add_argument("--checkpoint", default=None, help="Override: explicit checkpoint .pth path.")
    parser.add_argument("--config", default=None, help="Override: explicit config.json path (required if --checkpoint is set).")
    args = parser.parse_args()

    if args.checkpoint and not args.config:
        parser.error("--config is required when --checkpoint is set.")

    out_path = args.out or f"./synth_{args.lang}.wav"
    synthesize(args.lang, args.text, out_path, checkpoint=args.checkpoint, config=args.config)


if __name__ == "__main__":
    main()
