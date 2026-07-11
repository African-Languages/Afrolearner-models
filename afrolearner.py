"""
train_tts_coqui.py
================================================================
TEXT-TO-SPEECH training using coqui-tts — the real, training-capable
VITS toolkit. Replaces the earlier transformers-based attempt, whose
VitsModel.forward() unconditionally raises NotImplementedError the
moment `labels` is passed — confirmed in the installed transformers
source. transformers ships VITS *inference* only; it was never built
to train VITS. coqui-tts is the toolkit VITS/MMS were actually trained
with in the first place (Meta's own MMS-TTS release pipeline traces
back to this codebase's fairseq lineage).

================================================================
ENVIRONMENT — READ BEFORE INSTALLING
================================================================
coqui-tts requires a transformers version in a narrow compatible range.
Verified working: transformers==4.57.1
  - transformers >= 5.0 removes `isin_mps_friendly`, which coqui-tts's
    XTTS import path needs at package-import time (even though this
    script never uses XTTS) -> ImportError on `import TTS`.
  - transformers < 4.57 lacks `is_torchcodec_available`, which newer
    coqui-tts versions import at package-import time -> ImportError.
  - transformers==4.57.0 exists but is YANKED on PyPI (broken install).
    Use 4.57.1 or later within the 4.57.x/4.58.x range instead.

This pins a DIFFERENT transformers version than train_mms.py (the ASR
script) may use. Use a SEPARATE virtual environment for this script:

    python -m venv venv_tts
    source venv_tts/bin/activate
    pip install coqui-tts torch torchaudio soundfile librosa datasets
    pip install "transformers==4.57.1"   # AFTER coqui-tts, to pin the
                                          # right version over whatever
                                          # coqui-tts's own resolver picked

================================================================
WHAT THIS SCRIPT DOES
================================================================
1. EXPORT (once per language, cached): downloads each language's
   WaxalNLP split via `datasets`, decodes audio defensively (skips
   corrupted rows individually rather than crashing), and writes a
   local LJSpeech-style folder coqui-tts can consume:
       ./coqui_data/<lang>/wavs/*.wav
       ./coqui_data/<lang>/metadata.csv   (filename|text, pipe-delimited)

2. CHECKPOINT RESOLUTION per language, in priority order:
   a. PRETRAINED FAIRSEQ (yor, hau, pcm — confirmed generator repos
      exist): build a Vits model from config, then call
      model.load_fairseq_checkpoint() to load Meta's real G_100000.pth
      generator weights, using coqui-tts's own verified key-rehashing
      (TTS.tts.utils.fairseq.rehash_fairseq_vits_checkpoint) — this is
      the library's own production conversion logic, not a hand-built
      mapping like the one we couldn't safely verify for transformers.
   b. FULL SCRATCH (ibo, ful — fallback if fairseq files are missing
      or fail to load): Vits.init_from_config() with no pretrained
      weights at all. Needs more data and more epochs.

3. TRAIN: real coqui-tts Trainer/TrainerArgs, the library's own
   train_step/get_optimizer/get_criterion — generator + discriminator
   GAN training, verified to actually run (tested end-to-end on a
   synthetic 3-sample dataset before being handed to you).

================================================================
DOWNLOADING FAIRSEQ GENERATOR FILES (required for Track A languages)
================================================================
Each language needs config.json + G_100000.pth + vocab.txt in a local
folder, e.g. ./fairseq_checkpoints/yor/. Get them with:

    python -c "
from huggingface_hub import hf_hub_download
import shutil
for lang in ['yor', 'hau', 'pcm']:
    for fname in ['config.json', 'G_100000.pth', 'vocab.txt']:
        path = hf_hub_download(repo_id='facebook/mms-tts', subfolder=f'models/{lang}', filename=fname)
        shutil.copy(path, f'./fairseq_checkpoints/{lang}/{fname}')
"

ibo and ful do NOT have these files in facebook/mms-tts's models/ folder
as far as could be confirmed — the script tries anyway (cheap to check)
and falls back to full_scratch automatically if the directory or files
are missing.

================================================================
"""

import os
import csv
import json
import logging
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
from datasets import load_dataset, Audio as HFAudio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("train_tts_coqui.log")],
)
log = logging.getLogger(__name__)

# ======================
# CONFIG
# ======================
HF_DATASET_NAME   = "google/WaxalNLP"
EXPORT_ROOT       = "./coqui_data"            # local LJSpeech-style export per language
FAIRSEQ_CKPT_ROOT = "./fairseq_checkpoints"   # downloaded Meta generator files, per language
OUTPUT_ROOT       = "./coqui_output"          # trainer output per language
SAMPLE_RATE       = 16000

EPOCHS_PRETRAINED  = 200    # fine-tuning a pretrained fairseq generator
EPOCHS_FULL_SCRATCH = 1000  # training a randomly-initialized model from zero

BATCH_SIZE       = 16
MIN_AUDIO_SEC    = 0.5
MAX_AUDIO_SEC    = 12
MAX_TEXT_LEN     = 200

# Minimum exported samples to attempt full_scratch. Below this, full-scratch
# VITS is very unlikely to reach intelligible speech — there's no good
# fallback at that point short of finding a verified pretrained checkpoint.
FULL_SCRATCH_MIN_SAMPLES = 2000

LANGUAGES = {
    "yor": {"hf_config": "yor_tts", "fairseq_lang": "yor"},
    "ibo": {"hf_config": "ibo_tts", "fairseq_lang": "ibo"},
    "hau": {"hf_config": "hau_tts", "fairseq_lang": "hau"},
    "ful": {"hf_config": "ful_tts", "fairseq_lang": "fuv"},
    "pcm": {"hf_config": "pcm_tts", "fairseq_lang": "pcm"},
}


# ======================
# STEP 1 — EXPORT: HuggingFace parquet -> local LJSpeech-style folder
# ======================
def safe_decode_audio(audio_field):
    """
    Decodes one row's audio manually rather than relying on `datasets`'
    automatic Audio() decoding, which crashes one layer above any
    try/except we write if a row's bytes are corrupted (confirmed on at
    least one Fulfulde row in this dataset during earlier debugging).
    Returns (array, sampling_rate) or (None, None) on any failure.
    """
    try:
        if isinstance(audio_field, dict) and "array" in audio_field and audio_field["array"] is not None:
            arr = np.array(audio_field["array"], dtype=np.float32)
            sr = int(audio_field["sampling_rate"])
            return arr, sr

        raw_bytes = audio_field.get("bytes") if isinstance(audio_field, dict) else None
        path = audio_field.get("path") if isinstance(audio_field, dict) else None

        if raw_bytes:
            import io
            arr, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32", always_2d=False)
        elif path:
            arr, sr = librosa.load(path, sr=None, mono=True)
        else:
            return None, None

        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        return arr.astype(np.float32), int(sr)
    except Exception as e:
        log.debug(f"Audio decode failed (skipping row): {e}")
        return None, None


def export_language_to_local(lang: str, hf_config: str, force: bool = False) -> tuple[str, int]:
    """
    Downloads the HF split and writes:
        EXPORT_ROOT/<lang>/wavs/<id>.wav
        EXPORT_ROOT/<lang>/metadata.csv      (id|text, pipe-delimited, no header)

    Idempotent: skips re-export if metadata.csv already exists and force=False.
    Returns (export_dir, sample_count).
    """
    export_dir = os.path.join(EXPORT_ROOT, lang)
    wavs_dir = os.path.join(export_dir, "wavs")
    metadata_path = os.path.join(export_dir, "metadata.csv")

    if os.path.exists(metadata_path) and not force:
        with open(metadata_path, encoding="utf-8") as f:
            count = sum(1 for _ in f)
        log.info(f"  [{lang}] Already exported ({count} samples) -> {export_dir}. "
                 f"Set force=True to re-export.")
        return export_dir, count

    os.makedirs(wavs_dir, exist_ok=True)

    log.info(f"  [{lang}] Loading HF dataset {HF_DATASET_NAME}/{hf_config} ...")
    raw = load_dataset(HF_DATASET_NAME, hf_config, split="train", trust_remote_code=False)

    # Disable automatic audio decoding — decode manually below so a single
    # corrupted row gets skipped instead of crashing the whole export.
    if "audio" in raw.features:
        raw = raw.cast_column("audio", HFAudio(decode=False))

    written = 0
    skipped = 0
    with open(metadata_path, "w", encoding="utf-8", newline="") as meta_f:
        writer = csv.writer(meta_f, delimiter="|", lineterminator="\n")
        for i, row in enumerate(raw):
            text = row.get("text", "")
            if not isinstance(text, str):
                text = ""
            text = unicodedata.normalize("NFC", text.strip())
            if not text or len(text) > MAX_TEXT_LEN:
                skipped += 1
                continue

            arr, sr = safe_decode_audio(row.get("audio"))
            if arr is None:
                skipped += 1
                continue

            if sr != SAMPLE_RATE:
                arr = librosa.resample(arr, orig_sr=sr, target_sr=SAMPLE_RATE)
                sr = SAMPLE_RATE

            duration = len(arr) / sr
            if duration < MIN_AUDIO_SEC or duration > MAX_AUDIO_SEC:
                skipped += 1
                continue

            file_id = f"{lang}_{i:06d}"
            wav_path = os.path.join(wavs_dir, f"{file_id}.wav")
            sf.write(wav_path, arr, sr)
            writer.writerow([file_id, text])
            written += 1

    log.info(f"  [{lang}] Exported {written} samples ({skipped} skipped: "
             f"corrupted audio, empty/too-long text, or out-of-range duration) -> {export_dir}")
    return export_dir, written


# ======================
# STEP 2 — CUSTOM FORMATTER for the exported metadata.csv
# ======================
_waxal_formatter_registered = False


def register_waxal_formatter():
    """
    metadata.csv lines are `<file_id>|<text>` (pipe-delimited, 2 columns).
    This is intentionally simpler than coqui's built-in "ljspeech" formatter,
    which requires 3 columns (raw + normalized text) and only uses the
    third — using our own formatter avoids a silent mismatch from
    duplicating text into a column whose only purpose is being read.

    Guarded to register only once per process: register_formatter() raises
    if called twice with the same name, and this function is called once
    per language inside the training loop.
    """
    global _waxal_formatter_registered
    if _waxal_formatter_registered:
        return
    from TTS.tts.datasets import register_formatter

    def waxal_formatter(root_path, manifest_file, **kwargs):
        items = []
        txt_file = os.path.join(root_path, manifest_file)
        with open(txt_file, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                cols = line.split("|")
                if len(cols) < 2:
                    continue
                file_id, text = cols[0], cols[1]
                wav_file = os.path.join(root_path, "wavs", f"{file_id}.wav")
                items.append({
                    "text": text,
                    "audio_file": wav_file,
                    "speaker_name": "speaker",
                    "root_path": root_path,
                })
        return items

    register_formatter("waxal", waxal_formatter)
    _waxal_formatter_registered = True


# ======================
# STEP 3 — MODEL CONSTRUCTION: fairseq-pretrained or full-scratch
# ======================
def try_load_fairseq_generator(model, config, fairseq_dir: str) -> bool:
    """
    Attempts model.load_fairseq_checkpoint(). Returns True on success,
    False if the checkpoint directory or required files are missing —
    callers should fall back to full_scratch on False, not crash.
    """
    required = ["config.json", "vocab.txt"]
    # accepts either model.pth or G_100000.pth for the weights file
    has_weights = os.path.isfile(os.path.join(fairseq_dir, "model.pth")) or \
                  os.path.isfile(os.path.join(fairseq_dir, "G_100000.pth"))
    has_required = all(os.path.isfile(os.path.join(fairseq_dir, f)) for f in required)

    if not (os.path.isdir(fairseq_dir) and has_required and has_weights):
        log.warning(f"  Fairseq checkpoint incomplete or missing at {fairseq_dir} "
                    f"(need config.json, vocab.txt, and model.pth/G_100000.pth). "
                    f"Falling back to full_scratch training.")
        return False

    try:
        model.load_fairseq_checkpoint(config, fairseq_dir, eval=False, strict=True)
        log.info(f"  Loaded pretrained fairseq generator from {fairseq_dir} "
                 f"(discriminator NOT included — fairseq releases ship generator-only; "
                 f"a fresh discriminator is initialized by the Vits model class itself).")
        return True
    except Exception as e:
        log.warning(f"  Fairseq checkpoint load failed ({e}) — falling back to full_scratch training.")
        return False


def build_model_and_config(lang: str, fairseq_lang: str, train_samples: list, sample_count: int):
    """
    Returns (model, config, mode) where mode is "pretrained" or "full_scratch".
    """
    from TTS.tts.configs.vits_config import VitsConfig
    from TTS.tts.models.vits import Vits, VitsArgs, VitsAudioConfig

    audio_config = VitsAudioConfig(
        sample_rate=SAMPLE_RATE, win_length=1024, hop_length=256,
        num_mels=80, mel_fmin=0, mel_fmax=None,
    )
    vits_args = VitsArgs()

    fairseq_dir = os.path.join(FAIRSEQ_CKPT_ROOT, fairseq_lang)
    attempt_fairseq = os.path.isdir(fairseq_dir)

    epochs = EPOCHS_PRETRAINED if attempt_fairseq else EPOCHS_FULL_SCRATCH

    config = VitsConfig(
        model_args=vits_args,
        audio=audio_config,
        run_name=f"waxal_{lang}",
        batch_size=BATCH_SIZE,
        eval_batch_size=max(2, BATCH_SIZE // 2),
        num_loader_workers=2,
        num_eval_loader_workers=2,
        run_eval=sample_count > 20,
        epochs=epochs,
        text_cleaner="basic_cleaners",
        use_phonemes=False,           # character-level — matches our export, no
                                       # phonemizer/espeak dependency needed
        compute_input_seq_cache=True,
        print_step=25,
        output_path=os.path.join(OUTPUT_ROOT, lang),
        datasets=[],  # filled by caller via dataset_config, not needed here
        min_audio_len=int(MIN_AUDIO_SEC * SAMPLE_RATE),
        max_audio_len=int(MAX_AUDIO_SEC * SAMPLE_RATE),
        test_sentences=[],
    )

    model = Vits.init_from_config(config, train_samples)

    mode = "full_scratch"
    if attempt_fairseq:
        if try_load_fairseq_generator(model, config, fairseq_dir):
            mode = "pretrained"

    if mode == "full_scratch" and sample_count < FULL_SCRATCH_MIN_SAMPLES:
        log.warning(
            f"  [{lang}] Training full_scratch with only {sample_count} samples, "
            f"below the recommended {FULL_SCRATCH_MIN_SAMPLES}. The model is learning "
            f"phonetics, prosody, AND speech synthesis simultaneously from zero — "
            f"expect a long, rough training run. Proceeding anyway."
        )

    return model, config, mode


# ======================
# STEP 4 — TRAIN ONE LANGUAGE
# ======================
def train_language(lang: str, info: dict):
    from TTS.tts.configs.shared_configs import BaseDatasetConfig
    from TTS.tts.datasets import load_tts_samples
    from trainer import Trainer, TrainerArgs

    log.info(f"\n{'='*60}\nTTS TRAIN: {lang}\n{'='*60}")

    export_dir, sample_count = export_language_to_local(lang, info["hf_config"])
    if sample_count == 0:
        log.error(f"  [{lang}] 0 samples exported — skipping.")
        return

    register_waxal_formatter()
    dataset_config = BaseDatasetConfig(formatter="waxal", meta_file_train="metadata.csv", path=export_dir)

    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        eval_split=sample_count > 20,
        eval_split_size=0.02,
    )
    log.info(f"  [{lang}] {len(train_samples)} train / {len(eval_samples) if eval_samples else 0} eval samples")

    model, config, mode = build_model_and_config(lang, info["fairseq_lang"], train_samples, sample_count)
    log.info(f"  [{lang}] Training mode: {mode} | epochs: {config.epochs}")

    output_path = os.path.join(OUTPUT_ROOT, lang)
    os.makedirs(output_path, exist_ok=True)

    trainer = Trainer(
        TrainerArgs(continue_path=output_path if os.path.isdir(output_path) and
                    any(Path(output_path).rglob("*.pth")) else None),
        config,
        output_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    trainer.fit()
    log.info(f"  [{lang}] Training complete -> {output_path}")


# ======================
# MAIN
# ======================
def train_all():
    for lang, info in LANGUAGES.items():
        try:
            train_language(lang, info)
        except Exception as e:
            log.error(f"  {lang} failed: {e}", exc_info=True)
            continue


if __name__ == "__main__":
    train_all()