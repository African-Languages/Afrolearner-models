import os
import torch
import numpy as np
import librosa
import unicodedata
import json
from datasets import load_dataset, concatenate_datasets
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    get_cosine_schedule_with_warmup,
)
from torch.utils.data import DataLoader
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("train_mms.log")]
)
log = logging.getLogger(__name__)

# ======================
# CONFIG
# ======================
MODEL_NAME      = "facebook/mms-300m"
DATASET_NAME    = "google/WaxalNLP"
BATCH_SIZE      = 4
GRAD_ACCUM      = 4
EPOCHS          = 10

# Paths are env-var overridable so the same script runs unchanged locally or
# on Kaggle: point these at /kaggle/working/... (writable, session-persisted)
# in the notebook's first cell rather than editing this file.
CHECKPOINT_PATH  = os.environ.get("AFROLEARNER_MMS_CHECKPOINT", "./checkpoint_mms.pt")
BEST_MODEL_PATH  = os.environ.get("AFROLEARNER_MMS_BEST_MODEL", "./best_model_mms")
PROCESSOR_PATH   = os.environ.get("AFROLEARNER_MMS_PROCESSOR", "./mms_processor")
FINAL_MODEL_PATH = os.environ.get("AFROLEARNER_MMS_FINAL_MODEL", "./final_model_mms")

CHUNK_SIZE      = 1000
TOTAL_SIZE      = 20000
MAX_TEXT_LEN    = 100
SAMPLE_RATE     = 16000
MAX_AUDIO_SEC   = 30
MIN_AUDIO_SEC   = 1.5   # raised — very short clips cause CTC NaN loss
LR              = 3e-5  # lower LR — more stable for CTC
WEIGHT_DECAY    = 1e-2
WARMUP_STEPS    = 200
LOG_EVERY       = 20
SAVE_EVERY      = 200

# MMS-300m encoder downsampling factor is 320
# meaning 1 second of audio → 50 encoder frames
# CTC requires: encoder_frames > label_length
# We enforce this in preprocess below
MMS_DOWNSAMPLE  = 320

LANGUAGES = {
    "yor": {"config": "yor_tts", "mms_lang": "yor"},
    "ibo": {"config": "ibo_tts", "mms_lang": "ibo"},
    "hau": {"config": "hau_tts", "mms_lang": "hau"},
    "ful": {"config": "ful_tts", "mms_lang": "fuv"},
    "pcm": {"config": "pcm_tts", "mms_lang": "pcm"},
}

device  = "cuda" if torch.cuda.is_available() else "cpu"
use_amp = device == "cuda"
log.info(f"Device: {device} | AMP: {use_amp}")

# ======================
# TEXT NORMALIZATION
# ======================
def normalize_text(text):
    text = unicodedata.normalize("NFC", text.strip().lower())
    return " ".join(text.split())

# ======================
# BUILD VOCABULARY & PROCESSOR
# ======================
def build_vocab_and_processor():
    if os.path.exists(PROCESSOR_PATH):
        log.info(f"Loading existing processor from {PROCESSOR_PATH}")
        return Wav2Vec2Processor.from_pretrained(PROCESSOR_PATH)

    log.info("Building joint vocabulary...")
    chars = set()
    for lang, info in LANGUAGES.items():
        try:
            ds = load_dataset(DATASET_NAME, info["config"], split="train[:300]")
            for ex in ds:
                text = normalize_text(ex.get("text", ""))
                chars.update(text.replace(" ", "|"))
        except Exception as e:
            log.warning(f"  Vocab skip {lang}: {e}")

    special   = ["[PAD]", "[UNK]", "|"]
    normal    = sorted([c for c in chars if c not in special])
    all_tok   = special + normal
    vocab_dict = {v: i for i, v in enumerate(all_tok)}
    log.info(f"Vocabulary size: {len(vocab_dict)}")

    os.makedirs(PROCESSOR_PATH, exist_ok=True)
    with open(f"{PROCESSOR_PATH}/vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab_dict, f, ensure_ascii=False, indent=2)
    with open(f"{PROCESSOR_PATH}/tokenizer_config.json", "w") as f:
        json.dump({
            "unk_token": "[UNK]", "pad_token": "[PAD]",
            "word_delimiter_token": "|", "do_lower_case": False,
            "tokenizer_class": "Wav2Vec2CTCTokenizer",
        }, f)

    tokenizer = Wav2Vec2CTCTokenizer(
        f"{PROCESSOR_PATH}/vocab.json",
        unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|",
    )
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1, sampling_rate=SAMPLE_RATE,
        padding_value=0.0, do_normalize=True, return_attention_mask=True,
    )
    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor, tokenizer=tokenizer
    )
    processor.save_pretrained(PROCESSOR_PATH)
    log.info(f"Processor saved to {PROCESSOR_PATH}")
    return processor

# ======================
# DATASET CACHE
# ======================
DATASET_CACHE: dict = {}

def warm_dataset_cache():
    log.info("Warming dataset cache...")
    for lang, info in LANGUAGES.items():
        try:
            ds = load_dataset(DATASET_NAME, info["config"],
                              split="train", trust_remote_code=False)
            ds = ds.add_column("lang", [lang] * len(ds))
            DATASET_CACHE[lang] = ds
            log.info(f"  OK {lang}: {len(ds)} samples")
        except Exception as e:
            log.warning(f"  SKIP {lang}: {e}")
    if not DATASET_CACHE:
        raise RuntimeError("No languages loaded.")

def load_chunk(start: int, size: int):
    splits = []
    for lang, ds in DATASET_CACHE.items():
        end = min(start + size, len(ds))
        if start >= len(ds):
            continue
        splits.append(ds.select(range(start, end)))
    if not splits:
        raise RuntimeError(f"No data for chunk start={start}")
    return concatenate_datasets(splits)

# ======================
# SAFE AUDIO
# ======================
def safe_get_audio(example):
    try:
        audio = example["audio"]
        arr   = np.array(audio["array"], dtype=np.float32)
        sr    = int(audio["sampling_rate"])
        return arr, sr
    except Exception as e:
        log.debug(f"Audio decode failed: {e}")
        return None, None

# ======================
# PREPROCESS
# KEY FIX: enforce CTC constraint — encoder frames must exceed label length
# MMS-300m: encoder_frames = audio_samples // 320
# ======================
def make_preprocess(processor):
    def preprocess(example):
        invalid = {"valid": False, "input_values": None,
                   "attention_mask": None, "labels": None}

        speech, sr = safe_get_audio(example)
        if speech is None:
            return invalid

        try:
            if sr != SAMPLE_RATE:
                speech = librosa.resample(speech, orig_sr=sr, target_sr=SAMPLE_RATE)

            duration = len(speech) / SAMPLE_RATE
            if duration < MIN_AUDIO_SEC or duration > MAX_AUDIO_SEC:
                return invalid

            # Tokenize text first to get label length
            text   = normalize_text(example["text"])
            labels = processor.tokenizer(text).input_ids
            if len(labels) == 0:
                return invalid

            # CTC constraint: encoder output frames must be > label length
            # encoder_frames = num_samples // MMS_DOWNSAMPLE
            encoder_frames = len(speech) // MMS_DOWNSAMPLE
            if encoder_frames <= len(labels):
                return invalid   # audio too short for this transcript — skip

            # Audio features
            audio_inputs = processor.feature_extractor(
                speech, sampling_rate=SAMPLE_RATE,
                return_tensors="np", padding=False,
            )
            input_values   = audio_inputs.input_values[0]
            attention_mask = audio_inputs.attention_mask[0]

            return {
                "input_values":   input_values,
                "attention_mask": attention_mask,
                "labels":         labels,
                "valid":          True,
            }

        except Exception as e:
            log.debug(f"Preprocess error: {e}")
            return invalid

    return preprocess

# ======================
# DATA COLLATOR
# ======================
def make_collator(processor):
    def collate(batch):
        max_audio = max(len(x["input_values"]) for x in batch)
        padded_audio, padded_masks = [], []
        for x in batch:
            pad = max_audio - len(x["input_values"])
            padded_audio.append(np.pad(x["input_values"], (0, pad), constant_values=0.0))
            padded_masks.append(np.pad(x["attention_mask"], (0, pad), constant_values=0))

        input_values   = torch.tensor(np.stack(padded_audio), dtype=torch.float32)
        attention_mask = torch.tensor(np.stack(padded_masks), dtype=torch.long)

        labels    = [x["labels"] for x in batch]
        max_label = max(len(l) for l in labels)
        padded_labels = [l + [-100] * (max_label - len(l)) for l in labels]
        labels_tensor = torch.tensor(padded_labels, dtype=torch.long)

        return {
            "input_values":   input_values,
            "attention_mask": attention_mask,
            "labels":         labels_tensor,
        }
    return collate

# ======================
# TRAIN
# ======================
def train():
    warm_dataset_cache()
    processor  = build_vocab_and_processor()
    vocab_size = len(processor.tokenizer)
    log.info(f"Vocab size: {vocab_size}")

    log.info(f"Loading {MODEL_NAME}...")
    model = Wav2Vec2ForCTC.from_pretrained(
        MODEL_NAME,
        vocab_size=vocab_size,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        ignore_mismatched_sizes=True,
    )
    model.freeze_feature_encoder()
    model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Trainable parameters: {trainable:,}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=WEIGHT_DECAY
    )

    chunks_per_epoch  = TOTAL_SIZE // CHUNK_SIZE
    batches_per_chunk = (CHUNK_SIZE * len(LANGUAGES)) // BATCH_SIZE
    total_steps       = (EPOCHS * chunks_per_epoch * batches_per_chunk) // GRAD_ACCUM
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=total_steps
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch = 0
    start_chunk = 0
    global_step = 0
    best_loss   = float("inf")

    if os.path.exists(CHECKPOINT_PATH):
        log.info("Resuming from checkpoint...")
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt.get("epoch", 0)
        start_chunk = ckpt.get("chunk", 0)
        global_step = ckpt.get("step", 0)
        best_loss   = ckpt.get("best_loss", float("inf"))
        log.info(f"  Resumed: epoch={start_epoch} chunk={start_chunk} step={global_step}")

    preprocess = make_preprocess(processor)
    collate_fn = make_collator(processor)

    def save_checkpoint(epoch, chunk, step, loss):
        torch.save({
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch, "chunk": chunk, "step": step, "best_loss": best_loss,
        }, CHECKPOINT_PATH)
        log.info(f"  Checkpoint saved (step={step} loss={loss:.4f})")

    for epoch in range(start_epoch, EPOCHS):
        log.info(f"\n{'='*52}\nEPOCH {epoch+1}/{EPOCHS}\n{'='*52}")
        epoch_loss_sum, epoch_steps = 0.0, 0

        chunk_range = range(
            start_chunk if epoch == start_epoch else 0,
            TOTAL_SIZE, CHUNK_SIZE
        )

        for chunk_start in chunk_range:
            try:
                raw = load_chunk(chunk_start, CHUNK_SIZE)
            except Exception as e:
                log.error(f"Skipping chunk {chunk_start}: {e}")
                continue

            raw = raw.filter(
                lambda t: isinstance(t, str) and 0 < len(t.split()) < MAX_TEXT_LEN,
                input_columns=["text"]
            )

            processed = raw.map(
                preprocess, remove_columns=raw.column_names,
                desc=f"Chunk {chunk_start}", writer_batch_size=50
            )
            processed = processed.filter(lambda x: x["valid"])
            processed = processed.remove_columns(["valid"])

            if len(processed) == 0:
                log.warning(f"Chunk {chunk_start}: 0 valid — skipping.")
                continue

            log.info(f"Chunk {chunk_start}: {len(processed)}/{len(raw)} valid")

            loader = DataLoader(
                processed, batch_size=BATCH_SIZE, shuffle=True,
                collate_fn=collate_fn, num_workers=0, pin_memory=use_amp
            )

            model.train()
            optimizer.zero_grad()
            running_loss = 0.0

            for batch_idx, batch in enumerate(loader):
                input_values   = batch["input_values"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels         = batch["labels"].to(device)

                with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                    out  = model(input_values=input_values,
                                 attention_mask=attention_mask, labels=labels)
                    loss = out.loss

                # Skip NaN batches rather than propagating poison
                if torch.isnan(loss) or torch.isinf(loss):
                    log.debug(f"NaN/Inf loss at batch {batch_idx} — skipping batch")
                    optimizer.zero_grad()
                    continue

                (loss / GRAD_ACCUM).backward()
                running_loss += loss.item()

                if (batch_idx + 1) % GRAD_ACCUM == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], 1.0
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    avg_loss        = running_loss / GRAD_ACCUM
                    epoch_loss_sum += avg_loss
                    epoch_steps    += 1
                    running_loss    = 0.0

                    if global_step % LOG_EVERY == 0:
                        log.info(
                            f"Ep {epoch+1} | Chunk {chunk_start} | "
                            f"Step {global_step} | Loss {avg_loss:.4f} | "
                            f"LR {scheduler.get_last_lr()[0]:.2e}"
                        )

                    if global_step % SAVE_EVERY == 0:
                        save_checkpoint(epoch, chunk_start, global_step, avg_loss)
                        if avg_loss < best_loss:
                            best_loss = avg_loss
                            model.save_pretrained(BEST_MODEL_PATH)
                            processor.save_pretrained(BEST_MODEL_PATH)
                            log.info(f"  Best model saved (loss={best_loss:.4f})")

            log.info(f"  Chunk {chunk_start} done.")

        if epoch_steps > 0:
            epoch_avg = epoch_loss_sum / epoch_steps
            log.info(f"Epoch {epoch+1} avg loss: {epoch_avg:.4f}")
            save_checkpoint(epoch + 1, 0, global_step, epoch_avg)

    model.save_pretrained(FINAL_MODEL_PATH)
    processor.save_pretrained(FINAL_MODEL_PATH)
    log.info(f"Training complete — model saved to {FINAL_MODEL_PATH}")

if __name__ == "__main__":
    train()