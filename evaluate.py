import os
import torch
import numpy as np
import librosa
import unicodedata
import re
import warnings
warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from datasets import load_dataset
from jiwer import wer, cer

# ======================
# CONFIG — use absolute path to avoid HF path validation error
# ======================
MODEL_PATH  = os.path.abspath("./best_model_mms")
SAMPLE_RATE = 16000

LANGUAGES = {
    "yor": "yor_tts",
    "ibo": "ibo_tts",
    "hau": "hau_tts",
    "pcm": "pcm_tts",
}

# ======================
# LOAD MODEL
# ======================
print(f"Loading model from {MODEL_PATH}...")
processor = Wav2Vec2Processor.from_pretrained(MODEL_PATH)
model     = Wav2Vec2ForCTC.from_pretrained(MODEL_PATH)
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Running on: {device}\n")

# ======================
# TEXT NORMALIZATION
# ======================
def normalize(text):
    text = unicodedata.normalize("NFC", text.strip().lower())
    return re.sub(r'\s+', ' ', text)

def exact_match(ref, hyp):
    return normalize(ref) == normalize(hyp)

def prefix_match(ref, hyp):
    ref_words = normalize(ref).split()
    hyp_words = normalize(hyp).split()
    if not ref_words:
        return 0.0
    matches = sum(r == h for r, h in zip(ref_words, hyp_words))
    return matches / len(ref_words)

# ======================
# EVALUATE
# ======================
all_refs, all_hyps = [], []
results = {}

for lang, config in LANGUAGES.items():
    print(f"\nEvaluating [{lang}]...")

    ds, split_used = None, None
    for split in ["test", "validation"]:
        try:
            ds = load_dataset("google/WaxalNLP", config, split=split)
            split_used = split
            print(f"  Split: '{split}' ({len(ds)} samples)")
            break
        except Exception as e:
            print(f"  '{split}' unavailable: {e}")

    if ds is None:
        print(f"  Skipping {lang}")
        continue

    lang_refs, lang_hyps = [], []
    exact_matches = 0
    prefix_scores = []
    skipped       = 0

    for example in ds:
        try:
            ref_text = example.get("text") or example.get("transcription", "")
            if not ref_text.strip():
                skipped += 1
                continue

            audio  = example["audio"]
            speech = np.array(audio["array"], dtype=np.float32)
            sr     = int(audio["sampling_rate"])

            if sr != SAMPLE_RATE:
                speech = librosa.resample(speech, orig_sr=sr, target_sr=SAMPLE_RATE)

            duration = len(speech) / SAMPLE_RATE
            if duration < 0.5 or duration > 30:
                skipped += 1
                continue

            inputs = processor.feature_extractor(
                speech, sampling_rate=SAMPLE_RATE,
                return_tensors="pt", padding=True
            )
            input_values   = inputs.input_values.to(device)
            attention_mask = inputs.attention_mask.to(device)

            with torch.no_grad():
                logits = model(
                    input_values=input_values,
                    attention_mask=attention_mask
                ).logits

            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.tokenizer.decode(
                predicted_ids[0], skip_special_tokens=True
            ).strip()

            ref_norm = normalize(ref_text)
            hyp_norm = normalize(transcription)

            lang_refs.append(ref_norm)
            lang_hyps.append(hyp_norm)

            if exact_match(ref_text, transcription):
                exact_matches += 1
            prefix_scores.append(prefix_match(ref_text, transcription))

        except Exception:
            skipped += 1
            continue

    if not lang_refs:
        print(f"  No valid samples for {lang}")
        continue

    n          = len(lang_refs)
    lang_wer   = wer(lang_refs, lang_hyps)
    lang_cer   = cer(lang_refs, lang_hyps)
    exact_acc  = exact_matches / n * 100
    prefix_acc = np.mean(prefix_scores) * 100

    results[lang] = {
        "wer": lang_wer, "cer": lang_cer,
        "exact_acc": exact_acc, "prefix_acc": prefix_acc,
        "samples": n, "skipped": skipped, "split": split_used
    }
    all_refs.extend(lang_refs)
    all_hyps.extend(lang_hyps)

    print(f"  Samples  : {n} | Skipped: {skipped}")
    print(f"  Exact %  : {exact_acc:.1f}%")
    print(f"  Prefix % : {prefix_acc:.1f}%")
    print(f"  CER      : {lang_cer:.2%}")
    print(f"  WER      : {lang_wer:.2%}")

    print(f"\n  --- Sample predictions ({lang}) ---")
    for i in range(min(5, n)):
        mark = "✓" if normalize(lang_refs[i]) == normalize(lang_hyps[i]) else "✗"
        print(f"  {mark} REF : {lang_refs[i]}")
        print(f"    HYP : {lang_hyps[i]}")
        print()

# ======================
# SUMMARY
# ======================
print("\n" + "="*62)
print("EVALUATION SUMMARY")
print("="*62)
print(f"{'Lang':<6} {'Split':<10} {'N':<6} {'Exact%':<10} {'Prefix%':<10} {'CER':<8} {'WER'}")
print("-"*62)
for lang, r in results.items():
    print(
        f"{lang:<6} {r['split']:<10} {r['samples']:<6} "
        f"{r['exact_acc']:<10.1f} {r['prefix_acc']:<10.1f} "
        f"{r['cer']:<8.2%} {r['wer']:.2%}"
    )

if all_refs:
    overall_wer   = wer(all_refs, all_hyps)
    overall_cer   = cer(all_refs, all_hyps)
    total_exact   = sum(r["exact_acc"] * r["samples"] for r in results.values())
    total_n       = sum(r["samples"] for r in results.values())
    overall_exact = total_exact / total_n if total_n else 0
    print("-"*62)
    print(
        f"{'Overall':<6} {'':<10} {total_n:<6} "
        f"{overall_exact:<10.1f} {'':<10} "
        f"{overall_cer:<8.2%} {overall_wer:.2%}"
    )
    print("="*62)
    print("\nMetric Guide:")
    print("  Exact %  → % of clips transcribed perfectly")
    print("  Prefix % → % of starting words correct")
    print("  CER      → character error rate (fairer for tonal languages)")
    print("  WER      → word error rate (strictest)")