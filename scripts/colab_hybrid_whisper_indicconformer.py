
# # NPPE-2 Hybrid ASR Notebook
# 
# This notebook is the fast, practical route for the last few hours:
# 
# - detect language with a light Whisper model
# - send English to a strong Whisper model
# - send Indic audio to `ai4bharat/indic-conformer-600m-multilingual`
# - force routing fixes: `ur -> hi`, `ml -> ta`
# - lowercase English before submission
# - run a quick validation split before creating `submission.csv`
# 
# Notes:
# - `0.10` WER is **not guaranteed**. This is the highest-upside fast pipeline I can justify under your time limit.
# - The Indic model is gated on Hugging Face. Accept access on the model page once, then provide `HF_TOKEN` in Colab if needed.
# - If your own checkpoint beats the default Whisper model on validation, replace `WHISPER_MODEL_ID` with your HF repo id.


!pip install -q -U faster-whisper transformers datasets evaluate jiwer pandas numpy scikit-learn librosa soundfile torchaudio sentencepiece accelerate huggingface_hub "onnx==1.20.1" "onnxruntime==1.20.1" "onnxruntime-gpu==1.20.1"


import os
import re
import gc
import json
import math
import random
import string
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import librosa
import torchaudio
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from jiwer import wer
from faster_whisper import WhisperModel
from transformers import AutoModel
from huggingface_hub import login

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

RUNNING_IN_KAGGLE = False  # Set True only after moving notebook into Kaggle


# -------------------------------
# Paths and runtime config
# -------------------------------
HF_TOKEN = os.environ.get("HF_TOKEN", "")
if HF_TOKEN:
    login(token=HF_TOKEN, add_to_git_credential=False)
else:
    print("HF_TOKEN not found in env. Public models will still work. Set it if Indic Conformer or private checkpoints need access.")

if RUNNING_IN_KAGGLE:
    INPUT_PATH = "/kaggle/input/multilingual-speech-recognition"
    WORK_DIR = "/kaggle/working/nppe2_hybrid"
    TRAIN_CSV = os.path.join(INPUT_PATH, "train.csv")
    TEST_CSV = os.path.join(INPUT_PATH, "test.csv")
    TRAIN_DIR = os.path.join(INPUT_PATH, "competition_data", "train")
    TEST_DIR = os.path.join(INPUT_PATH, "competition_data", "test")
    SUBMISSION_PATH = "/kaggle/working/submission.csv"
else:
    from google.colab import drive
    if not os.path.exists('/content/drive'):
        drive.mount('/content/drive')

    ZIP_PATH = "/content/drive/MyDrive/multilingual-speech-recognition.zip"  # change if needed
    EXTRACT_ROOT = "/content/competition_data"
    WORK_DIR = "/content/nppe2_hybrid"

    if not os.path.exists(EXTRACT_ROOT):
        !unzip -q "$ZIP_PATH" -d "$EXTRACT_ROOT"
        print("Dataset extracted to", EXTRACT_ROOT)

    TRAIN_CSV = os.path.join(EXTRACT_ROOT, "train.csv")
    TEST_CSV = os.path.join(EXTRACT_ROOT, "test.csv")
    TRAIN_DIR = os.path.join(EXTRACT_ROOT, "competition_data", "train")
    TEST_DIR = os.path.join(EXTRACT_ROOT, "competition_data", "test")
    SUBMISSION_PATH = "/content/submission.csv"

os.makedirs(WORK_DIR, exist_ok=True)
print("TRAIN_CSV:", TRAIN_CSV)
print("TEST_CSV:", TEST_CSV)
print("TRAIN_DIR:", TRAIN_DIR)
print("TEST_DIR:", TEST_DIR)
print("WORK_DIR:", WORK_DIR)


# -------------------------------
# Main knobs
# -------------------------------
VAL_SIZE = 0.10
EVAL_MAX_ROWS = 220            # keep this modest for fast local validation
USE_CACHE = True
RUN_VALIDATION = True

LID_MODEL_ID = "small"        # faster-whisper model used only for language detection
WHISPER_MODEL_ID = "large-v3-turbo"  # replace with e.g. "sagandeep/whisper-medium-checkpoint-675" only if validation proves it better
INDIC_MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"
INDIC_FALLBACK_ID = "sunilmahendrakar/indic-conformer-600m-multilingual"

WHISPER_BEAM_SIZE = 5
WHISPER_COMPUTE_TYPE = "float16" if torch.cuda.is_available() else "int8"
WHISPER_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LID_CACHE_VAL = os.path.join(WORK_DIR, "val_lid.csv")
LID_CACHE_TEST = os.path.join(WORK_DIR, "test_lid.csv")
VAL_PRED_CACHE = os.path.join(WORK_DIR, "val_preds.csv")
TEST_PRED_CACHE = os.path.join(WORK_DIR, "test_preds.csv")


# -------------------------------
# Helpers
# -------------------------------
def resolve_audio_path(audio_name: str, split: str) -> str:
    base_dir = TRAIN_DIR if split == "train" else TEST_DIR
    candidates = [
        os.path.join(base_dir, audio_name),
        os.path.join(Path(base_dir).parent, audio_name),
        os.path.join(Path(base_dir).parent, split, audio_name),
        os.path.join(Path(TRAIN_CSV).parent, "competition_data", split, audio_name),
        os.path.join(Path(TRAIN_CSV).parent, split, audio_name),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Could not resolve path for {audio_name} ({split})")


def normalize_reference(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip())


def clean_submission_text(text: str, route_group: str) -> str:
    text = normalize_reference(text)
    if route_group == "en":
        text = text.lower()
    text = re.sub(r"[_]", " ", text)
    text = re.sub(r"[^\w\sऀ-ॿ஀-௿]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def detect_script(text: str) -> str:
    counts = {}
    for ch in str(text):
        if not ch.isalpha():
            continue
        name = unicodedata.name(ch, "")
        for script in ["LATIN", "DEVANAGARI", "TAMIL", "ARABIC", "BENGALI", "TELUGU", "MALAYALAM"]:
            if script in name:
                counts[script] = counts.get(script, 0) + 1
                break
    return max(counts, key=counts.get) if counts else "OTHER"


def route_language(pred_lang: str) -> str:
    pred_lang = (pred_lang or "").lower()
    if pred_lang == "en":
        return "en"
    if pred_lang in {"hi", "ur", "mr", "ne", "mai", "doi", "sa"}:
        return "hi"
    if pred_lang in {"ta", "ml"}:
        return "ta"
    return "fallback"


def clear_memory(*objs):
    for obj in objs:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# -------------------------------
# Load metadata and build split
# -------------------------------
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

audio_col = "audio" if "audio" in train_df.columns else train_df.columns[0]
text_col = "text" if "text" in train_df.columns else train_df.columns[-1]

train_df[text_col] = train_df[text_col].map(normalize_reference)
train_df["audio_path"] = train_df[audio_col].map(lambda x: resolve_audio_path(x, "train"))
test_df["audio_path"] = test_df[audio_col].map(lambda x: resolve_audio_path(x, "test"))

train_df["script"] = train_df[text_col].map(detect_script)
script_counts = train_df["script"].value_counts()
train_df["script_bucket"] = train_df["script"].where(train_df["script"].map(script_counts) >= 5, "RARE")

tr_df, va_df = train_test_split(
    train_df,
    test_size=VAL_SIZE,
    random_state=SEED,
    stratify=train_df["script_bucket"],
)
tr_df = tr_df.reset_index(drop=True)
va_df = va_df.reset_index(drop=True)

if EVAL_MAX_ROWS and len(va_df) > EVAL_MAX_ROWS:
    va_df = va_df.groupby("script_bucket", group_keys=False).apply(
        lambda x: x.sample(max(1, round(EVAL_MAX_ROWS * len(x) / len(va_df))), random_state=SEED)
    ).reset_index(drop=True)
    va_df = va_df.head(EVAL_MAX_ROWS).copy()

print("train rows:", len(train_df))
print("validation rows used:", len(va_df))
print(train_df["script_bucket"].value_counts())


# -------------------------------
# Step 1: language detection with a small Whisper model
# -------------------------------
def add_lid_predictions(df: pd.DataFrame, cache_path: str) -> pd.DataFrame:
    if USE_CACHE and os.path.exists(cache_path):
        cached = pd.read_csv(cache_path)
        if len(cached) == len(df) and set([audio_col, "pred_lang", "pred_lang_prob"]).issubset(cached.columns):
            merged = df.merge(cached[[audio_col, "pred_lang", "pred_lang_prob"]], on=audio_col, how="left")
            if merged["pred_lang"].notna().all():
                print(f"Loaded cached LID from {cache_path}")
                return merged

    lid_model = WhisperModel(LID_MODEL_ID, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE)
    pred_langs, pred_probs = [], []

    for path in tqdm(df["audio_path"].tolist(), desc=f"LID -> {Path(cache_path).name}"):
        segments, info = lid_model.transcribe(
            path,
            task="transcribe",
            beam_size=1,
            best_of=1,
            temperature=0.0,
            condition_on_previous_text=False,
            vad_filter=False,
        )
        _ = list(segments)
        pred_langs.append(getattr(info, "language", "unknown"))
        pred_probs.append(float(getattr(info, "language_probability", 0.0)))

    out = df.copy()
    out["pred_lang"] = pred_langs
    out["pred_lang_prob"] = pred_probs
    out[[audio_col, "pred_lang", "pred_lang_prob"]].to_csv(cache_path, index=False)
    clear_memory(lid_model)
    return out

if RUN_VALIDATION:
    va_df = add_lid_predictions(va_df, LID_CACHE_VAL)

test_df = add_lid_predictions(test_df, LID_CACHE_TEST)


# -------------------------------
# Step 2: model wrappers
# -------------------------------
def transcribe_with_whisper(paths, forced_language=None, desc="Whisper"):
    model = WhisperModel(WHISPER_MODEL_ID, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE)
    outputs = []
    for path in tqdm(paths, desc=desc):
        segments, info = model.transcribe(
            path,
            language=forced_language,
            task="transcribe",
            beam_size=WHISPER_BEAM_SIZE,
            best_of=WHISPER_BEAM_SIZE,
            temperature=0.0,
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        outputs.append(text)
    clear_memory(model)
    return outputs


def load_indic_model():
    model_id_to_try = [INDIC_MODEL_ID]
    if INDIC_FALLBACK_ID:
        model_id_to_try.append(INDIC_FALLBACK_ID)

    last_error = None
    for model_id in model_id_to_try:
        try:
            print("Loading Indic model:", model_id)
            kwargs = {"trust_remote_code": True}
            if HF_TOKEN:
                kwargs["token"] = HF_TOKEN
            model = AutoModel.from_pretrained(model_id, **kwargs)
            return model, model_id
        except Exception as exc:
            print(f"Failed to load {model_id}: {exc}")
            last_error = exc
    raise RuntimeError(f"Could not load any Indic model. Last error: {last_error}")


def load_wav_16k_mono(path: str):
    wav, sr = torchaudio.load(path)
    wav = torch.mean(wav, dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(wav)
    return wav


def transcribe_with_indic(paths, lang_code, desc="Indic"):
    model, loaded_id = load_indic_model()
    outputs = []
    for path in tqdm(paths, desc=f"{desc} [{lang_code}]"):
        wav = load_wav_16k_mono(path)
        text = model(wav, lang_code, "rnnt")
        outputs.append(text if isinstance(text, str) else str(text))
    clear_memory(model)
    print("Indic model used:", loaded_id)
    return outputs


# -------------------------------
# Step 3: routed inference
# -------------------------------
def run_routed_inference(df: pd.DataFrame, cache_path: str = None) -> pd.DataFrame:
    if cache_path and USE_CACHE and os.path.exists(cache_path):
        cached = pd.read_csv(cache_path)
        required = {audio_col, "pred_text", "route_group", "pred_lang", "pred_lang_prob"}
        if len(cached) == len(df) and required.issubset(cached.columns):
            print(f"Loaded cached predictions from {cache_path}")
            return cached

    work = df.copy()
    work["route_group"] = work["pred_lang"].map(route_language)
    work["pred_text"] = ""

    english_mask = work["route_group"] == "en"
    hindi_mask = work["route_group"] == "hi"
    tamil_mask = work["route_group"] == "ta"
    fallback_mask = work["route_group"] == "fallback"

    if english_mask.any():
        texts = transcribe_with_whisper(
            work.loc[english_mask, "audio_path"].tolist(),
            forced_language="en",
            desc="Whisper English",
        )
        work.loc[english_mask, "pred_text"] = texts

    if hindi_mask.any():
        texts = transcribe_with_indic(
            work.loc[hindi_mask, "audio_path"].tolist(),
            lang_code="hi",
            desc="Indic Hindi route",
        )
        work.loc[hindi_mask, "pred_text"] = texts

    if tamil_mask.any():
        texts = transcribe_with_indic(
            work.loc[tamil_mask, "audio_path"].tolist(),
            lang_code="ta",
            desc="Indic Tamil route",
        )
        work.loc[tamil_mask, "pred_text"] = texts

    if fallback_mask.any():
        texts = transcribe_with_whisper(
            work.loc[fallback_mask, "audio_path"].tolist(),
            forced_language=None,
            desc="Whisper fallback",
        )
        work.loc[fallback_mask, "pred_text"] = texts

    work["pred_text"] = [clean_submission_text(t, r) for t, r in zip(work["pred_text"], work["route_group"])]

    if cache_path:
        keep_cols = [c for c in [audio_col, "audio_path", text_col, "script_bucket", "pred_lang", "pred_lang_prob", "route_group", "pred_text"] if c in work.columns]
        work[keep_cols].to_csv(cache_path, index=False)
    return work


# -------------------------------
# Optional validation
# -------------------------------
if RUN_VALIDATION:
    val_pred_df = run_routed_inference(va_df, VAL_PRED_CACHE)
    val_pred_df[text_col] = val_pred_df[text_col].map(normalize_reference)

    overall_wer = wer(val_pred_df[text_col].tolist(), val_pred_df["pred_text"].tolist())
    print("
Overall validation WER:", round(overall_wer, 4))

    print("
Route counts:")
    print(val_pred_df["route_group"].value_counts(dropna=False))

    print("
Predicted language counts:")
    print(val_pred_df["pred_lang"].value_counts(dropna=False).head(20))

    for bucket in sorted(val_pred_df["script_bucket"].unique()):
        sub = val_pred_df[val_pred_df["script_bucket"] == bucket]
        if len(sub) == 0:
            continue
        bucket_wer = wer(sub[text_col].tolist(), sub["pred_text"].tolist())
        print(f"WER[{bucket}] = {bucket_wer:.4f} on {len(sub)} rows")

    display(val_pred_df[[audio_col, text_col, "pred_lang", "route_group", "pred_text"]].head(10))
else:
    print("Validation skipped by config.")


# -------------------------------
# Final test submission
# -------------------------------
test_pred_df = run_routed_inference(test_df, TEST_PRED_CACHE)
submission = test_pred_df[[audio_col]].copy()
submission["text"] = test_pred_df["pred_text"]
submission.to_csv(SUBMISSION_PATH, index=False)

print("Saved submission to:", SUBMISSION_PATH)
print(submission.head(10))
print("
Route counts on test:")
print(test_pred_df["route_group"].value_counts(dropna=False))
print("
Predicted language counts on test:")
print(test_pred_df["pred_lang"].value_counts(dropna=False).head(20))


# ## How to use this notebook
# 
# 1. Run all cells in Colab with GPU enabled.
# 2. If Indic loading fails, first accept the model terms on Hugging Face for `ai4bharat/indic-conformer-600m-multilingual`, then set `HF_TOKEN`.
# 3. Check the validation WER and route counts.
# 4. If validation looks worse than expected, try one change at a time:
#    - swap `WHISPER_MODEL_ID` to one of your saved checkpoints
#    - increase `EVAL_MAX_ROWS` for a more stable local estimate
#    - change `ml -> ta` or `ur -> hi` only if validation contradicts it
# 5. Upload the generated `submission.csv` to Kaggle.
