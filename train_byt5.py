"""
train_byt5.py
=============
Fine-tune google/byt5-small để sửa lỗi noise (OCR, typo, leet-speak).

Yêu cầu: chạy scripts/build_chunks.py TRƯỚC để tạo data_train/

Pipeline:
    data_train/train_chunks.csv  → train ByT5
    data_train/val_chunks.csv    → validate mỗi epoch
    data_train/test_chunks.csv   → đánh giá cuối (KHÔNG dùng trong training)

Lưu model (HuggingFace format):
    byt5_correction/config.json
    byt5_correction/model.safetensors
    byt5_correction/tokenizer.json
    Load lại: AutoModelForSeq2SeqLM.from_pretrained("byt5_correction")
"""

import json
import os
import random
import sys
from pathlib import Path

# Giảm phân mảnh VRAM – đặt TRƯỚC khi import torch
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)



# ─────────────────────────────────────────────
# Cấu hình
# ─────────────────────────────────────────────
MODEL_NAME  = "google/byt5-small"

# Các file chunks đã được build sẵn bởi scripts/build_chunks.py
TRAIN_CHUNKS = Path("/kaggle/input/datasets/thnhtrung5/data-set/train_chunks.csv")
VAL_CHUNKS   = Path("/kaggle/input/datasets/thnhtrung5/data-set/val_chunks.csv")
TEST_CHUNKS  = Path("/kaggle/input/datasets/thnhtrung5/data-set/test_chunks.csv")

OUTPUT_DIR  = Path("byt5_correction")
CACHE_DIR   = Path("byt5_correction/.tok_cache")

MAX_INPUT_LEN  = 512   # bytes (ByT5 limit) — phải khớp với CHUNK_SIZE trong build_chunks.py
MAX_TARGET_LEN = 512
CHUNK_SIZE     = 512   # phải khớp với CHUNK_SIZE trong build_chunks.py
OVERLAP        = 80    # phải khớp với OVERLAP trong build_chunks.py

# ── Điều chỉnh theo phần cứng ──────────────────────────────────
# | Môi trường         | BATCH | GRAD_ACCUM | eff_batch | Ghi chú             |
# |--------------------|-------|------------|-----------|---------------------|
# | Colab T4 (1 GPU)   |   2   |     8      |    16     |                     |
# | Kaggle T4×2 (2GPU) |   2   |     8      |    32     | ← KHUYẾN NGHỊ      |
# | Kaggle P100 (1GPU) |   2   |     8      |    16     | USE_BF16=False      |
MAX_TRAIN_SAMPLES = None  # None → dùng hết train_chunks.csv   
MAX_VAL_SAMPLES   = 1_000  
BATCH_SIZE        = 2      
EVAL_BATCH_SIZE   = 1     
GRAD_ACCUM        = 8     
EPOCHS            = 10
LR                = 1e-4
WARMUP_RATIO      = 0.06
SEED              = 42

# ── Eval mode ────────────────────────────────────────────────────
# False → dùng cross-entropy loss thật từ logits (nhanh, ít VRAM)
# True  → generate() để đánh giá chất lượng inference thực tế
USE_GENERATE_EVAL = True

# eval_accumulation_steps: sau bao nhiêu bước eval thì flush logits về CPU
# Đặt = 1 để KHÔNG tích luỹ logit 3D trong RAM → tránh OOM khi pad & concat
EVAL_ACCUM_STEPS  = 1

# T4 hỗ trợ bf16 tốt hơn fp16 cho ByT5 (tránh underflow gradient)
# Kaggle T4×2: True | Kaggle P100: False (P100 không hỗ trợ bf16)
USE_BF16 = True


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def load_split(csv_path: Path, max_samples: int | None = None, split_name: str = "") -> pd.DataFrame:
    """Đọc file chunks CSV (input=noisy, target=clean)."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Khong tim thay: {csv_path}\n"
            f"Hay chay truoc: python scripts/build_chunks.py"
        )
    df = pd.read_csv(csv_path)

    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=SEED).reset_index(drop=True)
        print(f"[INFO] {split_name}: sampled {max_samples:,} / {len(df)+max_samples:,} chunks")
    else:
        print(f"[INFO] {split_name}: {len(df):,} chunks")

    return df


def preprocess_fn(examples, tokenizer):
    """Tokenize input/target cho ByT5."""
    model_inputs = tokenizer(
        text=examples["input"],
        text_target=examples["target"],
        max_length=MAX_INPUT_LEN,
        max_target_length=MAX_TARGET_LEN,
        truncation=True,
        padding=False,
    )
    return model_inputs


class EpochLogCallback(TrainerCallback):
    """In exact match và CER sau mỗi epoch."""

    def __init__(self):
        self._train_acc_sum   = 0.0
        self._train_acc_steps = 0

    # Tích luỹ accuracy trên từng batch train
    def on_prediction_step(self, args, state: TrainerState, control: TrainerControl,
                           **kwargs):
        # Hook này chỉ gọi trong evaluate(), không dùng ở đây
        pass

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        # Không có logits ở hook này – dùng on_log thay thế
        pass

    def on_log(self, args, state: TrainerState, control: TrainerControl,
               logs=None, **kwargs):
        if logs is None:
            return
        # HuggingFace Trainer không tự log accuracy – ta tính ở compute_metrics
        # Hook này chỉ dùng để forward các giá trị đã có
        pass

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl,
                    metrics=None, **kwargs):
        """Gọi sau mỗi lần evaluate (= sau mỗi epoch)."""
        if metrics is None:
            return
        epoch = int(state.epoch) if state.epoch else "?"
        val_loss = metrics.get("eval_loss", None)
        val_exact_match = metrics.get("eval_exact_match", None)
        val_cer = metrics.get("eval_cer", None)
        print(f"\n[EPOCH {epoch}] ", end="")
        if val_loss  is not None:
            print(f"val_loss={val_loss:.4f}  ", end="")
        if val_exact_match is not None:
            print(f"val_exact_match={val_exact_match*100:.2f}%  ", end="")
        if val_cer is not None:
            print(f"val_cer={val_cer:.4f}", end="")
        print()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")

    # ── 1. Load 3 splits từ file đã build sẵn ─────────────────────
    print("\n[INFO] Loading pre-built chunk datasets...")
    train_df = load_split(TRAIN_CHUNKS, max_samples=MAX_TRAIN_SAMPLES, split_name="Train")
    val_df   = load_split(VAL_CHUNKS,   max_samples=MAX_VAL_SAMPLES,   split_name="Val  ")
    test_df  = load_split(TEST_CHUNKS,  max_samples=None,              split_name="Test ")

    print(f"\n[INFO] Sizes → Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

    # ── 2. Chuyển sang HuggingFace Dataset ────────────────────────
    train_ds = Dataset.from_pandas(train_df[["input", "target"]], preserve_index=False)
    val_ds   = Dataset.from_pandas(val_df[["input", "target"]],   preserve_index=False)
    test_ds  = Dataset.from_pandas(test_df[["input", "target"]],  preserve_index=False)

    # ── 3. Load tokenizer & model ─────────────────────────────────
    print(f"\n[INFO] Loading tokenizer & model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        tie_word_embeddings=False,
    )
    model.to(device)
    model.config.decoder_start_token_id = tokenizer.pad_token_id
    model.config.forced_bos_token_id = None

    model.generation_config.decoder_start_token_id = tokenizer.pad_token_id
    model.generation_config.forced_bos_token_id = None

    # ── 4. Tokenize (có cache) ─────────────────────────────────────
    print("[INFO] Tokenizing datasets...")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tok_fn = lambda ex: preprocess_fn(ex, tokenizer)

    train_tok = train_ds.map(
        tok_fn, batched=True,
        remove_columns=["input", "target"],
        cache_file_name=str(CACHE_DIR / f"train_{len(train_df)}.arrow"),
    )
    val_tok = val_ds.map(
        tok_fn, batched=True,
        remove_columns=["input", "target"],
        cache_file_name=str(CACHE_DIR / f"val_{len(val_df)}.arrow"),
    )
    test_tok = test_ds.map(
        tok_fn, batched=True,
        remove_columns=["input", "target"],
        cache_file_name=str(CACHE_DIR / f"test_{len(test_df)}.arrow"),
    )

    collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    # ── 5. Training arguments ──────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


    steps_per_epoch = max(1, len(train_tok) // (BATCH_SIZE * GRAD_ACCUM))
    total_steps     = steps_per_epoch * EPOCHS
    warmup_steps    = max(100, int(total_steps * WARMUP_RATIO))
    eff_batch       = BATCH_SIZE * GRAD_ACCUM
    print(f"\n[INFO] effective_batch={eff_batch} (per_device={BATCH_SIZE} x accum={GRAD_ACCUM})")
    print(f"[INFO] steps/epoch={steps_per_epoch} | total={total_steps} | warmup={warmup_steps}")

    # Chọn precision: bf16 > fp16 cho ByT5 trên T4/A100
    use_bf16 = USE_BF16 and (device == "cuda") and torch.cuda.is_bf16_supported()
    use_fp16 = (not use_bf16) and (device == "cuda")
    if device == "cuda":
        prec = "bf16" if use_bf16 else "fp16"
        print(f"[INFO] Precision: {prec}")

    # ── compute_metrics: tính token accuracy từ logits ────────────
    def compute_metrics(eval_preds):
        """
        eval_preds.predictions : token ids (batch, seq) khi predict_with_generate=True
        eval_preds.label_ids   : (batch, seq) với -100 ở padding

        Lưu ý: khi eval_accumulation_steps=1, Trainer trả về predictions
        dưới dạng list/tuple các chunk có seq_len khác nhau → NumPy không
        thể stack thành array đồng nhất → phải xử lý thủ công.
        """
        predictions, label_ids = eval_preds

        def _to_ids(obj):
            if isinstance(obj, np.ndarray):
                if obj.ndim in {2, 3}:
                    return list(obj)
                return list(obj)
            out = []
            for chunk in obj:
                chunk = np.asarray(chunk)
                if chunk.ndim >= 2:
                    out.extend(chunk)
                else:
                    out.append(chunk)
            return out

        def _pad_labels(labels):
            return [np.where(seq == -100, tokenizer.pad_token_id, seq) for seq in labels]

        def levenshtein(a: str, b: str) -> int:
            if a == b:
                return 0
            n, m = len(a), len(b)
            if n == 0:
                return m
            if m == 0:
                return n
            dp = list(range(m + 1))
            for i in range(1, n + 1):
                prev = dp[0]
                dp[0] = i
                for j in range(1, m + 1):
                    cur = dp[j]
                    if a[i - 1] == b[j - 1]:
                        dp[j] = prev
                    else:
                        dp[j] = min(prev + 1, dp[j] + 1, dp[j - 1] + 1)
                    prev = cur
            return dp[m]

        try:
            pred_list  = _to_ids(predictions)
            label_list = _to_ids(label_ids)
            label_list = _pad_labels(label_list)

            pred_texts = tokenizer.batch_decode(pred_list, skip_special_tokens=True)
            label_texts = tokenizer.batch_decode(label_list, skip_special_tokens=True)

            exact_matches = 0
            total_cer = 0.0
            total = 0
            for pred_text, label_text in zip(pred_texts, label_texts):
                pred_text = pred_text.strip()
                label_text = label_text.strip()
                exact_matches += int(pred_text == label_text)
                dist = levenshtein(label_text, pred_text)
                total_cer += dist / max(1, len(label_text))
                total += 1

            exact_match = exact_matches / total if total > 0 else 0.0
            cer = total_cer / total if total > 0 else 0.0
        except Exception as e:
            print(f"[WARN] compute_metrics error: {e}")
            exact_match = 0.0
            cer = 1.0

        return {
            "exact_match": exact_match,
            "cer": cer,
        }

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,   
        gradient_accumulation_steps=GRAD_ACCUM,
        eval_accumulation_steps=EVAL_ACCUM_STEPS,     
        gradient_checkpointing=True,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        weight_decay=0.02,
        label_smoothing_factor=0.1,   
        predict_with_generate=USE_GENERATE_EVAL,
        generation_max_length=512 if USE_GENERATE_EVAL else None,
        generation_num_beams=2 if USE_GENERATE_EVAL else 1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=3,
        logging_steps=max(10, steps_per_epoch // 5),
        report_to="none",
        seed=SEED,
        bf16=use_bf16,
        fp16=use_fp16,
        dataloader_num_workers=2,
        use_cpu=(device == "cpu"),
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=3),
        ],
    )

    # ── 6. Train ───────────────────────────────────────────────────
    print("\n[INFO] Starting training...")
    trainer.train()
    # ── 7. Đánh giá trên Test set ──────────────────────────────────
    print("\n[INFO] Evaluating on TEST set...")
    test_results = trainer.evaluate(eval_dataset=test_tok, metric_key_prefix="test")
    print(f"[TEST] test_loss = {test_results.get('test_loss', 'N/A'):.4f}")
    test_exact = test_results.get('test_exact_match', None)
    test_cer = test_results.get('test_cer', None)
    if test_exact is not None:
        print(f"[TEST] test_exact_match = {test_exact*100:.2f}%")
    if test_cer is not None:
        print(f"[TEST] test_cer = {test_cer:.4f}")
    # ── 8. Lưu model ──────────────────────────────────────────────
    print(f"\n[INFO] Saving best model to: {OUTPUT_DIR}/")
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    meta = {
        "base_model":     MODEL_NAME,
        "max_input_len":  MAX_INPUT_LEN,
        "max_target_len": MAX_TARGET_LEN,
        "chunk_size":     CHUNK_SIZE,    
        "overlap":        OVERLAP,        
        "train_samples":  len(train_df),
        "val_samples":    len(val_df),
        "test_samples":   len(test_df),
        "test_loss":      test_results.get("test_loss"),
        "epochs_trained": EPOCHS,
    }
    with open(OUTPUT_DIR / "train_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*55}")
    print(f"[OK] Training xong! Model luu tai: {OUTPUT_DIR}/")
    print(f"     test_loss     = {test_results.get('test_loss', 'N/A'):.4f}")
    test_exact = test_results.get('test_exact_match', None)
    test_cer = test_results.get('test_cer', None)
    if test_exact is not None:
        print(f"     test_exact_match = {test_exact*100:.2f}%")
    if test_cer is not None:
        print(f"     test_cer = {test_cer:.4f}")
    print(f"\n     Load lai: AutoModelForSeq2SeqLM.from_pretrained('{OUTPUT_DIR}')")
    print(f"     Test thu : python test_byt5.py")

if __name__ == "__main__":
    main()
