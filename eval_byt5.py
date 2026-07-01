"""
eval_byt5.py
============
Đánh giá model ByT5-small đã fine-tune trên test set.

Tính các metrics:
  - Exact Match
  - CER (Character Error Rate)
  - First Character Accuracy

Dựa trên notebook train_model(true).ipynb (cell 2).
Sử dụng: python eval_byt5.py [--model MODEL_PATH] [--test TEST_CSV] [--samples N] [--batch BATCH]
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ─────────────────────────────────────────────
# Cấu hình mặc định
# ─────────────────────────────────────────────
DEFAULT_MODEL  = "byt5_correction"
DEFAULT_TEST   = "data_train/test_chunks.csv"
DEFAULT_SAMPLES = 200
DEFAULT_BATCH   = 32
NUM_BEAMS       = 4


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def levenshtein(a: str, b: str) -> int:
    """Tính Levenshtein distance giữa 2 chuỗi."""
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


def safe_decode(token_ids, tokenizer) -> str:
    """Decode token IDs thành text, xử lý ByT5 byte-level tokens."""
    ids = [t for t in token_ids if t not in (0, 1)]
    byte_vals = []
    for t in ids:
        if 3 <= t <= 258:
            byte_vals.append(t - 3)
    raw_bytes = bytes(byte_vals)
    return raw_bytes.decode("utf-8", errors="replace")


def fix_first_char(pred: str, inp: str) -> str:
    """
    Sửa ký tự đầu tiên của prediction nếu bị lỗi.
    
    ByT5 đôi khi bị mất hoặc sai ký tự đầu do cách encode byte-level.
    Hàm này dùng heuristic để khôi phục từ input gốc.
    """
    if not pred or not inp:
        return pred

    LEET_MAP = {
        '0': 'o', '1': 'i', '3': 'e', '4': 'a', '5': 's',
        '6': 'g', '7': 't', '8': 'b', '9': 'g',
    }
    inp0 = inp[0]

    # Nếu ký tự đầu là replacement char (�) → model generate byte lỗi
    # → restore từ input
    if pred[0] == '\ufffd':
        first = LEET_MAP.get(inp0, inp0)
        if inp0.isupper():
            first = first.upper()
        else:
            first = first.lower()
        return first + pred[1:]

    # Nếu pred[0] khớp expected → chỉ fix case nếu cần
    expected = LEET_MAP.get(inp0, inp0)
    if pred[0].lower() == expected.lower():
        if inp0.isupper() and pred[0].islower():
            return pred[0].upper() + pred[1:]
        return pred

    # Nếu bị mất char đầu (pred bắt đầu bằng inp[1:3])
    if len(inp) >= 2 and pred[:2].lower() == inp[1:3].lower():
        first = LEET_MAP.get(inp0, inp0)
        if inp0.isupper():
            first = first.upper()
        return first + pred

    return pred


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ByT5 noise correction model")
    parser.add_argument("--model",   type=str, default=DEFAULT_MODEL,   help="Path to model directory")
    parser.add_argument("--test",    type=str, default=DEFAULT_TEST,    help="Path to test CSV")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES, help="Number of test samples (0 = all)")
    parser.add_argument("--batch",   type=int, default=DEFAULT_BATCH,   help="Batch size for inference")
    parser.add_argument("--beams",   type=int, default=NUM_BEAMS,       help="Number of beams for generation")
    parser.add_argument("--show-errors", type=int, default=20,          help="Max error examples to show")
    args = parser.parse_args()

    print("=" * 60)
    print("  ByT5-small Noise Correction — Evaluation")
    print("=" * 60)

    # ── Load model ────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Loading model: {args.model}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device)
    model.eval()

    # ── Load test data ────────────────────────────────────────────
    test_csv = Path(args.test)
    if not test_csv.exists():
        raise FileNotFoundError(
            f"Không tìm thấy: {test_csv}\n"
            f"Hãy chạy trước: python scripts/build_chunks.py"
        )

    test_df = pd.read_csv(test_csv)
    if args.samples > 0 and len(test_df) > args.samples:
        test_df = test_df.sample(n=args.samples, random_state=42).reset_index(drop=True)

    print(f"[INFO] Test samples: {len(test_df):,}\n")

    # ── Inference & Metrics ───────────────────────────────────────
    inputs_list = test_df["input"].astype(str).tolist()
    targets = test_df["target"].astype(str).tolist()

    exact_matches = 0
    total_cer = 0.0
    predictions = []

    first_char_matches = 0
    first_char_total = 0
    missing_first_char_count = 0
    wrong_first_char_examples = []
    missing_first_char_examples = []

    for start in tqdm(range(0, len(inputs_list), args.batch), desc="Evaluating"):
        batch_inputs = inputs_list[start:start + args.batch]
        batch_targets = targets[start:start + args.batch]

        encoded = tokenizer(
            batch_inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **encoded,
                max_new_tokens=512,
                num_beams=args.beams,
                early_stopping=True,
                decoder_start_token_id=tokenizer.pad_token_id,
            )

        preds = [safe_decode(out.tolist(), tokenizer) for out in outputs]

        for inp, pred, target in zip(batch_inputs, preds, batch_targets):
            pred = pred.strip()
            pred = fix_first_char(pred, inp)
            target = target.strip()
            predictions.append(pred)

            # Exact Match
            if pred == target:
                exact_matches += 1

            # CER
            dist = levenshtein(target, pred)
            total_cer += dist / max(1, len(target))

            # First Character Accuracy
            if len(pred) > 0 and len(target) > 0:
                first_char_total += 1
                if pred[0] == target[0]:
                    first_char_matches += 1
                elif len(wrong_first_char_examples) < args.show_errors:
                    wrong_first_char_examples.append({
                        "input": inp, "target": target, "pred": pred,
                    })

            # Missing first character check
            if len(target) >= 2 and pred == target[1:]:
                missing_first_char_count += 1
                if len(missing_first_char_examples) < args.show_errors:
                    missing_first_char_examples.append({
                        "input": inp, "target": target, "pred": pred,
                    })

    # ── Print Results ─────────────────────────────────────────────
    n = len(test_df)
    exact_match = exact_matches / n
    cer = total_cer / n
    first_char_acc = first_char_matches / first_char_total if first_char_total > 0 else 0

    print("\n" + "=" * 70)
    print(f"  Exact Match        : {exact_match:.4f}  ({exact_matches}/{n})")
    print(f"  CER                : {cer:.4f}")
    print(f"  First Char Accuracy: {first_char_acc:.4f}")
    print(f"  Missing First Char : {missing_first_char_count}")
    print("=" * 70)

    # ── Error analysis ────────────────────────────────────────────
    if wrong_first_char_examples:
        print(f"\nFIRST CHARACTER ERRORS ({len(wrong_first_char_examples)} shown)\n")
        for ex in wrong_first_char_examples:
            print("INPUT :", repr(ex["input"][:120]))
            print("TARGET:", repr(ex["target"][:120]))
            print("PRED  :", repr(ex["pred"][:120]))
            print()

    if missing_first_char_examples:
        print(f"\nMISSING FIRST CHARACTER EXAMPLES ({len(missing_first_char_examples)} shown)\n")
        for ex in missing_first_char_examples:
            print("INPUT :", repr(ex["input"][:120]))
            print("TARGET:", repr(ex["target"][:120]))
            print("PRED  :", repr(ex["pred"][:120]))
            print()

    # ── Detailed wrong predictions ────────────────────────────────
    print(f"\nDETAILED WRONG PREDICTIONS (top {args.show_errors})\n")
    wrong_count = 0
    for inp, pred, target in zip(inputs_list, predictions, targets):
        if pred != target:
            wrong_count += 1
            dist = levenshtein(target, pred)
            print("=" * 100)
            print("INPUT :", repr(inp))
            print("TARGET:", repr(target))
            print("PRED  :", repr(pred))
            print(f"Distance: {dist}")
            print(f"CER: {dist / max(1, len(target))}")

            if wrong_count >= args.show_errors:
                break

    print(f"\n[INFO] Evaluation complete. Total wrong: {n - exact_matches}/{n}")


if __name__ == "__main__":
    main()
