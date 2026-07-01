"""
test_byt5.py
============
Inference với model ByT5-small đã fine-tune.

Load model từ thư mục byt5_correction/ và sửa lỗi noise trong text.
Hỗ trợ text dài (tự động chunk → sửa → ghép lại).
"""

import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


# ─────────────────────────────────────────────
# Cấu hình
# ─────────────────────────────────────────────
MODEL_DIR  = Path("byt5_correction")
CHUNK_SIZE = 512
OVERLAP    = 80

LEET_MAP = {
    '0': 'o', '1': 'i', '3': 'e', '4': 'a', '5': 's',
    '6': 'g', '7': 't', '8': 'b', '9': 'g',
}


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def load_model(model_dir: Path):
    """Load model và tokenizer từ thư mục đã lưu."""
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Không tìm thấy model tại '{model_dir}'.\n"
            f"Hãy chạy: python train_byt5.py  trước!"
        )
    
    # Đọc metadata nếu có
    meta_path = model_dir / "train_meta.json"
    meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"[INFO] Model: {meta.get('base_model', 'byt5')}")
        print(f"       Trained on {meta.get('train_samples', '?')} samples")

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model     = AutoModelForSeq2SeqLM.from_pretrained(str(model_dir))
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    return tokenizer, model, device


def safe_decode(token_ids, tokenizer) -> str:
    """Decode token IDs thành text, xử lý ByT5 byte-level tokens.
    
    ByT5 encode bytes thành token IDs: byte_value + 3.
    Token 0 = pad, 1 = eos → bỏ qua.
    """
    ids = [t for t in token_ids if t not in (0, 1)]
    byte_vals = [t - 3 for t in ids if 3 <= t <= 258]
    return bytes(byte_vals).decode("utf-8", errors="replace")


def fix_first_char(pred: str, inp: str) -> str:
    """
    Sửa ký tự đầu tiên của prediction nếu bị lỗi.
    
    ByT5 đôi khi bị mất hoặc sai ký tự đầu do cách encode byte-level.
    Hàm này dùng heuristic để khôi phục từ input gốc.
    """
    if not pred or not inp:
        return pred

    inp0 = inp[0]

    # Nếu ký tự đầu là replacement char (�) → model generate byte lỗi
    # → restore từ input
    if pred[0] == '\ufffd':
        first = LEET_MAP.get(inp0, inp0)
        return (first.upper() if inp0.isupper() else first.lower()) + pred[1:]

    # Nếu pred[0] khớp expected → chỉ fix case nếu cần
    expected = LEET_MAP.get(inp0, inp0)
    if pred[0].lower() == expected.lower():
        if inp0.isupper() and pred[0].islower():
            return pred[0].upper() + pred[1:]
        return pred

    # Nếu bị mất char đầu (pred bắt đầu bằng inp[1:3])
    if len(inp) >= 2 and pred[:2].lower() == inp[1:3].lower():
        first = LEET_MAP.get(inp0, inp0)
        return (first.upper() if inp0.isupper() else first) + pred

    return pred


# ─────────────────────────────────────────────
# Chunking
# ─────────────────────────────────────────────

def make_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list[str]:
    """Cắt text thành các chunk theo byte boundary, giống build_chunks.py."""
    chunks = []
    start = 0
    encoded = text.encode("utf-8")
    while start < len(encoded):
        end = start + chunk_size
        chunk_bytes = encoded[start:end]
        # Tránh cắt giữa multibyte UTF-8
        while len(chunk_bytes) > 0:
            try:
                chunk_bytes.decode("utf-8")
                break
            except UnicodeDecodeError:
                chunk_bytes = chunk_bytes[:-1]
        chunks.append(chunk_bytes.decode("utf-8"))
        start += chunk_size - overlap
    return chunks


# ─────────────────────────────────────────────
# Correct batch
# ─────────────────────────────────────────────

def correct_batch(
    chunks: list[str],
    tokenizer,
    model,
    device: str,
    num_beams: int = 4,
    batch_size: int = 8,
) -> list[str]:
    """Correct một batch chunks, dùng safe_decode + fix_first_char."""
    preds = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=CHUNK_SIZE,
        ).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **encoded,
                max_new_tokens=CHUNK_SIZE,
                num_beams=num_beams,
                early_stopping=True,
            )
        for inp, out in zip(batch, outputs):
            pred = safe_decode(out.tolist(), tokenizer).strip()
            pred = fix_first_char(pred, inp)
            preds.append(pred)
    return preds


# ─────────────────────────────────────────────
# Stitch
# ─────────────────────────────────────────────

def stitch(preds: list[str], overlap: int = OVERLAP) -> str:
    """Ghép các chunk predictions lại, dùng overlap để vá phần đầu bị lỗi."""
    if not preds:
        return ""
    if len(preds) == 1:
        return preds[0]
    result = preds[0]
    for pred in preds[1:]:
        body = pred[overlap:] if len(pred) > overlap else ""
        result = result + body
    return result


# ─────────────────────────────────────────────
# Pipeline chính
# ─────────────────────────────────────────────

def correct_text(
    text: str,
    tokenizer,
    model,
    device: str,
    num_beams: int = 4,
) -> str:
    """
    Nhận 1 đoạn văn noisy bất kỳ, trả về văn bản đã sửa.

    Args:
        text     : đoạn văn thô cần sửa
        num_beams: beam search width (4 = cân bằng, 1 = nhanh hơn)

    Returns:
        Văn bản đã được sửa lỗi
    """
    chunks = make_chunks(text)
    if not chunks:
        return text
    preds = correct_batch(chunks, tokenizer, model, device, num_beams=num_beams)
    return stitch(preds)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  ByT5-small Noise Correction — Test Script")
    print("=" * 60)

    # Load model
    tokenizer, model, device = load_model(MODEL_DIR)
    print(f"[INFO] Device: {device}\n")


    print("─" * 60)
    print("Interactive mode (nhập text để sửa, Ctrl+C để thoát):\n")
    try:
        while True:
            user_input = input("Noisy text > ").strip()
            if not user_input:
                continue
            result = correct_text(user_input, tokenizer, model, device)
            print(f"Corrected  > {result}\n")
    except (KeyboardInterrupt, EOFError):
        print("\n[INFO] Kết thúc.")


if __name__ == "__main__":
    main()
