"""
build_chunks.py
===============
Chunk + thêm noise cho 3 tập train / val / test,
lưu kết quả vào thư mục data_train/ để dùng cho training.

Chạy script này MỘT LẦN trước khi train:
    python scripts/build_chunks.py

Đầu ra:
    data_train/train_chunks.csv   (input=noisy, target=clean)
    data_train/val_chunks.csv
    data_train/test_chunks.csv
"""

import random
import sys
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd

# Thêm project root vào sys.path để import noise module
sys.path.insert(0, str(Path(__file__).parent.parent))

from noise.config import NoiseConfig
from noise.pipeline import NoisePipeline

# ─────────────────────────────────────────────
# Cấu hình — phải khớp với train_byt5.py
# ─────────────────────────────────────────────
INPUT_DIR  = Path("data")
OUTPUT_DIR = Path("data_train")
TEXT_COL   = "full_text"    # cột chứa clean text trong data/ (từ data_cleaned.csv)

CHUNK_SIZE = 512            # ký tự mỗi chunk
OVERLAP    = 80             # overlap giữa chunk để giữ ngữ cảnh
MAX_TRIES  = 10             # số lần thử tạo noise khác clean (tăng để đảm bảo noise thực sự khác)
SEED       = 42

SPLITS = [
    ("train.csv", "train_chunks.csv"),
    (   "val.csv",   "val_chunks.csv"),
    (  "test.csv",  "test_chunks.csv"),
]


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def chunk_text(text: str) -> list[str]:
    """Cắt văn bản dài thành các chunk nhỏ có overlap.

    Luôn cắt tại ranh giới từ (khoảng trắng gần nhất) để tránh
    chunk bắt đầu/kết thúc giữa một từ đang bị cắt đôi.
    """
    chunks, start = [], 0
    n = len(text)
    while start < n:
        end = min(start + CHUNK_SIZE, n)

        
        if end < n:
            snap = text.rfind(' ', start, end)
            if snap > start:          
                end = snap            

        c = text[start:end].strip()
        if len(c) >= 10:
            chunks.append(c)

        if end >= n:
            break

       
        next_start = end - OVERLAP
        snap_fwd = text.find(' ', next_start)
        if 0 < snap_fwd < end:
            start = snap_fwd + 1     
        else:
            start = end             
    return chunks



CLEAN_RATIO = 0.40

def build_chunks(csv_path: Path, pipeline: NoisePipeline, split_name: str) -> pd.DataFrame:
    """Đọc split CSV, chunk từng bài, thêm noise, trả về DataFrame.
    
    Mỗi chunk sẽ có CLEAN_RATIO xác suất được giữ nguyên (clean→clean)
    để model học được "nếu input đã sạch thì không sửa gì cả".
    """
    df = pd.read_csv(csv_path)
    rows = []

    for text in df[TEXT_COL].astype(str):
        for chunk in chunk_text(text):
            # 10% giữ nguyên (clean pass-through), 90% thêm noise
            if random.random() < CLEAN_RATIO:
                rows.append({"input": chunk, "target": chunk})
                continue

            noisy = chunk
            for _ in range(MAX_TRIES):
                candidate = pipeline.apply_noise(chunk)
                if candidate != chunk:
                    noisy = candidate
                    break
            rows.append({"input": noisy, "target": chunk})

    result = pd.DataFrame(rows)
    result = result.sample(frac=1, random_state=SEED).reset_index(drop=True)

    print(f"  [{split_name}] {len(df):,} docs -> {len(result):,} chunks")
    return result


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main() -> None:
    random.seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Khởi tạo noise pipeline
    config = NoiseConfig(
        noise_ratio=1.0,          # luôn áp noise
        max_operations=4,         # tối đa 4 operations/chunk
        min_char_noise_rate=0.03, # ít nhất 5% ký tự bị đổi → chống copy-input
        prefix_noise_prob=0.05,   # 5% chunk đầu có noise (giữ lại để tương thích API, không dùng trực tiếp)
        prefix_len=20,            # 20 chars đầu được ưu tiên noise
    )
    pipeline = NoisePipeline(config)

    print(f"[INFO] Building chunks (CHUNK_SIZE={CHUNK_SIZE}, OVERLAP={OVERLAP})...\n")

    for in_file, out_file in SPLITS:
        in_path  = INPUT_DIR  / in_file
        out_path = OUTPUT_DIR / out_file

        if not in_path.exists():
            print(f"  [SKIP] {in_path} not found")
            continue

        df_chunks = build_chunks(in_path, pipeline, split_name=in_file)
        df_chunks.to_csv(out_path, index=False, encoding="utf-8")
        print(f"  Saved → {out_path}\n")

    print("[DONE] Tat ca chunks da duoc luu vao data_train/")
    print("       Tiep theo chay: python train_byt5.py")


if __name__ == "__main__":
    main()
