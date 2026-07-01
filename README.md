# ByT5 Vietnamese Text Noise Correction

Fine-tune [google/byt5-small](https://huggingface.co/google/byt5-small) để sửa lỗi noise trong văn bản tiếng Anh: **OCR errors**, **leet-speak** (ký tự số thay chữ), **typo bàn phím**, **swap/delete/insert ký tự**.

## 📋 Tổng quan

| Mục | Chi tiết |
|-----|----------|
| **Base model** | `google/byt5-small` (Seq2Seq, byte-level) |
| **Task** | Sequence-to-sequence text denoising |
| **Noise types** | OCR confusion, leet-speak, keyboard typo, random swap/delete/insert, word drop |
| **Metrics** | Exact Match, CER (Character Error Rate) |

## 🏗️ Cấu trúc project

```
├── noise/                      # Noise generation package
│   ├── config.py               #   NoiseConfig dataclass
│   ├── character_noise.py      #   Swap, delete, insert, keyboard typo
│   ├── word_noise.py           #   Word drop
│   ├── Confusion.py            #   OCR / leet-speak noise
│   └── pipeline.py             #   NoisePipeline orchestrator
│
├── scripts/                    # Data preparation
│   ├── split_data.py           #   Chia data_raw → data/ (80/10/10)
│   └── build_chunks.py         #   Chunk + noise → data_train/
│
├── train_byt5.py               # Fine-tune ByT5 trên data_train/
├── eval_byt5.py                # Đánh giá model trên test set
├── test_byt5.py                # Inference + interactive mode
├── train_model(true).ipynb     # Notebook gốc (Kaggle/Colab)
│
├── byt5_correction/            # Model weights (sau khi train)
│   ├── config.json
│   ├── tokenizer_config.json
│   └── model.safetensors      
│
├── requirements.txt
└── .gitignore
```
link của model.safetensors   :https://drive.google.com/drive/folders/1u8F0Ukl0COzRHfEDJGo1cbmwYfs-gHit?usp=sharing
## 🚀 Hướng dẫn sử dụng

### 1. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 2. Chuẩn bị data

**Bước 2a** — Đặt file data gốc:

Đặt file `data_cleaned.csv` (chứa cột `full_text`) vào thư mục `data_raw/`:
```
data_raw/
└── data_cleaned.csv
```

<!-- Nếu bạn host data trên Kaggle/HuggingFace, thêm link ở đây -->
<!-- > 📥 Tải data tại: [link] -->

**Bước 2b** — Chia train/val/test:

```bash
python scripts/split_data.py
```
Kết quả: `data/train.csv`, `data/val.csv`, `data/test.csv` (tỉ lệ 80/10/10).

**Bước 2c** — Chunk + thêm noise:

```bash
python scripts/build_chunks.py
```
Kết quả: `data_train/train_chunks.csv`, `data_train/val_chunks.csv`, `data_train/test_chunks.csv`.

Mỗi file chứa 2 cột:
- `input`: text đã thêm noise
- `target`: text gốc (clean)

### 3. Training

```bash
python train_byt5.py
```


Model sẽ được lưu tại `byt5_correction/`.

### 4. Đánh giá (Evaluation)

```bash
# Đánh giá 200 samples (mặc định)
python eval_byt5.py

# Đánh giá toàn bộ test set
python eval_byt5.py --samples 0

# Tuỳ chọn
python eval_byt5.py --model byt5_correction --test data_train/test_chunks.csv --samples 500 --batch 32 --beams 4
```

### 5. Inference (Test thử)

```bash
python test_byt5.py
```

Script sẽ:
1. Chạy qua một số test case mẫu (leet-speak, typo, clean text)
2. Vào **interactive mode** — nhập text bất kỳ để sửa lỗi

Ví dụ:
```
Noisy text > De4r Pr1nc1pal, I 7hink s7udents shou1d be a11owed
Corrected  > Dear Principal, I think students should be allowed
```

## 🔧 Pipeline tổng quan

```
data_raw/data_cleaned.csv
        │
        ▼  scripts/split_data.py
data/train.csv, val.csv, test.csv
        │
        ▼  scripts/build_chunks.py (chunk + noise)
data_train/train_chunks.csv, val_chunks.csv, test_chunks.csv
        │
        ▼  train_byt5.py (fine-tune ByT5)
byt5_correction/ (saved model)
        │
        ├─▶ eval_byt5.py  (metrics: Exact Match, CER)
        └─▶ test_byt5.py  (inference + interactive)
```

## Noise types

| Loại noise | Module | Mô tả |
|-----------|--------|--------|
| Character swap | `character_noise.py` | Đổi chỗ 2 ký tự liền kề |
| Character delete | `character_noise.py` | Xoá 1 ký tự ngẫu nhiên |
| Character insert | `character_noise.py` | Chèn 1 ký tự ngẫu nhiên |
| Keyboard typo | `character_noise.py` | Thay bằng phím lân cận trên bàn phím QWERTY |
| Word drop | `word_noise.py` | Xoá 1 từ ngẫu nhiên |
| OCR / leet-speak | `Confusion.py` | `o↔0`, `i↔1`, `e↔3`, `a↔4`, `s↔5`, `t↔7`, `rn→m`, v.v. |
| Prefix noise | `character_noise.py` | Ép noise vào đầu chunk (chống bias copy-input) |

## ⚙️ Cấu hình chính

Các hyperparameter quan trọng (trong `train_byt5.py`):

| Param | Giá trị | Mô tả |
|-------|---------|--------|
| `CHUNK_SIZE` | 512 | Kích thước mỗi chunk (bytes) |
| `OVERLAP` | 80 | Overlap giữa các chunk |
| `EPOCHS` | 10 | Số epoch training |
| `LR` | 1e-4 | Learning rate |
| `WARMUP_RATIO` | 0.06 | Warmup ratio |
| `CLEAN_RATIO` | 0.40 | 40% chunks giữ clean (trong `build_chunks.py`) |
