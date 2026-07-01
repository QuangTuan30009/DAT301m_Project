from dataclasses import dataclass


@dataclass
class NoiseConfig:
    # Xác suất áp dụng noise cho mỗi sample (1.0 = luôn noise)
    noise_ratio: float = 1.0

    # Số lượng operations tối đa áp dụng ngẫu nhiên lên toàn chunk
    max_operations: int = 6

    # Tỉ lệ ký tự bị thay thế tối thiểu trên toàn chunk
    # Đảm bảo noise đủ mạnh để model không "copy input"
    min_char_noise_rate: float = 0.05  # ít nhất 5% ký tự bị đổi

    # Xác suất ép noise vào prefix (50 chars đầu)
    # Giải quyết vấn đề model học "copy đầu chunk"
    prefix_noise_prob: float = 0.85   # 85% samples sẽ bị noise ở đầu

    # Độ dài prefix được coi là "đầu chunk" (chars)
    prefix_len: int = 60