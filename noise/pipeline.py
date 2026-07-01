import random

from .config import NoiseConfig
from .character_noise import CharacterNoise
from .word_noise import WordNoise
from .Confusion import OCRNoise


class NoisePipeline:
    def __init__(self, config: NoiseConfig):
        self.config = config

    def apply_noise(self, text: str) -> str:
        """
        Áp dụng noise pipeline với hai lớp bảo vệ chống "copy input" bias:

        Lớp 1 – PREFIX NOISE (ưu tiên cao):
            Với xác suất `prefix_noise_prob`, ép noise vào đầu chunk
            để model buộc phải học sửa lỗi ở vị trí đầu.

        Lớp 2 – GLOBAL OPERATIONS:
            Chọn ngẫu nhiên k operations từ danh sách và áp lên toàn chunk.

        Lớp 3 – MIN CHAR NOISE RATE (đảm bảo cường độ tối thiểu):
            Nếu sau tất cả operations, ít hơn `min_char_noise_rate` ký tự
            bị thay đổi so với bản gốc, áp dụng thêm per-char noise.
        """
        if random.random() >= self.config.noise_ratio:
            return text

        noisy_text = text

        # ── Lớp 1: PREFIX NOISE ─────────────────────────────────────────
        # Ép noise vào đầu chunk với xác suất cao
        if random.random() < self.config.prefix_noise_prob:
            noisy_text = CharacterNoise.apply_noise_to_prefix(
                noisy_text, prefix_len=self.config.prefix_len
            )

        # ── Lớp 2: GLOBAL OPERATIONS ────────────────────────────────────
        all_operations = [
            CharacterNoise.random_swap,
            CharacterNoise.random_delete,
            CharacterNoise.random_insert,
            CharacterNoise.keyboard_typo,
            WordNoise.drop_word,
            OCRNoise.apply_ocr_noise,
        ]

        k = random.randint(2, min(self.config.max_operations, len(all_operations)))
        operations = random.sample(all_operations, k)

        for op in operations:
            noisy_text = op(noisy_text)

        # ── Lớp 3: ĐẢM BẢO CỰC KỲ QUAN TRỌNG — MIN CHAR NOISE ─────────
        # Tính tỉ lệ ký tự thực sự bị thay đổi so với bản clean gốc.
        # Nếu quá ít → model dễ học lối tắt "copy input".
        # Khi đó, áp dụng thêm per-char noise để đảm bảo đủ cường độ.
        changed = self._count_char_diff_rate(text, noisy_text)
        if changed < self.config.min_char_noise_rate:
            # Áp noise rate cao hơn mức tối thiểu một chút để có buffer
            target_rate = self.config.min_char_noise_rate + 0.03
            noisy_text = CharacterNoise.apply_char_noise_rate(noisy_text, rate=target_rate)

        return noisy_text

    @staticmethod
    def _count_char_diff_rate(original: str, noisy: str) -> float:
        """
        Ước tính tỉ lệ ký tự khác nhau giữa original và noisy.
        Dùng so sánh theo vị trí (không phải edit distance) để nhanh.
        """
        if not original:
            return 0.0
        # So sánh theo vị trí đến min length
        min_len = min(len(original), len(noisy))
        diff = sum(1 for a, b in zip(original[:min_len], noisy[:min_len]) if a != b)
        # Tính thêm phần độ dài chênh lệch
        len_diff = abs(len(original) - len(noisy))
        total_changed = diff + len_diff
        return total_changed / max(1, len(original))