import random
import string


class CharacterNoise:
    KEYBOARD_NEIGHBORS = {
        "q": ["w", "a"],
        "w": ["q", "e", "s"],
        "e": ["w", "r", "d"],
        "r": ["e", "t", "f"],
        "t": ["r", "y", "g"],
        "y": ["t", "u", "g", "h"],
        "u": ["y", "i", "h", "j"],
        "i": ["u", "o", "j", "k"],
        "o": ["i", "p", "k", "l"],
        "p": ["o", "l"],
        "a": ["q", "s", "z"],
        "s": ["a", "w", "d", "x"],
        "d": ["s", "e", "f", "c"],
        "f": ["d", "r", "g", "v"],
        "g": ["f", "t", "h", "v", "b"],
        "h": ["g", "y", "j", "b", "n"],
        "j": ["h", "u", "k", "n", "m"],
        "k": ["j", "i", "l", "m"],
        "l": ["k", "o", "p"],
        "z": ["a", "x"],
        "x": ["z", "s", "c"],
        "c": ["x", "d", "v"],
        "v": ["c", "f", "g", "b"],
        "b": ["v", "g", "h", "n"],
        "n": ["b", "h", "j", "m"],
        "m": ["n", "j", "k"],
    }

    @staticmethod
    def random_swap(text: str) -> str:
        chars = list(text)

        if len(chars) < 2:
            return text

        idx = random.randint(0, len(chars) - 2)

        chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]

        return "".join(chars)

    @staticmethod
    def random_delete(text: str) -> str:
        if len(text) <= 1:
            return text

        idx = random.randint(0, len(text) - 1)

        return text[:idx] + text[idx + 1 :]

    @staticmethod
    def random_insert(text: str) -> str:
        idx = random.randint(0, len(text))

        random_char = random.choice(string.ascii_lowercase)

        return text[:idx] + random_char + text[idx:]

    @staticmethod
    def keyboard_typo(text: str) -> str:
        if len(text) < 1:
            return text

        for _ in range(5):
            idx = random.randint(0, len(text) - 1)
            ch = text[idx]
            lower = ch.lower()
            if lower not in CharacterNoise.KEYBOARD_NEIGHBORS:
                continue
            replacement = random.choice(CharacterNoise.KEYBOARD_NEIGHBORS[lower])
            if ch.isupper():
                replacement = replacement.upper()
            return text[:idx] + replacement + text[idx + 1 :]

        return text

    @staticmethod
    def apply_char_noise_rate(text: str, rate: float = 0.08) -> str:
        """
        Áp dụng noise per-char với xác suất `rate` cho mỗi ký tự.
        CHỈ dùng: swap / delete / typo keyboard.
        KHÔNG insert — tránh làm noisy dài hơn clean (gây lỗi dữ liệu).
        """
        if not text:
            return text

        chars = list(text)
        result = []
        i = 0
        while i < len(chars):
            ch = chars[i]
            if random.random() < rate:
                op = random.randint(0, 2)  # 0=swap, 1=delete, 2=typo (Bỏ insert)
                if op == 0 and i + 1 < len(chars):
                    # swap với ký tự kế tiếp
                    result.append(chars[i + 1])
                    result.append(ch)
                    i += 2
                    continue
                elif op == 1:
                    # xóa ký tự này
                    i += 1
                    continue
                else:
                    # keyboard typo
                    lower = ch.lower()
                    if lower in CharacterNoise.KEYBOARD_NEIGHBORS:
                        rep = random.choice(CharacterNoise.KEYBOARD_NEIGHBORS[lower])
                        if ch.isupper():
                            rep = rep.upper()
                        result.append(rep)
                    else:
                        result.append(ch)
                    i += 1
                    continue
            result.append(ch)
            i += 1
        return "".join(result)

    @staticmethod
    def apply_noise_to_prefix(text: str, prefix_len: int = 60) -> str:
        """
        Ép noise vào `prefix_len` ký tự ĐẦU TIÊN của text.
        Giải quyết vấn đề model không học được cách sửa lỗi ở đầu chunk
        vì noise uniform ít khi chạm vào đầu.

        Sẽ áp dụng ít nhất 1 trong các phép: swap, delete, insert, typo
        trực tiếp tại vị trí ngẫu nhiên trong đoạn prefix.
        """
        if len(text) < 4:
            return text

        prefix_end = min(prefix_len, len(text))
        prefix = text[:prefix_end]
        rest = text[prefix_end:]

        # Áp dụng noise rate cao hơn ở prefix để đảm bảo có thay đổi
        noisy_prefix = CharacterNoise.apply_char_noise_rate(prefix, rate=0.12)

        # Nếu sau rate-based noise vẫn không đổi gì (hiếm), force 1 operation
        if noisy_prefix == prefix:
            ops = [
                CharacterNoise.random_swap,
                CharacterNoise.random_delete,
                CharacterNoise.random_insert,
                CharacterNoise.keyboard_typo,
            ]
            noisy_prefix = random.choice(ops)(prefix)

        return noisy_prefix + rest