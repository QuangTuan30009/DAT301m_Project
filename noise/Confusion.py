import random


class OCRNoise:
    # Mapping hai chiều: chữ→số VÀ số→chữ (leet-speak style)
    OCR_MAPPING = {
        # Chữ thường → số/ký tự
        "o": "0",
        "l": "1",
        "i": "1",
        "e": "3",
        "a": "4",
        "s": "5",
        "t": "7",
        "g": "9",
        "b": "8",
       
        "0": "o",
        "1": "l",
        "3": "e",
        "4": "a",
        "5": "s",
        "7": "t",
        "9": "g",
        "8": "b",
    }

    
    VISUAL_CONFUSION = {
        "l": "I",   
        "I": "l",  
        "O": "0",
        "0": "O",
        # (xử lý ở cấp chuỗi bên dưới)
    }

    @classmethod
    def apply_ocr_noise(cls, text: str) -> str:
            
        result = []
        i = 0
        while i < len(text):
            if (
                i + 1 < len(text)
                and text[i] == "r"
                and text[i + 1] == "n"
                and random.random() < 0.20
            ):
                result.append("m")
                i += 2
                continue
            result.append(text[i])
            i += 1
        text = "".join(result)

        # Bước 2: leet-speak / số–chữ (xác suất 30% mỗi ký tự)
        chars = []
        for ch in text:
            if ch.lower() in cls.OCR_MAPPING and random.random() < 0.3:
                replacement = cls.OCR_MAPPING[ch.lower()]
                if ch.isupper() and replacement.isalpha():
                    replacement = replacement.upper()
                chars.append(replacement)
            else:
                chars.append(ch)
        text = "".join(chars)

        # Bước 3: visual confusion I↔l, O↔0 (xác suất 25% mỗi ký tự)
        chars = []
        for ch in text:
            if ch in cls.VISUAL_CONFUSION and random.random() < 0.25:
                chars.append(cls.VISUAL_CONFUSION[ch])
            else:
                chars.append(ch)

        return "".join(chars)