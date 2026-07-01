import random


class WordNoise:
    @staticmethod
    def drop_word(text: str) -> str:
        words = text.split()

        if len(words) <= 1:
            return text

        idx = random.randint(0, len(words) - 1)

        del words[idx]

        return " ".join(words)