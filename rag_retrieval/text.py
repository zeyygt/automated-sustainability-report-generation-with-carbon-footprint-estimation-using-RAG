"""Text normalization shared by embedding, keyword search, and reranking."""

from __future__ import annotations

import re
import unicodedata


_TURKISH_TRANSLATION = str.maketrans(
    {
        "ı": "i",
        "İ": "i",
        "ğ": "g",
        "Ğ": "g",
        "ü": "u",
        "Ü": "u",
        "ş": "s",
        "Ş": "s",
        "ö": "o",
        "Ö": "o",
        "ç": "c",
        "Ç": "c",
    }
)


def normalize_for_search(text: str) -> str:
    """Return lowercase ASCII-ish text for robust lexical matching."""

    translated = text.translate(_TURKISH_TRANSLATION).casefold()
    decomposed = unicodedata.normalize("NFKD", translated)
    without_marks = "".join(char for char in decomposed if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", without_marks).strip()


def search_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", normalize_for_search(text))

