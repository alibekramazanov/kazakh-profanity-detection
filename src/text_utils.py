"""Text preprocessing shared by training, evaluation and inference.

Kept deliberately light, because the subword tokenizer of the pretrained model
already handles punctuation, emoji, Cyrillic/Latin script and unknown words.
"""
import re

from src.prepare_data import scrub_pii


def normalize(text: str) -> str:
    """1) remove personal info (URLs, e-mails, @handles, phone numbers)
    2) shorten character floods: 'қотааааақ' -> 'қотааақ', '!!!!!!' -> '!!!'
    3) collapse whitespace / line breaks.
    Case is kept (the pretrained tokenizer is case-sensitive); punctuation and
    masks such as '*' and '@' are kept because they carry signal ('қ*тақ')."""
    t = scrub_pii(str(text))
    t = re.sub(r"(.)\1{3,}", r"\1\1\1", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t
