"""Rule-based Kazakh profanity lexicon used ONLY to pre-label data (weak labels).

The trained model never sees these rules. They produce candidate labels that are
then checked by hand for the ambiguous patterns (see data/manual_review.csv).

Matching is done on a "skeleton" of the text so that one rule covers
Kazakh Cyrillic, Russian-keyboard spellings and Latin transliteration:
  қ,к,q,k -> к   ғ,г,g -> г   ө,о,o -> о   ұ,ү,у,u,w -> у   ң,н,n -> н
  ә,а,a -> а     і,и,i -> и  (ы stays ы)   repeated letters collapsed (ккк -> к)
"""
import re

_LATIN_DIGRAPHS = [("zh", "ж"), ("sh", "ш"), ("ch", "ч"), ("dj", "ж"), ("ng", "н")]
_CHAR_MAP = {
    "қ": "к", "q": "к", "k": "к", "ғ": "г", "g": "г", "ө": "о", "o": "о",
    "ұ": "у", "ү": "у", "u": "у", "w": "у", "ң": "н", "n": "н", "ә": "а", "a": "а",
    "і": "и", "i": "и", "һ": "х", "h": "х", "ё": "е", "e": "е", "э": "е",
    "s": "с", "c": "с", "t": "т", "b": "б", "m": "м", "l": "л", "p": "п", "j": "ж",
    "z": "з", "d": "д", "r": "р", "y": "ы", "v": "в", "f": "ф", "x": "х",
    "ь": "", "ъ": "",
    # Kazakh Latin alphabet (2021) letters
    "ı": "ы", "ǵ": "г", "ý": "у", "ó": "о", "á": "а", "ń": "н", "ú": "у", "ş": "ш", "ç": "ч",
}
MASK_CHARS = "*@#$%"
MASKABLE = ("котак", "сигейин", "сигейн", "сигем", "сиктим", "амынды", "амшык", "жалаптар", "каншык", "ташак")
_LETTERS = "а-я"


def skeleton(text: str) -> str:
    t = str(text).lower()
    for a, b in _LATIN_DIGRAPHS:
        t = t.replace(a, b)
    t = "".join(_CHAR_MAP.get(ch, ch) for ch in t)
    t = re.sub(r"(.)\1+", r"\1", t)          # collapse repeated characters
    return t


B = rf"(?<![{_LETTERS}])"   # left word boundary on the skeleton
E = rf"(?![{_LETTERS}])"    # right word boundary

# Prefixes that glue onto the verb in curses: шешеңді-сігейін, анаңды-..., әкеңді-...
_CURSE_PREFIX = r"(?:(?:шеше|ана|ене|аке|ата)[а-я]{0,5})"   # шешеңді-, шешес-, анаңды-, әкеңді- ...
_SIK_SUFFIX = r"(?:ейин|ейн|еин|ей|ем|емин|ин|ил|илип|ип|ис|исетин|ти|тим|кен|пе|у|им|ер)"

# strong rules: auto-label 1
STRONG = {
    "котак": r"кота[кг]",                                    # қотақ, қотағым, котак, qotaq, қотақбас...
    "сик_verb": rf"(?:{B}|{_CURSE_PREFIX})си[кг]{_SIK_SUFFIX}{E}"
                rf"|(?:{B}|{_CURSE_PREFIX})сыг(?:ейин|ейн|еин|ем){E}",
    "ам_forms": rf"{B}ам(?:ынды|ына|ыны|шык|жалак|дар|ды){E}",
    "жалап_pl": r"жалап(?:тар|тар[а-я]*|сын|сын[а-я]*)",
    "каншык": rf"(?<!кара)каншык",
    "котин": rf"{B}кот(?:ине|инди|ин|ке){E}",
    "ташак": r"ташак",
    "curse_auzyn": rf"(?:аке|ата|шеше|ене|ана)[{_LETTERS}]{{0,6}}\s+аузын\s+(?:си[кг]|с[кг]|урайн|урайын|урам)",
}

# ambiguous rules: need a human decision (homonyms / meta usage / Russian homographs)
AMBIGUOUS = {
    "сгин_скин": rf"{B}с[кг]ин{E}",                          # Russian "сгинь", game "скин"
    "ам_short": rf"{B}ам(?:ын|ы|сын){E}",                     # also "аминь", typos of "аман"
    "жалап_sg": rf"{B}(?:д?жалап){E}",                       # also gerund of жалау "to lick"
    "бок": rf"{B}бок(?:ка|тан|пен|сын|тай|ты|тын|ке)?{E}",   # Russian "бок" (side)
    "боктау": r"{B}бокта".replace("{B}", B),                 # verb "to swear": often meta talk
    "кот": rf"{B}кот{E}",                                     # Russian "кот" (cat)
    "котен": rf"{B}котен(?:ге|ди|ин)?{E}",                   # Russian "котёнок"
}

# Russian obscenities (out of scope for label 1, kept as a flag)
RU_MAT = r"за[еий]?бал|заибал|питараз|пидараз|ху[йеяию]|пизд|пизж|бля|ебат|ебан|ебал|еба[лнт]|{B}еб[уи]|сука|сучк|гондон|долбае|далбае|пидор|мудак|залуп".replace("{B}", B)


def _masked_hits(text: str):
    """Masked tokens such as 'қ*тақ', 'с**ейін', 'қот@қбас': letters are kept and each
    mask character becomes a wildcard for 0-3 letters. The token must match the START
    of a profane word (suffixes allowed), so 'мың$' does not match 'амынды'."""
    hits = []
    for tok in re.findall(r"\S+", str(text).lower()):
        tok = tok.strip(".,!?;:()\"'«»")
        if not any(m in tok for m in MASK_CHARS) or re.search(r"\d", tok):
            continue
        if len(re.findall(r"[^\W\d_]", tok)) < 2 or not re.match(r"[^\W\d_]", tok):
            continue
        sk = skeleton(tok)
        pat = "".join(f"[{_LETTERS}]{{0,3}}" if ch in MASK_CHARS else re.escape(ch) for ch in sk)
        for word in MASKABLE:
            if re.fullmatch(pat + f"[{_LETTERS}]*", word) or re.fullmatch(pat, word[: len(sk) + 3]):
                hits.append(f"masked:{word}")
                break
    return hits


def match(text: str):
    """Return (strong_hits, ambiguous_hits, ru_mat_flag)."""
    sk = skeleton(text)
    strong = [k for k, p in STRONG.items() if re.search(p, sk)]
    strong += _masked_hits(text)
    amb = [k for k, p in AMBIGUOUS.items() if re.search(p, sk)]
    ru = bool(re.search(RU_MAT, sk))
    return strong, amb, ru
