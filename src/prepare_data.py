"""Build the Kazakh profanity dataset from KazSAnDRA.

Steps
  1. load the KazSAnDRA score-classification CSVs (train/valid/test, 06/09/10)
  2. remove personal information (URLs, e-mails, @handles, phone numbers)
  3. label: strong lexicon rule -> 1; ambiguous rule -> manual decision
     (data/manual_review.csv); no rule -> 0
  4. exact de-duplication, near-duplicate grouping (MinHash on character 4-grams)
  5. keep all positives + all hard negatives + a random sample of other negatives
  6. grouped, stratified 80/10/10 split (near-duplicates never cross splits)
  7. train-only synthetic augmentation (masked / Latin-script variants), flagged

Usage
  python -m src.prepare_data --kazsandra_dir data/raw --out_dir data/processed --seed 42
"""
import argparse
import json
import os
import re

import numpy as np
import pandas as pd
from datasketch import MinHash, MinHashLSH
from sklearn.model_selection import StratifiedGroupKFold

from src.lexicon import AMBIGUOUS, STRONG, match, skeleton

FILES = ["06_sc_train_ib.csv", "09_sc_valid.csv", "10_sc_test.csv"]

# ---------------------------------------------------------------- PII removal
URL_RE = re.compile(r"(https?://\S+|www\.\S+|\b\S+\.(?:kz|com|ru|org|net)\b\S*)", re.I)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
HANDLE_RE = re.compile(r"(?<!\w)@[\w.]{2,}")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\s\-()]*){7,}(?!\d)")


def scrub_pii(text: str) -> str:
    t = str(text)
    t = EMAIL_RE.sub("<email>", t)
    t = URL_RE.sub("<url>", t)
    t = HANDLE_RE.sub("<user>", t)
    t = PHONE_RE.sub(" <phone> ", t)
    return re.sub(r"[ \t]+", " ", t).strip()


# ---------------------------------------------------------------- near-duplicates
def dedup_key(text: str) -> str:
    """Letters-only skeleton: case, script variants, punctuation, emoji and
    repeated letters are ignored, so 'Котак!!!' and 'қотаққ' share a key."""
    return re.sub(r"[^а-яa-z]", "", skeleton(text))


def near_duplicate_groups(texts, threshold=0.8, seed=42):
    n = len(texts)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:                       # smaller index becomes the root -> deterministic ids
            parent[max(ra, rb)] = min(ra, rb)

    # 1) identical letter skeletons
    first = {}
    for i, t in enumerate(texts):
        k = dedup_key(t)
        if k in first:
            union(first[k], i)
        else:
            first[k] = i
    # 2) MinHash LSH on character 4-grams of the skeleton (texts with >= 20 letters)
    lsh = MinHashLSH(threshold=threshold, num_perm=64)
    mins = {}
    for i, t in enumerate(texts):
        k = dedup_key(t)
        if len(k) < 20:
            continue
        m = MinHash(num_perm=64, seed=seed)
        for j in range(len(k) - 3):
            m.update(k[j:j + 4].encode("utf8"))
        for other in sorted(lsh.query(m), key=int):
            union(int(other), i)
        lsh.insert(str(i), m)
        mins[i] = m
    return np.array([find(i) for i in range(n)])


# ---------------------------------------------------------------- labeling
def label_rows(df, manual):
    res = df["text"].map(match)
    df["rule_hits"] = res.map(lambda r: ";".join(r[0]))
    df["ambiguous_hits"] = res.map(lambda r: ";".join(r[1]))
    df["ru_profanity"] = res.map(lambda r: int(r[2]))

    man = manual.set_index("custom_id")["manual_label"].astype(str)
    text_of = df.set_index("custom_id")["text"]
    by_text = {text_of[c]: lab for c, lab in man.items() if c in text_of.index}

    labels, methods = [], []
    for t, s, a in zip(df["text"], df["rule_hits"], df["ambiguous_hits"]):
        if s:
            labels.append(1); methods.append("rule_strong")
        elif a:
            lab = by_text.get(t)
            if lab is None:
                labels.append(-1); methods.append("unreviewed")
            elif lab == "drop":
                labels.append(-1); methods.append("manual_drop")
            else:
                labels.append(int(lab)); methods.append("manual")
        else:
            labels.append(0); methods.append("rule_negative")
    df["label"] = labels
    df["label_method"] = methods
    return df


# ---------------------------------------------------------------- synthetic train-only data
def _mask_word(w, rng):
    letters = [i for i, ch in enumerate(w) if ch.isalpha()]
    if len(letters) < 3:
        return None
    inner = letters[1:-1]
    k = max(1, min(len(inner), int(rng.integers(1, 3))))
    pos = set(rng.choice(inner, size=k, replace=False).tolist())
    sym = str(rng.choice(["*", "*", "@", "#"]))
    return "".join(sym if i in pos else ch for i, ch in enumerate(w))


_CYR2LAT = str.maketrans({
    "а": "a", "ә": "a", "б": "b", "в": "v", "г": "g", "ғ": "g", "д": "d", "е": "e", "ж": "j", "з": "z",
    "и": "i", "й": "i", "к": "k", "қ": "q", "л": "l", "м": "m", "н": "n", "ң": "n", "о": "o", "ө": "o",
    "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ұ": "u", "ү": "u", "ф": "f", "х": "h", "һ": "h",
    "ц": "ts", "ч": "ch", "ш": "sh", "щ": "sh", "ы": "y", "і": "i", "э": "e", "ю": "iu", "я": "ia",
    "ь": "", "ъ": "",
})


def _profane_token_ids(tokens):
    pats = list(STRONG.values()) + list(AMBIGUOUS.values())
    return [i for i, tok in enumerate(tokens) if any(re.search(p, skeleton(tok)) for p in pats)]


def make_synthetic(train, rng, n_mask_pos=300, n_mask_neg=300, n_latin=150):
    rows = []
    pos = train[(train.label == 1)]
    neg = train[(train.label == 0) & (train.text.str.len() > 15)]
    # (a) masked profanity -> label 1
    for _, r in pos.sample(min(n_mask_pos, len(pos)), random_state=int(rng.integers(1e9))).iterrows():
        toks = r.text.split(" ")
        ids = _profane_token_ids(toks)
        if not ids:
            continue
        i = int(rng.choice(ids))
        m = _mask_word(toks[i], rng)
        if m is None:
            continue
        toks[i] = m
        rows.append({**r.to_dict(), "text": " ".join(toks), "is_synthetic": 1,
                     "synthetic_type": "masked_profanity", "label_method": "synthetic"})
    # (b) masked ordinary word in a clean review -> label 0 (so '*' alone is not a profanity cue)
    for _, r in neg.sample(min(n_mask_neg, len(neg)), random_state=int(rng.integers(1e9))).iterrows():
        toks = r.text.split(" ")
        cand = [i for i, t in enumerate(toks) if sum(ch.isalpha() for ch in t) >= 4]
        if not cand:
            continue
        i = int(rng.choice(cand))
        m = _mask_word(toks[i], rng)
        if m is None:
            continue
        toks[i] = m
        rows.append({**r.to_dict(), "text": " ".join(toks), "is_synthetic": 1,
                     "synthetic_type": "masked_clean", "label_method": "synthetic"})
    # (c) Latin-script transliteration of whole comments, both classes
    both = pd.concat([pos.sample(min(n_latin, len(pos)), random_state=int(rng.integers(1e9))),
                      neg.sample(min(n_latin, len(neg)), random_state=int(rng.integers(1e9)))])
    for _, r in both.iterrows():
        rows.append({**r.to_dict(), "text": r.text.lower().translate(_CYR2LAT), "is_synthetic": 1,
                     "synthetic_type": "latin_translit", "label_method": "synthetic"})
    out = pd.DataFrame(rows)
    out["id"] = [f"syn{i:05d}_{p}" for i, p in enumerate(out["id"])]
    return out


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kazsandra_dir", default="data/raw")
    ap.add_argument("--manual", default="data/manual_review.csv")
    ap.add_argument("--out_dir", default="data/processed")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_market_neg", type=int, default=8000)
    ap.add_argument("--n_other_neg", type=int, default=10000)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    stats = {"seed": args.seed}

    df = pd.concat([pd.read_csv(os.path.join(args.kazsandra_dir, f)) for f in FILES], ignore_index=True)
    stats["kazsandra_rows"] = len(df)
    df = df.rename(columns={"label": "score"})
    df["score"] = df["score"] + 1                      # KazSAnDRA stores 1..5 as 0..4
    df["text"] = df["text"].astype(str).map(scrub_pii)
    df = df[df.text.str.len() > 0]

    manual = pd.read_csv(args.manual)
    df = label_rows(df, manual)
    stats["label_method_counts_full"] = df.label_method.value_counts().to_dict()
    df = df[df.label >= 0].copy()

    # exact duplicates (after lower-casing and whitespace normalisation)
    df["norm"] = df.text.str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    before = len(df)
    df = df.drop_duplicates("norm").drop(columns="norm")
    stats["exact_duplicates_removed"] = before - len(df)

    # sampling: all positives, all hard negatives, random other negatives
    pos = df[df.label == 1]
    hard = df[(df.label == 0) & ((df.ru_profanity == 1) | (df.label_method == "manual"))]
    rest = df[(df.label == 0) & ~df.index.isin(hard.index)]
    market = rest[rest.domain == "market"]
    other = rest[rest.domain != "market"]
    sel = pd.concat([
        pos, hard,
        market.sample(min(args.n_market_neg, len(market)), random_state=args.seed),
        other.sample(min(args.n_other_neg, len(other)), random_state=args.seed),
    ])
    sel = sel.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    sel["id"] = sel["custom_id"]
    sel["is_synthetic"] = 0
    sel["synthetic_type"] = ""

    groups = near_duplicate_groups(sel.text.tolist(), seed=args.seed)
    sel["group"] = groups
    multi = pd.Series(groups).value_counts()
    stats["near_duplicate_groups_with_2plus"] = int((multi > 1).sum())
    stats["rows_in_near_duplicate_groups"] = int(multi[multi > 1].sum())

    # grouped stratified split: fold 0 -> test, fold 1 -> val, rest -> train
    sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=args.seed)
    fold = np.zeros(len(sel), dtype=int)
    for k, (_, idx) in enumerate(sgkf.split(sel, sel.label, sel.group)):
        fold[idx] = k
    sel["split"] = np.where(fold == 0, "test", np.where(fold == 1, "val", "train"))
    for a in ["train", "val", "test"]:
        for b in ["train", "val", "test"]:
            if a < b:
                assert not set(sel[sel.split == a].group) & set(sel[sel.split == b].group)

    train = sel[sel.split == "train"]
    syn = make_synthetic(train, rng)
    syn["split"] = "train"
    full = pd.concat([sel, syn], ignore_index=True)

    cols = ["id", "text", "label", "split", "domain", "score", "label_method", "rule_hits",
            "ambiguous_hits", "ru_profanity", "is_synthetic", "synthetic_type", "group"]
    full = full[cols]
    for s in ["train", "val", "test"]:
        full[full.split == s].drop(columns="split").to_csv(os.path.join(args.out_dir, f"{s}.csv"), index=False)

    stats["splits"] = {
        s: {"total": int((full.split == s).sum()),
            "profanity_1": int(((full.split == s) & (full.label == 1)).sum()),
            "clean_0": int(((full.split == s) & (full.label == 0)).sum()),
            "synthetic": int(((full.split == s) & (full.is_synthetic == 1)).sum())}
        for s in ["train", "val", "test"]}
    stats["total_rows"] = int(len(full))
    stats["total_real"] = int((full.is_synthetic == 0).sum())
    stats["class_distribution_real"] = full[full.is_synthetic == 0].label.value_counts().to_dict()
    stats["synthetic_types"] = full[full.is_synthetic == 1].synthetic_type.value_counts().to_dict()
    stats["domain_counts_real"] = full[full.is_synthetic == 0].domain.value_counts().to_dict()
    stats["ru_profanity_label0_real"] = int(((full.ru_profanity == 1) & (full.label == 0) & (full.is_synthetic == 0)).sum())
    stats["label_method_counts_selected"] = full.label_method.value_counts().to_dict()
    with open(os.path.join(args.out_dir, "dataset_stats.json"), "w", encoding="utf8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2, default=int)
    print(json.dumps(stats, ensure_ascii=False, indent=2, default=int))


if __name__ == "__main__":
    main()
