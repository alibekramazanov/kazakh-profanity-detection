"""Shared helpers: turning text into model input and getting probabilities.

Text -> model input (the same path at training, evaluation and inference):
  raw text --normalize()--> clean text --SentencePiece tokenizer--> subword ids
  --head+tail truncation--> [<s>] ids [</s>] --pad in batch--> input_ids + attention_mask
"""
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.text_utils import normalize

LABELS = {0: "clean", 1: "profanity"}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def encode(text, tokenizer, max_length=128, head_tokens=64):
    """Tokenize one comment. If it has more than max_length-2 subword tokens, keep the
    first `head_tokens` and the last (max_length-2-head_tokens) tokens: swearing is
    usually at the very start or end of a review, so this loses less than cutting the tail."""
    ids = tokenizer(normalize(text), add_special_tokens=False)["input_ids"]
    budget = max_length - 2                       # room for <s> and </s>
    truncated = len(ids) > budget
    if truncated:
        tail = budget - head_tokens
        ids = ids[:head_tokens] + ids[-tail:]
    start = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id
    end = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id
    return [start] + ids + [end], truncated       # XLM-R: <s> ... </s>


class CommentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128, head_tokens=64):
        enc = [encode(t, tokenizer, max_length, head_tokens) for t in texts]
        self.ids = [e[0] for e in enc]
        self.n_truncated = sum(e[1] for e in enc)
        self.labels = list(labels) if labels is not None else [0] * len(texts)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return self.ids[i], self.labels[i]


def make_collate(pad_id):
    def collate(batch):
        L = max(len(ids) for ids, _ in batch)
        input_ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
        mask = torch.zeros((len(batch), L), dtype=torch.long)
        for i, (ids, _) in enumerate(batch):
            input_ids[i, : len(ids)] = torch.tensor(ids)
            mask[i, : len(ids)] = 1
        labels = torch.tensor([y for _, y in batch], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": mask, "labels": labels}
    return collate


@torch.no_grad()
def predict_proba(model, dataset, pad_id, device, batch_size=64, fp16=False):
    """Probability of class 1 (profanity) = softmax(logits)[:, 1]."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=make_collate(pad_id))
    out = []
    for b in loader:
        b = {k: v.to(device) for k, v in b.items() if k != "labels"}
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=fp16 and device.type == "cuda"):
            logits = model(**b).logits
        out.append(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def load_inference_config(model_dir):
    path = os.path.join(model_dir, "inference_config.json")
    if os.path.exists(path):
        with open(path, encoding="utf8") as f:
            return json.load(f)
    return {"threshold": 0.5, "max_length": 128, "head_tokens": 64}
