"""Evaluate the saved model ONCE on the held-out test set.

Uses the threshold that was chosen on the validation set (inference_config.json).
Writes test metrics, confusion matrix, all test predictions, error lists and
subgroup breakdowns. Refuses to overwrite earlier test results unless --force,
so the test set is not used repeatedly for tuning.

Usage
  python -m src.evaluate --model_dir outputs/model --data_dir data/processed --out_dir outputs/results
"""
import argparse
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.model_utils import CommentDataset, load_inference_config, predict_proba


def scores(y, pred):
    if len(y) == 0:
        return {"n": 0}
    return {"precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)), "n": int(len(y)), "n_pos": int(np.sum(y))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="outputs/model")
    ap.add_argument("--data_dir", default="data/processed")
    ap.add_argument("--out_dir", default="outputs/results")
    ap.add_argument("--split", default="test")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_json = os.path.join(args.out_dir, f"{args.split}_metrics.json")
    if args.split == "test" and os.path.exists(out_json) and not args.force:
        raise SystemExit(f"{out_json} exists: the test set has already been evaluated. Use --force to overwrite.")

    cfg = load_inference_config(args.model_dir)
    thr = float(cfg["threshold"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device)

    df = pd.read_csv(os.path.join(args.data_dir, f"{args.split}.csv"))
    ds = CommentDataset(df.text, df.label, tok, cfg["max_length"], cfg["head_tokens"])
    p = predict_proba(model, ds, tok.pad_token_id, device, fp16=device.type == "cuda")
    df["prob"] = p
    df["pred"] = (p >= thr).astype(int)
    y, pred = df.label.values, df.pred.values

    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    m = {"split": args.split, "n_examples": int(len(df)), "n_profanity": int(y.sum()), "n_clean": int((1 - y).sum()),
         "threshold": thr, "threshold_source": "validation set (max F1)",
         "profanity_class": scores(y, pred),
         "accuracy": float((y == pred).mean()),
         "pr_auc": float(average_precision_score(y, p)), "roc_auc": float(roc_auc_score(y, p)),
         "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
         "metrics_at_threshold_0.5": scores(y, (p >= 0.5).astype(int))}

    # subgroup analysis
    has_mask = df.text.str.contains(r"\w[\*@#]+\w|\w[\*@#]{2,}", regex=True)
    latin = df.text.map(lambda t: len(re.findall(r"[a-zA-Z]", t)) > len(re.findall(r"[а-яА-ЯәіңғүұқөһӘІҢҒҮҰҚӨҺ]", t)))
    sub = {}
    for d, g in df.groupby("domain"):
        sub[f"domain={d}"] = scores(g.label.values, g.pred.values)
    part = lambda mask: scores(df.loc[mask, "label"].values, df.loc[mask, "pred"].values)
    sub["label_method=manual (ambiguous words)"] = part(df.label_method == "manual")
    neg_ru = df[(df.label == 0) & (df.ru_profanity == 1)]
    sub["clean but Russian obscenity (false-positive rate)"] = {"n": int(len(neg_ru)), "fp_rate": float(neg_ru.pred.mean()) if len(neg_ru) else None}
    sub["contains masked token"] = part(has_mask)
    sub["mostly Latin script"] = part(latin)
    m["subgroups"] = sub
    m["n_truncated_comments"] = int(ds.n_truncated)

    json.dump(m, open(out_json, "w", encoding="utf8"), indent=2, ensure_ascii=False)
    df[["id", "text", "label", "prob", "pred", "domain", "label_method", "ru_profanity"]].to_csv(
        os.path.join(args.out_dir, f"{args.split}_predictions.csv"), index=False)
    df[(df.label == 0) & (df.pred == 1)].sort_values("prob", ascending=False)[["id", "text", "prob", "label_method"]].to_csv(
        os.path.join(args.out_dir, f"{args.split}_false_positives.csv"), index=False)
    df[(df.label == 1) & (df.pred == 0)].sort_values("prob")[["id", "text", "prob", "rule_hits", "label_method"]].to_csv(
        os.path.join(args.out_dir, f"{args.split}_false_negatives.csv"), index=False)
    pd.DataFrame([[tn, fp], [fn, tp]], index=["true 0 clean", "true 1 profanity"],
                 columns=["pred 0 clean", "pred 1 profanity"]).to_csv(os.path.join(args.out_dir, f"{args.split}_confusion_matrix.csv"))

    # figures
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    cm = np.array([[tn, fp], [fn, tp]])
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
    ax.set_xticks([0, 1], ["pred clean", "pred profanity"]); ax.set_yticks([0, 1], ["true clean", "true profanity"])
    ax.set_title(f"{args.split} confusion matrix (thr={thr:.2f})")
    fig.tight_layout(); fig.savefig(os.path.join(args.out_dir, f"{args.split}_confusion_matrix.png"), dpi=150); plt.close(fig)
    pr, rc, _ = precision_recall_curve(y, p)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.plot(rc, pr); ax.set_xlabel("recall"); ax.set_ylabel("precision"); ax.set_title(f"{args.split} PR curve (AP={m['pr_auc']:.3f})")
    fig.tight_layout(); fig.savefig(os.path.join(args.out_dir, f"{args.split}_pr_curve.png"), dpi=150); plt.close(fig)

    print(json.dumps({k: m[k] for k in ["n_examples", "threshold", "profanity_class", "confusion_matrix", "pr_auc"]}, indent=2))


if __name__ == "__main__":
    main()
