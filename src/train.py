"""Fine-tune a pretrained transformer (default: xlm-roberta-base) for binary
Kazakh profanity classification.

Objective: cross-entropy over 2 classes (0 = clean, 1 = profanity).
Model selection: after every epoch, predict on the validation set, find the decision
threshold that maximises F1 of the profanity class, keep the epoch with the best
validation F1 (early stopping). The test set is NOT used here.

Usage
  python -m src.train --config configs/train_config.json --data_dir data/processed --out_dir outputs
"""
import argparse
import json
import os
import platform
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from src.model_utils import LABELS, CommentDataset, make_collate, predict_proba, set_seed


def best_threshold(y, p, grid):
    rows = []
    for t in np.arange(*grid):
        pred = (p >= t).astype(int)
        rows.append({"threshold": round(float(t), 4),
                     "precision": precision_score(y, pred, zero_division=0),
                     "recall": recall_score(y, pred, zero_division=0),
                     "f1": f1_score(y, pred, zero_division=0)})
    curve = pd.DataFrame(rows)
    # highest F1; ties -> the threshold closest to 0.5
    curve["dist"] = (curve.threshold - 0.5).abs()
    best = curve.sort_values(["f1", "dist"], ascending=[False, True]).iloc[0]
    return best.drop("dist").to_dict(), curve.drop(columns="dist")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_config.json")
    ap.add_argument("--data_dir", default="data/processed")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--model_name", default=None, help="override config (e.g. a local path)")
    ap.add_argument("--num_epochs", type=int, default=None)
    ap.add_argument("--max_train_samples", type=int, default=None, help="debug only")
    args = ap.parse_args()

    cfg = json.load(open(args.config, encoding="utf8"))
    if args.model_name:
        cfg["model_name"] = args.model_name
    if args.num_epochs:
        cfg["num_epochs"] = args.num_epochs
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fp16 = bool(cfg["fp16"]) and device.type == "cuda"
    model_dir = os.path.join(args.out_dir, "model")
    res_dir = os.path.join(args.out_dir, "results")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    train = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    val = pd.read_csv(os.path.join(args.data_dir, "val.csv"))
    if args.max_train_samples:
        train = train.sample(args.max_train_samples, random_state=cfg["seed"])
    print(f"train {len(train)} (pos {train.label.sum()}), val {len(val)} (pos {val.label.sum()}), device {device}")

    tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_name"], num_labels=2, id2label=LABELS, label2id={v: k for k, v in LABELS.items()})
    model.to(device)

    ds_tr = CommentDataset(train.text, train.label, tok, cfg["max_length"], cfg["head_tokens"])
    ds_va = CommentDataset(val.text, val.label, tok, cfg["max_length"], cfg["head_tokens"])
    print(f"comments longer than {cfg['max_length']} tokens (head+tail truncated): "
          f"train {ds_tr.n_truncated}, val {ds_va.n_truncated}")
    g = torch.Generator().manual_seed(cfg["seed"])
    loader = DataLoader(ds_tr, batch_size=cfg["train_batch_size"], shuffle=True, generator=g,
                        collate_fn=make_collate(tok.pad_token_id))

    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    params = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": cfg["weight_decay"]},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(params, lr=cfg["learning_rate"])
    total = len(loader) * cfg["num_epochs"]
    sched = get_linear_schedule_with_warmup(opt, int(cfg["warmup_ratio"] * total), total)
    scaler = torch.amp.GradScaler("cuda", enabled=fp16)

    history, best_f1, bad_epochs = [], -1.0, 0
    for epoch in range(1, cfg["num_epochs"] + 1):
        model.train()
        t0, losses = time.time(), []
        for step, batch in enumerate(loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=fp16):
                loss = model(**batch).loss            # cross-entropy over the 2 classes
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())
            if step % 100 == 0:
                print(f"  epoch {epoch} step {step}/{len(loader)} loss {np.mean(losses[-100:]):.4f}")

        p_val = predict_proba(model, ds_va, tok.pad_token_id, device, cfg["eval_batch_size"], fp16)
        best, curve = best_threshold(val.label.values, p_val, cfg["threshold_grid"])
        rec = {"epoch": epoch, "train_loss": float(np.mean(losses)),
               "val_pr_auc": float(average_precision_score(val.label, p_val)),
               "val_f1_at_0.5": float(f1_score(val.label, (p_val >= 0.5).astype(int))),
               "val_best_threshold": best["threshold"], "val_precision": best["precision"],
               "val_recall": best["recall"], "val_f1": best["f1"], "seconds": round(time.time() - t0, 1)}
        history.append(rec)
        print(json.dumps(rec))

        if rec["val_f1"] > best_f1:
            best_f1, bad_epochs = rec["val_f1"], 0
            model.save_pretrained(model_dir, safe_serialization=True)
            tok.save_pretrained(model_dir)
            json.dump({"threshold": best["threshold"], "max_length": cfg["max_length"],
                       "head_tokens": cfg["head_tokens"], "truncation": cfg["truncation"],
                       "base_model": cfg["model_name"], "best_epoch": epoch, "labels": LABELS,
                       "preprocessing": "src.text_utils.normalize (PII scrub, flood shortening, whitespace)"},
                      open(os.path.join(model_dir, "inference_config.json"), "w"), indent=2)
            curve.to_csv(os.path.join(res_dir, "val_threshold_curve.csv"), index=False)
            pd.DataFrame({"id": val.id, "label": val.label, "prob": p_val}).to_csv(
                os.path.join(res_dir, "val_predictions.csv"), index=False)
            print(f"  saved best model (val F1 {best_f1:.4f} at threshold {best['threshold']})")
        else:
            bad_epochs += 1
            if bad_epochs >= cfg["early_stopping_patience"]:
                print("early stopping")
                break

    json.dump({"config": cfg, "history": history, "best_val_f1": best_f1,
               "n_train": len(train), "n_val": len(val),
               "n_truncated": {"train": ds_tr.n_truncated, "val": ds_va.n_truncated}},
              open(os.path.join(res_dir, "train_history.json"), "w"), indent=2)
    import transformers, sklearn
    json.dump({"python": platform.python_version(), "torch": torch.__version__,
               "transformers": transformers.__version__, "scikit-learn": sklearn.__version__,
               "pandas": pd.__version__, "numpy": np.__version__,
               "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"},
              open(os.path.join(res_dir, "environment.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
