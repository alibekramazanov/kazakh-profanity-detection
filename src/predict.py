"""Predict Kazakh profanity for new comments with the saved model (no retraining).

Examples
  python -m src.predict --model_dir outputs/model --text "Өте жақсы тауар, рахмет!"
  python -m src.predict --model_dir <hf-user>/kazakh-profanity-xlmr --text "..." --text "..."
  python -m src.predict --model_dir outputs/model            # interactive: type comments, empty line to quit

In Python / a notebook cell:
  from src.predict import ProfanityDetector
  det = ProfanityDetector("outputs/model")
  det.predict(["Сапасы нашар екен", "..."])
"""
import argparse
import os

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.model_utils import LABELS, CommentDataset, load_inference_config, predict_proba


class ProfanityDetector:
    def __init__(self, model_dir, threshold=None, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).float().to(self.device)
        self.cfg = _config(model_dir)
        self.threshold = float(threshold if threshold is not None else self.cfg["threshold"])

    def predict(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        ds = CommentDataset(texts, None, self.tok, self.cfg["max_length"], self.cfg["head_tokens"])
        probs = predict_proba(self.model, ds, self.tok.pad_token_id, self.device)
        return [{"text": t, "label": int(p >= self.threshold), "label_name": LABELS[int(p >= self.threshold)],
                 "prob_profanity": round(float(p), 4), "threshold": self.threshold}
                for t, p in zip(texts, probs)]


def _config(model_dir):
    if os.path.isdir(model_dir):
        return load_inference_config(model_dir)
    # model hosted on the Hugging Face Hub: download inference_config.json next to the weights
    try:
        from huggingface_hub import hf_hub_download
        import json
        with open(hf_hub_download(model_dir, "inference_config.json"), encoding="utf8") as f:
            return json.load(f)
    except Exception:
        return {"threshold": 0.5, "max_length": 128, "head_tokens": 64}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="outputs/model", help="local folder or Hugging Face repo id")
    ap.add_argument("--text", action="append", help="comment to classify (can repeat)")
    ap.add_argument("--threshold", type=float, default=None)
    args = ap.parse_args()
    det = ProfanityDetector(args.model_dir, args.threshold)

    def show(rs):
        for r in rs:
            print(f"[{r['label']} {r['label_name']:9s}] p(profanity)={r['prob_profanity']:.3f}  {r['text']}")

    if args.text:
        show(det.predict(args.text))
    else:
        print(f"threshold = {det.threshold:.2f}. Type a Kazakh comment (empty line to quit).")
        while True:
            try:
                t = input("> ").strip()
            except EOFError:
                break
            if not t:
                break
            show(det.predict(t))


if __name__ == "__main__":
    main()
