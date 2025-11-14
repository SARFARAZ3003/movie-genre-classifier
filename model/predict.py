import os
import argparse
import joblib
import numpy as np
from typing import List, Dict, Any

# Cross-platform default path
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join("model", "model.joblib"))

def _softmax(x: np.ndarray) -> np.ndarray:
    # stable softmax for probs
    x = np.atleast_2d(x)
    x = x - x.max(axis=1, keepdims=True)
    ex = np.exp(x)
    return ex / ex.sum(axis=1, keepdims=True)

class MovieGenrePredictor:
    def __init__(self, model_path: str = MODEL_PATH):
        bundle = joblib.load(model_path)
        if isinstance(bundle, dict) and "pipeline" in bundle:
            self.pipeline = bundle["pipeline"]
            self.labeler = bundle.get("labeler")
            self.class_names = bundle.get("classes")
        else:
            self.pipeline = bundle
            self.labeler = None
            self.class_names = getattr(self.pipeline, "classes_", None)

    def predict(self, texts: List[str]) -> List[str]:
        preds = self.pipeline.predict(texts)
        if self.labeler is not None and np.issubdtype(np.array(preds).dtype, np.integer):
            preds = self.labeler.inverse_transform(preds)
        return [str(p) for p in preds]

    def predict_proba_topk(self, texts: List[str], k: int = 3) -> List[List[Dict[str, Any]]]:
        # Try true probs; else use decision_function + softmax
        probs = None
        if hasattr(self.pipeline, "predict_proba"):
            try:
                probs = self.pipeline.predict_proba(texts)
            except Exception:
                probs = None

        if probs is None:
            # LinearSVC case
            scores = self.pipeline.decision_function(texts)
            # binary case gives shape (n_samples,), expand to (n,2)
            if scores.ndim == 1:
                scores = np.vstack([-scores, scores]).T
            probs = _softmax(scores)

        classes = np.array(self.class_names) if self.class_names is not None else np.arange(probs.shape[1])

        out = []
        for row in probs:
            idx = np.argsort(row)[::-1][:k]
            out.append([{"label": str(classes[i]), "proba": float(row[i])} for i in idx])
        return out


def _cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="+")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--model", default=MODEL_PATH)
    args = ap.parse_args()

    pred = MovieGenrePredictor(args.model)
    res = pred.predict_proba_topk(args.text, k=args.topk)

    for i, r in enumerate(res, 1):
        print(f"\nINPUT {i}: {args.text[i-1]}")
        for j, it in enumerate(r, 1):
            print(f"  {j}. {it['label']}: {it['proba']:.3f}")


if __name__ == "__main__":
    _cli()
