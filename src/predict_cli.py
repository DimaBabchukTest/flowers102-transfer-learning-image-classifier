# src/predict_cli.py
from __future__ import annotations

import sys
import torch
from PIL import Image

from infer import load_checkpoint, build_eval_transform, predict_pil_image

def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: python src/predict_cli.py <ckpt_path> <image_path>")
        raise SystemExit(2)

    ckpt_path = sys.argv[1]
    image_path = sys.argv[2]

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model, meta = load_checkpoint(ckpt_path, device)
    tfm = build_eval_transform(int(meta["image_size"]))

    img = Image.open(image_path).convert("RGB")
    pred, probs = predict_pil_image(model, img, tfm, device)

    top_prob = float(probs[pred].item())
    print({"predicted_class_index": pred, "confidence": round(top_prob, 4), "model": meta["model_name"]})

if __name__ == "__main__":
    main()
