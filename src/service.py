from __future__ import annotations

import io
import random
from typing import Any, Dict, Tuple

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from PIL import Image
from torchvision.datasets import Flowers102
from torchvision import transforms

from infer import load_checkpoint, build_eval_transform, predict_pil_image

CKPT_PATH = "artifacts/flowers102_run_best.pt"

app = FastAPI(title="Flowers102 Classifier Demo")

device: torch.device
model: Any
tfm: Any
meta: Dict[str, Any]

# We'll keep the raw PIL images in memory by sample_id for the demo.
# (For production you'd store paths or use object storage.)
SAMPLES: Dict[str, Dict[str, Any]] = {}

def build_test_dataset(image_size: int) -> Flowers102:
    # IMPORTANT: we want the original PIL image for display, so transform=None here.
    # We will apply tfm only at inference time.
    return Flowers102(root="./data", split="test", download=True, transform=None)

test_ds: Flowers102 | None = None

@app.on_event("startup")
def startup() -> None:
    global device, model, tfm, meta, test_ds, class_names

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model, meta = load_checkpoint(CKPT_PATH, device)
    tfm = build_eval_transform(int(meta["image_size"]))

    # Load test dataset WITHOUT transforms (we want raw PIL images)
    test_ds = Flowers102(root="./data", split="test", download=True, transform=None)

    # Load class names (length = 102)
    class_names = test_ds.classes


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "device": str(device),
        "model_name": meta.get("model_name"),
        "num_classes": meta.get("num_classes"),
        "image_size": meta.get("image_size"),
        "cached_samples": len(SAMPLES),
    }

@app.get("/sample")
def sample() -> Dict[str, Any]:
    """
    Picks a random image from the test set, caches it in memory,
    and returns metadata + a URL to fetch the image bytes.
    """
    if test_ds is None:
        raise HTTPException(status_code=500, detail="Test dataset not initialized")

    idx = random.randrange(0, len(test_ds))
    img, true_label = test_ds[idx]  # img is PIL.Image, true_label is int

    sample_id = f"test_{idx}_{random.randint(1000,9999)}"
    SAMPLES[sample_id] = {"image": img, "true_label": int(true_label), "idx": idx}

    return {
        "sample_id": sample_id,
        "test_index": idx,
        "true_label": int(true_label),
        "image_url": f"/sample/{sample_id}/image",
        "predict_url": f"/predict_by_id/{sample_id}",
    }

@app.get("/sample/{sample_id}/image")
def sample_image(sample_id: str) -> Response:
    if sample_id not in SAMPLES:
        raise HTTPException(status_code=404, detail="Unknown sample_id")

    img: Image.Image = SAMPLES[sample_id]["image"]
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return Response(content=buf.getvalue(), media_type="image/jpeg")

@app.post("/predict_by_id/{sample_id}")
def predict_by_id(sample_id: str) -> Dict[str, Any]:
    if sample_id not in SAMPLES:
        raise HTTPException(status_code=404, detail="Unknown sample_id")

    img: Image.Image = SAMPLES[sample_id]["image"]
    true_label: int = int(SAMPLES[sample_id]["true_label"])

    pred_idx, probs = predict_pil_image(model, img, tfm, device)
    conf = float(probs[pred_idx].item())

    # Optional: top-5
    topk = torch.topk(probs, k=5)
    top5 = [
        {"class_index": int(i), "confidence": float(v)}
        for v, i in zip(topk.values.tolist(), topk.indices.tolist())
    ]

    return {
    "sample_id": sample_id,

    "true_label": true_label,
    "true_label_name": class_names[true_label],

    "predicted_label": pred_idx,
    "predicted_label_name": class_names[pred_idx],

    "confidence": round(conf, 6),

    "top5": [
        {
            "class_index": int(i),
            "class_name": class_names[int(i)],
            "confidence": round(float(v), 6),
        }
        for v, i in zip(topk.values, topk.indices)
    ],
    }
# curl http://127.0.0.1:8000/sample
# {"sample_id":"test_3975_9537","test_index":3975,"true_label":74,"image_url":"/sample/test_3975_9537/image","predict_url":"/predict_by_id/test_3975_9537"}%  

# http://127.0.0.1:8000/sample/<sample_id>/image

# curl -X POST http://127.0.0.1:8000/predict_by_id/<sample_id>

# curl -X POST http://127.0.0.1:8000/predict_by_id/test_3975_9537
# {"sample_id":"test_3975_9537","true_label":74,"predicted_label":74,"confidence":0.997265,"top5":[{"class_index":74,"confidence":0.9972649812698364},{"class_index":19,"confidence":0.0021987399086356163},{"class_index":75,"confidence":0.00023290578974410892},{"class_index":44,"confidence":0.00016733583470340818},{"class_index":42,"confidence":4.027093746117316e-05}]}

# Docker
# curl http://127.0.0.1:9100/sample
#{"sample_id":"test_1866_9557","test_index":1866,"true_label":45,"image_url":"/sample/test_1866_9557/image","predict_url":"/predict_by_id/test_1866_9557"}
#

# http://127.0.0.1:9100/sample/test_1866_9557/image
# curl -X POST http://127.0.0.1:9100/predict_by_id/test_1866_9557
#{"sample_id":"test_1866_9557","true_label":45,"predicted_label":45,"confidence":0.987932,"top5":[{"class_index":45,"confidence":0.9879321455955505},{"class_index":94,"confidence":0.004427148494869471},{"class_index":15,"confidence":0.004182374104857445},{"class_index":64,"confidence":0.0009252253803424537},{"class_index":4,"confidence":0.000712231791112572}]}%  