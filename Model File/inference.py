# inference.py

import json
import math
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models
from torchvision.models import ResNet50_Weights

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "model")
META_PATH  = os.path.join(MODEL_DIR, "model_meta.json")
CKPT_PATH  = os.path.join(MODEL_DIR, "best_resnet50.pth")
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TILE_SIZE  = 64     # pixels — each patch sliced from the satellite image

# ── Load metadata ─────────────────────────────────────────────────────────────
with open(META_PATH, "r") as f:
    META = json.load(f)

CLASS_NAMES   = META["class_names"]
IMG_SIZE      = META["img_size"]          # 224
MEAN          = META["imagenet_mean"]     # [0.485, 0.456, 0.406]
STD           = META["imagenet_std"]      # [0.229, 0.224, 0.225]
NUM_CLASSES   = META["num_classes"]       # 10

print(f"[inference] Device      : {DEVICE}")
print(f"[inference] Classes     : {CLASS_NAMES}")
print(f"[inference] Input size  : {IMG_SIZE}x{IMG_SIZE}")


# ══════════════════════════════════════════════════════════════════════════════
#  Model Definition — must match training architecture exactly
# ══════════════════════════════════════════════════════════════════════════════
class ResNet50Classifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        in_features   = self.backbone.fc.in_features   # 2048
        self.backbone.fc = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)


# ── Load weights ──────────────────────────────────────────────────────────────
def load_model() -> ResNet50Classifier:
    model = ResNet50Classifier(num_classes=NUM_CLASSES)
    state = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    print(f"[inference] Model loaded from {CKPT_PATH}")
    return model

MODEL = load_model()   # loaded once at import time


# ══════════════════════════════════════════════════════════════════════════════
#  Preprocessing — same pipeline as training val transform
# ══════════════════════════════════════════════════════════════════════════════
def preprocess_tile(crop_bgr: np.ndarray) -> torch.Tensor:
    """
    Takes a raw BGR numpy crop from cv2,
    resizes to IMG_SIZE x IMG_SIZE,
    normalises with ImageNet mean/std,
    returns a (1, 3, IMG_SIZE, IMG_SIZE) float tensor.
    """
    # BGR → RGB
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    # Resize to model input size
    resized  = cv2.resize(crop_rgb, (IMG_SIZE, IMG_SIZE),
                          interpolation=cv2.INTER_LINEAR)
    # Normalise
    img      = resized.astype(np.float32) / 255.0
    mean     = np.array(MEAN, dtype=np.float32)
    std      = np.array(STD,  dtype=np.float32)
    img      = (img - mean) / std
    # HWC → CHW → batch dim
    tensor   = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)
    return tensor.to(DEVICE)


# ══════════════════════════════════════════════════════════════════════════════
#  Single tile prediction
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def predict_tile(crop_bgr: np.ndarray) -> dict:
    """
    Returns:
        class_name  : predicted class string
        class_idx   : predicted class index
        confidence  : confidence % of predicted class
        all_probs   : list of 10 probabilities (%)
    """
    tensor = preprocess_tile(crop_bgr)
    logits = MODEL(tensor)                           # (1, 10)
    probs  = F.softmax(logits, dim=1).squeeze()      # (10,)
    idx    = int(probs.argmax().item())
    return {
        "class_name" : CLASS_NAMES[idx],
        "class_idx"  : idx,
        "confidence" : round(float(probs[idx].item()) * 100, 1),
        "all_probs"  : [round(float(p) * 100, 2) for p in probs],
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Tile the full satellite image & run inference on every patch
# ══════════════════════════════════════════════════════════════════════════════
def run_inference(image_path: str, tile_size: int = TILE_SIZE) -> dict:
    """
    Slices the input satellite image into non-overlapping tile_size×tile_size
    patches and runs predict_tile() on each one.

    Returns a results dict with:
        grid_rows, grid_cols  : grid dimensions
        tiles                 : flat list of per-tile dicts (row, col, x, y, prediction)
        image_shape           : (H, W) of the original image after crop-to-fit
        tile_size             : tile size used
        class_counts          : {class_name: count} across all tiles
        total_tiles           : total number of tiles
    """
    # ── Load image ────────────────────────────────────────────────────────────
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Could not load image: {image_path}")

    H, W = img_bgr.shape[:2]
    print(f"[inference] Image loaded : {W}×{H}  path={image_path}")

    # ── Crop to exact multiple of tile_size ───────────────────────────────────
    # (avoids partial edge tiles)
    H_fit = (H // tile_size) * tile_size
    W_fit = (W // tile_size) * tile_size
    img_bgr = img_bgr[:H_fit, :W_fit]

    grid_rows = H_fit // tile_size
    grid_cols = W_fit // tile_size
    total     = grid_rows * grid_cols
    print(f"[inference] Grid         : {grid_rows} rows × {grid_cols} cols = {total} tiles")

    # ── Predict each tile ─────────────────────────────────────────────────────
    tiles        = []
    class_counts = {cls: 0 for cls in CLASS_NAMES}

    for row in range(grid_rows):
        for col in range(grid_cols):
            y1 = row * tile_size
            x1 = col * tile_size
            y2 = y1 + tile_size
            x2 = x1 + tile_size

            crop   = img_bgr[y1:y2, x1:x2]
            result = predict_tile(crop)

            class_counts[result["class_name"]] += 1
            tiles.append({
                "row"        : row,
                "col"        : col,
                "x"          : x1,
                "y"          : y1,
                "class_name" : result["class_name"],
                "class_idx"  : result["class_idx"],
                "confidence" : result["confidence"],
                "all_probs"  : result["all_probs"],
            })

    print(f"[inference] Inference complete — {total} tiles processed")

    return {
        "grid_rows"   : grid_rows,
        "grid_cols"   : grid_cols,
        "tiles"       : tiles,
        "image_shape" : (H_fit, W_fit),
        "tile_size"   : tile_size,
        "class_counts": class_counts,
        "total_tiles" : total,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Quick sanity test — run directly:  python inference.py
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys
    test_path = sys.argv[1] if len(sys.argv) > 1 else None

    if test_path is None:
        # Generate a random noise image just to verify the pipeline works
        print("\n[test] No image path given — running with random noise tile...")
        dummy = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        pred  = predict_tile(dummy)
        print(f"[test] Prediction : {pred['class_name']}  ({pred['confidence']}%)")
        print("[test] All probs  :")
        for cls, prob in zip(CLASS_NAMES, pred["all_probs"]):
            print(f"         {cls:<28} {prob:.2f}%")
    else:
        results = run_inference(test_path)
        print(f"\n[test] Class distribution:")
        for cls, cnt in results["class_counts"].items():
            pct = cnt / results["total_tiles"] * 100
            print(f"  {cls:<28} {cnt:4d} tiles  ({pct:.1f}%)")