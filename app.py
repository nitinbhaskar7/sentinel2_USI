# app.py

import base64
import io
import json
import os
import uuid

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image
from flask_cors import CORS

from inference import CLASS_NAMES, TILE_SIZE, run_inference
from usi_score import compute_usi

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── Colour map for land cover classes ────────────────────────────────────────
CLASS_COLORS = {
    "AnnualCrop"           : "#FFD700",   # gold
    "Forest"               : "#228B22",   # forest green
    "HerbaceousVegetation" : "#90EE90",   # light green
    "Highway"              : "#A9A9A9",   # grey
    "Industrial"           : "#FF4500",   # red-orange
    "Pasture"              : "#98FB98",   # pale green
    "PermanentCrop"        : "#DAA520",   # goldenrod
    "Residential"          : "#DEB887",   # burlywood
    "River"                : "#4169E1",   # royal blue
    "SeaLake"              : "#00BFFF",   # deep sky blue
}


def generate_land_cover_map(inference_results: dict) -> str:
    """
    Draws a colour-coded grid of tile predictions onto the original image.
    Returns base64-encoded PNG string for embedding in HTML.
    """
    H, W      = inference_results["image_shape"]
    tile_size = inference_results["tile_size"]
    tiles     = inference_results["tiles"]

    # Create blank RGBA canvas
    canvas = np.zeros((H, W, 4), dtype=np.uint8)

    for tile in tiles:
        x1  = tile["x"]
        y1  = tile["y"]
        x2  = x1 + tile_size
        y2  = y1 + tile_size
        hex_col = CLASS_COLORS.get(tile["class_name"], "#FFFFFF")

        # Hex → RGBA
        r = int(hex_col[1:3], 16)
        g = int(hex_col[3:5], 16)
        b = int(hex_col[5:7], 16)

        canvas[y1:y2, x1:x2] = [r, g, b, 180]   # alpha=180 (semi-transparent)

    img_pil = Image.fromarray(canvas, mode="RGBA")
    buf     = io.BytesIO()
    img_pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def generate_grid_preview(image_path: str, inference_results: dict) -> str:
    """
    Overlays a cyan grid on the original image to show tile boundaries.
    Returns base64-encoded JPEG string.
    """
    img = cv2.imread(image_path)
    H, W      = inference_results["image_shape"]
    tile_size = inference_results["tile_size"]
    img       = img[:H, :W].copy()

    # Draw grid lines
    for row in range(0, H, tile_size):
        cv2.line(img, (0, row), (W, row), (0, 255, 255), 1)
    for col in range(0, W, tile_size):
        cv2.line(img, (col, 0), (col, H), (0, 255, 255), 1)

    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode("utf-8")


# ══════════════════════════════════════════════════════════════════════════════
#  Routes
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/")
def index():
    return render_template("./index.html", class_colors=CLASS_COLORS,
                           class_names=CLASS_NAMES)


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    
    selected_type = request.form.get("type", "urban") 

    # ── Save uploaded image ───────────────────────────────────────────────────
    ext      = os.path.splitext(file.filename)[1].lower() or ".jpg"
    filename = f"{uuid.uuid4().hex}{ext}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        # ── Run inference ─────────────────────────────────────────────────────
        inference_results = run_inference(filepath, tile_size=TILE_SIZE)
        usi               = compute_usi(inference_results, selected_type=selected_type)

        # ── Generate visualisations ───────────────────────────────────────────
        land_cover_b64 = generate_land_cover_map(inference_results)
        grid_b64       = generate_grid_preview(filepath, inference_results)

        # ── Prepare tile data for frontend (first 120 tiles max) ─────────────
        tiles_preview = inference_results["tiles"][:120]
        tiles_data    = [
            {
                "row"        : t["row"],
                "col"        : t["col"],
                "class_name" : t["class_name"],
                "confidence" : t["confidence"],
                "color"      : CLASS_COLORS.get(t["class_name"], "#fff"),
                "all_probs"  : {CLASS_NAMES[i]: t["all_probs"][i]
                                for i in range(len(CLASS_NAMES))},
            }
            for t in tiles_preview
        ]

        return jsonify({
            "success"        : True,
            "usi"            : usi,
            "land_cover_b64" : land_cover_b64,
            "grid_b64"       : grid_b64,
            "tiles"          : tiles_data,
            "class_colors"   : CLASS_COLORS,
            "image_path"     : f"/static/uploads/{filename}",
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("Starting Sentinel-2 USI App...")
    print("Open http://127.0.0.1:5000 in your browser")
    app.run(debug=True, host="0.0.0.0", port=5000)