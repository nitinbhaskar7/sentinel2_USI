# usi_score.py

# ══════════════════════════════════════════════════════════════════════════════
#  Urban Sustainability Index (USI) — Land Cover Based Scoring
#
#  Formula:
#    USI = Σ (class_weight × class_fraction)   clamped to [0.0, 1.0]
#
#  Weights rationale:
#    +1.0  → Forest         (best carbon sink, biodiversity)
#    +0.8  → HerbaceousVeg  (green cover, habitat)
#    +0.7  → Pasture        (open green land)
#    +0.6  → SeaLake        (clean water body)
#    +0.5  → River          (water corridor)
#    +0.3  → AnnualCrop     (productive land, mildly positive)
#    +0.2  → PermanentCrop  (agriculture, lower than annual)
#    -0.2  → Residential    (urban area, mild stress)
#    -0.5  → Highway        (fragmentation, pollution)
#    -0.8  → Industrial     (highest urban stress)
# ══════════════════════════════════════════════════════════════════════════════

from inference import CLASS_NAMES

# ── Sustainability weight per class ───────────────────────────────────────────
CLASS_WEIGHTS = {
    "Forest"               :  1.0,
    "HerbaceousVegetation" :  0.8,
    "Pasture"              :  0.7,
    "SeaLake"              :  0.6,
    "River"                :  0.5,
    "AnnualCrop"           :  0.3,
    "PermanentCrop"        :  0.2,
    "Residential"          : -0.2,
    "Highway"              : -0.5,
    "Industrial"           : -0.8,
}

# ── Status thresholds ─────────────────────────────────────────────────────────
# USI is normalised to [0, 1] after shifting from [-0.8, 1.0] raw range
THRESHOLDS = {
    "Healthy"  : 0.65,   # USI ≥ 0.65 → green
    "Moderate" : 0.45,   # USI ≥ 0.45 → yellow
    "At Risk"  : 0.25,   # USI ≥ 0.25 → orange
    # else       Critical → red
}

# ── Cover group definitions ───────────────────────────────────────────────────
GREEN_CLASSES  = {"Forest", "HerbaceousVegetation", "Pasture"}
WATER_CLASSES  = {"SeaLake", "River"}
URBAN_CLASSES  = {"Residential", "Highway", "Industrial"}
AGRI_CLASSES   = {"AnnualCrop", "PermanentCrop"}


def compute_usi(inference_results: dict) -> dict:
    """
    Takes the output dict from run_inference() and computes the USI score.

    Returns a comprehensive dict with:
        usi_raw        : raw weighted score  (can be negative)
        usi_score      : normalised 0–1 score
        usi_pct        : usi_score as percentage (0–100)
        status         : 'Healthy' / 'Moderate' / 'At Risk' / 'Critical'
        status_color   : hex colour for UI badge
        class_pct      : {class_name: percentage_of_total_tiles}
        green_cover    : % of green tiles
        water_cover    : % of water tiles
        urban_cover    : % of urban tiles
        agri_cover     : % of agriculture tiles
        dominant_class : class with most tiles
        total_tiles    : total tile count
        grid_rows      : grid rows
        grid_cols      : grid cols
        breakdown      : per-class contribution to USI score
    """
    total_tiles  = inference_results["total_tiles"]
    class_counts = inference_results["class_counts"]

    if total_tiles == 0:
        raise ValueError("No tiles found — image may be too small.")

    # ── Class fractions ───────────────────────────────────────────────────────
    class_frac = {
        cls: count / total_tiles
        for cls, count in class_counts.items()
    }

    class_pct = {
        cls: round(frac * 100, 2)
        for cls, frac in class_frac.items()
    }

    # ── Raw USI (weighted sum) ────────────────────────────────────────────────
    usi_raw = sum(
        CLASS_WEIGHTS.get(cls, 0.0) * frac
        for cls, frac in class_frac.items()
    )

    # ── Normalise to [0, 1] ───────────────────────────────────────────────────
    # Theoretical min = -0.8 (all industrial), max = +1.0 (all forest)
    # Shift: (usi_raw + 0.8) / (1.0 + 0.8)
    USI_MIN = -0.8
    USI_MAX =  1.0
    usi_score = (usi_raw - USI_MIN) / (USI_MAX - USI_MIN)
    usi_score = max(0.0, min(1.0, usi_score))   # clamp to [0, 1]
    usi_pct   = round(usi_score * 100, 2)

    # ── Status ────────────────────────────────────────────────────────────────
    if usi_score >= THRESHOLDS["Healthy"]:
        status       = "Healthy"
        status_color = "#22C55E"   # green
        status_icon  = "🟢"
    elif usi_score >= THRESHOLDS["Moderate"]:
        status       = "Moderate"
        status_color = "#EAB308"   # yellow
        status_icon  = "🟡"
    elif usi_score >= THRESHOLDS["At Risk"]:
        status       = "At Risk"
        status_color = "#F97316"   # orange
        status_icon  = "🟠"
    else:
        status       = "Critical"
        status_color = "#EF4444"   # red
        status_icon  = "🔴"

    # ── Cover group summaries ─────────────────────────────────────────────────
    green_cover = round(sum(
        class_pct.get(c, 0) for c in GREEN_CLASSES), 2)
    water_cover = round(sum(
        class_pct.get(c, 0) for c in WATER_CLASSES), 2)
    urban_cover = round(sum(
        class_pct.get(c, 0) for c in URBAN_CLASSES), 2)
    agri_cover  = round(sum(
        class_pct.get(c, 0) for c in AGRI_CLASSES),  2)

    # ── Dominant class ────────────────────────────────────────────────────────
    dominant_class = max(class_counts, key=class_counts.get)

    # ── Per-class USI contribution breakdown ─────────────────────────────────
    breakdown = {}
    for cls in CLASS_NAMES:
        weight       = CLASS_WEIGHTS.get(cls, 0.0)
        frac         = class_frac.get(cls, 0.0)
        contribution = weight * frac
        breakdown[cls] = {
            "weight"      : weight,
            "coverage_pct": round(frac * 100, 2),
            "contribution": round(contribution, 5),
            "tile_count"  : class_counts.get(cls, 0),
        }

    return {
        "usi_raw"       : round(usi_raw, 5),
        "usi_score"     : round(usi_score, 5),
        "usi_pct"       : usi_pct,
        "status"        : status,
        "status_color"  : status_color,
        "status_icon"   : status_icon,
        "class_pct"     : class_pct,
        "green_cover"   : green_cover,
        "water_cover"   : water_cover,
        "urban_cover"   : urban_cover,
        "agri_cover"    : agri_cover,
        "dominant_class": dominant_class,
        "total_tiles"   : total_tiles,
        "grid_rows"     : inference_results["grid_rows"],
        "grid_cols"     : inference_results["grid_cols"],
        "image_shape"   : inference_results["image_shape"],
        "tile_size"     : inference_results["tile_size"],
        "breakdown"     : breakdown,
    }


def print_usi_report(usi: dict) -> None:
    """Pretty-prints the full USI report to terminal."""
    print("\n" + "═" * 55)
    print("  URBAN SUSTAINABILITY INDEX REPORT")
    print("═" * 55)
    print(f"  USI Score   : {usi['usi_pct']:.2f}%  (raw: {usi['usi_raw']:.4f})")
    print(f"  Status      : {usi['status_icon']}  {usi['status']}")
    print(f"  Total Tiles : {usi['total_tiles']}  "
          f"({usi['grid_rows']} rows × {usi['grid_cols']} cols)")
    print("─" * 55)
    print(f"  🌿 Green Cover  : {usi['green_cover']:6.2f}%")
    print(f"  💧 Water Cover  : {usi['water_cover']:6.2f}%")
    print(f"  🏙  Urban Cover  : {usi['urban_cover']:6.2f}%")
    print(f"  🌾 Agri Cover   : {usi['agri_cover']:6.2f}%")
    print("─" * 55)
    print("  Per-Class Breakdown:")
    for cls, info in usi["breakdown"].items():
        if info["tile_count"] == 0:
            continue
        sign = "+" if info["contribution"] >= 0 else ""
        bar  = "█" * int(abs(info["coverage_pct"]) / 2)
        print(f"  {cls:<26} {info['coverage_pct']:5.1f}%  "
              f"w={info['weight']:+.1f}  "
              f"contrib={sign}{info['contribution']:.4f}  {bar}")
    print("═" * 55)


# ══════════════════════════════════════════════════════════════════════════════
#  Quick test — python usi_score.py <image_path>
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys
    from inference import run_inference

    if len(sys.argv) < 2:
        print("Usage: python usi_score.py <path_to_satellite_image>")
        sys.exit(1)

    image_path       = sys.argv[1]
    inference_result = run_inference(image_path)
    usi              = compute_usi(inference_result)
    print_usi_report(usi)