# usi_score.py

# ══════════════════════════════════════════════════════════════════════════════
#  Urban Sustainability Index (USI) — Marull-inspired scoring
#
#  Core formula:
#    USI_raw = alpha * GHS + (1 - alpha) * (1 - UPS)
#
#  GHS (Green-Hydro Score): weighted sum of vegetation + water cover
#  UPS (Urban Pressure Score): weighted sum of built-up + highways
#
#  Normalization:
#    USI_score = sigmoid(10 * (USI_raw - 0.5))  -> [0, 1]
#    USI_pct   = USI_score * 100
#
#  Notes:
#    - Urban/rural selection drives separate weights.
#    - Highway, Industrial, Residential -> UPS
#    - All other classes -> GHS
# ══════════════════════════════════════════════════════════════════════════════

import math

from inference import CLASS_NAMES

# ── Status thresholds ─────────────────────────────────────────────────────────
# USI is normalised to [0, 1] after sigmoid; thresholds remain on [0, 1]
THRESHOLDS = {
    "Healthy"  : 0.65,   # USI ≥ 0.65 → green
    "Moderate" : 0.45,   # USI ≥ 0.45 → yellow
    "At Risk"  : 0.25,   # USI ≥ 0.25 → orange
    # else       Critical → red
}

# ── Zone-specific weights ────────────────────────────────────────────────────
ALPHA_BY_ZONE = {
    "urban": 0.60,
    "rural": 0.50,
}

GHS_WEIGHTS_BY_ZONE = {
    "urban": {
        "V_nat":  0.45,  # Forest + HerbaceousVegetation
        "V_agri": 0.15,  # AnnualCrop + PermanentCrop + Pasture
        "R_riv":  0.30,  # River
        "R_lake": 0.10,  # SeaLake
    },
    "rural": {
        "V_nat":  0.40,
        "V_agri": 0.30,
        "R_riv":  0.20,
        "R_lake": 0.10,
    },
}

UPS_WEIGHTS_BY_ZONE = {
    "urban": {
        "B": 0.80,  # Residential + Industrial
        "H": 0.20,  # Highway
    },
    "rural": {
        "B": 0.85,
        "H": 0.15,
    },
}

HABITABILITY_MIN = 0.02
VALID_ZONES = {"urban", "rural"}

# ── Cover group definitions ───────────────────────────────────────────────────
GREEN_CLASSES  = {"Forest", "HerbaceousVegetation", "Pasture"}
WATER_CLASSES  = {"SeaLake", "River"}
URBAN_CLASSES  = {"Residential", "Highway", "Industrial"}
AGRI_CLASSES   = {"AnnualCrop", "PermanentCrop"}


def _normalize_zone(selected_type: str) -> str:
    zone = (selected_type or "urban").strip().lower()
    return zone if zone in VALID_ZONES else "urban"


def _effective_weight(class_name: str, zone: str, alpha: float) -> float:
    ghs_w = GHS_WEIGHTS_BY_ZONE[zone]
    ups_w = UPS_WEIGHTS_BY_ZONE[zone]

    if class_name in {"Forest", "HerbaceousVegetation"}:
        return alpha * ghs_w["V_nat"]
    if class_name in {"AnnualCrop", "PermanentCrop", "Pasture"}:
        return alpha * ghs_w["V_agri"]
    if class_name == "River":
        return alpha * ghs_w["R_riv"]
    if class_name == "SeaLake":
        return alpha * ghs_w["R_lake"]
    if class_name in {"Residential", "Industrial"}:
        return -(1 - alpha) * ups_w["B"]
    if class_name == "Highway":
        return -(1 - alpha) * ups_w["H"]
    return 0.0


def compute_usi(inference_results: dict, selected_type: str = "urban") -> dict:
    """
    Takes the output dict from run_inference() and computes the USI score.

    Returns a comprehensive dict with:
        usi_raw        : raw combined score (0–1, before sigmoid)
        usi_score      : normalised 0–1 score (after sigmoid)
        usi_pct        : usi_score as percentage (0–100)
        status         : 'Healthy' / 'Moderate' / 'At Risk' / 'Critical'
        status_color   : hex colour for UI badge
        zone           : 'urban' / 'rural' (selected)
        out_of_scope   : True if habitability gate fails
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

    zone  = _normalize_zone(selected_type)
    alpha = ALPHA_BY_ZONE[zone]

    v_nat  = class_frac.get("Forest", 0.0) + class_frac.get("HerbaceousVegetation", 0.0)
    v_agri = (class_frac.get("AnnualCrop", 0.0)
              + class_frac.get("PermanentCrop", 0.0)
              + class_frac.get("Pasture", 0.0))
    r_riv  = class_frac.get("River", 0.0)
    r_lake = class_frac.get("SeaLake", 0.0)
    b      = class_frac.get("Residential", 0.0) + class_frac.get("Industrial", 0.0)
    h      = class_frac.get("Highway", 0.0)

    out_of_scope = (b + h) < HABITABILITY_MIN

    if out_of_scope:
        ghs = None
        ups = None
        usi_raw = None
        usi_score = None
        usi_pct = None
        status = "Out of Scope"
        status_color = "#94A3B8"
        status_icon = "⚪"
    else:
        ghs_w = GHS_WEIGHTS_BY_ZONE[zone]
        ups_w = UPS_WEIGHTS_BY_ZONE[zone]

        ghs = (
            ghs_w["V_nat"] * v_nat
            + ghs_w["V_agri"] * v_agri
            + ghs_w["R_riv"] * r_riv
            + ghs_w["R_lake"] * r_lake
        )
        ups = (
            ups_w["B"] * b
            + ups_w["H"] * h
        )

        usi_raw = alpha * ghs + (1 - alpha) * (1 - ups)
        usi_raw = max(0.0, min(1.0, usi_raw))

        usi_score = 1 / (1 + math.exp(-10 * (usi_raw - 0.5)))
        usi_score = max(0.0, min(1.0, usi_score))
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
        weight       = _effective_weight(cls, zone, alpha)
        frac         = class_frac.get(cls, 0.0)
        contribution = weight * frac
        breakdown[cls] = {
            "weight"      : round(weight, 5),
            "coverage_pct": round(frac * 100, 2),
            "contribution": round(contribution, 5),
            "tile_count"  : class_counts.get(cls, 0),
        }

    return {
        "usi_raw"       : None if usi_raw is None else round(usi_raw, 5),
        "usi_score"     : None if usi_score is None else round(usi_score, 5),
        "usi_pct"       : usi_pct,
        "status"        : status,
        "status_color"  : status_color,
        "status_icon"   : status_icon,
        "zone"          : zone,
        "out_of_scope"  : out_of_scope,
        "alpha"         : alpha,
        "ghs"           : None if ghs is None else round(ghs, 5),
        "ups"           : None if ups is None else round(ups, 5),
        "urban_pressure": round((b + h) * 100, 2),
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
    if usi.get("out_of_scope"):
        print("  USI Score   : N/A  (out of scope)")
    else:
        print(f"  USI Score   : {usi['usi_pct']:.2f}%  (raw: {usi['usi_raw']:.4f})")
    print(f"  Status      : {usi['status_icon']}  {usi['status']}")
    print(f"  Zone        : {usi.get('zone', 'urban').title()}")
    print(f"  Total Tiles : {usi['total_tiles']}  "
          f"({usi['grid_rows']} rows × {usi['grid_cols']} cols)")
    print("─" * 55)
    print(f"  🌿 Green Cover  : {usi['green_cover']:6.2f}%")
    print(f"  💧 Water Cover  : {usi['water_cover']:6.2f}%")
    print(f"  🏙  Urban Cover  : {usi['urban_cover']:6.2f}%")
    print(f"  🌾 Agri Cover   : {usi['agri_cover']:6.2f}%")
    print("─" * 55)
    if not usi.get("out_of_scope"):
        print(f"  GHS          : {usi['ghs']:.4f}")
        print(f"  UPS          : {usi['ups']:.4f}")
        print(f"  Alpha        : {usi['alpha']:.2f}")
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