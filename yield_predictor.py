"""
yield_predictor.py - Yield Prediction Engine for Apple Orchards

Uses zone_fruit_counts.csv to compute predicted yield in tons/ha and
estimates the season progression (leaf year) via linear regression against
Pink Lady reference data.

Usage:
    python yield_predictor.py [--trees_per_ha 2500] [--avg_fruit_weight_g 185]
                              [--zone_csv outputs/zone_fruit_counts.csv]
                              [--output outputs/yield_prediction.json]
"""

import argparse
import csv
import json
import os
import math


# Pink Lady reference yield data: (leaf_year, tons_per_ha)
# Leaf year = orchard age proxy (years since establishment reaching production)
REFERENCE_DATA = [
    (2016, 23),
    (2017, 57),
    (2018, 77),
    (2019, 89),
    (2020, 131),
]


def linear_regression(xs, ys):
    """Simple least-squares linear regression. Returns (slope, intercept)."""
    n = len(xs)
    sx = sum(xs)
    sy = sum(ys)
    sxy = sum(x * y for x, y in zip(xs, ys))
    sxx = sum(x * x for x in xs)
    denom = n * sxx - sx * sx
    if denom == 0:
        return 0.0, sy / n
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    return slope, intercept


def load_zone_counts(zone_csv: str) -> dict:
    """Read zone_fruit_counts.csv, return dict with zone data."""
    if not os.path.exists(zone_csv):
        raise FileNotFoundError(f"Zone CSV not found: {zone_csv}")

    zones = []
    total_fruits = 0
    with open(zone_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            count = int(row["fruit_count"])
            zones.append({
                "zone_id": int(row["zone_id"]),
                "zone_label": row["zone_label"],
                "z_min_m": float(row["z_min_m"]),
                "z_max_m": float(row["z_max_m"]),
                "fruit_count": count,
            })
            total_fruits += count

    return {"zones": zones, "total_fruits": total_fruits}


def predict_yield(zone_csv: str, output_json: str,
                  trees_per_ha: int = 2500,
                  avg_fruit_weight_g: float = 185.0) -> dict:
    """
    Compute predicted yield and estimate leaf year.

    Yield formula:
        yield_tons_ha = total_fruits × avg_fruit_weight_kg × trees_per_ha / 1000

    Leaf year estimation:
        Fit linear regression on REFERENCE_DATA,
        then invert: year = (yield - intercept) / slope

    Args:
        zone_csv:            path to zone_fruit_counts.csv
        output_json:         path to write yield_prediction.json
        trees_per_ha:        planting density
        avg_fruit_weight_g:  average single fruit weight in grams

    Returns:
        dict with all prediction results
    """
    zone_data = load_zone_counts(zone_csv)
    total_fruits = zone_data["total_fruits"]
    avg_fruit_weight_kg = avg_fruit_weight_g / 1000.0

    # Yield per tree × density
    yield_tons_ha = (total_fruits * avg_fruit_weight_kg * trees_per_ha) / 1000.0

    # Linear regression on reference data
    ref_years = [r[0] for r in REFERENCE_DATA]
    ref_yields = [r[1] for r in REFERENCE_DATA]
    slope, intercept = linear_regression(ref_years, ref_yields)

    # Predict leaf year from current yield (inverse of regression)
    if slope != 0:
        estimated_year_float = (yield_tons_ha - intercept) / slope
        estimated_year = int(round(estimated_year_float))
    else:
        estimated_year_float = ref_years[-1]
        estimated_year = ref_years[-1]

    # Clamp to plausible range with a small buffer
    estimated_year = max(ref_years[0] - 5, min(ref_years[-1] + 5, estimated_year))

    # R² of the regression for reference
    mean_y = sum(ref_yields) / len(ref_yields)
    ss_tot = sum((y - mean_y) ** 2 for y in ref_yields)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(ref_years, ref_yields))
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Zone breakdown as percentage
    zone_breakdown = []
    for z in zone_data["zones"]:
        pct = (z["fruit_count"] / total_fruits * 100) if total_fruits > 0 else 0.0
        zone_breakdown.append({
            **z,
            "percentage": round(pct, 1),
        })

    result = {
        "inputs": {
            "total_fruits_detected": total_fruits,
            "trees_per_ha": trees_per_ha,
            "avg_fruit_weight_g": avg_fruit_weight_g,
        },
        "yield": {
            "predicted_tons_ha": round(yield_tons_ha, 2),
            "formula": "total_fruits × avg_fruit_weight_kg × trees_per_ha / 1000",
        },
        "season_estimation": {
            "estimated_year": estimated_year,
            "estimated_year_exact": round(estimated_year_float, 2),
            "regression_slope": round(slope, 4),
            "regression_intercept": round(intercept, 4),
            "r_squared": round(r_squared, 4),
            "reference_data": REFERENCE_DATA,
        },
        "zone_breakdown": zone_breakdown,
    }

    os.makedirs(os.path.dirname(output_json) if os.path.dirname(output_json) else ".", exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n--- Yield Prediction Results ---")
    print(f"Total fruits detected:   {total_fruits:,}")
    print(f"Avg fruit weight:        {avg_fruit_weight_g}g")
    print(f"Trees per ha:            {trees_per_ha:,}")
    print(f"Predicted yield:         {yield_tons_ha:.2f} tons/ha")
    print(f"Estimated leaf year:     {estimated_year} (R²={r_squared:.3f})")
    print(f"\nZone breakdown:")
    for z in zone_breakdown:
        print(f"  Zone {z['zone_id']} | {z['zone_label']:<24} | "
              f"{z['fruit_count']:>6} fruits ({z['percentage']:.1f}%)")
    print(f"\nSaved: {output_json}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Apple orchard yield predictor")
    parser.add_argument("--zone_csv", default="outputs/zone_fruit_counts.csv",
                        help="Path to zone_fruit_counts.csv")
    parser.add_argument("--output", default="outputs/yield_prediction.json",
                        help="Path to write yield_prediction.json")
    parser.add_argument("--trees_per_ha", type=int, default=2500,
                        help="Tree planting density (default: 2500)")
    parser.add_argument("--avg_fruit_weight_g", type=float, default=185.0,
                        help="Average fruit weight in grams (default: 185g for Pink Lady)")
    args = parser.parse_args()

    predict_yield(
        zone_csv=args.zone_csv,
        output_json=args.output,
        trees_per_ha=args.trees_per_ha,
        avg_fruit_weight_g=args.avg_fruit_weight_g,
    )


if __name__ == "__main__":
    main()
