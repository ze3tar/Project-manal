"""
zone_classifier.py - Wire-Zone Classifier for Guyot/Multi-Leader Orchard Systems

Reads fruit_labeled.csv (x, y, z, is_fruit) and bins fruit points into
5 vertical height zones based on Z coordinate. Outputs per-zone fruit count
to outputs/zone_fruit_counts.csv.

Zone definitions (Guyot/multi-leader wires):
  Zone 0: Ground      0.0 – 0.5 m
  Zone 1: Low wire    0.5 – 1.0 m
  Zone 2: Mid wire    1.0 – 1.5 m
  Zone 3: High wire   1.5 – 2.0 m
  Zone 4: Top wire    2.0 – 2.5 m
"""

import argparse
import os
import csv
import json


ZONE_EDGES = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
ZONE_LABELS = [
    "Ground (0.0–0.5m)",
    "Low wire (0.5–1.0m)",
    "Mid wire (1.0–1.5m)",
    "High wire (1.5–2.0m)",
    "Top wire (2.0–2.5m)",
]


def classify_zones(input_csv: str, output_csv: str) -> dict:
    """
    Read fruit_labeled.csv, bin fruit points by Z into 5 zones.

    Args:
        input_csv:  path to fruit_labeled.csv (columns: x, y, z, is_fruit)
        output_csv: path to write zone_fruit_counts.csv

    Returns:
        dict with keys 'zone_counts', 'total_fruits', 'total_points',
        'out_of_range_fruits'
    """
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    zone_counts = [0] * 5
    total_points = 0
    total_fruits = 0
    out_of_range = 0

    with open(input_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_points += 1
            try:
                is_fruit = int(float(row["is_fruit"]))
            except (KeyError, ValueError):
                continue

            if not is_fruit:
                continue

            total_fruits += 1
            try:
                z = float(row["z"])
            except (KeyError, ValueError):
                out_of_range += 1
                continue

            assigned = False
            for zone_idx in range(5):
                lo = ZONE_EDGES[zone_idx]
                hi = ZONE_EDGES[zone_idx + 1]
                if lo <= z < hi:
                    zone_counts[zone_idx] += 1
                    assigned = True
                    break
            if not assigned:
                out_of_range += 1

    os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else ".", exist_ok=True)

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["zone_id", "zone_label", "z_min_m", "z_max_m", "fruit_count"])
        for i in range(5):
            writer.writerow([
                i,
                ZONE_LABELS[i],
                ZONE_EDGES[i],
                ZONE_EDGES[i + 1],
                zone_counts[i],
            ])

    result = {
        "zone_counts": zone_counts,
        "zone_labels": ZONE_LABELS,
        "total_fruits": total_fruits,
        "total_points": total_points,
        "out_of_range_fruits": out_of_range,
    }

    print(f"\n--- Wire-Zone Classification Results ---")
    print(f"Total points:          {total_points:,}")
    print(f"Total fruit points:    {total_fruits:,}")
    print(f"Out-of-range fruits:   {out_of_range}")
    print(f"\nFruit counts per zone:")
    for i in range(5):
        bar = "█" * min(zone_counts[i], 40)
        print(f"  Zone {i} | {ZONE_LABELS[i]:<24} | {zone_counts[i]:>6}  {bar}")
    print(f"\nSaved: {output_csv}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Wire-zone fruit classifier")
    parser.add_argument(
        "--input", default="outputs/fruit_labeled.csv",
        help="Path to fruit_labeled.csv (default: outputs/fruit_labeled.csv)"
    )
    parser.add_argument(
        "--output", default="outputs/zone_fruit_counts.csv",
        help="Path to write zone_fruit_counts.csv"
    )
    args = parser.parse_args()

    classify_zones(args.input, args.output)


if __name__ == "__main__":
    main()
