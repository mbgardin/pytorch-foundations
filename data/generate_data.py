"""Generate the small, fictional classification data used in the notebooks."""

import csv
import math
import random
from pathlib import Path


def main():
    rng = random.Random(486)
    rows = []
    for _ in range(600):
        x1 = rng.uniform(-3, 3)
        x2 = rng.uniform(-3, 3)
        x3 = rng.gauss(0, 1)
        # A noisy nonlinear boundary: not a measurement of actual people or events.
        score = x1 + 0.8 * math.sin(x2) + 0.5 * x3 + rng.gauss(0, 0.8)
        rows.append([round(x1, 5), round(x2, 5), round(x3, 5), int(score > 0)])
    rng.shuffle(rows)
    destination = Path(__file__).with_name("observations.csv")
    with destination.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["feature_1", "feature_2", "feature_3", "label", "split"])
        for i, row in enumerate(rows):
            split = "train" if i < 420 else "validation" if i < 510 else "test"
            writer.writerow([*row, split])
    print(f"Wrote {len(rows)} rows to {destination}")


if __name__ == "__main__":
    main()
