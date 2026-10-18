"""Sample 100 prediction rows reproducibly, keeping every CSV column."""

import argparse
import csv
import random
from pathlib import Path


SAMPLE_SIZE = 100
SEED = 42


def sample_csv(input_csv: Path, output_csv: Path) -> None:
    with input_csv.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, strict=True)
        header = next(reader, [])
        if "source" not in header:
            raise ValueError("Input must be a prediction CSV with a 'source' column.")
        rows = [row for row in reader if row]

    if any(len(row) != len(header) for row in rows):
        raise ValueError("Some rows have a different number of columns from the header.")
    if len(rows) < SAMPLE_SIZE:
        raise ValueError(f"Need at least {SAMPLE_SIZE} sentences; found {len(rows)}.")

    sampled = random.Random(SEED).sample(rows, SAMPLE_SIZE)
    with output_csv.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(sampled)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path, help="Path to the full prediction CSV.")
    args = parser.parse_args()
    output_csv = args.input_csv.with_name("error_analysis.csv")
    try:
        sample_csv(args.input_csv, output_csv)
    except (OSError, ValueError, csv.Error) as error:
        parser.error(str(error))
    print(f"Saved {SAMPLE_SIZE} sentences (seed={SEED}) to {output_csv}")


if __name__ == "__main__":
    main()
