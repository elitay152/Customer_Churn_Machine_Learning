"""Export a cleaned, numerically encoded CSV from the raw customer data."""

import argparse
from pathlib import Path

import pandas as pd


def prepare(source, destination):
    frame = pd.read_csv(source, index_col=0)
    cleaned = pd.get_dummies(
        frame.drop(columns=["CustomerId", "Surname"]).drop_duplicates(),
        drop_first=True, dtype=int,
    )
    output = cleaned.to_csv().encode()
    destination = Path(destination)
    if destination.exists():
        existing = pd.read_csv(destination, index_col=0)
        pd.testing.assert_frame_equal(existing, cleaned, check_dtype=False)
        return  # Preserve equivalent existing data.
    destination.write_bytes(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="Path to Kaggle's Churn_Modelling.csv")
    parser.add_argument("--output", default="Cleaned_data.csv")
    args = parser.parse_args()
    prepare(args.source, args.output)
