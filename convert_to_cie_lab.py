"""Append OpenCV CIE LAB vector columns to stabilized palette datasets."""

import argparse
import ast
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


DEFAULT_INPUTS = [
    Path("palette_and_text_train_balanced.csv"),
    Path("palette_and_text_val_normalized.csv"),
    Path("palette_and_text_test_normalized.csv"),
]
RGB_COLUMN = "palette"
LAB_COLUMNS = ("CIE_L", "CIE_a", "CIE_b")


def palette_rgb_to_cie_columns(value):
    """Convert one serialized RGB palette into three serialized LAB vectors."""
    palette_rgb = np.asarray(ast.literal_eval(str(value)), dtype=np.uint8)
    if palette_rgb.ndim != 2 or palette_rgb.shape[1] != 3:
        raise ValueError("Each palette must contain colors shaped as [R, G, B]")

    palette_lab = cv2.cvtColor(
        palette_rgb.reshape(1, -1, 3),
        cv2.COLOR_RGB2LAB
    ).reshape(-1, 3)

    return {
        column: json.dumps(palette_lab[:, index].tolist(), separators=(",", ":"))
        for index, column in enumerate(LAB_COLUMNS)
    }


def convert_dataset(input_path, output_path):
    dataframe = pd.read_csv(input_path)
    if RGB_COLUMN not in dataframe.columns:
        raise ValueError(f"{input_path} must contain a {RGB_COLUMN!r} column")

    converted = dataframe[RGB_COLUMN].map(palette_rgb_to_cie_columns)
    for column in LAB_COLUMNS:
        dataframe[column] = converted.map(lambda values: values[column])

    dataframe.to_csv(output_path, index=False)
    print(f"Converted {input_path} -> {output_path} ({len(dataframe)} rows)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        default=DEFAULT_INPUTS,
        help="Input CSV files; defaults to the stabilized train, validation, and test files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory for converted files.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for input_path in args.inputs:
        output_path = args.output_dir / f"{input_path.stem}_cie_lab.csv"
        convert_dataset(input_path, output_path)


if __name__ == "__main__":
    main()
