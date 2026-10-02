from pathlib import Path
import pandas as pd


def load_dataset(filename):
    path = Path(__file__).parent / filename

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    required_columns = [
        "text_input",
        "palette",
        "palette_lab",
        "palette_lab_reorder"
    ]

    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(
            f"Dataset is missing columns: {missing}"
        )

    return df