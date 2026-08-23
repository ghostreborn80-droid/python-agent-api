"""Load raw sales data from CSV files into pandas DataFrames."""

import pandas as pd

REQUIRED_COLUMNS = ["order_id", "date", "region", "product", "quantity", "revenue"]


def load_data(path):
    """Read a sales CSV file into a DataFrame."""
    return pd.read_csv(path)


def validate_columns(df, required=None):
    """Raise ValueError if any required column is missing from df."""
    required = required or REQUIRED_COLUMNS
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return True
