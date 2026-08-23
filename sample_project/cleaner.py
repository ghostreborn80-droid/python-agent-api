"""Cleaning and normalization steps for the sales dataset."""
import pandas as pd

def clean_data(df):
    """Drop rows with missing revenue and compute per-order totals.

    NOTE: this function contains an INJECTED BUG (see comment below).
    """
    cleaned = df.dropna(subset=['revenue']).copy()
    cleaned['total'] = cleaned['revenue'] * cleaned['quantity']
    return cleaned

def normalize_regions(df):
    """Lowercase region names and fill missing regions with 'unknown'."""
    df = df.copy()
    df['region'] = df['region'].fillna('unknown').str.lower()
    return df