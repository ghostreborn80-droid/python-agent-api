"""Aggregation analytics over cleaned sales data."""

import pandas as pd


def revenue_by_region(df):
    """Total revenue per region, sorted high to low."""
    return df.groupby("region")["total"].sum().sort_values(ascending=False)


def top_products(df, n=3):
    """Top-n products by total revenue."""
    totals = df.groupby("product")["total"].sum().sort_values(ascending=False)
    return totals.head(n)


def summarize(df):
    """Headline statistics for the whole dataset."""
    return {
        "orders": int(len(df)),
        "revenue": float(df["total"].sum()),
        "avg_order": float(df["total"].mean()),
    }
