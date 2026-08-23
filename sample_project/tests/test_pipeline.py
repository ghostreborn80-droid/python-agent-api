"""Pipeline tests. INTENTIONALLY RED right now:
cleaner.clean_data contains an injected bug (KeyError: 'qty')."""

import pandas as pd

from analyzer import revenue_by_region
from cleaner import clean_data, normalize_regions
from data_loader import load_data, validate_columns
from report import generate_report

DATA = "sales.csv"


def test_load_data_columns():
    df = load_data(DATA)
    assert "revenue" in df.columns
    assert len(df) == 10


def test_validate_columns_ok():
    assert validate_columns(load_data(DATA)) is True


def test_clean_data_drops_missing_revenue():
    cleaned = clean_data(load_data(DATA))
    assert cleaned["revenue"].notna().all()


def test_clean_data_adds_total():
    cleaned = clean_data(load_data(DATA))
    assert "total" in cleaned.columns


def test_clean_data_total_nonnegative():
    cleaned = clean_data(load_data(DATA))
    assert (cleaned["total"] >= 0).all()


def test_normalize_regions():
    df = pd.DataFrame({"region": ["North", "SOUTH", None]})
    assert set(normalize_regions(df)["region"]) == {"north", "south", "unknown"}


def test_revenue_by_region_positive():
    cleaned = normalize_regions(clean_data(load_data(DATA)))
    assert revenue_by_region(cleaned).sum() > 0


def test_full_report_html():
    html = generate_report(DATA)
    assert "<html>" in html
    assert "Sales Report" in html
