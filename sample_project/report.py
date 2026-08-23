"""End-to-end pipeline: load -> clean -> analyze -> HTML report."""

from analyzer import revenue_by_region, summarize, top_products
from cleaner import clean_data, normalize_regions
from data_loader import load_data
from utils import format_currency


def generate_report(path):
    """Run the full sales pipeline and return an HTML report string."""
    df = load_data(path)
    df = clean_data(df)
    df = normalize_regions(df)
    stats = summarize(df)
    by_region = revenue_by_region(df)
    products = top_products(df)
    return to_html(stats, by_region, products)


def to_html(stats, by_region, products):
    """Render pipeline outputs as a small standalone HTML page."""
    rows = "".join(
        f"<tr><td>{region}</td><td>{format_currency(total)}</td></tr>"
        for region, total in by_region.items()
    )
    items = "".join(f"<li>{name}</li>" for name in products.index)
    return (
        "<html><body><h1>Sales Report</h1>"
        f"<p>Orders: {stats['orders']} | "
        f"Revenue: {format_currency(stats['revenue'])} | "
        f"Avg order: {format_currency(stats['avg_order'])}</p>"
        "<h2>Revenue by Region</h2>"
        "<table border='1'><tr><th>Region</th><th>Revenue</th></tr>"
        f"{rows}</table>"
        f"<h2>Top Products</h2><ul>{items}</ul>"
        "</body></html>"
    )
