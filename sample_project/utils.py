"""Small shared helpers used across the sales project."""


def format_currency(value):
    """Format a number as a USD currency string."""
    return f"${value:,.2f}"


def safe_divide(a, b, default=0.0):
    """Divide a by b, returning `default` instead of crashing on zero."""
    if b == 0:
        return default
    return a / b
