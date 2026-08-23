"""Green-from-the-start tests for utils."""

from utils import format_currency, safe_divide


def test_format_currency():
    assert format_currency(1234.5) == "$1,234.50"
    assert format_currency(0) == "$0.00"


def test_safe_divide():
    assert safe_divide(10, 2) == 5.0
    assert safe_divide(1, 0) == 0.0
