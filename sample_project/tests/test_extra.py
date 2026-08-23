from utils import format_currency

def test_format_currency_zero_added_by_agent():
    assert format_currency(0) == "$0.00"
