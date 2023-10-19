import pandas as pd
import pytest

from src.app import is_integer_series, is_string_series, is_float_series

def test_is_integer_series():
    # Test a series of integers
    series1 = pd.Series([1, 2, 3, 4])
    assert is_integer_series(series1) == True

    # Test a series of floats
    series2 = pd.Series([1.0, 2.0, 3.0, 4.0])
    assert is_integer_series(series2) == True

    # Test a series of strings
    series3 = pd.Series(['1', '2', '3', '4'])
    assert is_integer_series(series3) == True

    # Test a series of mixed types
    series4 = pd.Series([1, '2', 3.0, '4'])
    assert is_integer_series(series4) == True

    # Test a series of floats with decimal
    series4 = pd.Series([1.8, 2.3, 3.0, 4.1])
    assert is_integer_series(series4) == False

def test_is_string_series():
    # Test a series of strings
    series1 = pd.Series(['apple', 'banana', 'cherry'])
    assert is_string_series(series1) == True

    # Test a series of integers
    series2 = pd.Series([1, 2, 3, 4])
    assert is_string_series(series2) == False

    # Test a series of floats
    series3 = pd.Series([1.0, 2.0, 3.0, 4.0])
    assert is_string_series(series3) == False

    # Test a series of 'strings' that are actually floats
    series3 = pd.Series(['1.0', '2.0', '3.0', '4.0'])
    assert is_string_series(series3) == False

    # Test a series of mixed types
    series4 = pd.Series([1, '2', 3.0, 'four'])
    assert is_string_series(series4) == True