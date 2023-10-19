import pandas as pd
import numpy as np

def is_integer_series(series):
    """Check if a series is an integer series"""
    mask = series.astype('float').dropna().astype(int) == series.astype('float').dropna()
    return mask.all()

def is_string_series(series):
    """Check if a series is a string series and cannot be converted to a number"""
    try:
        series.astype('float')
        return False
    except ValueError:
        return True


if __name__ == "__main__":
    v1 = is_integer_series(pd.Series([1.0,2.0,3.0]))
    v2 = is_integer_series(pd.Series([1.0,2.0,3.1]))
    v3 = is_integer_series(pd.Series([1.0,2.0,3.1, np.nan]))
    v4 = is_integer_series(pd.Series([1, 2, 3]))
    v5 = is_integer_series(pd.Series([1, 2, 3.0]))
    v6 = is_integer_series(pd.Series(['1', '2', '3']))
    print(v1, v2, v3, v4, v5, v6)