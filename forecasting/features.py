"""Causal calendar and shifted history features, shared by training and recursion."""
import numpy as np
import pandas as pd


def build_features(daily):
    if not isinstance(daily.index, pd.DatetimeIndex) or not daily.index.is_monotonic_increasing or not daily.index.is_unique:
        raise ValueError("daily must have a sorted unique DatetimeIndex")
    if len(daily) > 1 and not (daily.index[1:] - daily.index[:-1] == pd.Timedelta(days=1)).all():
        raise ValueError("daily must have a continuous daily calendar")
    result = pd.DataFrame(index=daily.index)
    result["day_of_week"] = daily.index.dayofweek
    result["day_of_month"] = daily.index.day
    result["month"] = daily.index.month
    result["weekend"] = (daily.index.dayofweek >= 5).astype(int)
    result["month_end"] = daily.index.is_month_end.astype(int)
    result["time"] = (daily.index - pd.Timestamp("2000-01-01")).days
    for column in ["income", "expense"] + sorted(c for c in daily if c.startswith("category::")):
        previous = daily[column].shift(1)
        for lag in (1, 7, 14, 28):
            result[f"{column}_lag_{lag}"] = daily[column].shift(lag)
        for window in (7, 28):
            result[f"{column}_mean_{window}"] = previous.rolling(window, min_periods=window).mean()
            result[f"{column}_sum_{window}"] = previous.rolling(window, min_periods=window).sum()
    return result.replace([np.inf, -np.inf], np.nan)
