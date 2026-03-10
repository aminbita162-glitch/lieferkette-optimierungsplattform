from typing import List
import numpy as np
import pandas as pd


def forecast_demand(history: List[float], days: int = 7) -> List[int]:
    """
    Simple demand forecast using moving average.
    Returns predicted demand for next N days.
    """

    if not history:
        return [0 for _ in range(days)]

    try:
        series = pd.Series(history, dtype=float)
    except Exception:
        return [0 for _ in range(days)]

    # moving average
    if len(series) < 3:
        avg = series.mean()
    else:
        avg = series.rolling(window=3).mean().iloc[-1]

    if pd.isna(avg):
        avg = series.mean()

    forecast = []

    for _ in range(days):
        noise = np.random.normal(0, 2)
        value = max(0, int(avg + noise))
        forecast.append(value)

    return forecast