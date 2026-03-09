import numpy as np
import pandas as pd


def forecast_demand(history: list, days: int = 7):
    """
    Simple demand forecast using moving average.
    """

    if not history:
        return [0] * days

    series = pd.Series(history, dtype=float)

    if len(series) < 3:
        avg = series.mean()
    else:
        avg = series.rolling(window=3).mean().iloc[-1]

    if pd.isna(avg):
        avg = series.mean()

    forecast = [max(0, int(avg + np.random.normal(0, 2))) for _ in range(days)]

    return forecast