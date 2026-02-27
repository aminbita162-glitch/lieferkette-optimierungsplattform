from datetime import datetime
import pandas as pd


def generate_calendar(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Generate a daily calendar between start_date and end_date.

    Parameters
    ----------
    start_date : str
        Start date in format YYYY-MM-DD
    end_date : str
        End date in format YYYY-MM-DD

    Returns
    -------
    pd.DataFrame
        Calendar dataframe with time-based features
    """

    date_range = pd.date_range(start=start_date, end=end_date, freq="D")

    calendar = pd.DataFrame({"date": date_range})

    calendar["year"] = calendar["date"].dt.year
    calendar["month"] = calendar["date"].dt.month
    calendar["day"] = calendar["date"].dt.day
    calendar["weekday"] = calendar["date"].dt.weekday  # Monday=0, Sunday=6
    calendar["is_weekend"] = calendar["weekday"].isin([5, 6]).astype(int)

    # Example payday rule: 25th of each month
    calendar["is_payday"] = (calendar["day"] == 25).astype(int)

    return calendar


if __name__ == "__main__":
    df = generate_calendar("2021-01-01", "2021-12-31")
    print(df.head())