import pandas as pd

from quellcode.datensimulation.kalender_erzeugen import generate_calendar


def test_generate_calendar_has_expected_columns():
    df = generate_calendar("2021-01-01", "2021-01-10")

    expected = {"date", "year", "month", "day", "weekday", "is_weekend", "is_payday"}
    assert expected.issubset(set(df.columns))


def test_generate_calendar_row_count_inclusive():
    df = generate_calendar("2021-01-01", "2021-01-10")
    # inclusive range: 10 days
    assert len(df) == 10


def test_weekend_flag_is_binary():
    df = generate_calendar("2021-01-01", "2021-01-31")
    assert set(df["is_weekend"].unique()).issubset({0, 1})


def test_payday_rule_day_25():
    df = generate_calendar("2021-01-01", "2021-01-31")
    payday_days = df.loc[df["is_payday"] == 1, "day"].unique().tolist()
    assert payday_days == [25]