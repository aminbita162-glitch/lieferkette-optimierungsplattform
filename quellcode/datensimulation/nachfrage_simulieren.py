from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DemandSimulationConfig:
    """
    Configuration for demand simulation.
    """

    seed: int = 42

    # Scale
    n_skus: int = 5000
    n_warehouses: int = 3

    # Demand distribution
    use_negative_binomial: bool = True
    nb_dispersion_alpha: float = 0.25  # higher => more variance (over-dispersion)

    # Seasonality strengths
    weekly_seasonality_strength: float = 0.25
    yearly_seasonality_strength: float = 0.35

    # Trend
    trend_strength: float = 0.0002  # small daily trend on log-scale

    # Promotions
    promo_probability: float = 0.08
    promo_uplift_strength: float = 0.45

    # Pricing & elasticity
    price_noise_std: float = 0.03
    elasticity_mean: float = -1.2
    elasticity_std: float = 0.5

    # Base demand
    base_demand_log_mean: float = 1.2
    base_demand_log_std: float = 1.0

    # Limits
    max_daily_units_cap: int = 500


def _fourier_yearly_terms(dates: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """
    Basic yearly seasonality using sine/cosine of day-of-year.
    """
    doy = dates.dt.dayofyear.to_numpy()
    angle = 2.0 * np.pi * doy / 365.25
    return np.sin(angle), np.cos(angle)


def _weekly_term(weekday: pd.Series) -> np.ndarray:
    """
    Weekly seasonality: deterministic pattern by weekday.
    """
    w = weekday.to_numpy()
    pattern = np.array([0.00, 0.02, 0.03, 0.03, 0.02, 0.08, 0.10])
    return pattern[w]


def _negative_binomial_rng(mu: np.ndarray, alpha: float, rng: np.random.Generator) -> np.ndarray:
    """
    Negative binomial via Gamma-Poisson mixture.
    """
    mu = np.maximum(mu, 1e-9)
    shape = 1.0 / max(alpha, 1e-9)
    scale = alpha * mu
    lam = rng.gamma(shape=shape, scale=scale)
    return rng.poisson(lam=lam)


def simulate_demand(
    calendar: pd.DataFrame,
    config: DemandSimulationConfig,
    sku_master: Optional[pd.DataFrame] = None,
    warehouse_master: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Simulate daily demand and realized sales.
    """
    rng = np.random.default_rng(config.seed)

    if sku_master is None:
        sku_master = pd.DataFrame(
            {
                "sku_id": np.arange(1, config.n_skus + 1),
                "base_price": rng.uniform(3.0, 50.0, size=config.n_skus).round(2),
            }
        )

        abc_probs = np.array([0.15, 0.35, 0.50])
        sku_master["abc_class"] = rng.choice(["A", "B", "C"], size=config.n_skus, p=abc_probs)

    if warehouse_master is None:
        warehouse_master = pd.DataFrame({"warehouse_id": np.arange(1, config.n_warehouses + 1)})

    base_log = rng.normal(config.base_demand_log_mean, config.base_demand_log_std, size=len(sku_master))

    abc_multiplier = sku_master["abc_class"].map({"A": 1.8, "B": 1.0, "C": 0.6}).to_numpy()

    base_mu = np.exp(base_log) * abc_multiplier

    elasticity = rng.normal(config.elasticity_mean, config.elasticity_std, size=len(sku_master))
    elasticity = np.clip(elasticity, -3.5, -0.1)

    results = []
    cal = calendar.copy()
    cal["date"] = pd.to_datetime(cal["date"])

    weekly = _weekly_term(cal["weekday"]) * config.weekly_seasonality_strength
    y_sin, y_cos = _fourier_yearly_terms(cal["date"])
    yearly = (0.6 * y_sin + 0.4 * y_cos) * config.yearly_seasonality_strength

    t = np.arange(len(cal), dtype=float)
    trend = config.trend_strength * t

    payday = 0.03 * cal["is_payday"].to_numpy()

    day_effect_log = weekly + yearly + trend + payday

    sku_ids = sku_master["sku_id"].to_numpy()
    base_prices = sku_master["base_price"].to_numpy()

    for wh_id in warehouse_master["warehouse_id"].to_numpy():

        n_days = len(cal)
        n_skus = len(sku_master)

        eps_day = rng.normal(0.0, config.price_noise_std, size=n_days)
        eps_sku_day = rng.normal(0.0, config.price_noise_std, size=(n_days, n_skus))

        price = base_prices[None, :] * np.exp(eps_day[:, None] + eps_sku_day)
        price = np.maximum(price, 0.5)

        promo_flag = rng.random(size=(n_days, n_skus)) < config.promo_probability
        promo_depth = rng.uniform(0.05, 0.40, size=(n_days, n_skus)) * promo_flag

        promo_uplift = config.promo_uplift_strength * (promo_depth / 0.40)

        log_price_ratio = np.log(price / base_prices[None, :])
        price_effect = elasticity[None, :] * log_price_ratio

        noise = rng.normal(0.0, 0.10, size=(n_days, n_skus))

        log_mu = (
            np.log(base_mu[None, :])
            + day_effect_log[:, None]
            + promo_uplift
            + price_effect
            + noise
        )

        mu = np.exp(log_mu)

        if config.use_negative_binomial:
            true_demand = _negative_binomial_rng(mu=mu, alpha=config.nb_dispersion_alpha, rng=rng)
        else:
            true_demand = rng.poisson(lam=mu)

        true_demand = np.clip(true_demand, 0, config.max_daily_units_cap).astype(int)

        units_sold = true_demand.copy()

        df = pd.DataFrame(
            {
                "date": np.repeat(cal["date"].to_numpy(), n_skus),
                "sku_id": np.tile(sku_ids, n_days),
                "warehouse_id": wh_id,
                "price": price.reshape(-1).round(2),
                "promo_flag": promo_flag.reshape(-1).astype(int),
                "promo_depth": promo_depth.reshape(-1).round(3),
                "true_demand": true_demand.reshape(-1),
                "units_sold": units_sold.reshape(-1),
            }
        )

        results.append(df)

    return pd.concat(results, ignore_index=True)


if __name__ == "__main__":
    # FIXED IMPORT (important for package structure)
    from quellcode.datensimulation.kalender_erzeugen import generate_calendar

    cfg = DemandSimulationConfig(n_skus=50, n_warehouses=2, seed=7)

    cal_df = generate_calendar("2022-01-01", "2022-03-31")

    out = simulate_demand(cal_df, cfg)

    print(out.head())
    print(out.describe(include="all"))