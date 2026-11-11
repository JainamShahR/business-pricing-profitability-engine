"""Small, explainable pricing analytics engine for monthly product sales.

This portfolio project estimates *associations* in sample historical data.
Predictions and scenario results are illustrative, not causal estimates.
"""

from dataclasses import dataclass
import sqlite3

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

REQUIRED_COLUMNS = (
    "date", "product", "selling_price", "unit_cost", "units_sold", "marketing_spend"
)

SQL_SUMMARY = """
SELECT
    product,
    SUM(units_sold) AS units_sold,
    ROUND(SUM(selling_price * units_sold), 2) AS revenue,
    ROUND(SUM(unit_cost * units_sold), 2) AS cost_of_goods,
    ROUND(SUM(marketing_spend), 2) AS marketing_spend,
    ROUND(SUM((selling_price - unit_cost) * units_sold - marketing_spend), 2)
        AS contribution_profit
FROM sales
GROUP BY product
ORDER BY contribution_profit DESC;
"""


def validate_sales(raw: pd.DataFrame) -> pd.DataFrame:
    """Require one row per product and calendar month; return clean data."""
    missing = set(REQUIRED_COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
    df = raw.loc[:, REQUIRED_COLUMNS].copy()
    try:
        df["date"] = pd.to_datetime(df["date"], errors="raise")
    except (ValueError, TypeError) as exc:
        raise ValueError("date values must be valid dates, e.g. 2025-01-01") from exc
    if df["date"].isna().any():
        raise ValueError("date cannot be blank")
    df["date"] = df["date"].dt.to_period("M").dt.to_timestamp()
    df["product"] = df["product"].astype("string").str.strip()
    if df["product"].isna().any() or df["product"].eq("").any():
        raise ValueError("product cannot be blank")
    for col in ("selling_price", "unit_cost", "units_sold", "marketing_spend"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isna().any() or not np.isfinite(df[col]).all():
            raise ValueError(f"{col} must contain only finite numeric values")
    if (df["selling_price"] <= 0).any():
        raise ValueError("selling_price must be greater than zero")
    if (df[["unit_cost", "units_sold", "marketing_spend"]] < 0).any().any():
        raise ValueError("cost, units, and marketing spend cannot be negative")
    if (df["units_sold"] % 1 != 0).any():
        raise ValueError("units_sold must be a whole number")
    if df.duplicated(["product", "date"]).any():
        raise ValueError("Provide only one monthly row for each product")
    return df.sort_values(["product", "date"]).reset_index(drop=True)


def add_financials(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["revenue"] = out["selling_price"] * out["units_sold"]
    out["cost_of_goods"] = out["unit_cost"] * out["units_sold"]
    # Excludes overhead, tax, shipping and other operating costs: NOT net profit.
    out["contribution_profit"] = (
        out["revenue"] - out["cost_of_goods"] - out["marketing_spend"]
    )
    return out


def sql_product_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate key financial metrics with a real SQL query via SQLite."""
    with sqlite3.connect(":memory:") as connection:
        df_for_sql = df.copy()
        df_for_sql["date"] = df_for_sql["date"].dt.strftime("%Y-%m-%d")
        df_for_sql.to_sql("sales", connection, index=False, if_exists="replace")
        summary = pd.read_sql_query(SQL_SUMMARY, connection)
    summary["contribution_margin_pct"] = np.where(
        summary["revenue"] > 0,
        100 * summary["contribution_profit"] / summary["revenue"],
        0.0,
    )
    return summary


def _feature_matrix(price, marketing, dates) -> np.ndarray:
    """Log-price model; its price coefficient approximates elasticity."""
    prices = np.asarray(price, dtype=float).reshape(-1)
    budgets = np.asarray(marketing, dtype=float).reshape(-1)
    months = pd.DatetimeIndex(pd.to_datetime(dates)).month.to_numpy()
    if not (len(prices) == len(budgets) == len(months)):
        raise ValueError("price, marketing, and dates must have equal length")
    return np.column_stack([
        np.log(prices),
        np.log1p(budgets),
        np.sin(2 * np.pi * months / 12),
        np.cos(2 * np.pi * months / 12),
    ])


@dataclass
class DemandFit:
    model: LinearRegression
    holdout_mae_units: float
    baseline_mae_units: float
    price_elasticity: float
    test_months: int


def fit_demand_model(product_rows: pd.DataFrame) -> DemandFit:
    """Evaluate on the latest months, then refit on all available rows."""
    df = product_rows.sort_values("date").reset_index(drop=True)
    if len(df) < 12:
        raise ValueError("Demand model requires at least 12 monthly records")
    holdout = min(6, max(3, len(df) // 5))
    X = _feature_matrix(df["selling_price"], df["marketing_spend"], df["date"])
    y = np.log(np.maximum(df["units_sold"].to_numpy(dtype=float), 1.0))
    train_end = len(df) - holdout
    initial = LinearRegression().fit(X[:train_end], y[:train_end])
    predicted_units = np.exp(np.clip(initial.predict(X[train_end:]), -10, 18))
    actual_units = df["units_sold"].to_numpy(dtype=float)[train_end:]
    mae = float(mean_absolute_error(actual_units, predicted_units))
    naive_prediction = np.full(holdout, float(df["units_sold"].iloc[:train_end].mean()))
    baseline_mae = float(mean_absolute_error(actual_units, naive_prediction))
    final_model = LinearRegression().fit(X, y)
    return DemandFit(
        model=final_model,
        holdout_mae_units=mae,
        baseline_mae_units=baseline_mae,
        price_elasticity=float(final_model.coef_[0]),
        test_months=holdout,
    )


def predict_units(fit: DemandFit, price: float, marketing_spend: float, date) -> float:
    if price <= 0 or marketing_spend < 0:
        raise ValueError("price must be positive and marketing must be nonnegative")
    X = _feature_matrix([price], [marketing_spend], [date])
    return float(np.exp(np.clip(fit.model.predict(X)[0], -10, 18)))


def make_scenario(
    fit: DemandFit, base_price: float, unit_cost: float,
    marketing_spend: float, date, change_pct: float,
) -> dict:
    """Compare estimated monthly contribution profit at a changed price."""
    if change_pct <= -100:
        raise ValueError("Price change cannot reduce the price to zero or below")
    selling_price = base_price * (1 + change_pct / 100.0)
    units = predict_units(fit, selling_price, marketing_spend, date)
    revenue = selling_price * units
    # Model-estimated contribution, not full business net profit.
    profit = (selling_price - unit_cost) * units - marketing_spend
    return {
        "price_change_pct": float(change_pct),
        "selling_price": selling_price,
        "estimated_units": units,
        "estimated_revenue": revenue,
        "estimated_contribution_profit": profit,
    }
