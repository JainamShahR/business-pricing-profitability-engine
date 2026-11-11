import pandas as pd
import pytest

from generate_data import build_sample_data
from pricing_engine import (
    add_financials, fit_demand_model, make_scenario,
    sql_product_summary, validate_sales,
)


def test_contribution_profit_and_sql_summary():
    df = pd.DataFrame([{
        "date": "2025-01-01", "product": "Example", "selling_price": 100,
        "unit_cost": 60, "units_sold": 10, "marketing_spend": 50,
    }])
    clean = validate_sales(df)
    result = add_financials(clean)
    assert result.iloc[0]["revenue"] == 1000
    assert result.iloc[0]["contribution_profit"] == 350
    summary = sql_product_summary(clean)
    assert summary.iloc[0]["contribution_profit"] == 350
    assert summary.iloc[0]["contribution_margin_pct"] == 35


def test_reject_duplicate_months():
    row = {
        "date": "2025-01-01", "product": "Example", "selling_price": 100,
        "unit_cost": 60, "units_sold": 10, "marketing_spend": 50,
    }
    with pytest.raises(ValueError, match="one monthly row"):
        validate_sales(pd.DataFrame([row, row]))


def test_fitted_model_and_pricing_scenario():
    df = validate_sales(build_sample_data())
    rows = df[df["product"] == "Wireless Earbuds"].sort_values("date")
    fit = fit_demand_model(rows)
    assert fit.test_months > 0
    assert fit.holdout_mae_units >= 0
    assert fit.price_elasticity < 0  # deterministic fictional dataset
    latest = rows.iloc[-1]
    scenario = make_scenario(
        fit, latest["selling_price"], latest["unit_cost"],
        latest["marketing_spend"], latest["date"], 5,
    )
    assert scenario["selling_price"] == pytest.approx(latest["selling_price"] * 1.05)
    assert scenario["estimated_units"] > 0
