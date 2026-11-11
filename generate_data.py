"""Create fictional monthly sales data so this starter runs without private data."""

from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT = PROJECT_DIR / "data" / "sample_sales.csv"

CATALOG = [
    # product, regular price INR, unit cost INR, monthly units, elasticity, monthly marketing INR
    ("Wireless Earbuds", 2499, 1150, 360, 1.6, 18000),
    ("Laptop Stand", 1399, 620, 310, 1.3, 10000),
    ("Office Chair", 6999, 4100, 135, 1.8, 16000),
    ("Water Bottle", 799, 310, 500, 1.1, 8500),
]


def build_sample_data(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=36, freq="MS")
    rows = []
    for name, regular_price, cost, typical_units, elasticity, typical_marketing in CATALOG:
        for month_index, date in enumerate(dates):
            # Deliberate, controlled variation lets us demonstrate model fitting.
            price = round(regular_price * rng.uniform(0.88, 1.12), 2)
            marketing = round(typical_marketing * rng.uniform(0.75, 1.25), 2)
            seasonal = 1 + 0.08 * np.sin(2 * np.pi * date.month / 12)
            marketing_effect = (marketing / typical_marketing) ** 0.22
            expected_units = (
                typical_units * (regular_price / price) ** elasticity
                * seasonal * marketing_effect * (1 + 0.003 * month_index)
            )
            units = max(0, int(round(expected_units * rng.normal(1, 0.035))))
            rows.append({
                "date": date.strftime("%Y-%m-%d"),
                "product": name,
                "selling_price": price,
                "unit_cost": cost,
                "units_sold": units,
                "marketing_spend": marketing,
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df = build_sample_data()
    df.to_csv(OUTPUT, index=False)
    print(f"Created {OUTPUT} with {len(df)} fictional product-month rows")
