# Business Pricing & Profitability Decision Engine

An approachable Python + SQL + Streamlit portfolio starter for analyzing product profitability and comparing pricing scenarios.

## What it does

- **SQL financial analysis:** calculates revenue, units sold, cost of goods sold, marketing spend, and contribution profit for each product.
- **Historical reporting:** displays monthly revenue and contribution profit.
- **Demand modelling:** fits an interpretable log-linear regression to each product's price, marketing spend, and seasonality. It tests on the most recent months before refitting.
- **What-if pricing:** simulates price changes between **−10% and +10%**, estimating demand, revenue, and contribution profit.

**Important:** The included dataset is **synthetic and fictional**. Forecasts are estimates based on observed associations, not proven causal effects. The dashboard does **not** claim a real-world revenue increase, and contribution profit is **not net profit** (other overheads and taxes are excluded).

## Tech stack

Python · SQL (SQLite for a no-setup demo) · Pandas · NumPy · Scikit-learn · Streamlit · Plotly · pytest

## Run locally (macOS/Linux)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python generate_data.py
streamlit run app.py
```

Then open the local URL printed by Streamlit. On Windows activate with `.venv\\Scripts\\activate`.

You can also run unit tests:

```bash
pytest -q
```

## Data format

You can use the built-in `data/sample_sales.csv` or upload a CSV to the app. Uploads need **one row per product per month** with the following columns:

| Column | Meaning | Example |
|---|---|---|
| `date` | Month (YYYY-MM-01) | `2025-01-01` |
| `product` | Product name | `Laptop Stand` |
| `selling_price` | Actual price per unit (INR) | `1399.00` |
| `unit_cost` | Cost of goods per unit (INR) | `620.00` |
| `units_sold` | Units sold in month | `310` |
| `marketing_spend` | Monthly product marketing cost (INR) | `10000` |

Use at least **12 monthly records per product** to enable the forecasting section.

## How the calculations work

- `Revenue = Selling price × Units sold`
- `Cost of goods sold = Unit cost × Units sold`
- `Contribution profit = Revenue − Cost of goods sold − Marketing spend`
- `Contribution margin % = 100 × Contribution profit / Revenue`

Demand uses a simple regression on the log of units with log price, log marketing spend, and seasonal features. A price coefficient of `−1.5` suggests an approximate 1.5% fall in estimated demand for a small 1% increase in price **in this data**, holding other included factors constant. This is not a validated causal elasticity estimate.

For the scenario comparison, the simulator keeps the latest month's unit cost, marketing spend, and season constant while changing selling price. It compares all scenarios against a **model-predicted 0% price-change baseline** rather than against historical actuals.

## Folder layout

```text
pricing_profitability_starter/
├── app.py                 # Streamlit dashboard
├── pricing_engine.py      # SQL summary, demand regression, scenario calculations
├── generate_data.py       # Reproducible sample data generator
├── data/
│   └── sample_sales.csv   # Fictional monthly product sales
├── tests/
│   └── test_pricing_engine.py
├── requirements.txt
├── pyproject.toml         # Test configuration
├── .gitignore
└── README.md
```
