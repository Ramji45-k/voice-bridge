# Dynamic Pricing System: First-Review Prototype

This Streamlit business-analytics app provides product catalog management, interactive sales analysis, and margin-aware price recommendations:

`Sales CSV -> product catalog -> interactive analytics -> demand model -> gross-profit optimization -> Excel decision history`

## Verified dataset profile

- 76,000 rows and 16 columns
- No missing values in the supplied file
- No duplicate rows in the supplied file
- Date range: 2022-01-01 to 2023-12-31
- Numeric inputs include price, discount, inventory, sales, competitor pricing, promotion, and demand
- Categorical inputs include product, category, region, weather, and seasonality

## Run the prototype

```powershell
python -m streamlit run app.py
```

The app starts with internal product IDs because the supplied sales data has no product names. In **Product catalog**, assign names to those source products or register new products with a name and market attributes. New products receive a generated internal reference ID; recommendations and Excel history retain both the readable name and reference ID. Names and product defaults persist in `product_catalog.csv`.

The **Overview** tab has date, category, and region filters plus interactive monthly revenue, category revenue, price-demand, and inventory-demand charts. Charts support zoom, pan, zoom in/out, reset, fullscreen, and PNG download. Filtered analytics can also be downloaded as CSV.

The advanced engine evaluates demand predictions on a chronological holdout (earlier 80% for training, later 20% for evaluation), then trains on all available rows for recommendations. It excludes historical units sold from model inputs and evaluates 31 candidate prices within +/-15% of current price, constrained to be at or above unit cost. It selects the price with the highest model-predicted gross profit. The dataset has no unit cost, so the initial cost is an illustrative 50% of median price; replace it with actual cost before interpreting the profit estimate. Save detailed decisions to `pricing_predictions.xlsx` for later Power BI analysis.

The recommended profit is a model-based scenario estimate, not proof that changing price will cause the predicted demand. Validate with a controlled pricing experiment before using recommendations in a live business.

## Pricing methodology

The rule engine uses demand and inventory bands, then adjusts for competitor pricing, promotions, and high existing discounts. The final change is bounded to +/-15% of the current price so the first-review prototype remains controlled and explainable.