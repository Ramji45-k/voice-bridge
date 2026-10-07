from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


MODEL_FEATURES = [
    "Price",
    "Inventory Level",
    "Units Ordered",
    "Discount",
    "Promotion",
    "Competitor Pricing",
    "Epidemic",
    "Product ID",
    "Category",
    "Region",
    "Weather Condition",
    "Seasonality",
]
NUMERIC_FEATURES = [
    "Price",
    "Inventory Level",
    "Units Ordered",
    "Discount",
    "Promotion",
    "Competitor Pricing",
    "Epidemic",
]
CATEGORICAL_FEATURES = [column for column in MODEL_FEATURES if column not in NUMERIC_FEATURES]


def _normalize_context(context) -> dict:
    if isinstance(context, pd.DataFrame):
        if context.empty:
            raise ValueError("Context data must include at least one row.")
        context = context.iloc[0]
    elif isinstance(context, dict):
        context = pd.Series(context)
    elif not isinstance(context, pd.Series):
        raise TypeError("Context must be a pandas Series, a single-row DataFrame, or a dictionary.")
    return context.to_dict()


def _validate_training_columns(data: pd.DataFrame) -> None:
    required_columns = ["Date", "Demand", *MODEL_FEATURES]
    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Training data is missing required columns: {', '.join(missing_columns)}")


def _prepare_context(context) -> dict:
    context_dict = _normalize_context(context)
    for feature in MODEL_FEATURES:
        context_dict.setdefault(feature, 0 if feature in NUMERIC_FEATURES else "Unknown")
    context_dict["Promotion"] = int(bool(context_dict.get("Promotion", False)))
    return context_dict


@dataclass(frozen=True)
class ModelMetrics:
    mean_absolute_error: float
    r2_score: float


@dataclass(frozen=True)
class AdvancedPricingResult:
    recommended_price: float
    predicted_demand: float
    expected_revenue: float
    expected_gross_profit: float
    baseline_demand: float
    baseline_gross_profit: float
    candidates: pd.DataFrame


def _build_demand_model() -> Pipeline:
    numeric_transformer = "passthrough"
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")
    preprocessor = ColumnTransformer(
        [("numeric", numeric_transformer, NUMERIC_FEATURES), ("categorical", categorical_transformer, CATEGORICAL_FEATURES)]
    )
    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(n_estimators=120, max_depth=18, min_samples_leaf=3, random_state=42, n_jobs=-1)),
        ]
    )

    return model


def train_demand_model(data: pd.DataFrame) -> tuple[Pipeline, ModelMetrics]:
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Training data must be provided as a pandas DataFrame.")
    _validate_training_columns(data)

    ordered_data = data.sort_values("Date", kind="stable")
    split_index = int(len(ordered_data) * 0.8)
    if split_index <= 0 or split_index >= len(ordered_data):
        raise ValueError("At least two dated records are required to train the demand model.")

    train_data = ordered_data.iloc[:split_index]
    test_data = ordered_data.iloc[split_index:]
    evaluation_model = _build_demand_model()
    evaluation_model.fit(train_data[MODEL_FEATURES], train_data["Demand"])
    predictions = evaluation_model.predict(test_data[MODEL_FEATURES])
    metrics = ModelMetrics(
        mean_absolute_error=round(float(mean_absolute_error(test_data["Demand"], predictions)), 2),
        r2_score=round(float(r2_score(test_data["Demand"], predictions)), 3),
    )
    model = _build_demand_model()
    model.fit(ordered_data[MODEL_FEATURES], ordered_data["Demand"])
    return model, metrics


def optimize_price(
    model: Pipeline,
    context: pd.Series | dict | pd.DataFrame,
    current_price: float,
    inventory_level: float,
    discount: int,
    promotion: bool,
    competitor_price: float,
    unit_cost: float,
) -> AdvancedPricingResult:
    if current_price <= 0 or unit_cost < 0:
        raise ValueError("Current price must be positive and unit cost cannot be negative.")
    minimum_price = max(0.01, current_price * 0.85, unit_cost)
    maximum_price = current_price * 1.15
    if minimum_price > maximum_price:
        raise ValueError("Unit cost must be below the maximum candidate price.")

    normalized_context = _prepare_context(context)
    candidate_prices = np.round(np.linspace(minimum_price, maximum_price, 31), 2)
    candidates = pd.DataFrame([normalized_context] * len(candidate_prices))
    candidates["Price"] = candidate_prices
    candidates["Inventory Level"] = inventory_level
    candidates["Discount"] = discount
    candidates["Promotion"] = int(promotion)
    candidates["Competitor Pricing"] = competitor_price
    candidates["Predicted Demand"] = np.maximum(model.predict(candidates[MODEL_FEATURES]), 0)
    candidates["Expected Revenue"] = candidates["Price"] * candidates["Predicted Demand"]
    candidates["Expected Gross Profit"] = (candidates["Price"] - unit_cost) * candidates["Predicted Demand"]
    best = candidates.loc[candidates["Expected Gross Profit"].idxmax()]
    baseline = pd.DataFrame([normalized_context])
    baseline["Price"] = current_price
    baseline["Inventory Level"] = inventory_level
    baseline["Discount"] = discount
    baseline["Promotion"] = int(promotion)
    baseline["Competitor Pricing"] = competitor_price
    baseline_demand = max(float(model.predict(baseline[MODEL_FEATURES])[0]), 0)
    return AdvancedPricingResult(
        recommended_price=float(best["Price"]),
        predicted_demand=float(best["Predicted Demand"]),
        expected_revenue=float(best["Expected Revenue"]),
        expected_gross_profit=float(best["Expected Gross Profit"]),
        baseline_demand=baseline_demand,
        baseline_gross_profit=float((current_price - unit_cost) * baseline_demand),
        candidates=candidates[["Price", "Predicted Demand", "Expected Revenue", "Expected Gross Profit"]],
    )