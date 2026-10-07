from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = [
    "Date",
    "Store ID",
    "Product ID",
    "Category",
    "Region",
    "Inventory Level",
    "Units Sold",
    "Units Ordered",
    "Price",
    "Discount",
    "Weather Condition",
    "Promotion",
    "Competitor Pricing",
    "Seasonality",
    "Epidemic",
    "Demand",
]
NUMERIC_COLUMNS = [
    "Inventory Level",
    "Units Sold",
    "Units Ordered",
    "Price",
    "Discount",
    "Promotion",
    "Competitor Pricing",
    "Epidemic",
    "Demand",
]
TEXT_COLUMNS = [
    "Store ID",
    "Product ID",
    "Category",
    "Region",
    "Weather Condition",
    "Seasonality",
]


def load_and_clean_data(path: str | Path) -> pd.DataFrame:
    """Load sales data, normalize its columns, and remove invalid or duplicate rows."""
    data = pd.read_csv(path)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Sales data is missing required columns: {', '.join(missing_columns)}")

    cleaned = data.copy()
    cleaned["Date"] = pd.to_datetime(cleaned["Date"], errors="coerce")
    for column in NUMERIC_COLUMNS:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    for column in TEXT_COLUMNS:
        cleaned[column] = cleaned[column].astype("string").str.strip()

    cleaned = cleaned.dropna(subset=REQUIRED_COLUMNS).drop_duplicates()
    return cleaned.reset_index(drop=True)


def dataset_profile(raw_data: pd.DataFrame, cleaned_data: pd.DataFrame) -> dict[str, int]:
    """Summarize the input and cleaned dataset for the app's sidebar."""
    return {
        "clean_rows": len(cleaned_data),
        "columns": len(raw_data.columns),
        "missing_values": int(raw_data.isna().sum().sum()),
        "duplicates_removed": int(raw_data.duplicated().sum()),
    }