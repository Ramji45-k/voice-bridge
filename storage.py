from datetime import datetime
from pathlib import Path

import pandas as pd


CATALOG_COLUMNS = [
    "Product ID",
    "Product Name",
    "Category",
    "Region",
    "Weather Condition",
    "Seasonality",
    "Default Price",
    "Default Unit Cost",
]


def load_product_catalog(path: str | Path, data: pd.DataFrame) -> pd.DataFrame:
    output_path = Path(path)
    defaults = (
        data.groupby("Product ID", as_index=False)
        .agg(
            Category=("Category", "first"),
            Region=("Region", "first"),
            Weather_Condition=("Weather Condition", "first"),
            Seasonality=("Seasonality", "first"),
            Default_Price=("Price", "median"),
        )
        .rename(columns={"Weather_Condition": "Weather Condition", "Default_Price": "Default Price"})
    )
    defaults["Product ID"] = defaults["Product ID"].astype(str)
    defaults["Product Name"] = ""
    defaults["Default Unit Cost"] = defaults["Default Price"] * 0.5
    catalog = defaults[CATALOG_COLUMNS].set_index("Product ID")

    if output_path.exists():
        saved = pd.read_csv(output_path, dtype={"Product ID": str}).reindex(columns=CATALOG_COLUMNS)
        for _, row in saved.iterrows():
            product_id = str(row["Product ID"])
            values = row.drop(labels="Product ID").dropna().to_dict()
            if product_id in catalog.index:
                for column, value in values.items():
                    if value != "":
                        catalog.loc[product_id, column] = value
            else:
                catalog.loc[product_id, list(values)] = list(values.values())

    return catalog.reset_index().reindex(columns=CATALOG_COLUMNS)


def save_product_catalog(path: str | Path, catalog: pd.DataFrame) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    catalog.reindex(columns=CATALOG_COLUMNS).to_csv(output_path, index=False)


def append_prediction(path: str | Path, values: dict) -> None:
    output_path = Path(path)
    record = {"Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), **values}
    new_data = pd.DataFrame([record])
    if output_path.exists():
        history = pd.read_excel(output_path)
        new_data = pd.concat([history, new_data], ignore_index=True)
    new_data.to_excel(output_path, index=False)