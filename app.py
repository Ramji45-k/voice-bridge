from pathlib import Path
from uuid import uuid4

import pandas as pd
import plotly.express as px
import streamlit as st

from advanced_pricing import optimize_price, train_demand_model
from data_prep import dataset_profile, load_and_clean_data
from pricing_engine import recommend_price
from pricing_storage import append_prediction, load_product_catalog, save_product_catalog


DATA_PATH = Path(__file__).with_name("sales_data.csv")
HISTORY_PATH = Path(__file__).with_name("pricing_predictions.xlsx")
CATALOG_PATH = Path(__file__).with_name("product_catalog.csv")
PLOT_CONFIG = {
    "displayModeBar": True,
    "displaylogo": False,
    "scrollZoom": True,
    "toImageButtonOptions": {
        "format": "png",
        "filename": "dynamic-pricing-chart",
        "height": 800,
        "width": 1400,
        "scale": 2,
    },
}

st.set_page_config(page_title="Dynamic Pricing System", page_icon="₹", layout="wide")
st.title("Dynamic Pricing Intelligence")
st.caption("Commercial performance, named product management, and margin-aware price scenarios")

raw_data = pd.read_csv(DATA_PATH)
data = load_and_clean_data(DATA_PATH)
profile = dataset_profile(raw_data, data)
catalog = load_product_catalog(CATALOG_PATH, data)
if not CATALOG_PATH.exists():
    save_product_catalog(CATALOG_PATH, catalog)


@st.cache_resource(show_spinner="Training demand model...")
def get_demand_model(clean_data: pd.DataFrame):
    return train_demand_model(clean_data)


demand_model, model_metrics = get_demand_model(data)
data["Revenue"] = data["Price"] * data["Units Sold"]
catalog_by_id = catalog.set_index("Product ID", drop=False)
name_by_id = catalog.set_index("Product ID")["Product Name"].fillna("").to_dict()
unnamed_position = {
    product_id: position + 1
    for position, product_id in enumerate(catalog.loc[catalog["Product Name"].fillna("").eq(""), "Product ID"])
}


def product_label(product_id: str) -> str:
    name = str(name_by_id.get(product_id, "")).strip()
    return name if name else f"Name required · product {unnamed_position.get(product_id, 0):03d}"


data["Product Name"] = data["Product ID"].astype(str).map(
    lambda product_id: product_label(product_id)
)

with st.sidebar:
    st.header("Dataset profile")
    st.metric("Rows", f"{profile['clean_rows']:,}")
    st.metric("Columns", profile["columns"])
    st.metric("Missing values", profile["missing_values"])
    st.metric("Duplicates removed", profile["duplicates_removed"])
    st.caption(f"Dates: {data['Date'].min():%d %b %Y} to {data['Date'].max():%d %b %Y}")
    st.divider()
    st.caption("Analytics filters")
    date_range = st.date_input(
        "Date range",
        value=(data["Date"].min().date(), data["Date"].max().date()),
        min_value=data["Date"].min().date(),
        max_value=data["Date"].max().date(),
    )
    category_options = sorted(data["Category"].dropna().unique())
    region_options = sorted(data["Region"].dropna().unique())
    selected_categories = st.multiselect("Categories", category_options, default=category_options)
    selected_regions = st.multiselect("Regions", region_options, default=region_options)

filtered_data = data.copy()
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    filtered_data = filtered_data[
        filtered_data["Date"].between(pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1]) + pd.Timedelta(days=1))
    ]
filtered_data = filtered_data[
    filtered_data["Category"].isin(selected_categories) & filtered_data["Region"].isin(selected_regions)
]

overview_tab, pricing_tab, catalog_tab, history_tab = st.tabs(
    ["Overview", "Price recommendations", "Product catalog", "Prediction history"]
)

with overview_tab:
    st.subheader("Commercial performance")
    if filtered_data.empty:
        st.info("No records match the selected filters.")
    else:
        kpi_one, kpi_two, kpi_three, kpi_four = st.columns(4)
        kpi_one.metric("Revenue", f"₹{filtered_data['Revenue'].sum():,.0f}")
        kpi_two.metric("Units sold", f"{filtered_data['Units Sold'].sum():,.0f}")
        kpi_three.metric("Average selling price", f"₹{filtered_data['Price'].mean():,.2f}")
        kpi_four.metric("Low-stock records", f"{(filtered_data['Inventory Level'] <= filtered_data['Inventory Level'].quantile(0.1)).sum():,}")

        monthly_revenue = (
            filtered_data.assign(Month=filtered_data["Date"].dt.to_period("M").dt.to_timestamp())
            .groupby("Month", as_index=False)["Revenue"]
            .sum()
        )
        category_revenue = (
            filtered_data.groupby("Category", as_index=False)["Revenue"]
            .sum()
            .sort_values("Revenue", ascending=True)
        )
        category_revenue["Revenue (₹)"] = category_revenue.pop("Revenue")
        inventory_demand = (
            filtered_data.groupby(["Product Name", "Category"], as_index=False)
            .agg(Inventory=("Inventory Level", "mean"), Demand=("Demand", "mean"), Revenue=("Revenue", "sum"))
        )
        scatter_data = filtered_data.sample(min(5000, len(filtered_data)), random_state=42)

        chart_left, chart_right = st.columns(2)
        with chart_left:
            st.caption("Monthly revenue")
            figure = px.line(monthly_revenue, x="Month", y="Revenue", markers=True, template="plotly_white")
            figure.update_layout(height=360, margin=dict(l=15, r=15, t=25, b=15), yaxis_title="Revenue (₹)")
            st.plotly_chart(figure, config=PLOT_CONFIG, use_container_width=True)
        with chart_right:
            st.caption("Revenue by category")
            figure = px.bar(category_revenue, x="Revenue (₹)", y="Category", orientation="h", template="plotly_white")
            figure.update_layout(height=360, margin=dict(l=15, r=15, t=25, b=15))
            st.plotly_chart(figure, config=PLOT_CONFIG, use_container_width=True)

        chart_left, chart_right = st.columns(2)
        with chart_left:
            st.caption("Price and observed demand")
            figure = px.scatter(
                scatter_data,
                x="Price",
                y="Demand",
                color="Category",
                hover_name="Product Name",
                hover_data=["Region", "Discount", "Inventory Level"],
                opacity=0.65,
                template="plotly_white",
            )
            figure.update_layout(height=390, margin=dict(l=15, r=15, t=25, b=15))
            st.plotly_chart(figure, config=PLOT_CONFIG, use_container_width=True)
        with chart_right:
            st.caption("Inventory and demand by product")
            figure = px.scatter(
                inventory_demand,
                x="Inventory",
                y="Demand",
                size="Revenue",
                color="Category",
                hover_name="Product Name",
                size_max=38,
                template="plotly_white",
            )
            figure.update_layout(height=390, margin=dict(l=15, r=15, t=25, b=15))
            st.plotly_chart(figure, config=PLOT_CONFIG, use_container_width=True)

        with st.expander("Low inventory watchlist"):
            threshold = filtered_data["Inventory Level"].quantile(0.1)
            watchlist = (
                filtered_data[filtered_data["Inventory Level"] <= threshold]
                .groupby(["Product Name", "Category", "Region"], as_index=False)
                .agg(Inventory=("Inventory Level", "mean"), Demand=("Demand", "mean"), Units_Sold=("Units Sold", "sum"))
                .sort_values(["Demand", "Inventory"], ascending=[False, True])
                .head(25)
            )
            st.dataframe(watchlist, use_container_width=True, hide_index=True)
        st.download_button(
            "Download filtered analytics CSV",
            filtered_data.to_csv(index=False).encode("utf-8"),
            file_name="pricing_analytics.csv",
            mime="text/csv",
        )

with pricing_tab:
    st.subheader("Price recommendation")
    product_ids = catalog.sort_values("Product Name", key=lambda names: names.fillna("").str.casefold())["Product ID"].tolist()
    selected_product_id = st.selectbox("Product name", product_ids, format_func=product_label)
    selected_product = catalog_by_id.loc[selected_product_id]
    existing_rows = data[data["Product ID"].astype(str) == selected_product_id]
    if existing_rows.empty:
        category_rows = data[data["Category"] == selected_product["Category"]]
        source_rows = category_rows if not category_rows.empty else data
        context = source_rows.iloc[0].copy()
        product_defaults = source_rows.median(numeric_only=True)
        context["Product ID"] = selected_product_id
        for column in ("Category", "Region", "Weather Condition", "Seasonality"):
            context[column] = selected_product[column]
    else:
        source_rows = existing_rows
        context = source_rows.iloc[0].copy()
        product_defaults = source_rows.median(numeric_only=True)

    name_default = str(selected_product["Product Name"]) if pd.notna(selected_product["Product Name"]) else ""
    product_name = st.text_input(
        "Product name (required)",
        value=name_default,
        key=f"product_name_{selected_product_id}",
        help="This name is used in the app and saved with every prediction. The internal product ID remains in the record for traceability.",
    ).strip()
    left, right = st.columns(2)
    with left:
        current_price = st.number_input("Current price", min_value=0.01, value=float(selected_product["Default Price"]), step=0.01)
        unit_cost = st.number_input(
            "Unit cost",
            min_value=0.01,
            value=float(selected_product["Default Unit Cost"]),
            step=0.01,
            help="The source dataset has no cost data. Replace the default with your actual unit cost.",
        )
        inventory_level = st.number_input("Inventory level", min_value=0.0, value=float(product_defaults["Inventory Level"]), step=1.0)
        demand = st.number_input("Demand", min_value=0.0, value=float(product_defaults["Demand"]), step=1.0)
    with right:
        competitor_price = st.number_input("Competitor price", min_value=0.01, value=float(product_defaults["Competitor Pricing"]), step=0.01)
        discount = st.slider("Discount (%)", min_value=0, max_value=100, value=int(product_defaults["Discount"]))
        promotion = st.toggle("Promotion active", value=bool(selected_product_id in data["Product ID"].astype(str).values and context["Promotion"]))

    predict_clicked = st.button("Calculate recommendation", type="primary", use_container_width=True)
    if predict_clicked:
        known_names = catalog.loc[catalog["Product ID"] != selected_product_id, "Product Name"].fillna("").astype(str)
        if not product_name:
            st.error("Enter a product name before calculating a recommendation.")
        elif product_name.casefold() in {name.casefold() for name in known_names if name.strip()}:
            st.error("That product name is already in the catalog. Product names must be unique.")
        elif unit_cost > current_price * 1.15:
            st.error("Unit cost is above the optimizer's maximum candidate price. Lower the cost or raise the current price.")
        else:
            catalog.loc[catalog["Product ID"] == selected_product_id, "Product Name"] = product_name
            save_product_catalog(CATALOG_PATH, catalog)
            result = recommend_price(current_price, inventory_level, demand, competitor_price, discount, promotion)
            advanced_result = optimize_price(
                demand_model,
                context,
                current_price,
                inventory_level,
                discount,
                promotion,
                competitor_price,
                unit_cost,
            )
            profit_change = advanced_result.expected_gross_profit - advanced_result.baseline_gross_profit
            st.session_state["pricing_result"] = {
                "signature": (selected_product_id, product_name, current_price, unit_cost, inventory_level, demand, competitor_price, discount, promotion),
                "result": result,
                "advanced_result": advanced_result,
                "profit_change": profit_change,
            }

    stored_result = st.session_state.get("pricing_result")
    current_signature = (selected_product_id, product_name, current_price, unit_cost, inventory_level, demand, competitor_price, discount, promotion)
    if stored_result and stored_result["signature"] == current_signature:
        result = stored_result["result"]
        advanced_result = stored_result["advanced_result"]
        profit_change = stored_result["profit_change"]
        st.divider()
        st.caption(f"Product: **{product_name}** · Internal reference: `{selected_product_id}` · {selected_product['Category']} · {selected_product['Region']}")
        metric_one, metric_two, metric_three, metric_four, metric_five = st.columns(5)
        metric_one.metric("Current price", f"₹{current_price:,.2f}")
        change_percent = (advanced_result.recommended_price / current_price - 1) * 100
        metric_two.metric("Recommended price", f"₹{advanced_result.recommended_price:,.2f}", f"{change_percent:+.1f}%")
        metric_three.metric("Unit cost", f"₹{unit_cost:,.2f}")
        metric_four.metric("Predicted gross profit", f"₹{advanced_result.expected_gross_profit:,.2f}", f"₹{profit_change:+,.2f} vs current")
        metric_five.metric("Predicted demand", f"{advanced_result.predicted_demand:,.0f}")
        st.info(f"Rule signal: **{result.demand_band} demand** and **{result.inventory_band} inventory**. {result.reason}")
        st.caption("Estimated scenario only. Price-response estimates are not causal evidence; test changes under controlled conditions before deployment.")
        profit_figure = px.line(
            advanced_result.candidates,
            x="Price",
            y="Expected Gross Profit",
            markers=True,
            template="plotly_white",
        )
        profit_figure.update_layout(height=390, margin=dict(l=15, r=15, t=25, b=15), yaxis_title="Predicted gross profit (₹)")
        st.plotly_chart(profit_figure, config=PLOT_CONFIG, use_container_width=True)
        if st.button("Save full prediction details to Excel"):
            append_prediction(
                HISTORY_PATH,
                {
                    "Product Name": product_name,
                    "Product ID": selected_product_id,
                    "Category": selected_product["Category"],
                    "Region": selected_product["Region"],
                    "Weather Condition": selected_product["Weather Condition"],
                    "Seasonality": selected_product["Seasonality"],
                    "Current Price": current_price,
                    "Unit Cost": unit_cost,
                    "Competitor Price": competitor_price,
                    "Inventory Level": inventory_level,
                    "Demand Input": demand,
                    "Discount": discount,
                    "Promotion": promotion,
                    "Recommended Price": advanced_result.recommended_price,
                    "Predicted Demand": advanced_result.predicted_demand,
                    "Expected Revenue": advanced_result.expected_revenue,
                    "Expected Gross Profit": advanced_result.expected_gross_profit,
                    "Gross Profit at Current Price": advanced_result.baseline_gross_profit,
                    "Gross Profit Uplift": profit_change,
                },
            )
            st.success(f"Detailed prediction saved to {HISTORY_PATH.name}.")

with catalog_tab:
    st.subheader("Product catalog")
    st.caption("Assign readable names to source products or register new products with their own names and market attributes.")
    rename_col, add_col = st.columns(2)
    with rename_col:
        st.markdown("**Name an existing product**")
        existing_ids = catalog.loc[catalog["Product ID"].isin(data["Product ID"].astype(str)), "Product ID"].tolist()
        rename_id = st.selectbox("Source product", existing_ids, format_func=product_label, key="rename_product_id")
        rename_default = str(catalog.loc[catalog["Product ID"] == rename_id, "Product Name"].iloc[0] or "")
        rename_name = st.text_input("Product name", value=rename_default, key=f"rename_name_{rename_id}").strip()
        if st.button("Save product name", use_container_width=True):
            names = catalog.loc[catalog["Product ID"] != rename_id, "Product Name"].fillna("").astype(str)
            if not rename_name:
                st.error("Product name is required.")
            elif rename_name.casefold() in {name.casefold() for name in names if name.strip()}:
                st.error("That name is already in use.")
            else:
                catalog.loc[catalog["Product ID"] == rename_id, "Product Name"] = rename_name
                save_product_catalog(CATALOG_PATH, catalog)
                st.success("Product name saved.")
                st.rerun()

    with add_col:
        st.markdown("**Register a new product**")
        with st.form("new_product_form", clear_on_submit=True):
            new_name = st.text_input("New product name (required)").strip()
            category = st.selectbox("Category", sorted(data["Category"].dropna().unique()))
            region = st.selectbox("Region", sorted(data["Region"].dropna().unique()))
            weather = st.selectbox("Weather condition", sorted(data["Weather Condition"].dropna().unique()))
            seasonality = st.selectbox("Seasonality", sorted(data["Seasonality"].dropna().unique()))
            new_price = st.number_input("Default selling price", min_value=0.01, value=float(data["Price"].median()), step=0.01)
            new_cost = st.number_input("Default unit cost", min_value=0.01, value=float(data["Price"].median() * 0.5), step=0.01)
            add_product_clicked = st.form_submit_button("Add product to catalog", use_container_width=True)
        if add_product_clicked:
            names = catalog["Product Name"].fillna("").astype(str)
            if not new_name:
                st.error("Product name is required.")
            elif new_name.casefold() in {name.casefold() for name in names if name.strip()}:
                st.error("That product name is already in use.")
            elif new_cost >= new_price:
                st.error("Default unit cost must be below the default selling price.")
            else:
                new_product = pd.DataFrame(
                    [[f"NEW-{uuid4().hex[:10].upper()}", new_name, category, region, weather, seasonality, new_price, new_cost]],
                    columns=catalog.columns,
                )
                catalog = pd.concat([catalog, new_product], ignore_index=True)
                save_product_catalog(CATALOG_PATH, catalog)
                st.success(f"{new_name} was added. It is ready for price recommendations.")
                st.rerun()

    catalog_view = catalog.copy()
    catalog_view["Catalog status"] = "Active"
    st.dataframe(
        catalog_view[["Product Name", "Product ID", "Catalog status", "Category", "Region", "Default Price", "Default Unit Cost"]].rename(
            columns={"Product ID": "Internal Product ID"}
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.download_button(
        "Download product catalog CSV",
        catalog.to_csv(index=False).encode("utf-8"),
        file_name="product_catalog.csv",
        mime="text/csv",
    )

with history_tab:
    st.subheader("Prediction history")
    if HISTORY_PATH.exists():
        history = pd.read_excel(HISTORY_PATH)
        st.dataframe(history.tail(100), use_container_width=True, hide_index=True)
        st.download_button(
            "Download prediction history CSV",
            history.to_csv(index=False).encode("utf-8"),
            file_name="pricing_predictions.csv",
            mime="text/csv",
        )
    else:
        st.info("Saved pricing decisions will appear here with the product name, internal reference, market context, and profit estimates.")
    with st.expander("Model evaluation and data fields"):
        st.write(f"Chronological holdout: MAE **{model_metrics.mean_absolute_error}**, R² **{model_metrics.r2_score}**")
        st.write("Numeric variables")
        st.code(", ".join(data.select_dtypes(include="number").columns))
        st.write("Categorical variables")
        st.code(", ".join(data.select_dtypes(exclude="number").columns))
        st.dataframe(data.head(10), use_container_width=True, hide_index=True)