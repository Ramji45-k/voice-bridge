from dataclasses import dataclass


@dataclass(frozen=True)
class PricingResult:
    recommended_price: float
    change_percent: float
    demand_band: str
    inventory_band: str
    reason: str


def recommend_price(
    current_price: float,
    inventory_level: float,
    demand: float,
    competitor_price: float,
    discount: float = 0,
    promotion: bool = False,
) -> PricingResult:
    """Calculate a bounded recommendation using interpretable business rules."""
    if current_price <= 0 or competitor_price <= 0:
        raise ValueError("Current and competitor prices must be greater than zero.")
    if inventory_level < 0 or demand < 0:
        raise ValueError("Inventory and demand cannot be negative.")

    demand_band = "High" if demand >= 150 else "Medium" if demand >= 80 else "Low"
    inventory_band = "Low" if inventory_level <= 75 else "Medium" if inventory_level <= 180 else "High"
    adjustment = 0.0
    reasons: list[str] = []

    if demand_band == "High" and inventory_band == "Low":
        adjustment += 0.08
        reasons.append("high demand and low inventory")
    elif demand_band == "High":
        adjustment += 0.04
        reasons.append("high demand")
    elif demand_band == "Low" and inventory_band == "High":
        adjustment -= 0.08
        reasons.append("low demand and high inventory")
    elif demand_band == "Low":
        adjustment -= 0.04
        reasons.append("low demand")
    else:
        reasons.append("balanced demand and inventory")

    competitor_gap = (competitor_price - current_price) / current_price
    if competitor_gap >= 0.05:
        adjustment += 0.03
        reasons.append("competitor price is higher")
    elif competitor_gap <= -0.05:
        adjustment -= 0.03
        reasons.append("competitor price is lower")

    if promotion:
        adjustment -= 0.02
        reasons.append("promotion is active")
    if discount >= 15:
        adjustment -= 0.01
        reasons.append("discount is already high")

    adjustment = max(-0.15, min(0.15, adjustment))
    recommended_price = round(current_price * (1 + adjustment), 2)
    change_percent = round(adjustment * 100, 1)
    reason = "; ".join(reasons).capitalize() + "."
    return PricingResult(recommended_price, change_percent, demand_band, inventory_band, reason)