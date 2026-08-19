import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_style("whitegrid")
os.makedirs("notebooks", exist_ok=True)

# 1. Define the Surge Multiplier Curve
def calculate_surge_multiplier(demand_supply_ratio: float) -> float:
    """
    Calculates dynamic surge multiplier based on the rider/driver ratio:
    - Balanced (ratio <= 1.0): 1.0x (base price)
    - Moderate Demand (1.0 < ratio <= 1.5): Smooth ramp from 1.0x to 1.35x
    - High Surge (1.5 < ratio <= 2.5): Steeper ramp up to 1.8x
    - Extreme Surge (ratio > 2.5): Capped at 2.2x to prevent customer churn
    """
    if demand_supply_ratio <= 1.0:
        return 1.0
    elif demand_supply_ratio <= 1.5:
        return round(1.0 + 0.7 * (demand_supply_ratio - 1.0), 3)
    elif demand_supply_ratio <= 2.5:
        return round(1.35 + 0.45 * (demand_supply_ratio - 1.5), 3)
    else:
        # Cap multiplier
        return round(min(1.8 + 0.2 * (demand_supply_ratio - 2.5), 2.2), 3)

def calculate_dynamic_price(base_price: float, demand_supply_ratio: float) -> tuple[float, float]:
    multiplier = calculate_surge_multiplier(demand_supply_ratio)
    dynamic_price = round(base_price * multiplier, 2)
    return dynamic_price, multiplier

# 2. Visualize the Surge Curve
ratios = np.linspace(0.2, 3.5, 200)
multipliers = [calculate_surge_multiplier(r) for r in ratios]

plt.figure(figsize=(8, 5))
plt.plot(ratios, multipliers, color="#1F4E79", lw=2.5, label="Surge Multiplier Curve")
plt.axvline(1.0, color="gray", linestyle="--", alpha=0.7, label="Equilibrium (1.0)")
plt.axvline(2.5, color="red", linestyle="--", alpha=0.7, label="Cap Threshold (2.5)")
plt.title("Dynamic Pricing: Surge Multiplier vs. Demand/Supply Ratio")
plt.xlabel("Demand / Supply Ratio (Riders / Drivers)")
plt.ylabel("Surge Multiplier")
plt.legend()
plt.tight_layout()
plt.savefig("notebooks/surge_curve.png")
print("Saved surge curve visualization to: notebooks/surge_curve.png")

# 3. Test Sample Scenarios
test_cases = [
    {"riders": 20, "drivers": 40, "base_price": 250.0},
    {"riders": 30, "drivers": 30, "base_price": 250.0},
    {"riders": 45, "drivers": 30, "base_price": 250.0},
    {"riders": 60, "drivers": 30, "base_price": 250.0},
    {"riders": 90, "drivers": 25, "base_price": 250.0},
]

print("\n" + "=" * 70)
print(f"{'Riders':<8} | {'Drivers':<8} | {'Ratio':<8} | {'Base Price':<12} | {'Multiplier':<12} | {'Dynamic Price':<12}")
print("=" * 70)
for case in test_cases:
    ratio = case["riders"] / case["drivers"]
    dyn_price, mult = calculate_dynamic_price(case["base_price"], ratio)
    print(f"{case['riders']:<8} | {case['drivers']:<8} | {ratio:<8.2f} | ${case['base_price']:<11.2f} | {mult:<11.2f}x | ${dyn_price:<11.2f}")
print("=" * 70)