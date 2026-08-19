"""
Step 1: Data Loading + Exploratory Data Analysis
Dynamic Ride Pricing Engine
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100

# ---- Load data ----
df = pd.read_csv("data/dynamicprice.csv")  # adjust path if needed

print("=" * 60)
print("SHAPE:", df.shape)
print("=" * 60)
print("\nDTYPES:\n", df.dtypes)
print("\nMISSING VALUES:\n", df.isnull().sum())
print("\nDUPLICATES:", df.duplicated().sum())
print("\nDESCRIBE (numeric):\n", df.describe().T)

print("\n" + "=" * 60)
print("CATEGORICAL VALUE COUNTS")
print("=" * 60)
for col in ["Location_Category", "Customer_Loyalty_Status", "Time_of_Booking", "Vehicle_Type"]:
    print(f"\n{col}:\n{df[col].value_counts()}")

# ---- Feature engineering: the core economic signal ----
df["Demand_Supply_Ratio"] = df["Number_of_Riders"] / df["Number_of_Drivers"]

print("\n" + "=" * 60)
print("DEMAND_SUPPLY_RATIO stats")
print("=" * 60)
print(df["Demand_Supply_Ratio"].describe())

# ---- Correlation of numeric features with target ----
numeric_cols = ["Number_of_Riders", "Number_of_Drivers", "Demand_Supply_Ratio",
                 "Number_of_Past_Rides", "Average_Ratings", "Expected_Ride_Duration",
                 "Historical_Cost_of_Ride"]
print("\n" + "=" * 60)
print("CORRELATION WITH TARGET (Historical_Cost_of_Ride)")
print("=" * 60)
print(df[numeric_cols].corr()["Historical_Cost_of_Ride"].sort_values(ascending=False))

# ---- Save cleaned + engineered version for next steps ----
df.to_csv("data/dynamicprice_engineered.csv", index=False)
print("\nSaved engineered dataset.")

# ---- Plots ----
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

axes[0, 0].hist(df["Historical_Cost_of_Ride"], bins=40, color="#1F4E79")
axes[0, 0].set_title("Distribution of Ride Cost (Target)")
axes[0, 0].set_xlabel("Historical Cost of Ride")

axes[0, 1].scatter(df["Demand_Supply_Ratio"], df["Historical_Cost_of_Ride"], alpha=0.3, color="#1F4E79")
axes[0, 1].set_title("Cost vs Demand/Supply Ratio")
axes[0, 1].set_xlabel("Demand_Supply_Ratio")
axes[0, 1].set_ylabel("Historical Cost of Ride")

sns.boxplot(data=df, x="Vehicle_Type", y="Historical_Cost_of_Ride", ax=axes[1, 0],
            hue="Vehicle_Type", palette=["#1F4E79", "#8FAADC"], legend=False)
axes[1, 0].set_title("Cost by Vehicle Type")

sns.boxplot(data=df, x="Time_of_Booking", y="Historical_Cost_of_Ride", ax=axes[1, 1],
            hue="Time_of_Booking", palette="Blues", legend=False)
axes[1, 1].set_title("Cost by Time of Booking")
axes[1, 1].tick_params(axis="x", rotation=20)

plt.tight_layout()
plt.savefig("notebooks/eda_overview.png")
print("\nSaved eda_overview.png")

# ---- Correlation heatmap ----
plt.figure(figsize=(7, 5))
sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap="Blues")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("notebooks/correlation_heatmap.png")
print("Saved correlation_heatmap.png")