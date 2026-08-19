import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

os.makedirs("notebooks", exist_ok=True)

# 1. Load Data and Tuned Pipeline Artifact
df = pd.read_csv("data/dynamicprice_engineered.csv")
pipeline = joblib.load("models/model.pkl")

target_col = "Historical_Cost_of_Ride"
X = df.drop(columns=[target_col])
y = df[target_col]

# 2. Extract Preprocessor and XGBoost Regressor
preprocessor = pipeline.named_steps["preprocessor"]
model = pipeline.named_steps["regressor"]

# Transform features through pipeline's ColumnTransformer
X_transformed = preprocessor.transform(X)

# Retrieve transformed feature names
categorical_cols = ["Location_Category", "Customer_Loyalty_Status", "Time_of_Booking", "Vehicle_Type"]
numeric_cols = [col for col in X.columns if col not in categorical_cols]
cat_encoder = preprocessor.named_transformers_["cat"]
encoded_cat_names = cat_encoder.get_feature_names_out(categorical_cols).tolist()
feature_names = numeric_cols + encoded_cat_names

# Convert transformed array to a structured DataFrame for SHAP
X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names)

# 3. Initialize SHAP TreeExplainer
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_transformed_df)

# 4. Global Feature Importance (Beeswarm Summary Plot)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_transformed_df, show=False)
plt.title("SHAP Global Feature Importance (Beeswarm)", fontsize=14, pad=15)
plt.tight_layout()
plt.savefig("notebooks/shap_global_summary.png", bbox_inches="tight")
plt.close()
print("Saved global summary plot to: notebooks/shap_global_summary.png")

# 5. Global Feature Importance (Bar Plot)
plt.figure(figsize=(10, 6))
shap.plots.bar(shap_values, show=False)
plt.title("SHAP Mean Absolute Feature Impact", fontsize=14, pad=15)
plt.tight_layout()
plt.savefig("notebooks/shap_global_bar.png", bbox_inches="tight")
plt.close()
print("Saved global bar plot to: notebooks/shap_global_bar.png")

# 6. Local / Per-Prediction Waterfall Plot (Sample #0)
sample_idx = 0
plt.figure(figsize=(10, 6))
shap.plots.waterfall(shap_values[sample_idx], show=False)
plt.title(f"SHAP Waterfall Explanation for Ride #{sample_idx}", fontsize=14, pad=15)
plt.tight_layout()
plt.savefig("notebooks/shap_local_waterfall.png", bbox_inches="tight")
plt.close()
print("Saved sample waterfall plot to: notebooks/shap_local_waterfall.png")

# 7. Print Sample Prediction Breakdown
base_val = explainer.expected_value
pred_val = model.predict(X_transformed_df.iloc[[sample_idx]])[0]
print("\n" + "=" * 60)
print(f"SHAP EXPLANATION BREAKDOWN (Ride #{sample_idx})")
print("=" * 60)
print(f"Base Value (Average Dataset Target): ${base_val:.2f}")
print(f"Model Predicted Base Price:        ${pred_val:.2f}")
print(f"Net SHAP Adjustment:               ${(pred_val - base_val):+.2f}")
print("=" * 60)