import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

os.makedirs("models", exist_ok=True)
os.makedirs("notebooks", exist_ok=True)

# 1. Load engineered dataset
df = pd.read_csv("data/dynamicprice_engineered.csv")

target_col = "Historical_Cost_of_Ride"
X = df.drop(columns=[target_col])
y = df[target_col]

categorical_cols = ["Location_Category", "Customer_Loyalty_Status", "Time_of_Booking", "Vehicle_Type"]
numeric_cols = [col for col in X.columns if col not in categorical_cols]

# 2. Train/Test Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Preprocessing Setup
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
    ]
)

# 4. Train Models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.08, random_state=42)
}

trained_pipelines = {}

print("=" * 65)
print(f"{'Model':<20} | {'RMSE':<12} | {'MAE':<12} | {'R2 Score':<10}")
print("=" * 65)

for name, model in models.items():
    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", model)
    ])
    
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    trained_pipelines[name] = pipe
    print(f"{name:<20} | {rmse:<12.4f} | {mae:<12.4f} | {r2:<10.4f}")

print("=" * 65)

# 5. Extract Feature Importance for Winning Tree Model (XGBoost)
winning_model_name = "XGBoost"
winning_pipe = trained_pipelines[winning_model_name]
regressor = winning_pipe.named_steps["regressor"]

cat_encoder = winning_pipe.named_steps["preprocessor"].named_transformers_["cat"]
encoded_cat_names = cat_encoder.get_feature_names_out(categorical_cols).tolist()
feature_names = numeric_cols + encoded_cat_names

feat_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": regressor.feature_importances_
}).sort_values("Importance", ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(data=feat_df.head(8), x="Importance", y="Feature", palette="Blues_r")
plt.title(f"Top Feature Importances ({winning_model_name})")
plt.tight_layout()
plt.savefig("notebooks/feature_importance.png")
print("Saved feature importance chart to: notebooks/feature_importance.png")

# 6. Save Model Artifact for Step 3 & Serving
joblib.dump(winning_pipe, "models/model.pkl")
print("Saved winning XGBoost pipeline to: models/model.pkl")

# 7. Written Justification
print("\n" + "=" * 60)
print("WINNER JUSTIFICATION")
print("=" * 60)
print("XGBoost is selected as the winning architecture because it models multi-variable interactions between duration, location tier, and vehicle type without requiring manual feature cross-products.")
print("While Linear Regression fits the dominant duration signal, XGBoost enables non-linear boundary adjustments and directly integrates with SHAP TreeExplainer for Layer 3 interpretability.")
print("=" * 60)