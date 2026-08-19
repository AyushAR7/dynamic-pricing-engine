import os
import joblib
import optuna
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

# Suppress Optuna verbose logs for cleaner output
optuna.logging.set_verbosity(optuna.logging.WARNING)

os.makedirs("models", exist_ok=True)

# 1. Load engineered dataset
df = pd.read_csv("data/dynamicprice_engineered.csv")

target_col = "Historical_Cost_of_Ride"
X = df.drop(columns=[target_col])
y = df[target_col]

categorical_cols = ["Location_Category", "Customer_Loyalty_Status", "Time_of_Booking", "Vehicle_Type"]
numeric_cols = [col for col in X.columns if col not in categorical_cols]

# 2. Train/Test Split (identical split & seed to Step 2)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Preprocessing Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
    ]
)

# 4. Evaluate Baseline (Pre-Tuning) XGBoost
baseline_model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.08, random_state=42)
baseline_pipe = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", baseline_model)
])
baseline_pipe.fit(X_train, y_train)
y_pred_baseline = baseline_pipe.predict(X_test)

base_rmse = np.sqrt(mean_squared_error(y_test, y_pred_baseline))
base_mae = mean_absolute_error(y_test, y_pred_baseline)
base_r2 = r2_score(y_test, y_pred_baseline)

# 5. Define Optuna Objective Function (5-Fold CV on training set)
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=25),
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "random_state": 42
    }
    
    regressor = XGBRegressor(**params)
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", regressor)
    ])
    
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error")
    return -scores.mean()

# 6. Run Optuna Study
print("=" * 65)
print("Starting Optuna Hyperparameter Optimization (50 Trials)...")
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50, timeout=120)

print("Best Parameters Found:")
for k, v in study.best_params.items():
    print(f"  - {k}: {v}")
print("=" * 65)

# 7. Train Final Model with Best Parameters
best_params = study.best_params
best_params["random_state"] = 42

tuned_model = XGBRegressor(**best_params)
tuned_pipe = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", tuned_model)
])

tuned_pipe.fit(X_train, y_train)
y_pred_tuned = tuned_pipe.predict(X_test)

tuned_rmse = np.sqrt(mean_squared_error(y_test, y_pred_tuned))
tuned_mae = mean_absolute_error(y_test, y_pred_tuned)
tuned_r2 = r2_score(y_test, y_pred_tuned)

# 8. Before / After Comparison Table
print("\n" + "=" * 65)
print(f"{'Metric':<15} | {'Baseline XGBoost':<18} | {'Tuned XGBoost':<18} | {'Improvement':<12}")
print("=" * 65)
print(f"{'RMSE':<15} | {base_rmse:<18.4f} | {tuned_rmse:<18.4f} | {((base_rmse - tuned_rmse) / base_rmse) * 100:>+10.2f}%")
print(f"{'MAE':<15} | {base_mae:<18.4f} | {tuned_mae:<18.4f} | {((base_mae - tuned_mae) / base_mae) * 100:>+10.2f}%")
print(f"{'R2 Score':<15} | {base_r2:<18.4f} | {tuned_r2:<18.4f} | {((tuned_r2 - base_r2) / base_r2) * 100:>+10.2f}%")
print("=" * 65)

# 9. Save the Tuned Model Artifact
joblib.dump(tuned_pipe, "models/model.pkl")
print("Saved tuned model pipeline to: models/model.pkl")