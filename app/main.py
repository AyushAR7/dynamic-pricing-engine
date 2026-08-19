import joblib
import pandas as pd
import numpy as np
import shap
from fastapi import FastAPI, HTTPException
from app.schemas import RideRequest, BasePriceResponse, DynamicPriceResponse, ExplainResponse

app = FastAPI(
    title="Dynamic Ride Pricing Engine API",
    description="Layered pricing API: Prediction -> Optimization -> Explainability",
    version="1.0.0"
)

# 1. Load the trained pipeline and initialize TreeExplainer
try:
    pipeline = joblib.load("models/model.pkl")
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["regressor"]
    explainer = shap.TreeExplainer(model)
except Exception as e:
    raise RuntimeError(f"Failed to load model pipeline: {e}")

# Retrieve feature columns for SHAP mapping
categorical_cols = ["Location_Category", "Customer_Loyalty_Status", "Time_of_Booking", "Vehicle_Type"]
cat_encoder = preprocessor.named_transformers_["cat"]
encoded_cat_names = cat_encoder.get_feature_names_out(categorical_cols).tolist()

def prepare_dataframe(ride: RideRequest) -> pd.DataFrame:
    data = ride.dict()
    data["Demand_Supply_Ratio"] = data["Number_of_Riders"] / data["Number_of_Drivers"]
    return pd.DataFrame([data])

def get_surge_multiplier(ratio: float) -> float:
    if ratio <= 1.0:
        return 1.0
    elif ratio <= 1.5:
        return round(1.0 + 0.7 * (ratio - 1.0), 3)
    elif ratio <= 2.5:
        return round(1.35 + 0.45 * (ratio - 1.5), 3)
    else:
        return round(min(1.8 + 0.2 * (ratio - 2.5), 2.2), 3)

@app.get("/")
def health_check():
    return {"status": "healthy", "service": "Dynamic Ride Pricing API"}

@app.post("/predict", response_model=BasePriceResponse)
def predict_base_price(ride: RideRequest):
    df_input = prepare_dataframe(ride)
    base_price = float(pipeline.predict(df_input)[0])
    return BasePriceResponse(base_price=round(base_price, 2))

@app.post("/optimize", response_model=DynamicPriceResponse)
def optimize_price(ride: RideRequest):
    df_input = prepare_dataframe(ride)
    base_price = float(pipeline.predict(df_input)[0])
    ratio = df_input["Demand_Supply_Ratio"].iloc[0]
    multiplier = get_surge_multiplier(ratio)
    dynamic_price = round(base_price * multiplier, 2)
    return DynamicPriceResponse(
        base_price=round(base_price, 2),
        demand_supply_ratio=round(ratio, 3),
        surge_multiplier=multiplier,
        dynamic_price=dynamic_price
    )

@app.post("/explain", response_model=ExplainResponse)
def explain_prediction(ride: RideRequest):
    df_input = prepare_dataframe(ride)
    base_price = float(pipeline.predict(df_input)[0])
    
    numeric_cols = [col for col in df_input.columns if col not in categorical_cols]
    all_features = numeric_cols + encoded_cat_names
    
    transformed_input = preprocessor.transform(df_input)
    shap_vals = explainer(pd.DataFrame(transformed_input, columns=all_features))
    
    # Map top contributors
    feature_contributions = dict(zip(all_features, [round(float(v), 2) for v in shap_vals.values[0]]))
    
    return ExplainResponse(
        base_price=round(base_price, 2),
        expected_value=round(float(explainer.expected_value), 2),
        shap_contributions=feature_contributions
    )