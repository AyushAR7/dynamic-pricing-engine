import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Dynamic Ride Pricing Engine",
    page_icon="🚗",
    layout="wide"
)

import os
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.title("🚗 Dynamic Ride Pricing & Explainability Engine")
st.markdown("Predict fair base prices, apply real-time demand-driven surge multipliers, and inspect SHAP explainability.")

# Sidebar - Input Form
st.sidebar.header("Ride Parameters")

with st.sidebar.form("ride_form"):
    num_riders = st.number_input("Number of Riders (Demand)", min_value=1, max_value=150, value=45)
    num_drivers = st.number_input("Number of Drivers (Supply)", min_value=1, max_value=150, value=30)
    duration = st.slider("Expected Ride Duration (minutes)", min_value=5.0, max_value=120.0, value=35.0, step=1.0)
    vehicle_type = st.selectbox("Vehicle Type", ["Economy", "Premium"])
    location_cat = st.selectbox("Location Category", ["Urban", "Suburban", "Rural"])
    loyalty_status = st.selectbox("Customer Loyalty Status", ["Regular", "Silver", "Gold"])
    time_booking = st.selectbox("Time of Booking", ["Morning", "Afternoon", "Evening", "Night"])
    past_rides = st.number_input("Number of Past Rides", min_value=0, max_value=500, value=25)
    rating = st.slider("Average Rating", min_value=1.0, max_value=5.0, value=4.5, step=0.1)

    submitted = st.form_submit_button("Calculate Dynamic Price")

if submitted:
    payload = {
        "Number_of_Riders": int(num_riders),
        "Number_of_Drivers": int(num_drivers),
        "Location_Category": location_cat,
        "Customer_Loyalty_Status": loyalty_status,
        "Number_of_Past_Rides": int(past_rides),
        "Average_Ratings": float(rating),
        "Time_of_Booking": time_booking,
        "Vehicle_Type": vehicle_type,
        "Expected_Ride_Duration": float(duration)
    }

    try:
        # 1. Fetch Optimization Data
        opt_res = requests.post(f"{API_URL}/optimize", json=payload)
        opt_data = opt_res.json()

        # 2. Fetch SHAP Explanations
        exp_res = requests.post(f"{API_URL}/explain", json=payload)
        exp_data = exp_res.json()

        st.subheader("📊 Pricing Decision Breakdown")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Predicted Base Price", f"${opt_data['base_price']:.2f}")
        col2.metric("Demand / Supply Ratio", f"{opt_data['demand_supply_ratio']:.2f}")
        col3.metric("Surge Multiplier", f"{opt_data['surge_multiplier']:.2f}x")
        col4.metric("Final Dynamic Price", f"${opt_data['dynamic_price']:.2f}")

        st.divider()

        # 3. SHAP Visualization
        st.subheader("🔍 SHAP Base Price Feature Contributions")
        st.markdown(f"**Global Base Value:** USD {exp_data['expected_value']:.2f} &nbsp;|&nbsp; **Net Adjustment:** {opt_data['base_price'] - exp_data['expected_value']:+.2f} USD")

        contributions = exp_data["shap_contributions"]
        shap_df = pd.DataFrame(list(contributions.items()), columns=["Feature", "Impact ($)"])
        
        # Filter to active features with meaningful contribution
        shap_df["Abs_Impact"] = shap_df["Impact ($)"].abs()
        shap_df = shap_df[shap_df["Abs_Impact"] > 0.05].sort_values("Impact ($)", ascending=True)

        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ["#D9534F" if val < 0 else "#2ECC71" for val in shap_df["Impact ($)"]]
        bars = ax.barh(shap_df["Feature"], shap_df["Impact ($)"], color=colors, height=0.6)
        ax.axvline(0, color="gray", linestyle="--", alpha=0.7)
        ax.set_xlabel("Price Impact (USD)", fontsize=10)
        ax.set_title("Local Feature Attribution (Pushed Price Up / Down)", fontsize=12, pad=10)

        # Set dynamic x-axis bounds with padding so text never overlaps y-axis
        min_val = min(shap_df["Impact ($)"].min(), 0)
        max_val = max(shap_df["Impact ($)"].max(), 0)
        x_padding = max(abs(min_val), abs(max_val)) * 0.18
        ax.set_xlim(min_val - x_padding, max_val + x_padding)

        # Annotate labels neatly
        for bar in bars:
            val = bar.get_width()
            if val < 0:
                ax.annotate(f"{val:+.2f}", 
                            (val, bar.get_y() + bar.get_height() / 2),
                            xytext=(-6, 0), textcoords="offset points",
                            va="center", ha="right", fontsize=9, fontweight="bold")
            else:
                ax.annotate(f"{val:+.2f}", 
                            (val, bar.get_y() + bar.get_height() / 2),
                            xytext=(6, 0), textcoords="offset points",
                            va="center", ha="left", fontsize=9, fontweight="bold")

        plt.tight_layout()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Could not connect to FastAPI server. Ensure Uvicorn is active at {API_URL}. Error: {e}")