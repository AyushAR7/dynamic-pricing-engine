# 🚗 Dynamic Ride Pricing & Explainability Engine

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ayush-pricing-model.streamlit.app)
[![FastAPI Docs](https://img.shields.io/badge/FastAPI-API%20Docs-009688?logo=fastapi&logoColor=white)](https://romans-prices.onrender.com/docs)
[![Python Version](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning pipeline that predicts fair base ride costs, applies bounded real-time surge optimization based on demand-supply ratios, and provides transparent local/global feature attributions using SHAP.

🔗 **Live Interactive App:** [ayush-pricing-model.streamlit.app](https://ayush-pricing-model.streamlit.app)  
⚡ **Production API Swagger Docs:** [romans-prices.onrender.com/docs](https://romans-prices.onrender.com/docs)

---

## 📌 Problem Overview

Standard ride-hailing algorithms frequently struggle to balance driver supply with dynamic passenger demand while maintaining pricing transparency. This project solves that tradeoff by combining:
1. **Accurate Baseline Pricing:** Supervised regression modeling based on trip duration, vehicle tier, and historical ride context.
2. **Surge Multiplier Optimization:** A bounded non-linear pricing adjustment engine driven by real-time localized supply-demand elasticity.
3. **Model Explainability:** SHAP TreeExplainer integration to break down exact feature contributions per prediction and eliminate black-box decision making.

---

## 🏗️ Architecture & Pipeline

```text
       [ Raw Ride Data ]
               │
               ▼
   [ Feature Engineering & EDA ] ───► Ratio Metrics (Demand / Supply)
               │
               ▼
    [ Model Selection & Tuning ] ───► XGBoost Regressor (Optuna CV Optimization)
               │
               ▼
     [ Serialized Pipeline ] ────► models/model.pkl
               │
       ┌───────┴───────────────────────────┐
       ▼                                   ▼
[ FastAPI Backend ]               [ SHAP Interpretability ]
  ├── /predict                      ├── Global Summary Attributions
  ├── /optimize                     └── Local Waterfall Decomposition
  └── /explain                             │
       │                                   ▼
       └─────────────────────────► [ Streamlit Frontend ]