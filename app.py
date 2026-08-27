import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Hotel Reservation Cancellation Predictor & Dynamic Overbooking Strategy",
    page_icon="🤖",
    layout="wide"
)

st.title("🎯 Hotel Reservation Cancellation Predictor & Dynamic Overbooking Strategy")
st.markdown("**Domain**: `Hospitality & Travel Tech` | **Tech Stack**: `LightGBM, XGBoost, Streamlit`")
st.markdown("**Author**: [Arjuna Fransesco](https://github.com/ArjunaFransesco) | **GitHub**: [Portfolio Repositories](https://github.com/ArjunaFransesco?tab=repositories)")
st.markdown("---")

col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("⚙️ Domain Input Telemetry")
    lead_time_days = st.slider("Lead Time Days", int(1), int(365), int(65))
    avg_daily_rate_adr = st.slider("Avg Daily Rate Adr", float(35.0), float(350.0), float(120.0))
    total_stay_nights = st.slider("Total Stay Nights", int(1), int(14), int(3))
    special_requests_count = st.slider("Special Requests Count", int(0), int(5), int(1))
    previous_cancellations_count = st.slider("Previous Cancellations Count", int(0), int(4), int(0))
    booking_channel_score = st.slider("Booking Channel Score", int(1), int(4), int(2))

with col2:
    st.subheader("🔮 Predictive Model Inference")
    model_path = os.path.join(os.path.dirname(__file__), "models/model_pipeline.joblib")
    scaler_path = os.path.join(os.path.dirname(__file__), "models/scaler.joblib")
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        input_df = pd.DataFrame([{"lead_time_days": lead_time_days, "avg_daily_rate_adr": avg_daily_rate_adr, "total_stay_nights": total_stay_nights, "special_requests_count": special_requests_count, "previous_cancellations_count": previous_cancellations_count, "booking_channel_score": booking_channel_score}])
        input_scaled = scaler.transform(input_df)
        pred = model.predict(input_scaled)[0]
        
        st.markdown("#### Real-Time Prediction Output")
        st.info(f"Predicted `is_canceled`: **{pred}**")
        
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_scaled)[0]
            st.progress(float(probs[1]) if len(probs) > 1 else float(probs[0]))
            st.caption(f"Confidence Probability Score: **{np.max(probs):.2%}**")
    else:
        st.warning("Model or Scaler artifact not found in models/ directory.")

st.markdown("---")
st.markdown("### 📊 Benchmark Metrics")
metrics_path = os.path.join(os.path.dirname(__file__), "reports/metrics.json")
if os.path.exists(metrics_path):
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics_data = json.load(f)
    st.json(metrics_data)
