"""
app.py — Streamlit Web Application
====================================
Flight Price Prediction System
AIC354 Machine Learning — Assignment 3

Usage:
    streamlit run app.py

The app loads 'models/flight_model.pkl' (the best trained model
saved from the notebook) and lets the user input flight parameters
interactively to receive a real-time price prediction.
"""

import os
import numpy  as np
import pandas as pd
import joblib
import streamlit as st

# ─────────────────────────────────────────────────────────────
#  Page Configuration
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="✈️ Flight Price Predictor",
    page_icon="✈️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
#  Constants — must match the training dataset vocabulary
# ─────────────────────────────────────────────────────────────
AIRLINES       = ["Air_India", "AirAsia", "GO_FIRST", "Indigo", "SpiceJet", "Vistara"]
CITIES         = ["Bangalore", "Chennai", "Delhi", "Hyderabad", "Kolkata", "Mumbai"]
TIMES          = ["Early_Morning", "Morning", "Afternoon", "Evening", "Night", "Late_Night"]
STOPS          = ["zero", "one", "two_or_more"]
CLASSES        = ["Economy", "Business"]
MODEL_PATH     = os.path.join(os.path.dirname(__file__), "models", "flight_model.pkl")

# ─────────────────────────────────────────────────────────────
#  Load Model (cached to avoid reloading on every interaction)
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(path: str):
    """Load the saved joblib model pipeline."""
    if not os.path.exists(path):
        return None
    return joblib.load(path)

model = load_model(MODEL_PATH)

# ─────────────────────────────────────────────────────────────
#  Header Section
# ─────────────────────────────────────────────────────────────
st.markdown("""
<h1 style='text-align: center; color: #1F4E79;'>✈️ Flight Price Predictor</h1>
<p style='text-align: center; color: #555; font-size: 1.05rem;'>
    AIC354 ML Assignment 3 | Ridge · Lasso · Extra Trees · AdaBoost · XGBoost
</p>
<hr>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
#  Sidebar — Model Status
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This web application uses a trained **Machine Learning model** to predict
    Indian domestic flight prices based on:
    - Airline carrier
    - Route (source → destination)
    - Departure & Arrival times
    - Number of stops
    - Travel class
    - Flight duration
    - Days left before departure
    """)
    st.divider()

    if model is not None:
        st.success(f"✅ Model loaded successfully")
        st.caption(f"Path: `{MODEL_PATH}`")
    else:
        st.error("❌ Model not found!")
        st.caption(f"Expected at: `{MODEL_PATH}`")
        st.caption("Run the notebook first to train and save the model.")

    st.divider()
    st.markdown("**Course:** AIC354 — Machine Learning")
    st.markdown("**University:** COMSATS Islamabad")

# ─────────────────────────────────────────────────────────────
#  Main Input Form
# ─────────────────────────────────────────────────────────────
st.subheader("🔍 Enter Flight Details")

with st.form(key="prediction_form"):

    # ── Row 1: Airline + Class ──────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        airline = st.selectbox(
            "✈️ Airline",
            options=AIRLINES,
            index=AIRLINES.index("Vistara"),
            help="Select the airline carrier."
        )
    with col2:
        flight_class = st.selectbox(
            "💺 Class",
            options=CLASSES,
            index=0,
            help="Economy or Business class."
        )

    # ── Row 2: Source + Destination ──────────────────────────
    col3, col4 = st.columns(2)
    with col3:
        source_city = st.selectbox(
            "🛫 Source City",
            options=CITIES,
            index=CITIES.index("Delhi"),
            help="City of departure."
        )
    with col4:
        destination_city = st.selectbox(
            "🛬 Destination City",
            options=CITIES,
            index=CITIES.index("Mumbai"),
            help="City of arrival."
        )

    # ── Row 3: Departure + Arrival Time ──────────────────────
    col5, col6 = st.columns(2)
    with col5:
        departure_time = st.selectbox(
            "🌅 Departure Time",
            options=TIMES,
            index=TIMES.index("Morning"),
        )
    with col6:
        arrival_time = st.selectbox(
            "🌆 Arrival Time",
            options=TIMES,
            index=TIMES.index("Afternoon"),
        )

    # ── Row 4: Stops + Duration ──────────────────────────────
    col7, col8 = st.columns(2)
    with col7:
        stops = st.selectbox(
            "🔄 Number of Stops",
            options=STOPS,
            format_func=lambda x: x.replace("_", " ").title(),
            index=0,
        )
    with col8:
        duration = st.number_input(
            "⏱️ Flight Duration (hours)",
            min_value=0.5,
            max_value=50.0,
            value=2.5,
            step=0.25,
            format="%.2f",
            help="Total travel duration in hours."
        )

    # ── Row 5: Days Left ─────────────────────────────────────
    days_left = st.slider(
        "📅 Days Left Before Departure",
        min_value=1,
        max_value=49,
        value=15,
        help="How many days from today until the flight."
    )

    # ── Submit Button ─────────────────────────────────────────
    submitted = st.form_submit_button(
        "🔮 Predict Price",
        use_container_width=True,
        type="primary"
    )

# ─────────────────────────────────────────────────────────────
#  Prediction Logic
# ─────────────────────────────────────────────────────────────
if submitted:

    # Validation: source ≠ destination
    if source_city == destination_city:
        st.error("⚠️ Source and destination cities cannot be the same!")
    elif model is None:
        st.error("❌ Model is not loaded. Please run the notebook to generate `models/flight_model.pkl`.")
    else:
        # Build input DataFrame matching training schema exactly
        input_data = pd.DataFrame([{
            "airline"          : airline,
            "source_city"      : source_city,
            "destination_city" : destination_city,
            "departure_time"   : departure_time,
            "stops"            : stops,
            "arrival_time"     : arrival_time,
            "class"            : flight_class,
            "duration"         : float(duration),
            "days_left"        : int(days_left),
        }])

        # Predict (model returns log-transformed price)
        with st.spinner("🧠 Predicting price ..."):
            pred_log   = model.predict(input_data)[0]
            pred_price = np.expm1(pred_log)   # inverse log transform → ₹

        # ── Display Result ────────────────────────────────────
        st.divider()
        st.markdown("### 🎯 Prediction Result")

        result_col1, result_col2, result_col3 = st.columns(3)
        with result_col1:
            st.metric("Predicted Price", f"₹{pred_price:,.0f}")
        with result_col2:
            price_usd = pred_price / 83.5    # approx INR → USD
            st.metric("Approx. (USD)", f"${price_usd:,.0f}")
        with result_col3:
            # Price category
            if pred_price < 5000:
                category = "🟢 Budget"
            elif pred_price < 15000:
                category = "🟡 Mid-Range"
            elif pred_price < 40000:
                category = "🟠 Premium"
            else:
                category = "🔴 Luxury"
            st.metric("Category", category)

        # ── Input Summary ─────────────────────────────────────
        st.divider()
        st.markdown("### 📋 Input Summary")
        summary_data = {
            "Parameter"  : ["Airline", "Route", "Class", "Departure",
                             "Arrival", "Stops", "Duration", "Days Left"],
            "Value"      : [
                airline,
                f"{source_city} → {destination_city}",
                flight_class,
                departure_time.replace("_", " "),
                arrival_time.replace("_", " "),
                stops.replace("_", " ").title(),
                f"{duration:.2f} hours",
                f"{days_left} days",
            ]
        }
        st.table(pd.DataFrame(summary_data).set_index("Parameter"))

        # ── Confidence Note ───────────────────────────────────
        st.info(
            "💡 **Note:** This prediction is based on a model trained with "
            "10-Fold Cross-Validation. Actual prices may vary based on real-time "
            "availability, demand surges, and airline-specific fare rules."
        )

# ─────────────────────────────────────────────────────────────
#  Footer
# ─────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<p style='text-align:center; color:#999; font-size:0.85rem;'>"
    "AIC354 Machine Learning · Assignment 3 · COMSATS University Islamabad"
    "</p>",
    unsafe_allow_html=True
)
