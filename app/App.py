import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from datetime import date, timedelta

# Load model and data

BASE = Path(__file__).resolve().parent.parent

model = joblib.load(BASE / "models" / "best_model.pkl")

df = pd.read_csv(BASE / "data" / "final_hotel_bookings.csv")

df["agent"] = df["agent"].astype(str)
df["company"] = df["company"].astype(str)

# App title
st.title("🏨 Hotel Booking Prediction")

st.write("Enter booking information to predict cancellation.")

# Default values for required model features
features = list(model.feature_names_in_)
data = {}

for col in features:
    if pd.api.types.is_numeric_dtype(df[col]):
        data[col] = df[col].median()
    else:
        data[col] = df[col].mode()[0]

# User inputs
data["hotel"] = st.selectbox(
    "Hotel Type", df["hotel"].unique()
)

arrival_date = st.date_input(
    "Arrival Date",
    value=date.today(),
    min_value=date.today()
)

data["adults"] = st.number_input(
    "Number of Adults", min_value=1, value=2
)

data["children"] = st.number_input(
    "Number of Children", min_value=0, value=0
)

data["total_nights"] = st.number_input(
    "Total Nights", min_value=1, value=3
)

data["adr"] = st.number_input(
    "Room Price Per Night (€)", min_value=0.0, value=100.0
)

data["deposit_type"] = st.selectbox(
    "Deposit Type", df["deposit_type"].unique()
)

# Calculate arrival information
data["lead_time"] = (arrival_date - date.today()).days
data["arrival_date_year"] = arrival_date.year
data["arrival_date_month"] = arrival_date.month
data["arrival_date_day_of_month"] = arrival_date.day
data["arrival_date_week_number"] = arrival_date.isocalendar()[1]

# Calculate departure date
departure_date = arrival_date + timedelta(
    days=int(data["total_nights"])
)

st.write("Departure Date:", departure_date.strftime("%d/%m/%Y"))

# Calculate weekend and weekday nights
weekend_nights = 0

for i in range(int(data["total_nights"])):
    stay_date = arrival_date + timedelta(days=i)

    if stay_date.weekday() in [5, 6]:
        weekend_nights += 1

data["stays_in_weekend_nights"] = weekend_nights

data["stays_in_week_nights"] = (
    data["total_nights"] - weekend_nights
)

# Calculate engineered features
data["total_guests"] = (
    data["adults"] + data["children"] + data["babies"]
)

data["estimated_total_price"] = (
    data["adr"] * data["total_nights"]
)

data["is_family"] = int(
    data["children"] + data["babies"] > 0
)

# Lead time category
if "lead_time_category" in features:
    categories = df["lead_time_category"].dropna().unique()

    for category in categories:
        rows = df[df["lead_time_category"] == category]

        if rows["lead_time"].min() <= data["lead_time"] <= rows["lead_time"].max():
            data["lead_time_category"] = category
            break

# Prediction
if st.button("Predict Cancellation"):

    input_data = pd.DataFrame([data])[features]

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("Booking is likely to be cancelled.")
    else:
        st.success("Booking is unlikely to be cancelled.")

    st.metric(
        "Estimated Cancellation Probability",
        f"{probability * 100:.2f}%"
    )

    st.metric(
        "Estimated Total Price",
        f"€{data['estimated_total_price']:.2f}"
    )

    st.caption(
        "Other booking details use typical dataset values."
    )

st.write("---")
st.caption("CDS4003 Final Project")