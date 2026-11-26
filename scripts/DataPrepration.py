import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
# 1. Load Dataset
df = pd.read_csv(BASE / "data" / "hotel_bookings.csv")
print("Original dataset shape:", df.shape)

# 2. Handle Missing Values
print("\nMissing values before cleaning:")
print(df.isnull().sum())

# Numerical values: Mean (Average)
df["children"] = df["children"].fillna(
    round(df["children"].mean())
)

# Categorical values: Mode
df["country"] = df["country"].fillna(
    df["country"].mode()[0]
)

df["agent"] = df["agent"].fillna(
    df["agent"].mode()[0]
)

df["company"] = df["company"].fillna(
    df["company"].mode()[0]
)

# 3. Handle Duplicates
print("\nDuplicate rows:", df.duplicated().sum())

df = df.drop_duplicates()

# 4. Handle Inconsistent Values and Formats
df.columns = df.columns.str.strip().str.lower()

for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].str.strip().str.lower()

# Remove bookings with no guests
df = df[
    (df["adults"] + df["children"] + df["babies"]) > 0
]

# Remove invalid negative values
df = df[
    (df["adr"] >= 0) &
    (df["lead_time"] >= 0) &
    (df["adults"] >= 0) &
    (df["children"] >= 0) &
    (df["babies"] >= 0) &
    (df["stays_in_weekend_nights"] >= 0) &
    (df["stays_in_week_nights"] >= 0)
].copy()

# 5. Detect Outliers Using IQR
columns = [
    "adr",
    "lead_time",
    "stays_in_weekend_nights",
    "stays_in_week_nights"
]

print("\nOutlier Detection:")

for col in columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[
        (df[col] < lower) | (df[col] > upper)
    ]

    print(col, "Outliers:", len(outliers))

# Remove extreme room prices using a proposed limit
df = df[df["adr"] <= 1000].copy()

# 6. Convert Data Types
df["children"] = df["children"].astype(int)
df["agent"] = df["agent"].astype(int)
df["company"] = df["company"].astype(int)
df["is_canceled"] = df["is_canceled"].astype(int)

df["reservation_status_date"] = pd.to_datetime(
    df["reservation_status_date"],
    errors="coerce"
)

# 7. Convert Month Names to Numbers
months = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12
}

df["arrival_date_month"] = df["arrival_date_month"].map(months)

# 8. Feature Engineering
df["total_guests"] = (
    df["adults"] + df["children"] + df["babies"]
)

df["total_nights"] = (
    df["stays_in_weekend_nights"] +
    df["stays_in_week_nights"]
)

df["estimated_total_price"] = (
    df["adr"] * df["total_nights"]
)

# 9. Remove Data Leakage Columns
df = df.drop(columns=[
    "reservation_status",
    "reservation_status_date",
    "assigned_room_type",
    "booking_changes",
    "days_in_waiting_list"
])

# Treat agent and company IDs as categories
df["agent"] = df["agent"].astype(str)
df["company"] = df["company"].astype(str)

# 10. One-Hot Encoding (for demonstration)
df_encoded = pd.get_dummies(df, dtype=int)

# 11. Check Final Results
print("\nFinal Dataset Shape:", df.shape)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

print("\nRemaining Identical Rows:", df.duplicated().sum())

print("\nData Types:")
print(df.dtypes)

print("\nStatistical Summary:")
print(df.describe())

print("\nFirst 5 Rows:")
print(df.head())

print("\nEncoded Dataset Shape:", df_encoded.shape)

# 12. Save Cleaned Dataset
df.to_csv(BASE / "data" / "cleaned_hotel_bookings.csv", index=False)

print("\nData Cleaning Completed Successfully!")