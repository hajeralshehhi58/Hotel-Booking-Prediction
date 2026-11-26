
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Cleaned Dataset
df = pd.read_csv(BASE / "data" / "cleaned_hotel_bookings.csv")
sns.set_style("whitegrid")

# 2. Descriptive Statistics
print("Dataset Shape:", df.shape)

print("\nStatistical Summary:")
print(df.describe())

print("\nCategorical Summary:")
print(df.describe(include="object"))

print("\nCancellation Percentage:")
print(df["is_canceled"].value_counts(normalize=True) * 100)

# 3. Cancelled vs Not Cancelled Bookings
sns.countplot(data=df, x="is_canceled")
plt.title("Cancelled vs Not Cancelled Bookings")
plt.xlabel("Booking Status")
plt.ylabel("Number of Bookings")
plt.xticks([0, 1], ["Not Cancelled", "Cancelled"])
plt.show()

# 4. Cancellation by Hotel Type
sns.barplot(data=df, x="hotel", y="is_canceled")
plt.title("Cancellation Rate by Hotel Type")
plt.xlabel("Hotel Type")
plt.ylabel("Cancellation Rate")
plt.show()

# 5. Cancellation by Arrival Month
sns.barplot(data=df, x="arrival_date_month", y="is_canceled")
plt.title("Cancellation Rate by Month")
plt.xlabel("Month")
plt.ylabel("Cancellation Rate")
plt.show()

# 6. Lead Time Distribution
sns.histplot(data=df, x="lead_time", bins=30, kde=True)
plt.title("Lead Time Distribution")
plt.xlabel("Lead Time (Days)")
plt.ylabel("Frequency")
plt.show()

# 7. Lead Time vs Cancellation
sns.boxplot(data=df, x="is_canceled", y="lead_time")
plt.title("Lead Time vs Cancellation")
plt.xlabel("Booking Status")
plt.ylabel("Lead Time (Days)")
plt.show()

# 8. Room Price Distribution
sns.histplot(data=df, x="adr", bins=30, kde=True)
plt.title("Room Price Distribution")
plt.xlabel("Average Daily Rate")
plt.ylabel("Frequency")
plt.show()

# 9. Deposit Type vs Cancellation
sns.barplot(data=df, x="deposit_type", y="is_canceled")
plt.title("Cancellation Rate by Deposit Type")
plt.xlabel("Deposit Type")
plt.ylabel("Cancellation Rate")
plt.show()

# 10. Market Segment vs Cancellation
sns.barplot(data=df, x="market_segment", y="is_canceled")
plt.title("Cancellation Rate by Market Segment")
plt.xlabel("Market Segment")
plt.ylabel("Cancellation Rate")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 11. Customer Type vs Cancellation
sns.barplot(data=df, x="customer_type", y="is_canceled")
plt.title("Cancellation Rate by Customer Type")
plt.xlabel("Customer Type")
plt.ylabel("Cancellation Rate")
plt.show()

# 12. Feature Engineering

# Total number of guests
df["total_guests"] = (
    df["adults"] + df["children"] + df["babies"]
)

# Total number of nights
df["total_nights"] = (
    df["stays_in_weekend_nights"] +
    df["stays_in_week_nights"]
)

# Estimated total booking price
df["estimated_total_price"] = (
    df["adr"] * df["total_nights"]
)

# Identify bookings with children or babies
df["is_family"] = (
    (df["children"] + df["babies"]) > 0
).astype(int)

# Categorize booking lead time
df["lead_time_category"] = pd.cut(
    df["lead_time"],
    bins=[-1, 7, 30, 90, np.inf],
    labels=["Last Minute", "Short", "Medium", "Long"]
)

# 13. Total Nights vs Cancellation
sns.boxplot(data=df, x="is_canceled", y="total_nights")
plt.title("Total Nights vs Cancellation")
plt.xlabel("Booking Status")
plt.ylabel("Total Nights")
plt.show()

# 14. Family Bookings vs Cancellation
sns.barplot(data=df, x="is_family", y="is_canceled")
plt.title("Cancellation Rate for Family Bookings")
plt.xlabel("Family Booking")
plt.ylabel("Cancellation Rate")
plt.xticks([0, 1], ["No", "Yes"])
plt.show()

# 15. Lead Time Category vs Cancellation
sns.barplot(
    data=df,
    x="lead_time_category",
    y="is_canceled"
)
plt.title("Cancellation Rate by Lead Time Category")
plt.xlabel("Lead Time Category")
plt.ylabel("Cancellation Rate")
plt.show()

# 16. Correlation Heatmap
plt.figure(figsize=(18, 12))

numeric_df = df.select_dtypes(include=np.number)

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    annot_kws={"size": 7}
)

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# 17. Correlation with Target Variable
print("\nCorrelation with Cancellation:")
print(
    numeric_df.corr()["is_canceled"]
    .sort_values(ascending=False)
)

# 18. Group Statistics
print("\nCancellation Rate by Hotel:")
print(df.groupby("hotel")["is_canceled"].mean() * 100)

print("\nCancellation Rate by Deposit Type:")
print(df.groupby("deposit_type")["is_canceled"].mean() * 100)

print("\nAverage Lead Time by Booking Status:")
print(df.groupby("is_canceled")["lead_time"].mean())

# 19. Display New Features
print("\nNew Features:")
print(df[[
    "total_guests",
    "total_nights",
    "estimated_total_price",
    "is_family",
    "lead_time_category"
]].head())

# 20. Save Final Dataset
df.to_csv(BASE / "data" / "final_hotel_bookings.csv", index=False)
print("\nFinal Dataset Shape:", df.shape)
print("EDA and Feature Engineering Completed!")