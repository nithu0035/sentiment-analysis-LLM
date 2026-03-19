
"""
sentiment_pipeline.py

Full sentiment pipeline for the Employee Sentiment Analysis project.

Steps:
- Load data from employee_feedback.csv
- Clean and prepare columns
- Compute sentiment scores & labels (TextBlob)
- Monthly sentiment aggregation
- Employee ranking and flight-risk flag
- Linear regression trend over monthly sentiment
- Save outputs to CSV files
"""

import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from textblob import TextBlob
from sklearn.linear_model import LinearRegression

load_dotenv()

DATA_PATH = os.getenv("DATA_PATH", "./data/employee_feedback.csv")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Loading data from: {DATA_PATH}")

# Load dataset (CSV)
df = pd.read_csv(DATA_PATH)

# === Column mapping ===
df = df.rename(columns={
    "from": "employee_id",
    "body": "feedback_text",
    "date": "date"
})

# Keep only relevant columns
df = df[["employee_id", "feedback_text", "date"]].copy()
df = df.dropna(subset=["feedback_text"])
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"])

# Sentiment helpers
def get_polarity(text: str) -> float:
    return TextBlob(str(text)).sentiment.polarity

def label_sentiment(score: float) -> str:
    if score > 0.05:
        return "Positive"
    elif score < -0.05:
        return "Negative"
    else:
        return "Neutral"

# Sentiment
df["sentiment_score"] = df["feedback_text"].apply(get_polarity)
df["sentiment_label"] = df["sentiment_score"].apply(label_sentiment)

# Monthly aggregation
df["year_month"] = df["date"].dt.to_period("M").dt.to_timestamp()
monthly = (
    df.groupby("year_month")
      .agg(avg_sentiment=("sentiment_score", "mean"),
           count=("sentiment_score", "count"))
      .reset_index()
)

# Employee ranking and flight risk
employee_stats = (
    df.groupby("employee_id")
      .agg(avg_sentiment=("sentiment_score", "mean"),
           feedback_count=("sentiment_score", "count"))
      .reset_index()
)

NEGATIVE_THRESHOLD = -0.1
MIN_FEEDBACKS = 3

employee_stats["flight_risk"] = np.where(
    (employee_stats["avg_sentiment"] <= NEGATIVE_THRESHOLD) &
    (employee_stats["feedback_count"] >= MIN_FEEDBACKS),
    "High",
    "Low"
)

# Linear regression over monthly sentiment
monthly_sorted = monthly.sort_values("year_month").reset_index(drop=True)
if len(monthly_sorted) > 1:
    monthly_sorted["t"] = np.arange(len(monthly_sorted))
    X = monthly_sorted[["t"]]
    y = monthly_sorted["avg_sentiment"]

    model = LinearRegression()
    model.fit(X, y)
    monthly_sorted["predicted_sentiment"] = model.predict(X)
    slope = model.coef_[0]
    intercept = model.intercept_
    print(f"Trend slope: {slope:.4f} (positive = improving sentiment)")
else:
    print("Not enough monthly data points for regression.")
    slope = None

# Save outputs
df.to_csv(os.path.join(OUTPUT_DIR, "feedback_with_sentiment.csv"), index=False)
employee_stats.to_csv(os.path.join(OUTPUT_DIR, "employee_ranking_and_flight_risk.csv"), index=False)
monthly_sorted.to_csv(os.path.join(OUTPUT_DIR, "monthly_sentiment_trend.csv"), index=False)

print("Outputs saved to:", OUTPUT_DIR)
