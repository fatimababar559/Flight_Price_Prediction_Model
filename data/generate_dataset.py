"""
generate_dataset.py
====================
Generates a synthetic but realistic Flight Price Dataset
that mirrors the structure of the Kaggle 'Flight Price Prediction'
dataset (airline, source_city, destination_city, departure_time,
stops, arrival_time, class, duration, days_left, price).

Run this ONLY if you do not have access to the real Kaggle dataset.
The real dataset can be downloaded via:
    kaggle datasets download -d shubhambhattarai/flight-price-prediction

Usage:
    python data/generate_dataset.py
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 300_000  # ~300k rows to simulate a large real-world dataset

AIRLINES      = ["SpiceJet", "AirAsia", "Vistara", "GO_FIRST",
                  "Indigo", "Air_India"]
SOURCE_CITIES = ["Delhi", "Mumbai", "Bangalore", "Kolkata", "Hyderabad", "Chennai"]
DEST_CITIES   = ["Delhi", "Mumbai", "Bangalore", "Kolkata", "Hyderabad", "Chennai"]
TIMES         = ["Early_Morning", "Morning", "Afternoon", "Evening", "Night", "Late_Night"]
STOPS         = ["zero", "one", "two_or_more"]
CLASSES       = ["Economy", "Business"]

airline      = np.random.choice(AIRLINES, N)
source_city  = np.random.choice(SOURCE_CITIES, N)
dest_city    = np.random.choice(DEST_CITIES, N)
departure    = np.random.choice(TIMES, N)
arrival      = np.random.choice(TIMES, N)
stops        = np.random.choice(STOPS, N, p=[0.50, 0.35, 0.15])
flight_class = np.random.choice(CLASSES, N, p=[0.75, 0.25])
duration     = np.round(np.random.uniform(0.83, 49.83, N), 2)   # hours
days_left    = np.random.randint(1, 49, N)

# --- Simulate a realistic price relationship ---
base_price = 3000.0

# Airline premium
airline_premium = {"SpiceJet": 0, "AirAsia": 200, "Vistara": 800,
                   "GO_FIRST": 100, "Indigo": 300, "Air_India": 600}

# Stops discount / surcharge
stops_factor = {"zero": 1.2, "one": 1.0, "two_or_more": 0.85}

# Class multiplier
class_factor = {"Economy": 1.0, "Business": 2.8}

# Days-left: last-minute is expensive
days_penalty = np.where(days_left <= 7, 1.4,
               np.where(days_left <= 14, 1.15, 1.0))

airline_adj  = np.array([airline_premium[a] for a in airline])
stops_adj    = np.array([stops_factor[s]    for s in stops])
class_adj    = np.array([class_factor[c]    for c in flight_class])

price = (
    base_price
    + airline_adj
    + duration * 450
    + days_penalty * 500
) * stops_adj * class_adj

# Add Gaussian noise (realistic scatter)
price += np.random.normal(0, price * 0.08)
price  = np.clip(price, 1000, 120_000).astype(int)

df = pd.DataFrame({
    "airline":          airline,
    "source_city":      source_city,
    "destination_city": dest_city,
    "departure_time":   departure,
    "stops":            stops,
    "arrival_time":     arrival,
    "class":            flight_class,
    "duration":         duration,
    "days_left":        days_left,
    "price":            price,
})

# Remove same source == destination rows
df = df[df["source_city"] != df["destination_city"]].reset_index(drop=True)

out_path = "data/flight_price_dataset.csv"
df.to_csv(out_path, index=False)
print(f"✅  Dataset saved → {out_path}  ({len(df):,} rows)")
print(df.head())
print(df.describe())
