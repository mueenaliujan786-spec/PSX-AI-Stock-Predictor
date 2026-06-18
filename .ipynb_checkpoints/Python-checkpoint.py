#!/usr/bin/env python
# Python.py — Fixed Version
# Changes made:
#   1. Removed 3 duplicate pip install lines — kept one clean comment
#   2. Changed AAPL (Apple US stock) → OGDC.KA (Pakistan stock — matches project)
#   3. Changed end date to date.today() — always fresh data
#   4. Changed shift(-1) → shift(-3) — matches app's 3-day prediction
#   5. Added StandardScaler
#   6. Added proper train/test split without shuffle
#   7. Cleaned up overall structure

# ── Install once if needed ──
# !pip install yfinance pandas numpy matplotlib scikit-learn --user

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date

print("All libraries loaded successfully!")

# ── Load PSX Stock Data ──
# FIX: using OGDC.KA (Pakistan) instead of AAPL (US)
# FIX: end=date.today() so data is always up-to-date
data = yf.download("OGDC.KA", start="2023-01-01", end=date.today())
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

print("Data loaded!")
print(data.head())

# ── Feature Engineering ──
data = data[['Close']].copy()
data['MA5']  = data['Close'].rolling(window=5).mean()

# FIX: shift(-3) for 3-day ahead prediction (matches the app)
data['Target'] = data['Close'].shift(-3)
data = data.dropna()
print("Features created!")
print(data.head())

# ── Train/Test Split and Model ──
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

X = data[['Close', 'MA5']]
y = data['Target']

# FIX: time-based split — no shuffle for time series data
split   = int(len(X) * 0.8)
X_train = X.iloc[:split]
X_test  = X.iloc[split:]
y_train = y.iloc[:split]
y_test  = y.iloc[split:]

# FIX: StandardScaler so features are equally weighted
scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_s, y_train)
print("Model trained!")

# ── Evaluate ──
predictions = model.predict(X_test_s)

results = pd.DataFrame({
    'Actual':    y_test.values.flatten(),
    'Predicted': predictions.flatten()
})
print(results.head(10))

mae = mean_absolute_error(y_test, predictions)
r2  = r2_score(y_test, predictions)

print(f"Mean Absolute Error: {mae:.2f} PKR")
print(f"R² Score           : {r2:.4f}")

# ── Plot Actual vs Predicted ──
plt.figure(figsize=(12, 6))
plt.plot(y_test.values,  label='Actual Price',    color='blue',   linewidth=2)
plt.plot(predictions,    label='Predicted Price', color='orange', linewidth=2, linestyle='--')
plt.title('OGDC.KA Stock Price: Actual vs Predicted (3 days ahead)')
plt.xlabel('Days')
plt.ylabel('Price (PKR)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ── Compare with Random Forest ──
from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_s, y_train)

rf_predictions = rf_model.predict(X_test_s)

rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_r2  = r2_score(y_test, rf_predictions)

print(f"\nRandom Forest  — MAE: {rf_mae:.2f} PKR, R²: {rf_r2:.4f}")
print(f"Linear Regression — MAE: {mae:.2f} PKR, R²: {r2:.4f}")