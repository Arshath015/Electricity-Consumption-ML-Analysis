import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

print("Loading dataset...")

file_path = "data/household_power_consumption.txt"

df = pd.read_csv(
    file_path,
    sep=";",
    na_values="?",
    low_memory=False
)

print("Dataset loaded")

# Combine Date and Time
df["Datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], dayfirst=True)

# Convert to numeric
df["Global_active_power"] = pd.to_numeric(df["Global_active_power"], errors="coerce")

df = df.dropna(subset=["Global_active_power"])

# Reduce dataset size for training
print("Reducing dataset size for faster training...")
df = df.sample(n=100000, random_state=42).sort_values("Datetime")

# Sort by time
df = df.sort_values("Datetime")

print("Creating lag features...")

# Lag features
df["lag1"] = df["Global_active_power"].shift(1)
df["lag2"] = df["Global_active_power"].shift(2)
df["lag3"] = df["Global_active_power"].shift(3)

df = df.dropna()

# Features and target
X = df[["lag1", "lag2", "lag3"]]
y = df["Global_active_power"]

print("Splitting dataset...")

split_index = int(len(df) * 0.8)

X_train = X[:split_index]
X_test = X[split_index:]

y_train = y[:split_index]
y_test = y[split_index:]

print("Training Random Forest model...")

model = RandomForestRegressor(
    n_estimators=10,
    random_state=42,
    n_jobs=1
)

model.fit(X_train, y_train)

print("Model trained")

# Predictions
predictions = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("\nModel Evaluation")
print("MAE:", mae)
print("RMSE:", rmse)

# Plot predicted vs actual

output_dir = "task2_forecasting/output"
os.makedirs(output_dir, exist_ok=True)

plt.figure(figsize=(12,6))

plt.plot(y_test.values[:1000], label="Actual")
plt.plot(predictions[:1000], label="Predicted")

plt.title("Actual vs Predicted Power Consumption")
plt.legend()

plt.savefig(f"{output_dir}/prediction_vs_actual.png")
plt.close()

print("Saved prediction plot")