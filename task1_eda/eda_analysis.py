import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("Loading dataset...")

file_path = "data/household_power_consumption.txt"

df = pd.read_csv(
    file_path,
    sep=";",
    na_values="?",
    low_memory=False
)

print("Dataset Loaded")
print("Shape:", df.shape)

# Combine Date and Time
print("Creating datetime column...")

df["Datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], dayfirst=True)

# Convert important column to numeric
df["Global_active_power"] = pd.to_numeric(df["Global_active_power"], errors="coerce")

# Drop rows with missing values
df = df.dropna(subset=["Global_active_power"])

print("After cleaning:", df.shape)

# Create output folder
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# -----------------------------------
# 1. Time Series Trend
# -----------------------------------

print("Generating time series plot...")

plt.figure(figsize=(14,6))
plt.plot(df["Datetime"], df["Global_active_power"], color="blue")
plt.title("Global Active Power Over Time")
plt.xlabel("Time")
plt.ylabel("Global Active Power (kilowatts)")
plt.tight_layout()

plt.savefig(f"{output_dir}/power_trend.png")
plt.close()

print("Saved power_trend.png")

# -----------------------------------
# 2. Hourly Usage Pattern
# -----------------------------------

print("Analyzing hourly usage...")

df["Hour"] = df["Datetime"].dt.hour

hourly_usage = df.groupby("Hour")["Global_active_power"].mean()

plt.figure(figsize=(10,5))
sns.lineplot(x=hourly_usage.index, y=hourly_usage.values)
plt.title("Average Hourly Electricity Usage")
plt.xlabel("Hour of Day")
plt.ylabel("Average Global Active Power")
plt.tight_layout()

plt.savefig(f"{output_dir}/hourly_usage.png")
plt.close()

print("Saved hourly_usage.png")

# -----------------------------------
# 3. Daily Usage Pattern
# -----------------------------------

print("Analyzing daily usage...")

df["Date_only"] = df["Datetime"].dt.date

daily_usage = df.groupby("Date_only")["Global_active_power"].mean()

plt.figure(figsize=(14,6))
plt.plot(daily_usage.index, daily_usage.values)
plt.title("Daily Average Electricity Usage")
plt.xlabel("Date")
plt.ylabel("Average Global Active Power")
plt.tight_layout()

plt.savefig(f"{output_dir}/daily_usage.png")
plt.close()

print("Saved daily_usage.png")

# -----------------------------------
# 4. Missing Values Check
# -----------------------------------

print("\nMissing Values Summary:")
print(df.isnull().sum())

print("\nEDA Completed Successfully")