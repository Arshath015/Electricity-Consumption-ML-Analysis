import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
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

df["Datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], dayfirst=True)

df["Global_active_power"] = pd.to_numeric(df["Global_active_power"], errors="coerce")

df = df.dropna(subset=["Global_active_power"])

print("Reducing dataset size...")

df = df.sample(n=100000, random_state=42).sort_values("Datetime")

output_dir = "task3_unsupervised/output"
os.makedirs(output_dir, exist_ok=True)

# -----------------------------------
# 1 Anomaly Detection
# -----------------------------------

print("Running anomaly detection...")

model = IsolationForest(
    contamination=0.01,
    random_state=42
)

df["anomaly"] = model.fit_predict(df[["Global_active_power"]])

anomalies = df[df["anomaly"] == -1]

plt.figure(figsize=(12,6))

plt.scatter(
    df["Datetime"],
    df["Global_active_power"],
    s=1,
    label="Normal"
)

plt.scatter(
    anomalies["Datetime"],
    anomalies["Global_active_power"],
    color="red",
    s=5,
    label="Anomaly"
)

plt.legend()
plt.title("Electricity Usage Anomaly Detection")

plt.savefig(f"{output_dir}/anomalies.png")
plt.close()

print("Saved anomalies plot")

# -----------------------------------
# 2 Daily Clustering
# -----------------------------------

print("Running clustering...")

df["Date_only"] = df["Datetime"].dt.date

daily_usage = df.groupby("Date_only")["Global_active_power"].mean()

daily_usage = daily_usage.reset_index()

kmeans = KMeans(n_clusters=3, random_state=42)

daily_usage["cluster"] = kmeans.fit_predict(
    daily_usage[["Global_active_power"]]
)

plt.figure(figsize=(10,6))

sns.scatterplot(
    data=daily_usage,
    x="Date_only",
    y="Global_active_power",
    hue="cluster",
    palette="viridis"
)

plt.title("Daily Electricity Usage Clusters")

plt.savefig(f"{output_dir}/clusters.png")
plt.close()

print("Saved clusters plot")

print("\nTask 3 Completed")