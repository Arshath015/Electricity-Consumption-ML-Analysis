import pandas as pd
from sklearn.ensemble import RandomForestRegressor

print("Loading dataset...")

file_path = "data/household_power_consumption.txt"

df = pd.read_csv(
    file_path,
    sep=";",
    na_values="?",
    low_memory=False
)

# Create datetime
df["Datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], dayfirst=True)

# Convert to numeric
df["Global_active_power"] = pd.to_numeric(df["Global_active_power"], errors="coerce")

df = df.dropna(subset=["Global_active_power"])

# Reduce dataset for faster processing
df = df.sample(n=50000, random_state=42).sort_values("Datetime")

print("Preparing lag features...")

df["lag1"] = df["Global_active_power"].shift(1)
df["lag2"] = df["Global_active_power"].shift(2)
df["lag3"] = df["Global_active_power"].shift(3)

df = df.dropna()

X = df[["lag1", "lag2", "lag3"]]
y = df["Global_active_power"]

split_index = int(len(df) * 0.8)

X_train = X[:split_index]
X_test = X[split_index:]

y_train = y[:split_index]

print("Training model...")

model = RandomForestRegressor(
    n_estimators=10,
    random_state=42
)

model.fit(X_train, y_train)

# Generate prediction
predictions = model.predict(X_test)

predicted_power = predictions[0]

print("\nPredicted Global Active Power:", predicted_power)


# Rule-Based Categorization
def categorize_usage(power_value):

    if power_value < 1:
        category = "Low Usage"
        suggestion = "Energy usage is efficient. Maintain current consumption habits."

    elif power_value < 2.5:
        category = "Medium Usage"
        suggestion = "Monitor appliance usage to optimize electricity consumption."

    else:
        category = "High Usage"
        suggestion = "Consider reducing heavy appliance usage to save energy."

    return category, suggestion


category, suggestion = categorize_usage(predicted_power)

print("Usage Category:", category)
print("Suggestion:", suggestion)