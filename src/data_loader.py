import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


ticker = "THYAO.IS"

data = yf.download(
    ticker,
    start="2025-01-01",
    end="2026-01-01",
    auto_adjust=False
)

# Clean column structure
if hasattr(data.columns, "levels"):
    data.columns = data.columns.get_level_values(0)

data.columns.name = None

# Feature Engineering 
data["Daily_Return"] = data["Close"].pct_change() #Daily return

data["MA20"] = data["Close"].rolling(window=20).mean() #20 days moving area

data["Next_Close"] = data["Close"].shift(-1) 
data["Target"] = (data["Next_Close"] > data["Close"]).astype(int) # 1=up / 0=down

data = data.dropna() #remove rows with missing values

# Define features and target
features = ["Close", "Daily_Return", "MA20"]

X = data[features]
y = data["Target"]

print("\nFeature data (X):")
print(X.head())

print("\nTarget data (y):")
print(y.head())

print("Data shape:")
print(data.shape)

print("\nMissing values:")
print(data.isnull().sum())

print("\nFirst 5 rows:")
print(data[
    ["Close", "Daily_Return", "MA20", "Next_Close", "Target"]
].head())

# Time-based train-test split
split_index = int(len(data) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]


# Scale features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the model
model = LogisticRegression()

model.fit(X_train_scaled, y_train)

print("\nLogistic Regression model trained successfully.")

# Make predictions on the test data
y_pred = model.predict(X_test_scaled)

# Baseline model: always predict the majority class in training data
majority_class = y_train.mode()[0]

baseline_pred = [majority_class] * len(y_test)

baseline_accuracy = accuracy_score(y_test, baseline_pred)

print("\n--- Baseline Model ---")
print(f"Majority class: {majority_class}")
print(f"Baseline Accuracy: {baseline_accuracy:.4f}")

y_proba = model.predict_proba(X_test_scaled)[:, 1]
# Inspect prediction probabilities
print("\n--- Prediction Probability Analysis ---")
print(f"Minimum UP probability: {y_proba.min():.4f}")
print(f"Maximum UP probability: {y_proba.max():.4f}")
print(f"Mean UP probability:    {y_proba.mean():.4f}")

print("\nFirst 10 predictions:")
prediction_results = pd.DataFrame(
    {
        "Actual": y_test,
        "Predicted": y_pred,
        "UP_Probability": y_proba,
    },
    index=y_test.index,
)

print(prediction_results.head(10))

# Evaluate model performance
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

print("\n--- Model Evaluation ---")

print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred, zero_division=0):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred, zero_division=0):.4f}")

if y_test.nunique() == 2:
    print(f"ROC-AUC:   {roc_auc_score(y_test, y_proba):.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
# Analyze prediction distribution
print("\n--- Prediction Distribution ---")

print("Actual labels:")
print(y_test.value_counts().sort_index().rename(index={
    0: "DOWN",
    1: "UP"
}))

print("\nPredicted labels:")
print(
    pd.Series(y_pred)
    .value_counts()
    .sort_index()
    .rename(index={
        0: "DOWN",
        1: "UP"
    })
)


print("\nTraining data:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nTesting data:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

print("\nTraining period:")
print(X_train.index.min(), "to", X_train.index.max())

print("\nTesting period:")
print(X_test.index.min(), "to", X_test.index.max())