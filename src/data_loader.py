import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


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
y_proba = model.predict_proba(X_test_scaled)[:, 1]

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