import pandas as pd
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

print("Starting deployment preparation...")

# Load dataset
df = pd.read_csv("data/creditcard.csv")

print("Dataset loaded successfully!")

# Separate features and target
X = df.drop("Class", axis=1)
y = df["Class"]

# Same split used during training
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Test data prepared...")

# Load trained model
model = joblib.load("models/fraud_model.pkl")

print("Model loaded successfully!")

# Make predictions
y_pred = model.predict(X_test)

# Calculate metrics
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)

# Get demo transactions
legitimate_sample = (
    df[df["Class"] == 0]
    .iloc[0]
    .drop("Class")
    .tolist()
)

fraud_sample = (
    df[df["Class"] == 1]
    .iloc[0]
    .drop("Class")
    .tolist()
)

# Create deployment data
deployment_data = {
    "model": "Random Forest",

    "total_transactions": int(len(df)),
    "fraud_transactions": int(y.sum()),
    "legitimate_transactions": int(len(df) - y.sum()),
    "fraud_rate": float((y.sum() / len(df)) * 100),

    "precision": float(precision),
    "recall": float(recall),
    "f1": float(f1),

    "confusion_matrix": cm.tolist(),

    "samples": {
        "legitimate": legitimate_sample,
        "fraud": fraud_sample
    }
}

# Save JSON
with open("app_data.json", "w") as file:
    json.dump(deployment_data, file, indent=4)

print()
print("===================================")
print("Deployment data created successfully!")
print("===================================")
print("File created: app_data.json")
print()
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print()
print("Confusion Matrix:")
print(cm)