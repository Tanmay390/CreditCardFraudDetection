import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier


# ==========================================
# 1. LOAD DATASET
# ==========================================

print("Loading dataset...")

df = pd.read_csv("data/creditcard.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# ==========================================
# 2. SEPARATE FEATURES AND TARGET
# ==========================================

X = df.drop("Class", axis=1)
y = df["Class"]

print("\nOriginal class distribution:")
print(y.value_counts())


# ==========================================
# 3. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)


# ==========================================
# 4. APPLY SMOTE ONLY TO TRAINING DATA
# ==========================================

print("\nApplying SMOTE...")

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

print("SMOTE completed!")

print("\nClass distribution after SMOTE:")
print(y_train_smote.value_counts())


# ==========================================
# 5. TRAIN RANDOM FOREST
# ==========================================

print("\nTraining Random Forest...")

random_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    class_weight=None
)

random_forest.fit(X_train_smote, y_train_smote)

rf_predictions = random_forest.predict(X_test)


# ==========================================
# 6. EVALUATE RANDOM FOREST
# ==========================================

rf_precision = precision_score(y_test, rf_predictions)
rf_recall = recall_score(y_test, rf_predictions)
rf_f1 = f1_score(y_test, rf_predictions)

print("\n========== RANDOM FOREST ==========")
print("Precision:", rf_precision)
print("Recall:", rf_recall)
print("F1 Score:", rf_f1)

print("\nClassification Report:")
print(classification_report(y_test, rf_predictions))


# ==========================================
# 7. TRAIN XGBOOST
# ==========================================

print("\nTraining XGBoost...")

xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    eval_metric="logloss",
    n_jobs=-1
)

xgb_model.fit(X_train_smote, y_train_smote)

xgb_predictions = xgb_model.predict(X_test)


# ==========================================
# 8. EVALUATE XGBOOST
# ==========================================

xgb_precision = precision_score(y_test, xgb_predictions)
xgb_recall = recall_score(y_test, xgb_predictions)
xgb_f1 = f1_score(y_test, xgb_predictions)

print("\n========== XGBOOST ==========")
print("Precision:", xgb_precision)
print("Recall:", xgb_recall)
print("F1 Score:", xgb_f1)

print("\nClassification Report:")
print(classification_report(y_test, xgb_predictions))


# ==========================================
# 9. TRAIN LOGISTIC REGRESSION
# ==========================================

print("\nScaling data for Logistic Regression...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_smote)
X_test_scaled = scaler.transform(X_test)


print("Training Logistic Regression...")

logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

logistic_model.fit(X_train_scaled, y_train_smote)

lr_predictions = logistic_model.predict(X_test_scaled)


# ==========================================
# 10. EVALUATE LOGISTIC REGRESSION
# ==========================================

lr_precision = precision_score(y_test, lr_predictions)
lr_recall = recall_score(y_test, lr_predictions)
lr_f1 = f1_score(y_test, lr_predictions)

print("\n========== LOGISTIC REGRESSION ==========")
print("Precision:", lr_precision)
print("Recall:", lr_recall)
print("F1 Score:", lr_f1)

print("\nClassification Report:")
print(classification_report(y_test, lr_predictions))


# ==========================================
# 11. COMPARE MODELS
# ==========================================

results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest",
        "XGBoost"
    ],
    "Precision": [
        lr_precision,
        rf_precision,
        xgb_precision
    ],
    "Recall": [
        lr_recall,
        rf_recall,
        xgb_recall
    ],
    "F1 Score": [
        lr_f1,
        rf_f1,
        xgb_f1
    ]
})

print("\n\n========== MODEL COMPARISON ==========")
print(results)


# ==========================================
# 12. SELECT BEST MODEL
# ==========================================

best_model_name = results.loc[
    results["F1 Score"].idxmax(),
    "Model"
]

print("\nBest model:", best_model_name)


if best_model_name == "XGBoost":

    best_model = xgb_model

elif best_model_name == "Random Forest":

    best_model = random_forest

else:

    best_model = logistic_model


# ==========================================
# 13. SAVE BEST MODEL
# ==========================================

joblib.dump(
    best_model,
    "models/fraud_model.pkl"
)

print("\nBest model saved successfully!")
print("Location: models/fraud_model.pkl")


# Save scaler for Logistic Regression
joblib.dump(
    scaler,
    "models/scaler.pkl"
)

print("Scaler saved successfully!")

print("\n========== TRAINING COMPLETE ==========")