import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.metrics import confusion_matrix


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load("models/fraud_model.pkl")


@st.cache_data
def load_app_data():
    with open("app_data.json", "r") as file:
        return json.load(file)


model = load_model()
app_data = load_app_data()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("💳 Fraud Detection System")

st.sidebar.markdown(
    """
    ### Navigation

    Use the menu below to explore the system.
    """
)

page = st.sidebar.radio(
    "Select Page",
    [
        "🏠 Dashboard",
        "🔍 Single Prediction",
        "📂 Batch Prediction",
        "📊 Model Performance"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    "Machine Learning model: Random Forest"
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.title("💳 Credit Card Fraud Detection System")

    st.markdown(
        """
        ### AI-powered transaction monitoring

        This system uses a **Random Forest machine learning model**
        to identify potentially fraudulent credit card transactions.
        """
    )

    st.markdown("---")

    # Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Transactions",
            f"{app_data['total_transactions']:,}"
        )

    with col2:
        st.metric(
            "Fraud Transactions",
            f"{app_data['fraud_transactions']:,}"
        )

    with col3:
        st.metric(
            "Legitimate Transactions",
            f"{app_data['legitimate_transactions']:,}"
        )

    with col4:
        st.metric(
            "Fraud Rate",
            f"{app_data['fraud_rate']:.3f}%"
        )

    st.markdown("---")

    # Transaction distribution
    st.subheader("📈 Transaction Distribution")

    distribution = pd.DataFrame(
        {
            "Transaction Type": [
                "Legitimate",
                "Fraud"
            ],
            "Count": [
                app_data["legitimate_transactions"],
                app_data["fraud_transactions"]
            ]
        }
    )

    st.bar_chart(
        distribution.set_index("Transaction Type")
    )

    st.markdown("---")

    # Model summary
    st.subheader("🤖 Model Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Precision",
            f"{app_data['precision'] * 100:.2f}%"
        )

    with col2:
        st.metric(
            "Recall",
            f"{app_data['recall'] * 100:.2f}%"
        )

    with col3:
        st.metric(
            "F1 Score",
            f"{app_data['f1'] * 100:.2f}%"
        )

    st.success(
        f"Best performing model: {app_data['model']}"
    )


# ============================================================
# SINGLE PREDICTION
# ============================================================

elif page == "🔍 Single Prediction":

    st.title("🔍 Single Transaction Prediction")

    st.write(
        "Enter the transaction details below and the machine learning "
        "model will determine whether the transaction is potentially fraudulent."
    )

    st.markdown("---")

    # Feature names
    feature_names = [
        "Time"
    ] + [f"V{i}" for i in range(1, 29)] + [
        "Amount"
    ]

    # Demo transaction buttons
    st.subheader("🎯 Demo Transactions")

    col1, col2 = st.columns(2)

    with col1:
        legitimate_button = st.button(
            "Load Legitimate Transaction",
            use_container_width=True
        )

    with col2:
        fraud_button = st.button(
            "Load Fraud Transaction",
            use_container_width=True
        )

    # Initialize values
    if "transaction_values" not in st.session_state:
        st.session_state.transaction_values = [0.0] * 30

    if legitimate_button:
        st.session_state.transaction_values = app_data["samples"]["legitimate"]
        st.success("Legitimate sample transaction loaded.")

    if fraud_button:
        st.session_state.transaction_values = app_data["samples"]["fraud"]
        st.warning("Fraud sample transaction loaded.")

    st.markdown("---")

    st.subheader("Transaction Features")

    values = st.session_state.transaction_values

    # Store user inputs
    input_values = []

    # First row
    col1, col2 = st.columns(2)

    with col1:
        time_value = st.number_input(
            "Time",
            value=float(values[0]),
            key="time_input"
        )
        input_values.append(time_value)

    with col2:
        amount_value = st.number_input(
            "Amount",
            value=float(values[29]),
            key="amount_input"
        )

    # V1 - V28
    st.markdown("### V1 - V28 Features")

    cols = st.columns(4)

    for i in range(1, 29):

        with cols[(i - 1) % 4]:

            value = st.number_input(
                f"V{i}",
                value=float(values[i]),
                key=f"v{i}"
            )

            input_values.append(value)

    # Reconstruct correct order
    transaction = [time_value]

    for i in range(1, 29):
        transaction.append(
            st.session_state[f"v{i}"]
        )

    transaction.append(amount_value)

    st.markdown("---")

    if st.button(
        "🔎 Check Transaction",
        type="primary",
        use_container_width=True
    ):

        input_array = np.array(transaction).reshape(1, -1)

        prediction = model.predict(input_array)[0]

        # Probability
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_array)[0][1]
        else:
            probability = 1.0 if prediction == 1 else 0.0

        st.markdown("---")

        if prediction == 1:

            st.error(
                "🚨 FRAUDULENT TRANSACTION DETECTED"
            )

            st.metric(
                "Fraud Probability",
                f"{probability * 100:.2f}%"
            )

            if probability >= 0.75:
                risk = "HIGH"
            elif probability >= 0.40:
                risk = "MEDIUM"
            else:
                risk = "LOW"

            st.warning(
                f"Risk Level: {risk}"
            )

        else:

            st.success(
                "✅ TRANSACTION APPEARS LEGITIMATE"
            )

            st.metric(
                "Fraud Probability",
                f"{probability * 100:.2f}%"
            )

            st.info(
                "The model does not detect strong evidence of fraud "
                "in this transaction."
            )


# ============================================================
# BATCH PREDICTION
# ============================================================

elif page == "📂 Batch Prediction":

    st.title("📂 Batch Transaction Prediction")

    st.write(
        "Upload a CSV file containing multiple transactions. "
        "The system will analyze every transaction automatically."
    )

    st.markdown("---")

    feature_names = [
        "Time"
    ] + [f"V{i}" for i in range(1, 29)] + [
        "Amount"
    ]

    st.subheader("Required CSV Columns")

    st.code(
        ", ".join(feature_names)
    )

    uploaded_file = st.file_uploader(
        "Upload Transaction CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            df = pd.read_csv(uploaded_file)

            missing_columns = [
                column
                for column in feature_names
                if column not in df.columns
            ]

            if missing_columns:

                st.error(
                    "The following required columns are missing:"
                )

                st.write(missing_columns)

            else:

                st.success(
                    f"File uploaded successfully: {len(df):,} transactions"
                )

                X_batch = df[feature_names]

                predictions = model.predict(X_batch)

                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(X_batch)[:, 1]
                else:
                    probabilities = predictions.astype(float)

                result = df.copy()

                result["Fraud Probability"] = probabilities

                result["Prediction"] = np.where(
                    predictions == 1,
                    "Fraud",
                    "Legitimate"
                )

                result["Risk Level"] = pd.cut(
                    probabilities,
                    bins=[-0.01, 0.40, 0.75, 1.0],
                    labels=["Low", "Medium", "High"]
                )

                st.markdown("---")

                st.subheader("📊 Prediction Summary")

                fraud_count = int(
                    np.sum(predictions == 1)
                )

                legitimate_count = int(
                    np.sum(predictions == 0)
                )

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Total",
                        len(result)
                    )

                with col2:
                    st.metric(
                        "Fraud",
                        fraud_count
                    )

                with col3:
                    st.metric(
                        "Legitimate",
                        legitimate_count
                    )

                st.markdown("---")

                st.subheader("Prediction Results")

                st.dataframe(
                    result,
                    use_container_width=True
                )

                csv = result.to_csv(index=False)

                st.download_button(
                    label="⬇️ Download Prediction Results",
                    data=csv,
                    file_name="fraud_predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        except Exception as e:

            st.error(
                f"Error processing file: {str(e)}"
            )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "📊 Model Performance":

    st.title("📊 Model Performance")

    st.write(
        "Performance of the trained machine learning model "
        "on the held-out test dataset."
    )

    st.markdown("---")

    # Metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Precision",
            f"{app_data['precision'] * 100:.2f}%"
        )

    with col2:
        st.metric(
            "Recall",
            f"{app_data['recall'] * 100:.2f}%"
        )

    with col3:
        st.metric(
            "F1 Score",
            f"{app_data['f1'] * 100:.2f}%"
        )

    st.markdown("---")

    st.subheader("🎯 Confusion Matrix")

    cm = np.array(
        app_data["confusion_matrix"]
    )

    cm_df = pd.DataFrame(
        cm,
        index=["Actual Legitimate", "Actual Fraud"],
        columns=["Predicted Legitimate", "Predicted Fraud"]
    )

    st.dataframe(
        cm_df,
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("🏆 Selected Model")

    st.success(
        f"Best Model: {app_data['model']}"
    )

    st.markdown(
        """
        The model was selected based on the **F1-score**, which balances
        precision and recall and is particularly useful for highly
        imbalanced fraud detection problems.
        """
    )

    st.markdown("---")

    st.subheader("📌 Model Metrics Explanation")

    st.write(
        "**Precision:** Of the transactions predicted as fraud, "
        "how many were actually fraudulent."
    )

    st.write(
        "**Recall:** Of all actual fraudulent transactions, "
        "how many were successfully detected."
    )

    st.write(
        "**F1 Score:** Harmonic mean of precision and recall."
    )