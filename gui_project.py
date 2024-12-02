import pandas as pd
from joblib import load
import matplotlib.pyplot as plt
import streamlit as st

# Streamlit app setup
st.title("DDoS Detection with Random Forest and XGBoost")

# Step 1: File upload
uploaded_file = st.file_uploader("Upload a CSV file for DDoS detection:", type="csv")
if uploaded_file is not None:
    # Load the dataset
    df_test = pd.read_csv(uploaded_file)
    df_test.columns = df_test.columns.str.strip()  # Remove leading/trailing spaces

    # Step 2: Handle Non-Numeric Data
    df_test = df_test.apply(pd.to_numeric, errors='coerce')  # Convert non-numeric values to NaN

    # Step 3: Load trained models
    try:
        rf_model = load("/Change to your model's path, random_forest_model.joblib")
        xgb_model = load("/Change to your model's path, xgboost_model.joblib")
    except Exception as e:
        st.error("Error loading models. Ensure model files are in the specified directory.")
        st.stop()

    # Step 4: Align the features with the trained models
    try:
        common_columns = [col for col in rf_model.feature_names_in_]  # Features used during training
        X_test = df_test[common_columns]  # Select features for prediction
    except KeyError as e:
        st.error(f"Feature mismatch: {e}")
        st.stop()

    # Step 5: Random Forest Section
    st.header("Random Forest")
    
    # Predict Labels using Random Forest
    rf_predictions = rf_model.predict(X_test)
    label_mapping = {0: 'BENIGN', 1: 'DDoS'}
    df_test['RF_Label'] = [label_mapping[pred] for pred in rf_predictions]

    # Calculate percentages
    rf_percentage = df_test['RF_Label'].value_counts(normalize=True).round(5) * 100

    # Visualization for Random Forest Predictions
    st.subheader("Random Forest Predictions")
    rf_fig, rf_ax = plt.subplots(figsize=(8, 6))
    rf_percentage.plot(kind='bar', color=['blue', 'green'], alpha=0.7, ax=rf_ax)
    rf_ax.set_title("Random Forest Predictions (%)")
    rf_ax.set_ylabel("Percentage")
    rf_ax.set_ylim(0, 100)
    st.pyplot(rf_fig)

    # Feature Importance for Random Forest
    st.subheader("Random Forest Feature Importances")
    rf_feature_importances = pd.DataFrame({
        "Feature": rf_model.feature_names_in_,
        "Importance": rf_model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    rf_fig2, rf_ax2 = plt.subplots(figsize=(10, 6))
    rf_feature_importances.plot(kind='barh', x='Feature', y='Importance', ax=rf_ax2, legend=False)
    rf_ax2.set_title("Random Forest Feature Importances")
    rf_ax2.set_xlabel("Importance")
    rf_ax2.set_ylabel("Feature")
    st.pyplot(rf_fig2)

    # Explanation for Random Forest
    st.write("### Explanation:")
    st.write(
        "**Avg Bwd Segment Size** is a critical feature because it measures the average size of backward packets in a connection. "
        "DDoS attacks often involve large amounts of traffic, and this feature helps differentiate normal traffic (Benign) from malicious activity (DDoS) "
        "by identifying anomalies in backward packet sizes."
    )

    # Step 6: XGBoost Section
    st.header("XGBoost")
    
    # Predict Labels using XGBoost
    xgb_predictions = xgb_model.predict(X_test)
    df_test['XGB_Label'] = [label_mapping[pred] for pred in xgb_predictions]

    # Calculate percentages
    xgb_percentage = df_test['XGB_Label'].value_counts(normalize=True).round(5) * 100

    # Visualization for XGBoost Predictions
    st.subheader("XGBoost Predictions")
    xgb_fig, xgb_ax = plt.subplots(figsize=(8, 6))
    xgb_percentage.plot(kind='bar', color=['red', 'orange'], alpha=0.7, ax=xgb_ax)
    xgb_ax.set_title("XGBoost Predictions (%)")
    xgb_ax.set_ylabel("Percentage")
    xgb_ax.set_ylim(0, 100)
    st.pyplot(xgb_fig)

    # Feature Importance for XGBoost
    st.subheader("XGBoost Feature Importances")
    xgb_feature_importances = pd.DataFrame({
        "Feature": xgb_model.feature_names_in_,
        "Importance": xgb_model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    xgb_fig2, xgb_ax2 = plt.subplots(figsize=(10, 6))
    xgb_feature_importances.plot(kind='barh', x='Feature', y='Importance', ax=xgb_ax2, legend=False, color="orange")
    xgb_ax2.set_title("XGBoost Feature Importances")
    xgb_ax2.set_xlabel("Importance")
    xgb_ax2.set_ylabel("Feature")
    st.pyplot(xgb_fig2)

    # Explanation for XGBoost
    st.write("### Explanation:")
    st.write(
       "**Bwd Packet Length Std** (Backward Packet Length Standard Deviation) is the most important feature in XGBoost. "
    "This metric represents the variability in the length of backward packets within a network flow. "
    "DDoS attacks often generate high variability in packet sizes due to the malicious nature of traffic, "
    "making this feature highly effective in identifying such attacks."
    )

else:
    st.info("Please upload a CSV file to proceed.")
