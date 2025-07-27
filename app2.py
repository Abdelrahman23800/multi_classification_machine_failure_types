import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model
with open("RandomForestClassifier.pkl", "rb") as f:
    model = pickle.load(f)

# Load label encoder for 'Type'
with open("Type_label_encoder.pkl", "rb") as f:
    type_encoder = pickle.load(f)

# Define features and their scalers
feature_names = [
    'Type', 'Air temperature [K]', 'Process temperature [K]',
    'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]'
]

# Load scalers
scalers = {}
for col in feature_names:
    scaler_filename = f"{col}_MinMaxScaler.pkl"
    with open(scaler_filename, "rb") as f:
        scalers[col] = pickle.load(f)

# Failure labels (order must match model target columns)
failure_labels = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']

# Streamlit UI
st.title("🔧 Machine Failure Type Prediction")
st.markdown("Use this tool to predict potential **machine failure types** based on current operating conditions.")
# Input form
with st.expander("🛠️ Enter Machine Parameters", expanded=True):
    type_input = st.selectbox("Product Type", ["L", "M", "H"])
    air_temp = st.number_input("Air temperature [K]", min_value=250.0, max_value=400.0, value=300.0)
    process_temp = st.number_input("Process temperature [K]", min_value=250.0, max_value=1000.0, value=310.0)
    rot_speed = st.number_input("Rotational speed [rpm]", min_value=0, max_value=3000, value=1500)
    torque = st.number_input("Torque [Nm]", min_value=0.0, max_value=100.0, value=40.0)
    tool_wear = st.number_input("Tool wear [min]", min_value=0, max_value=300, value=10)

# Predict button
if st.button("🔍 Predict"):
    # Encode and prepare input
    type_encoded = type_encoder.transform([type_input])[0]

    input_dict = {
        'Type': [type_encoded],
        'Air temperature [K]': [air_temp],
        'Process temperature [K]': [process_temp],
        'Rotational speed [rpm]': [rot_speed],
        'Torque [Nm]': [torque],
        'Tool wear [min]': [tool_wear]
    }

    input_df = pd.DataFrame(input_dict)

    # Apply scaling
    for col in feature_names:
        input_df[col] = scalers[col].transform(input_df[[col]])

    # Prediction
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)

    # Predicted labels
    predicted_failures = [label for label, val in zip(failure_labels, prediction) if val == 1]

    st.subheader("📋 Predicted Failure Types:")
    if predicted_failures:
        st.success("⚠️ " + ", ".join(predicted_failures))
    else:
        st.success("✅ No failure predicted.")

    # Show probabilities
    st.subheader("📊 Prediction Probabilities:")
    for label, prob in zip(failure_labels, probabilities):
        # prob is a list of arrays like [[0.9, 0.1]] -> prob[0][1] is prob of failure
        prob_percent = prob[0][1] * 100
        st.write(f"**{label}**: {prob_percent:.2f}% chance")
        st.progress(min(int(prob_percent), 100))
