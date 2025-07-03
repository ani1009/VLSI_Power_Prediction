# VLSI Power Consumption Prediction using ML (Streamlit App)

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------------------------------
# Streamlit page config
# -------------------------------------------------
st.set_page_config(page_title="VLSI Power Prediction", layout="centered")
st.title("🔋 VLSI Power Consumption Predictor")

# -------------------------------------------------
# Upload CSV file
# -------------------------------------------------
uploaded_file = st.file_uploader("Upload RTL Power Dataset (CSV)", type=["csv"])

# -------------------------------------------------
# Train model (only once per new dataset) and cache in session_state
# -------------------------------------------------
if uploaded_file:
    # Load data
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # Calculate a simple hash of the uploaded file to detect change
    data_hash = hash(uploaded_file.getvalue())

    # Train only if model not stored or data changed
    if (
        "model" not in st.session_state or
        st.session_state.get("data_hash", None) != data_hash
    ):
        # Features and target
        X = df.drop("power_mW", axis=1)
        y = df["power_mW"]

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)

        # Store in session_state
        st.session_state.model = model
        st.session_state.scaler = scaler
        st.session_state.features = list(X.columns)
        st.session_state.data_hash = data_hash

        # Evaluate once
        y_pred = model.predict(X_test_scaled)
        st.session_state.mse = mean_squared_error(y_test, y_pred)
        st.session_state.r2 = r2_score(y_test, y_pred)
        st.session_state.importances = model.feature_importances_

    # Retrieve from session_state
    model = st.session_state.model
    scaler = st.session_state.scaler
    features = st.session_state.features

    # -------------------------------------------------
    # Show model performance and feature importance
    # -------------------------------------------------
    st.markdown("---")
    st.subheader("📈 Model Performance")
    st.write(f"**Mean Squared Error:** {st.session_state.mse:.4f}")
    st.write(f"**R² Score:** {st.session_state.r2:.4f}")

    # Feature importance plot (only first render)
    fig, ax = plt.subplots()
    ax.barh(features, st.session_state.importances, color="teal")
    ax.set_xlabel("Importance")
    ax.set_title("🔍 Feature Importance")
    st.pyplot(fig)

    # -------------------------------------------------
    # Prediction form (sliders) – runs only on button click
    # -------------------------------------------------
    st.markdown("---")
    st.subheader("🔮 Predict Power")

    with st.form("predict_form"):
        input_values = []
        for feat in features:
            col_min = float(df[feat].min())
            col_max = float(df[feat].max())
            col_mean = float(df[feat].mean())
            # Slider for each feature
            val = st.slider(
                label=feat.replace("_", " ").title(),
                min_value=col_min,
                max_value=col_max,
                value=col_mean,
                step=(col_max - col_min) / 100
            )
            input_values.append(val)

        submitted = st.form_submit_button("Predict")

    # If user pressed Predict button
    if submitted:
        input_df = pd.DataFrame([input_values], columns=features)
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]

        # Display power estimate and interpretation
        st.success(f"Estimated Power: {prediction:.3f} mW")
        if prediction < 5:
            st.info("🟢 Low Power – Good for battery-powered use")
        elif prediction < 10:
            st.warning("🟠 Moderate Power – Consider minor optimizations")
        else:
            st.error("🔴 High Power – Needs optimization")

else:
    st.info("👆 Please upload a CSV file with RTL features (including 'power_mW') before predicting.")
