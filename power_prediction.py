# VLSI Power Consumption Prediction using ML (Complete Project)

# Step 1: Load and Explore the Data
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st

st.set_page_config(page_title="VLSI Power Prediction", layout="centered")
st.title("🔋 VLSI Power Consumption Predictor")

# Upload CSV data file
uploaded_file = st.file_uploader("Upload RTL Power Dataset (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # Feature and Target
    X = df.drop("power_mW", axis=1)
    y = df["power_mW"]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Predict
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.markdown("---")
    st.subheader("📈 Model Performance")
    st.write(f"**Mean Squared Error:** {mse:.4f}")
    st.write(f"**R² Score:** {r2:.4f}")

    # Feature Importance
    importances = model.feature_importances_
    features = X.columns
    fig, ax = plt.subplots()
    ax.barh(features, importances, color='teal')
    ax.set_xlabel("Importance")
    ax.set_title("🔍 Feature Importance")
    st.pyplot(fig)

    # Prediction UI
    st.markdown("---")
    st.subheader("🔮 Predict Power")

    input_values = []
    for feature in X.columns:
        val = st.number_input(f"{feature.replace('_', ' ').title()}", value=float(df[feature].mean()))
        input_values.append(val)

    input_df = pd.DataFrame([input_values], columns=X.columns)
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]

    # Display power estimate
    st.success(f"Estimated Power: {prediction:.3f} mW")

    # Traffic light-style interpretation
    if prediction < 5:
        st.info("🟢 Low Power – Good for battery-powered use")
    elif prediction < 10:
        st.warning("🟠 Moderate Power – Consider minor optimizations")
    else:
        st.error("🔴 High Power – Needs optimization")

else:
    st.info("👆 Please upload a CSV file with RTL features and power values.")
