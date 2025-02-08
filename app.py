import streamlit as st
import pandas as pd
import pickle
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("solar_wind_energy_weather_2021_2025.csv")

# Drop 'Date' column
df = df.drop(columns=['Date'])

# Encode 'State' column
le = LabelEncoder()
df['State'] = le.fit_transform(df['State'])

# Define features and targets
features = ['State', 'Temperature_C', 'Humidity_%', 'Wind_Speed_mps', 'Cloud_Cover_%']
target_solar = 'Solar_Energy_MWh'
target_wind = 'Wind_Energy_MWh'

X = df[features]
y_solar = df[target_solar]
y_wind = df[target_wind]

# Train-test split
from sklearn.model_selection import train_test_split
X_train_solar, X_test_solar, y_train_solar, y_test_solar = train_test_split(X, y_solar, test_size=0.2, random_state=42)
X_train_wind, X_test_wind, y_train_wind, y_test_wind = train_test_split(X, y_wind, test_size=0.2, random_state=42)

# Define models
models = {
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Linear Regression": LinearRegression()
}

# Train and select best models
from sklearn.metrics import r2_score

best_solar_model, best_wind_model = None, None
best_solar_acc, best_wind_acc = float('-inf'), float('-inf')

for name, model in models.items():
    model.fit(X_train_solar, y_train_solar)
    solar_pred = model.predict(X_test_solar)
    solar_acc = r2_score(y_test_solar, solar_pred) * 100

    if solar_acc > best_solar_acc:
        best_solar_acc = solar_acc
        best_solar_model = model

    model.fit(X_train_wind, y_train_wind)
    wind_pred = model.predict(X_test_wind)
    wind_acc = r2_score(y_test_wind, wind_pred) * 100

    if wind_acc > best_wind_acc:
        best_wind_acc = wind_acc
        best_wind_model = model

# Save the best models
with open("best_solar_model.pkl", "wb") as f:
    pickle.dump(best_solar_model, f)

with open("best_wind_model.pkl", "wb") as f:
    pickle.dump(best_wind_model, f)

# Load saved models
with open("best_solar_model.pkl", "rb") as f:
    best_solar_model = pickle.load(f)

with open("best_wind_model.pkl", "rb") as f:
    best_wind_model = pickle.load(f)

# Streamlit App UI
st.title("Solar & Wind Energy Prediction 🌞💨")
st.markdown("Enter the weather conditions to predict energy generation.")

# User input form
state_name = st.selectbox("Select State", ["Maharashtra", "Rajasthan", "Tamil Nadu", "Karnataka", "Gujarat"])
temp = st.number_input("Enter Temperature (°C)", min_value=-10.0, max_value=50.0, step=0.1)
humidity = st.number_input("Enter Humidity (%)", min_value=0.0, max_value=100.0, step=0.1)
wind_speed = st.number_input("Enter Wind Speed (mps)", min_value=0.0, max_value=50.0, step=0.1)
cloud_cover = st.number_input("Enter Cloud Cover (%)", min_value=0.0, max_value=100.0, step=0.1)

if st.button("Predict Energy"):
    # Encode state input
    state_encoded = le.transform([state_name])[0]

    # Create DataFrame for prediction
    input_data = pd.DataFrame([[state_encoded, temp, humidity, wind_speed, cloud_cover]], columns=features)

    # Predictions
    solar_prediction = best_solar_model.predict(input_data)[0]
    wind_prediction = best_wind_model.predict(input_data)[0]

    # Display results
    st.success(f"🌞 Predicted **Solar Energy Generation**: {solar_prediction:.2f} MWh")
    st.success(f"💨 Predicted **Wind Energy Generation**: {wind_prediction:.2f} MWh")
