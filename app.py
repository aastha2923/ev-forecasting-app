import streamlit as st
import numpy as np
import pickle

st.title("EV Sales Prediction App")

# ✅ Trained model ko load karo
with open("ev_model.pkl", "rb") as f:
    model = pickle.load(f)

# User inputs
average_price = st.number_input("Average Price (INR)", min_value=0, value=85000)
revenue = st.number_input("Revenue (INR)", min_value=0, value=10000000)
total_vehicles_sold = st.number_input("Total Vehicles Sold", min_value=0, value=10000)
avg_state_ev_sales = st.number_input("Average State EV Sales", min_value=0.0, value=500.0)
ev_charging_stations = st.number_input("EV Charging Stations", min_value=0, value=50)
revenue_per_ev = st.number_input("Revenue per EV", min_value=0, value=85000)
vehicle_category_encoded = st.selectbox("Vehicle Category", ["2-Wheelers", "4-Wheelers"])
vehicle_category_encoded = 0 if vehicle_category_encoded == "2-Wheelers" else 1
state_encoded = st.number_input("State Encoded (0-36)", min_value=0, max_value=36, value=10)
maker_encoded = st.number_input("Maker Encoded (0-30)", min_value=0, max_value=30, value=5)
lag_1_ev_sales = st.number_input("Last Month EV Sales (Lag 1)", min_value=0.0, value=500.0)
lag_2_ev_sales = st.number_input("Second Last Month EV Sales (Lag 2)", min_value=0.0, value=450.0)
rolling_3month_avg_ev_sales = st.number_input("Rolling 3-Month Avg EV Sales", min_value=0.0, value=475.0)

if st.button("Predict Future EV Sales"):
    features = np.array([[average_price, revenue, total_vehicles_sold, avg_state_ev_sales,
                          ev_charging_stations, revenue_per_ev, vehicle_category_encoded,
                          state_encoded, maker_encoded, lag_1_ev_sales, lag_2_ev_sales,
                          rolling_3month_avg_ev_sales]])
    prediction = model.predict(features)
    st.success(f"📈 Predicted Future EV Sales: {int(prediction[0])}")
