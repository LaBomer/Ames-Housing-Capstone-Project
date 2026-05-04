import streamlit as st
import joblib
import pandas as pd
import json
import matplotlib.pyplot as plt

#state
if "saved_houses" not in st.session_state:
    st.session_state.saved_houses = []

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

#load data & model
model = joblib.load("model/model.pkl")

with open("model/metrics.json", "r") as f:
    metrics = json.load(f)

viz_data = pd.read_csv("model/viz_data.csv")
feat_data = pd.read_csv("model/feature_importance.csv")

#title
st.title("Ames House Prediction")

#performace metrics
st.subheader("Performance Metrics")
col1, col2, col3 = st.columns(3)

col1.metric("RMSE", f"${metrics['RMSE']:,.0f}")
col2.metric("MAE", f"${metrics['MAE']:,.0f}")
col3.metric("R² Score", f"{metrics['R2']:.2f}")

st.write("---")

#input
st.subheader("Enter House Details")

sqft = st.number_input("Living Area (sq ft)", value=2000)
bedrooms = st.number_input("Bedrooms", value=3)
bathrooms = st.number_input("Full Bathrooms", value=2)
garage = st.number_input("Garage Capacity", value=2)
year = st.number_input("Year Built", value=2000)

quality = st.slider("Overall Quality (1 = Poor, 10 = Excellent)", 1, 10, 5)

#prediction
if st.button("Predict Price"):

    input_data = pd.DataFrame({
        "Gr Liv Area": [sqft],
        "Bedroom AbvGr": [bedrooms],
        "Full Bath": [bathrooms],
        "Garage Cars": [garage],
        "Year Built": [year],
        "Overall Qual": [quality]
    })

    #confidence interval
    all_preds = [tree.predict(input_data)[0] for tree in model.estimators_]

    prediction = sum(all_preds) / len(all_preds)
    std_dev = pd.Series(all_preds).std()

    lower = prediction - (1.96 * std_dev)
    upper = prediction + (1.96 * std_dev)

    #save last prediction
    st.session_state.last_prediction = {
        "SqFt": sqft,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Garage": garage,
        "Year": year,
        "Quality": quality,
        "Price": prediction,
        "Lower": lower,
        "Upper": upper
    }

#display previous prediction
if st.session_state.last_prediction:

    pred = st.session_state.last_prediction

    st.success(f"Estimated Price: ${pred['Price']:,.2f}")

    #save button
    if st.button("Save House"):
        st.session_state.saved_houses.append({
            "SqFt": pred["SqFt"],
            "Bedrooms": pred["Bedrooms"],
            "Bathrooms": pred["Bathrooms"],
            "Garage": pred["Garage"],
            "Year": pred["Year"],
            "Quality": pred["Quality"],
            "Price": pred["Price"]
        })
        st.success("House saved")

st.write("---")

#save
st.subheader("Saved Houses")

if st.session_state.saved_houses:
    saved_df = pd.DataFrame(st.session_state.saved_houses)
    saved_df = saved_df.sort_values(by="Price", ascending=False)
    st.dataframe(saved_df)
else:
    st.write("No saved predictions yet")

#clear save
if st.button("Clear Saved"):
    st.session_state.saved_houses = []

st.write("---")

#actual and predicted graph
st.subheader("Actual vs Predicted Prices")

fig1, ax1 = plt.subplots()
ax1.scatter(viz_data["Actual"], viz_data["Predicted"])
ax1.set_xlabel("Actual Prices")
ax1.set_ylabel("Predicted Prices")
ax1.set_title("Model Accuracy")

st.pyplot(fig1)

#importance graph
st.subheader("Feature Importance")

fig2, ax2 = plt.subplots()
ax2.barh(feat_data["Feature"], feat_data["Importance"])
ax2.set_xlabel("Importance")
ax2.set_title("Feature Contribution")

st.pyplot(fig2)