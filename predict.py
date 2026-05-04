import joblib
import pandas as pd

MODEL_PATH = "model/model.pkl"

model = joblib.load(MODEL_PATH)

print("Ames Prediction")

sqft = float(input("Living Area (sq ft): "))
bedrooms = int(input("Bedrooms: "))
bathrooms = int(input("Full Bathrooms: "))
garage = int(input("Garage Capacity (cars): "))
year = int(input("Year Built: "))

input_data = pd.DataFrame({
    "Gr Liv Area": [sqft],
    "Bedroom AbvGr": [bedrooms],
    "Full Bath": [bathrooms],
    "Garage Cars": [garage],
    "Year Built": [year]
})

prediction = model.predict(input_data)

print(f"\nPredicted House Price: ${prediction[0]:,.2f}")