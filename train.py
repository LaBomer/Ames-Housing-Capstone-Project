import pandas as pd
import numpy as np
import joblib
import os
import json
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# pathway for files
DATA_PATH = "data/AmesHousing.csv"
MODEL_PATH = "model/model.pkl"
METRICS_PATH = "model/metrics.json"

# loading data
housing = pd.read_csv(DATA_PATH)

# features from dataset
features = [
    "Gr Liv Area",
    "Bedroom AbvGr",
    "Full Bath",
    "Garage Cars",
    "Year Built",
    "Overall Qual"
]

X = housing[features]
y = housing["SalePrice"]

# train testing split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# model creation
model = RandomForestRegressor(n_estimators=100, random_state=42)

# model training
model.fit(X_train, y_train)

# prediction
predic = model.predict(X_test)

#performance metrics
mse = mean_squared_error(y_test, predic)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, predic)
r2 = r2_score(y_test, predic)

print("\n=== Model Performance ===")
print(f"MSE: ${mse:,.2f}")
print(f"RMSE: ${rmse:,.2f}")
print(f"MAE: ${mae:,.2f}")
print(f"R² Score: {r2:.4f}")

# save to model folder
os.makedirs("model", exist_ok=True)
joblib.dump(model, MODEL_PATH)

print("\nmodel saved ", MODEL_PATH)

# streamlit metrics
metrics = {
    "MSE": mse,
    "RMSE": rmse,
    "MAE": mae,
    "R2": r2
}

with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f)

print("metrics saved ", METRICS_PATH)

#training graph
plt.scatter(y_test, predic)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.show()

# Save predictions for visualization
viz_data = pd.DataFrame({
    "Actual": y_test,
    "Predicted": predic
})

viz_data.to_csv("model/viz_data.csv", index=False)

# Save feature importance
importances = model.feature_importances_
feature_names = X.columns

feat_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})

feat_df.to_csv("model/feature_importance.csv", index=False)