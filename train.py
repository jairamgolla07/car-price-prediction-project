import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn import metrics
import joblib
import os

# Load dataset
data = pd.read_csv("data/cardata.xls")

# Feature engineering
date_time = datetime.datetime.now()
data['Age'] = date_time.year - data['Year']
data.drop('Year', axis=1, inplace=True)

# Remove outliers
data = data[~(data['Selling_Price'] >= 33.0) & (data['Selling_Price'] <= 35.0)]

# Encode categorical columns
data['Fuel_Type'] = data['Fuel_Type'].map({'Petrol': 0, 'Diesel': 1, 'CNG': 2})
data['Seller_Type'] = data['Seller_Type'].map({'Dealer': 0, 'Individual': 1})
data['Transmission'] = data['Transmission'].map({'Manual': 0, 'Automatic': 1})

# Feature matrix and target
X = data.drop(['Car_Name', 'Selling_Price'], axis=1)
y = data['Selling_Price']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Train models
lr = LinearRegression().fit(X_train, y_train)
rf = RandomForestRegressor().fit(X_train, y_train)
gbr = GradientBoostingRegressor().fit(X_train, y_train)
xg = XGBRegressor().fit(X_train, y_train)

# Evaluate
print("Model Performance (R² scores):")
print("Linear Regression:", metrics.r2_score(y_test, lr.predict(X_test)))
print("Random Forest:    ", metrics.r2_score(y_test, rf.predict(X_test)))
print("Gradient Boosting:", metrics.r2_score(y_test, gbr.predict(X_test)))
print("XGBoost:          ", metrics.r2_score(y_test, xg.predict(X_test)))

# Save final model (XGBoost best performer)
os.makedirs("models", exist_ok=True)
xg_final = XGBRegressor().fit(X, y)
joblib.dump(xg_final, "models/cardata")
xg_final.save_model("models/xgb_model.json")

print("\n✅ Model saved in 'models/' folder.")
 