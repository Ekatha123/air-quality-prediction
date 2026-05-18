import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.feature_engineering import compute_aqi, create_features

print("Loading dataset...")
df = pd.read_csv("data/ancona_data.csv")
df['Date'] = pd.to_datetime(df['Date'])

print("Engineering features...")
df = compute_aqi(df)
df = create_features(df)

X = df[['lag1','lag2','rolling_mean_3','rolling_std_3']]
y = df['AQI']

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, shuffle=False)

print("Training Random Forest model...")
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Evaluating model...")
pred = model.predict(X_test)

mae = mean_absolute_error(y_test, pred)
rmse = rmse = mean_squared_error(y_test, pred) ** 0.5
r2 = r2_score(y_test, pred)

print("\nModel Performance:")
print("MAE  :", round(mae,3))
print("RMSE :", round(rmse,3))
print("R2   :", round(r2,4))

joblib.dump(model, "models/rf_model.pkl")
print("\nModel saved successfully in models/rf_model.pkl")
