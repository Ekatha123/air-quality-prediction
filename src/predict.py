import pandas as pd
import joblib
from feature_engineering import compute_aqi, create_features

model = joblib.load("models/rf_model.pkl")

def predict_next_aqi(csv_path):
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = compute_aqi(df)
    df = create_features(df)

    latest = df.iloc[-1]
    X = [[
        latest['lag1'],
        latest['lag2'],
        latest['rolling_mean_3'],
        latest['rolling_std_3']
    ]]

    pred = model.predict(X)[0]
    return pred

if __name__ == "__main__":
    pred = predict_next_aqi("data/ancona_data.csv")
    print("Predicted AQI:", round(pred,2))
