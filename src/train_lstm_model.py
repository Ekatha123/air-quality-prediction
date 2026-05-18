import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from sklearn.preprocessing import MinMaxScaler

print("Loading dataset...")
df = pd.read_csv("data/ancona_data.csv")

df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
df = df.ffill()

# Compute AQI (scaled to realistic range)
df['AQI_raw'] = (
    0.35 * df['PM2.5'] +
    0.25 * df['PM10'] +
    0.20 * df['NO2'] +
    0.20 * df['O3']
)

# Scale AQI to approximate 0-300 range
df['AQI'] = df['AQI_raw'] * 10

df = df.dropna()

# Clip extreme outliers
df['AQI'] = df['AQI'].clip(lower=df['AQI'].quantile(0.01), upper=df['AQI'].quantile(0.99))

# Log transform for stability
aqi = np.log1p(df['AQI'].values.reshape(-1,1))

scaler = MinMaxScaler()
scaled = scaler.fit_transform(aqi)

def make_sequences(data, step=24):
    X, y = [], []
    for i in range(step, len(data)):
        X.append(data[i-step:i])
        y.append(data[i])
    return np.array(X), np.array(y)

X, y = make_sequences(scaled)

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print("Training Stable LSTM model...")

model = Sequential([
    Input(shape=(X.shape[1],1)),
    LSTM(32, return_sequences=True),
    Dropout(0.2),
    LSTM(16),
    Dense(1)
])

model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss='mse'
)

model.fit(
    X_train, y_train,
    epochs=12,
    batch_size=256,
    validation_split=0.1,
    verbose=2
)

model.save("models/lstm_model.keras")
print("\nStable LSTM Model saved to models/lstm_model.keras")

# ------------------------------
# NOTE: Using this model in Streamlit
# ------------------------------
# To get the predicted AQI in the original scale, do:
# pred_scaled = model.predict(X_input)[0]
# pred_aqi = np.expm1(scaler.inverse_transform([[pred_scaled]]))[0][0]
# Then you can display pred_aqi in Streamlit
