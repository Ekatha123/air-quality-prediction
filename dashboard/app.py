import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datetime import datetime
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from src.feature_engineering import compute_aqi, create_features
from src.utils import aqi_category
from streamlit_autorefresh import st_autorefresh

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="AI Air Quality Monitoring", layout="wide")
st.title("🌍 AI Powered Air Quality Monitoring System")

# -------------------------------
# Auto Refresh (every 10 sec)
# -------------------------------
st_autorefresh(interval=10000, key="refresh")

# -------------------------------
# Load Dataset
# -------------------------------
data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'ancona_data.csv')
df = pd.read_csv(data_path)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').ffill()

# -------------------------------
# Feature Engineering
# -------------------------------
df = compute_aqi(df)
df = create_features(df)

# -------------------------------
# Load Model
# -------------------------------
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'rf_model.pkl')
model = joblib.load(model_path)

# -------------------------------
# Prediction
# -------------------------------
latest = df.iloc[-1]
X = pd.DataFrame([{
    'lag1': latest['lag1'],
    'lag2': latest['lag2'],
    'rolling_mean_3': latest['rolling_mean_3'],
    'rolling_std_3': latest['rolling_std_3']
}])

pred = model.predict(X)[0]

# -------------------------------
# Layout
# -------------------------------
c1, c2, c3 = st.columns(3)

with c1:
    st.metric("🌫 Predicted AQI", round(pred,2))

with c2:
    st.metric("🏥 Health Status", aqi_category(pred))

with c3:
    st.metric("📅 Last Updated", datetime.now().strftime("%d %b %Y, %H:%M:%S"))


# -------------------------------
# AQI Gauge Meter
# -------------------------------
fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=pred,
    title={'text': "Live AQI Meter"},
    gauge={
        'axis': {'range': [0, 300]},
        'bar': {'color': "darkblue"},
        'steps': [
            {'range': [0, 50], 'color': "green"},
            {'range': [50, 100], 'color': "yellow"},
            {'range': [100, 150], 'color': "orange"},
            {'range': [150, 200], 'color': "red"},
            {'range': [200, 300], 'color': "darkred"}
        ]
    }
))

st.plotly_chart(fig_gauge, use_container_width=True)

# -------------------------------
# Interactive AQI Trend
# -------------------------------
st.subheader("📈 Interactive AQI Trend")

fig_trend = px.line(
    df.tail(1500),
    x='Date',
    y='AQI',
    title="AQI Trend Over Time",
    markers=False
)

st.plotly_chart(fig_trend, use_container_width=True)

# -------------------------------
# Daily Pattern Heatmap
# -------------------------------
st.subheader("🔥 Daily & Hourly AQI Heatmap")

df['hour'] = df['Date'].dt.hour
df['day'] = df['Date'].dt.day

pivot = df.pivot_table(values='AQI', index='day', columns='hour', aggfunc='mean')

fig_heat = px.imshow(pivot, aspect='auto', color_continuous_scale='Turbo')

st.plotly_chart(fig_heat, use_container_width=True)

# -------------------------------
# Pollutant Breakdown
# -------------------------------
st.subheader("🧪 Pollutant Contribution")

poll = latest[['PM2.5','PM10','NO2','O3']]

fig_bar = px.bar(
    x=poll.index,
    y=poll.values,
    title="Current Pollutant Levels"
)

st.plotly_chart(fig_bar, use_container_width=True)
