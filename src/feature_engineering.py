import pandas as pd
import numpy as np

def compute_aqi(df):
    """
    Compute AQI using weighted pollutants and scale to realistic range.
    This gives AQI values roughly comparable to official AQI scale (0-500).
    """
    # Weighted sum of pollutants
    df['AQI_raw'] = (
        0.35 * df['PM2.5'] +
        0.25 * df['PM10'] +
        0.20 * df['NO2'] +
        0.20 * df['O3']
    )
    
    # Scale raw AQI to approximate 0-300 range
    # Adjust factor if needed based on data distribution
    df['AQI'] = df['AQI_raw'] * 10
    
    # Clip extreme outliers for stability
    df['AQI'] = df['AQI'].clip(lower=df['AQI'].quantile(0.01),
                                upper=df['AQI'].quantile(0.99))
    
    return df

def create_features(df):
    """
    Create time-series lag and rolling features for LSTM input.
    """
    df['lag1'] = df['AQI'].shift(1)
    df['lag2'] = df['AQI'].shift(2)
    df['rolling_mean_3'] = df['AQI'].rolling(3).mean()
    df['rolling_std_3'] = df['AQI'].rolling(3).std()
    
    df.dropna(inplace=True)
    return df

if __name__ == "__main__":
    # Load dataset
    df = pd.read_csv("data/ancona_data.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Compute AQI and features
    df = compute_aqi(df)
    df = create_features(df)
    
    # Display sample
    print(df[['Date','AQI','lag1','lag2','rolling_mean_3','rolling_std_3']].head())
