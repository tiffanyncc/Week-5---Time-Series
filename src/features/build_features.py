import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler

def preprocess_data(data):
    data['Date'] = pd.to_datetime(data['Date'])
    df = data.iloc[:-2, 0:2]
    df = df.set_index('Date')
    return df

def prepare_ml_data(data):
    data = yf.download("AAPL", start="2000-01-01", end="2022-05-31")
    data['Next_day'] = data['Close'].shift(-1)
    data['Target'] = (data['Next_day'] > data['Close']).astype(int)
    data.dropna(inplace=True)
  
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    scaler = StandardScaler()
    data[features] = scaler.fit_transform(data[features])
    
    return data, features
