import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBClassifier
from src.models.predict import predict

def train_arima_model(series, order):
    try:
        model = ARIMA(series, order=order)
        ar_model = model.fit()
        return ar_model
    except Exception as e:
        raise RuntimeError(f"Error training ARIMA model: {e}")

def train_arimax_model(series, exog, order):
    model = ARIMA(series, exog=exog, order=order)
    arimax_model = model.fit()
    return arimax_model

def train_xgboost_model(train, features):
    try:
        model = XGBClassifier(max_depth=3, n_estimators=100, random_state=42)
        model.fit(train[features], train['Target'])
        return model
    except Exception as e:
        raise RuntimeError(f"Error training XGBoost model: {e}")

def backtest(data, model, features, start=5031, step=120):
    all_predictions = []

    for i in range(start, data.shape[0], step):
        train = data.iloc[:i].copy()
        test = data.iloc[i:(i+step)].copy()
        model_preds = predict(train, test, features, model)
        all_predictions.append(model_preds)

    return pd.concat(all_predictions)
