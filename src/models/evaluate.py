from sklearn.metrics import mean_absolute_error, precision_score
from statsmodels.tsa.stattools import adfuller

def evaluate_arima_model(actual, predicted):
    try:
        mae = mean_absolute_error(actual, predicted)
        return mae
    except Exception as e:
        raise RuntimeError(f"Error evaluating ARIMA model: {e}")

def evaluate_arimax_model(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    return mae

def evaluate_xgboost_model(test, predictions):
    try:
        precision = precision_score(test['Target'], predictions)
        return precision
    except Exception as e:
        raise RuntimeError(f"Error evaluating XGBoost model: {e}")

def perform_adf_test(series):
    results = adfuller(series)
    p_value = results[1]
    if p_value > 0.05:
        differenced_series = series.diff().dropna()
        differenced_results = adfuller(differenced_series)
        differenced_p_value = differenced_results[1]
        return p_value, differenced_series, differenced_p_value
    return p_value, series, None