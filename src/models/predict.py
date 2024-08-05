import pandas as pd

def forecast_arima_model(model, steps, exog=None):
    try:
        forecast = model.get_forecast(steps, exog=exog)
        ypred = forecast.predicted_mean
        conf_int = forecast.conf_int(alpha=0.05)
        return ypred, conf_int
    except Exception as e:
        raise RuntimeError(f"Error forecasting ARIMA model: {e}")

def forecast_arimax_model(model, exog, steps):
    forecast = model.get_forecast(steps, exog=exog)
    ypred = forecast.predicted_mean
    conf_int = forecast.conf_int(alpha=0.05)
    return ypred, conf_int

def predict(train, test, features, model):
    model.fit(train[features], train['Target'])
    model_preds = model.predict(test[features])
    model_preds = pd.Series(model_preds, index=test.index, name='predictions')
    combine = pd.concat([test['Target'], model_preds], axis=1)
    return combine
