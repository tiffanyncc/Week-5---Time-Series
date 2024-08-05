import logging
import pandas as pd
import statsmodels
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from src.data.make_dataset import load_data
from src.features.build_features import preprocess_data, prepare_ml_data
from src.models.train import train_arima_model, train_arimax_model, train_xgboost_model, backtest
from src.models.predict import forecast_arima_model, forecast_arimax_model, predict
from src.models.evaluate import evaluate_arima_model, evaluate_arimax_model, evaluate_xgboost_model, perform_adf_test
from src.visualization.visualize import plot_series, plot_decomposition, plot_forecast, plot_model_performance, plot_acf_pacf, plot_differenced_series, plot_aapl_stock_price

# Setup logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

def main():
    # Load data
    try:
        data = load_data('src/data/AAPL.csv')
        logging.info('Data loaded successfully.')
    except Exception as e:
        logging.error(f'Error loading data: {e}')
        return

    # Preprocess data
    try:
        df = preprocess_data(data)
        logging.info('Data preprocessed successfully.')
    except Exception as e:
        logging.error(f'Error preprocessing data: {e}')
        return

    # Plot series
    try:
        plot_series(df['AAPL'], 'AAPL Stock Prices', 'aapl_stock_prices_plot')
        logging.info('Series plotted successfully.')
    except Exception as e:
        logging.error(f'Error plotting series: {e}')
    
    # Decomposition
    try:
        from statsmodels.tsa.seasonal import seasonal_decompose
        decomposed = seasonal_decompose(df['AAPL'])
        plot_decomposition(decomposed.trend, decomposed.seasonal, decomposed.resid, df['AAPL'], 'decomposition_plot')
        logging.info('Decomposition plotted successfully.')
    except Exception as e:
        logging.error(f'Error plotting decomposition: {e}')

    # ACF and PACF plots
    try:
        plot_acf_pacf(df['AAPL'], lags=11, filename='acf_pacf_plots')
        logging.info('ACF and PACF plots displayed and saved.')
    except Exception as e:
        logging.error(f'Error plotting ACF and PACF: {e}')

    # ADF test and differencing
    try:
        p_value, differenced_series, differenced_p_value = perform_adf_test(df['AAPL'])
        logging.info(f'ADF test p-value: {p_value}')
        if differenced_p_value:
            logging.info(f'1st order differenced ADF test p-value: {differenced_p_value}')
            plot_differenced_series(differenced_series, '1st order differenced series', 'differenced_series_plot')
    except Exception as e:
        logging.error(f'Error performing ADF test: {e}')
        
    # Train ARIMA model
    try:
        ar_model = train_arima_model(df['AAPL'], (1, 1, 1))
        print(ar_model.summary())
        logging.info('ARIMA model trained successfully.')
    except Exception as e:
        logging.error(f'Error training ARIMA model: {e}')
        return

    # Forecast with ARIMA model
    try:
        ypred, conf_int = forecast_arima_model(ar_model, 2)
        forecast_df = pd.DataFrame({
            'price_actual': ['184.40', '185.04'],
            'price_predicted': ypred,
            'lower_int': conf_int['lower AAPL'],
            'upper_int': conf_int['upper AAPL']
        }, index=pd.to_datetime(['2024-01-01', '2024-02-01']))
        plot_forecast(df['AAPL'], forecast_df['price_predicted'], forecast_df['lower_int'], forecast_df['upper_int'], 'ARIMA Model Performance', 'model_performance_plot')
        logging.info('ARIMA forecast plotted successfully.')
    except Exception as e:
        logging.error(f'Error forecasting with ARIMA model: {e}')

    # Evaluate ARIMA model
    try:
        arima_mae = evaluate_arima_model(forecast_df['price_actual'], forecast_df['price_predicted'])
        logging.info(f'ARIMA MAE: {arima_mae}')
    except Exception as e:
        logging.error(f'Error evaluating ARIMA model: {e}')

    # Bivariate analysis with ARIMAX model
    try:
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.set_index('Date')
        dfx = data.iloc[0:-2, 0:3]
        arimax_model = train_arimax_model(dfx['AAPL'], dfx['TXN'], (1, 1, 1))
        print(arimax_model.summary())
        ex = data['TXN'].iloc[-2:].values
        ypred, conf_int = forecast_arimax_model(arimax_model, ex, 2)
        dpx = pd.DataFrame({
            'price_actual': [184.40, 185.04],
            'price_predicted': ypred,
            'lower_int': conf_int['lower AAPL'],
            'upper_int': conf_int['upper AAPL']
        }, index=pd.to_datetime(['2024-01-01', '2024-02-01']))
        plot_forecast(data['AAPL'], dpx['price_predicted'], dpx['lower_int'], dpx['upper_int'], 'ARIMAX Model Performance', 'model_performance_plot2')
        logging.info('ARIMAX forecast plotted successfully.')
    except Exception as e:
        logging.error(f'Error in ARIMAX bivariate analysis: {e}')
    
    # Evaluate ARIMAX model
    try:
        arimax_mae = evaluate_arimax_model(dpx['price_actual'], dpx['price_predicted'])
        logging.info(f'ARIMAX MAE: {arimax_mae}')
    except Exception as e:
        logging.error(f'Error evaluating ARIMAX model: {e}')

        # Additional plot
    try:
        ml_data, features = prepare_ml_data(data)
        plot_aapl_stock_price(ml_data, '20yrs_aapl_stock_price_plot')
        logging.info('AAPL stock price plot displayed successfully.')
    except Exception as e:
        logging.error(f'Error displaying AAPL stock price plot: {e}')
        
    # Prepare data for XGBoost
    try:
        ml_data, features = prepare_ml_data(data)
        logging.info('Data prepared for XGBoost model successfully.')
    except Exception as e:
        logging.error(f'Error preparing data for XGBoost model: {e}')
        return

    # Train XGBoost model
    try:
        train_data = ml_data.iloc[:-30]
        test_data = ml_data.iloc[-30:]
        xgb_model = train_xgboost_model(train_data, features)
        logging.info('XGBoost model trained successfully.')
    except Exception as e:
        logging.error(f'Error training XGBoost model: {e}')
        return

    # Predict with XGBoost model
    try:
        model1_preds = predict(train_data, test_data, features, xgb_model)
        logging.info('XGBoost predictions made successfully.')
    except Exception as e:
        logging.error(f'Error predicting with XGBoost model: {e}')

    # Evaluate XGBoost model
    try:
        precision = evaluate_xgboost_model(test_data, model1_preds['predictions'])
        logging.info(f'XGBoost Precision: {precision}')
        plot_model_performance(test_data['Target'], model1_preds['predictions'], 'XGBoost Model Performance','xgboost_model_performance_plot')
    except Exception as e:
        logging.error(f'Error evaluating XGBoost model: {e}')

    # Backtesting
    try:
        backtest_predictions = backtest(ml_data, xgb_model, features)
        logging.info('Backtesting completed successfully.')
        backtest_precision = evaluate_xgboost_model(backtest_predictions, backtest_predictions['predictions'])
        logging.info(f'Backtesting Precision: {backtest_precision}')
    except Exception as e:
        logging.error(f'Error during backtesting: {e}')

if __name__ == '__main__':
    main()