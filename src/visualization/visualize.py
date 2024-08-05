import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def plot_series(df, title, filename):
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    plt.savefig(f'src/visualization/images/{filename}.png')
    plt.show()

def plot_decomposition(trend, seasonal, residual, original, filename):
    plt.figure(figsize=(12, 8))
    plt.subplot(411)
    plt.plot(original, label='Original', color='black')
    plt.legend(loc='upper left')
    plt.subplot(412)
    plt.plot(trend, label='Trend', color='red')
    plt.legend(loc='upper left')
    plt.subplot(413)
    plt.plot(seasonal, label='Seasonal', color='blue')
    plt.legend(loc='upper left')
    plt.subplot(414)
    plt.plot(residual, label='Residual', color='black')
    plt.legend(loc='upper left')
    plt.savefig(f'src/visualization/images/{filename}.png')
    plt.show()

def plot_acf_pacf(series, lags, filename):
    plt.rcParams.update({'figure.figsize': (7, 4), 'figure.dpi': 80})
    
    fig, axes = plt.subplots(2, 1)
    
    plot_acf(series.dropna(), ax=axes[0])
    axes[0].set_title('ACF Plot')
    
    plot_pacf(series.dropna(), lags=lags, ax=axes[1])
    axes[1].set_title('PACF Plot')
    
    plt.tight_layout()
    plt.savefig(f'src/visualization/images/{filename}.png')
    plt.show()

def plot_differenced_series(differenced_series, title, filename):
    plt.plot(differenced_series)
    plt.title(title)
    plt.xlabel('Date')
    plt.xticks(rotation=30)
    plt.ylabel('Price (USD)')
    plt.savefig(f'src/visualization/images/{filename}.png')
    plt.show()
    
def plot_forecast(data, forecast, lower_int, upper_int, title, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(data, label='Actual')
    plt.plot(forecast, color='orange', label='Forecast')
    plt.fill_between(forecast.index, lower_int, upper_int, color='k', alpha=0.15)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.xticks(rotation=45)
    plt.savefig(f'src/visualization/images/{filename}.png')
    plt.show()

def plot_model_performance(test, predictions, title, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(test, label='Actual')
    plt.plot(predictions, label='Predicted')
    plt.legend()
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.savefig(f'src/visualization/images/{filename}.png')
    plt.show()

def plot_aapl_stock_price(data, filename):
    data['Close'].plot(kind='line', figsize=(8, 4), title='AAPL Stock Prices')
    plt.gca().spines[['top', 'right']].set_visible(False)
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.savefig(f'src/visualization/images/{filename}.png')
    plt.show()