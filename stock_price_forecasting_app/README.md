# 📈 Stock Price Forecasting App

A Streamlit web application that downloads live stock market data and forecasts future prices using Facebook's Prophet time series forecasting model.

## 🌟 Features

- **Live Stock Data**: Automatically downloads historical stock prices using `yfinance`
- **Interactive Stock Selection**: Choose from 8 major stocks (AAPL, GOOGL, MSFT, TSLA, AMZN, META, NFLX, NVDA)
- **Flexible Date Range**: Select custom start and end dates for analysis
- **Historical Visualization**: Line charts with 30-day and 90-day moving averages
- **Prophet Forecasting**: AI-powered price predictions with confidence intervals
- **Key Metrics**: Current price, 52-week high/low, and price changes
- **Export Functionality**: Download forecast data as CSV
- **Business Day Awareness**: Forecasts skip weekends, matching real market behavior

## 🚀 Live Demo

[Add your Streamlit Cloud URL here after deployment]

## 📋 Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`:
  - streamlit
  - yfinance
  - pandas
  - numpy
  - matplotlib
  - prophet

## 🛠️ Local Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/stock-forecast-app.git
cd stock-forecast-app
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the app:
```bash
streamlit run app.py
```

4. Open your browser to `http://localhost:8501`

## 📊 How to Use

1. **Select a Stock**: Use the sidebar dropdown to choose from available stocks
2. **Set Date Range**: Pick start and end dates for historical data
3. **Configure Forecast**: Use the slider to set forecast horizon (7-120 days)
4. **Fetch Data**: Click the "Fetch Data & Forecast" button
5. **Analyze Results**: View historical charts, forecast plots, and key metrics
6. **Export Data**: Download the forecast as a CSV file if needed

## 🏗️ Project Structure

```
stock-forecast-app/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md          # Project documentation
```

## 🌐 Deployment to Streamlit Cloud

1. Create a public GitHub repository and push your code
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with your GitHub account
4. Click "Create app" → "Deploy a public app from GitHub"
5. Configure:
   - Repository: `yourusername/stock-forecast-app`
   - Branch: `main`
   - Main file path: `app.py`
   - App URL: Choose a custom URL
6. Click "Deploy"
7. Wait for deployment (usually 2-5 minutes)

## 📈 About the Technologies

### yfinance
A Python library that provides easy access to Yahoo Finance data. It downloads historical market data including Open, High, Low, Close prices, and Volume.

### Prophet
Developed by Facebook's Core Data Science team, Prophet is designed for forecasting time series data with strong seasonal patterns and multiple seasonality. It handles:
- Weekly seasonality (market patterns within weeks)
- Yearly seasonality (annual market cycles)
- Trend changes and outliers
- Holiday effects

## ⚠️ Disclaimer

This application is for **educational purposes only**. The forecasts provided should not be used as the sole basis for investment decisions. Stock market predictions are inherently uncertain, and past performance does not guarantee future results.

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests

## 📝 License

This project is open source and available under the MIT License.

## 👨‍💻 Author

Prepared by Kawchar Husain

## 🙏 Acknowledgments

- **yfinance**: For providing easy access to stock market data
- **Prophet**: For the powerful forecasting algorithm
- **Streamlit**: For the amazing web framework
- **Yahoo Finance**: For the underlying data source
