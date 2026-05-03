import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="Stock Price Forecasting",
    page_icon="📈",
    layout="wide"
)

# Title and description
st.title("📈 Stock Price Forecasting App")
st.markdown("Download live stock data and forecast future prices using Prophet")

# Sidebar for inputs
st.sidebar.header("Configuration")

# Stock selector
ticker = st.sidebar.selectbox(
    "Select Stock",
    options=["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "META", "NFLX", "NVDA"],
    format_func=lambda x: {
        "AAPL": "Apple (AAPL)",
        "GOOGL": "Google (GOOGL)",
        "MSFT": "Microsoft (MSFT)",
        "TSLA": "Tesla (TSLA)",
        "AMZN": "Amazon (AMZN)",
        "META": "Meta (META)",
        "NFLX": "Netflix (NFLX)",
        "NVDA": "NVIDIA (NVDA)"
    }[x]
)

# Date range selector
st.sidebar.subheader("Date Range")
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input(
        "Start Date",
        value=datetime(2020, 1, 1),
        max_value=datetime.now()
    )
with col2:
    end_date = st.date_input(
        "End Date",
        value=datetime.now(),
        max_value=datetime.now()
    )

# Forecast horizon
st.sidebar.subheader("Forecast Settings")
horizon = st.sidebar.slider("Forecast Horizon (days)", 7, 120, 90)

# Download button
if st.sidebar.button("🔄 Fetch Data & Forecast", type="primary"):
    try:
        # Download data
        with st.spinner(f"Downloading {ticker} data..."):
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if data.empty:
                st.error("No data available for the selected date range.")
            else:
                # Prepare data
                df = data[['Close']].copy()
                df.columns = ['Price']
                
                # Get stock info for metrics
                stock_info = yf.Ticker(ticker)
                
                # Display key metrics
                st.subheader(f"{ticker} Key Metrics")
                metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
                
                current_price = df['Price'].iloc[-1]
                price_52w_high = df['Price'].rolling(window=252).max().iloc[-1]
                price_52w_low = df['Price'].rolling(window=252).min().iloc[-1]
                price_change = df['Price'].iloc[-1] - df['Price'].iloc[-2] if len(df) > 1 else 0
                price_change_pct = (price_change / df['Price'].iloc[-2] * 100) if len(df) > 1 else 0
                
                with metrics_col1:
                    st.metric("Current Price", f"${current_price:.2f}", f"{price_change_pct:+.2f}%")
                with metrics_col2:
                    st.metric("52-Week High", f"${price_52w_high:.2f}")
                with metrics_col3:
                    st.metric("52-Week Low", f"${price_52w_low:.2f}")
                with metrics_col4:
                    st.metric("Data Points", len(df))
                
                # Historical price chart
                st.subheader("📊 Historical Closing Prices")
                
                # Calculate rolling mean
                df['MA_30'] = df['Price'].rolling(window=30).mean()
                df['MA_90'] = df['Price'].rolling(window=90).mean()
                
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(df.index, df['Price'], label='Close Price', linewidth=2, color='#1f77b4')
                ax.plot(df.index, df['MA_30'], label='30-Day MA', linewidth=1.5, linestyle='--', color='orange', alpha=0.7)
                ax.plot(df.index, df['MA_90'], label='90-Day MA', linewidth=1.5, linestyle='--', color='green', alpha=0.7)
                ax.set_xlabel('Date', fontsize=12)
                ax.set_ylabel('Price ($)', fontsize=12)
                ax.set_title(f'{ticker} Historical Closing Prices', fontsize=14, fontweight='bold')
                ax.legend(loc='best')
                ax.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                
                # Prepare data for Prophet
                st.subheader("🔮 Price Forecast using Prophet")
                
                with st.spinner("Training Prophet model..."):
                    df_prophet = df['Price'].reset_index()
                    df_prophet.columns = ['ds', 'y']
                    
                    # Fit Prophet model
                    model = Prophet(
                        weekly_seasonality=True,
                        yearly_seasonality=True,
                        daily_seasonality=False
                    )
                    model.fit(df_prophet)
                    
                    # Make future dataframe
                    future = model.make_future_dataframe(periods=horizon, freq='B')
                    forecast = model.predict(future)
                
                # Plot forecast
                fig2, ax2 = plt.subplots(figsize=(12, 6))
                
                # Plot historical data
                ax2.plot(df_prophet['ds'], df_prophet['y'], 'k.', label='Actual', markersize=3)
                
                # Plot forecast
                ax2.plot(forecast['ds'], forecast['yhat'], 'b-', label='Forecast', linewidth=2)
                ax2.fill_between(forecast['ds'], 
                                forecast['yhat_lower'], 
                                forecast['yhat_upper'], 
                                alpha=0.2, 
                                color='blue',
                                label='Confidence Interval')
                
                # Add vertical line at forecast start
                forecast_start = df_prophet['ds'].iloc[-1]
                ax2.axvline(x=forecast_start, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Forecast Start')
                
                ax2.set_xlabel('Date', fontsize=12)
                ax2.set_ylabel('Price ($)', fontsize=12)
                ax2.set_title(f'{ticker} Price Forecast ({horizon} days ahead)', fontsize=14, fontweight='bold')
                ax2.legend(loc='best')
                ax2.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig2)
                
                # Display forecast statistics
                st.subheader("📈 Forecast Summary")
                
                # Get future forecast only (after last historical date)
                future_forecast = forecast[forecast['ds'] > df_prophet['ds'].max()].copy()
                
                if not future_forecast.empty:
                    summary_col1, summary_col2, summary_col3 = st.columns(3)
                    
                    predicted_price = future_forecast['yhat'].iloc[-1]
                    predicted_change = predicted_price - current_price
                    predicted_change_pct = (predicted_change / current_price) * 100
                    
                    with summary_col1:
                        st.metric(
                            f"Predicted Price ({horizon} days)",
                            f"${predicted_price:.2f}",
                            f"{predicted_change_pct:+.2f}%"
                        )
                    with summary_col2:
                        st.metric(
                            "Predicted High",
                            f"${future_forecast['yhat_upper'].max():.2f}"
                        )
                    with summary_col3:
                        st.metric(
                            "Predicted Low",
                            f"${future_forecast['yhat_lower'].min():.2f}"
                        )
                    
                    # Download forecast data
                    st.subheader("💾 Export Forecast")
                    
                    # Prepare export data
                    export_df = future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
                    export_df.columns = ['Date', 'Predicted_Price', 'Lower_Bound', 'Upper_Bound']
                    export_df['Date'] = export_df['Date'].dt.strftime('%Y-%m-%d')
                    
                    csv = export_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Forecast as CSV",
                        data=csv,
                        file_name=f"{ticker}_forecast_{horizon}days.csv",
                        mime="text/csv"
                    )
                    
                    # Show forecast table
                    with st.expander("View Forecast Data"):
                        st.dataframe(export_df, use_container_width=True)
                
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please check your inputs and try again.")

else:
    st.info("👈 Configure your settings in the sidebar and click 'Fetch Data & Forecast' to begin")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Built with Streamlit, Prophet, and yfinance</p>
        <p><small>Disclaimer: This app is for educational purposes only. Do not use for actual investment decisions.</small></p>
    </div>
    """,
    unsafe_allow_html=True
)
