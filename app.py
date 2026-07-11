import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from ta.momentum import RSIIndicator
from ta.volume import VolumeWeightedAveragePrice
import plotly.graph_objects as go
import warnings

# Suppress warnings for cleaner terminal output
warnings.filterwarnings('ignore')

# -------------------------------------------------------------------
# 1. UI SETUP & CUSTOM CSS (Premium Look)
# -------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="Quant Momentum AI", page_icon="📈")

st.markdown("""
<style>
    /* Premium Dark Mode background for the whole app */
    .stApp {
        background-color: #0E1117;
    }
    
    /* Styling the Metric Cards */
    div[data-testid="metric-container"] {
        background-color: #1E1E2F;
        border: 1px solid #33334D;
        padding: 5% 5% 5% 10%;
        border-radius: 12px;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.4);
        transition: transform 0.2s ease-in-out;
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px);
        border-color: #4CAF50;
    }

    /* Title styling */
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: -webkit-linear-gradient(#4CAF50, #2E7D32);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .sub-title {
        color: #8888A0;
        font-size: 1.2rem;
        margin-bottom: 30px;
        font-weight: 400;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">Intraday Momentum Engine</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">ANN-powered Volume & Momentum Classification</p>', unsafe_allow_html=True)

# -------------------------------------------------------------------
# 2. DATA PIPELINE (Robust & Error Handled)
# -------------------------------------------------------------------
TICKERS = ['AAPL', 'MSFT', 'TSLA', 'NVDA', 'GOOGL', 'META', 'AMZN', 'NFLX', 'AMD', 'INTC']

@st.cache_data(ttl=300) # Cache data for 5 minutes to prevent spamming Yahoo Finance
def fetch_and_process_data():
    all_data = []
    
    for ticker in TICKERS:
        try:
            # Download 5-minute intraday data for the last 5 days
            df = yf.download(ticker, period="5d", interval="5m", progress=False)
            
            # Error Handling 1: Skip if data is missing or empty
            if df.empty or len(df) < 50:
                continue
                
            # Error Handling 2: Flatten multi-index columns if yfinance returns them
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns]
                
            # Error Handling 3: Forward fill missing values (gaps in trading)
            df = df.ffill()
                
            # Calculate Indicators
            # 1. RSI (Relative Strength Index)
            rsi_indicator = RSIIndicator(close=df['Close'], window=14)
            df['RSI'] = rsi_indicator.rsi()
            
            # 2. VWAP (Volume Weighted Average Price)
            vwap_indicator = VolumeWeightedAveragePrice(
                high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume'], window=14
            )
            df['VWAP'] = vwap_indicator.volume_weighted_average_price()
            
            # 3. Feature: Percentage deviation of current price from VWAP
            df['VWAP_Dev'] = (df['Close'] - df['VWAP']) / df['VWAP']
            
            # Target Variable: 1 if price goes UP in 3 periods (15 mins), else 0
            df['Future_Close'] = df['Close'].shift(-3)
            df['Target_Up'] = (df['Future_Close'] > df['Close']).astype(int)
            
            # Data Cleaning: Drop NaN values
            df = df.dropna()
            df['Ticker'] = ticker
            all_data.append(df)
            
        except Exception as e:
            pass
            
    return pd.concat(all_data) if all_data else pd.DataFrame()

# -------------------------------------------------------------------
# 3. MACHINE LEARNING ENGINE (ANN)
# -------------------------------------------------------------------
def train_and_predict(df):
    features = ['RSI', 'VWAP_Dev']
    X = df[features]
    y = df['Target_Up']
    
    # Scale data: Neural Networks are sensitive to scale.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # The Artificial Neural Network (MLPClassifier)
    ann = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', max_iter=500, random_state=42)
    ann.fit(X_scaled, y)
    
    latest_probs = {}
    for ticker in TICKERS:
        ticker_df = df[df['Ticker'] == ticker]
        if not ticker_df.empty:
            last_row = ticker_df.iloc[-1:][features]
            last_row_scaled = scaler.transform(last_row)
            prob_up = ann.predict_proba(last_row_scaled)[0][1]
            latest_probs[ticker] = prob_up
            
    # Sort stocks by highest probability of going up
    ranked_stocks = sorted(latest_probs.items(), key=lambda x: x[1], reverse=True)
    return ranked_stocks, df

# -------------------------------------------------------------------
# 4. DASHBOARD RENDER & VISUALIZATION
# -------------------------------------------------------------------
if st.button("🚀 Run Intraday Momentum Scan", type="primary", use_container_width=True):
    with st.spinner("Fetching live market data and computing ANN inferences..."):
        data = fetch_and_process_data()
        
        if data.empty:
            st.error("Market data unavailable. Please check your internet connection or try again during market hours.")
        else:
            ranked_stocks, full_data = train_and_predict(data)
            
            # SAVE TO MEMORY (Session State)
            st.session_state['ranked_stocks'] = ranked_stocks
            st.session_state['full_data'] = full_data

# Check if data exists in memory to draw the charts
if 'ranked_stocks' in st.session_state and 'full_data' in st.session_state:
    ranked_stocks = st.session_state['ranked_stocks']
    full_data = st.session_state['full_data']
    
    st.markdown("### 🔥 Top 3 Intraday Focus Stocks")
    st.caption("AI-predicted probability of upward momentum over the next 15 minutes.")
    
    col1, col2, col3 = st.columns(3)
    top_3 = ranked_stocks[:3]
    
    for i, col in enumerate([col1, col2, col3]):
        if i < len(top_3):
            ticker, prob = top_3[i]
            current_price = full_data[full_data['Ticker'] == ticker]['Close'].iloc[-1]
            
            with col:
                st.metric(
                    label=f"{ticker} (AI Confidence)", 
                    value=f"${current_price:.2f}", 
                    delta=f"{prob * 100:.1f}% Bullish Probability"
                )
    
    st.markdown("<br><hr><br>", unsafe_allow_html=True)
    st.markdown("### 📊 Advanced Order Flow Visualization")
    
    selected_ticker = st.selectbox("Analyze Specific Asset:", TICKERS)
    chart_df = full_data[full_data['Ticker'] == selected_ticker].tail(100) 
    
    fig = go.Figure()
    
    # Candlesticks
    fig.add_trace(go.Candlestick(
        x=chart_df.index,
        open=chart_df['Open'],
        high=chart_df['High'],
        low=chart_df['Low'],
        close=chart_df['Close'],
        name="Price Action",
        increasing_line_color='#26A69A', decreasing_line_color='#EF5350'
    ))
    
    # VWAP Line
    fig.add_trace(go.Scatter(
        x=chart_df.index,
        y=chart_df['VWAP'],
        line=dict(color='#FFA726', width=2, dash='dot'),
        name="VWAP"
    ))
    
    # FIX OVERNIGHT GAPS
    fig.update_xaxes(
        rangebreaks=[
            dict(bounds=[16, 9.5], pattern="hour"), # Hide between 4:00 PM and 9:30 AM
            dict(bounds=[6, 1], pattern="day of week") # Hide weekends (Using spaces!)
        ]
    )
    
    fig.update_layout(
        title=dict(text=f"{selected_ticker} Intraday Microstructure (5m)", font=dict(size=20, color='#FFFFFF')),
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        height=600,
        paper_bgcolor='#0E1117',
        plot_bgcolor='#0E1117',
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)