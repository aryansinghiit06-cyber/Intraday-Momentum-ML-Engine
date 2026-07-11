# Intraday Momentum ML Engine

## Overview
A Python-based machine learning pipeline designed to ingest historical market data, engineer technical financial features, and predict short-term price momentum. The engine utilizes a Multi-Layer Perceptron (MLP) Artificial Neural Network to classify future price movements based on calculated technical indicators.

## Core Architecture

### 1. Data Ingestion & Preprocessing
* Ingests real-time and historical market data directly via the `yfinance` API.
* Implements robust data cleaning pipelines using `pandas`, handling missing market-close data via forward-fill (`.ffill()`) techniques to ensure dataset integrity.

### 2. Feature Engineering
Calculates key technical indicators to feed as features into the neural network:
* **RSI (Relative Strength Index):** Measures the magnitude of recent price changes to evaluate overbought or oversold conditions.
* **VWAP (Volume Weighted Average Price):** Provides the average price a security has traded at throughout the day, based on both volume and price.

### 3. Predictive Modeling
* Uses `scikit-learn` to build and train a **Multi-Layer Perceptron (MLP)** Neural Network.
* Maps the engineered features (RSI, VWAP, moving averages) to binary classification targets (Up/Down momentum).
* Evaluates model accuracy using standardized test/train splits and classification metrics.

## Tech Stack
* **Language:** Python
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (`MLPClassifier`)
* **Data Sourcing:** yfinance

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/aryansinghiit06-cyber/Intraday-Momentum-ML-Engine.git
   ```
2. Install the required dependencies:
   ```bash
   pip install pandas scikit-learn yfinance numpy
   ```
3. Execute the ML pipeline:
   ```bash
   python main.py
   ```
