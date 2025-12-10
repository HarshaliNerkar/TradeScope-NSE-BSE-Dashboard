📊 TradeScope: Dynamic NSE & BSE Portfolio Dashboard
A comprehensive, interactive financial dashboard for tracking, calculating, and analyzing equity portfolio performance using Indian (NSE & BSE) stock market data, powered by Streamlit and yfinance.

✨ Key Features
TradeScope provides four dedicated tabs for in-depth analysis:

1. Portfolio
Performance Visualization: Displays historical price or percentage change for all selected assets (stocks and benchmarks) using Line, Area, or Candlestick charts.

Individual Stock Analysis: Provides key metrics (Latest Price, 50-Day Avg, 1-Year High/Low) for each selected ticker.

Technical Indicator: Option to overlay the 50-Day Moving Average (MA50).

2. Calculator
Goal Simulation: Calculates the simulated total portfolio value based on user-defined investment amounts and historical performance.

Profit/Loss Tracking: Shows the total profit or loss and the percentage return for the selected period.

Goal Tracking: Visualizes the progress toward a monetary goal and identifies the date when the goal was achieved (historically).

3. Analysis
Risk & Return Metrics: Calculates essential performance indicators for the portfolio and benchmarks, including:

CAGR (Compounded Annual Growth Rate)

Volatility

Sharpe Ratio (Risk-adjusted return)

Max Drawdown

95% Value at Risk (VaR)

Comparative Charts: Compares the Cumulative Returns and Rolling Volatility of the portfolio against chosen benchmarks (Nifty 50, Sensex).

4. Diversification
Correlation Heatmap: A dedicated visual tool to assess portfolio diversification by showing the correlation of daily returns between all selected assets. Lower correlation indicates better diversification.

⚙️ Local Setup and Deployment
Prerequisites
Python (3.7+)

Git (Optional, for cloning)

Installation Steps
Clone the Repository (or Download ZIP):

Bash

git clone https://github.com/YourUsername/TradeScope-Dashboard.git
cd TradeScope-Dashboard
Create and Activate a Virtual Environment (Recommended):

Bash

# Create environment
python -m venv venv

# Activate environment
source venv/bin/activate  # On macOS/Linux
.\venv\Scripts\activate  # On Windows
Install Required Packages:

Bash

pip install -r requirements.txt
(The requirements.txt file contains streamlit, pandas, numpy, yfinance, and plotly.)

Run the Application:

Bash

streamlit run dashboard.py
Your application will automatically open in your web browser, typically at http://localhost:8501.

🛠 Technology Stack
Core: Python

Web Framework: Streamlit

Data Source: yfinance library (Yahoo Finance API)

Data Manipulation: pandas and numpy

Visualization: plotly.express and plotly.graph_objects

🚀 Deployment
This application is designed for easy deployment on Streamlit Community Cloud (share.streamlit.io). Simply connect your GitHub repository to the Streamlit platform, specifying dashboard.py as the main file, and deploy!
