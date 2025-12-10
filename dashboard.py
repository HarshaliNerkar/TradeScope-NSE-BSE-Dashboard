# ------------------------ Imports ------------------------#
import streamlit as st
import pandas as pd
import datetime as dt
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# ------------------------ ANALYSIS FUNCTIONS (NEW) ------------------------#
def calculate_max_drawdown(cumulative_returns):
    """Calculates the Max Drawdown from a series of cumulative returns."""
    if cumulative_returns.empty:
        return 0
    
    # Calculate the running maximum (peak)
    peak = cumulative_returns.expanding(min_periods=1).max()
    
    # Calculate the drawdown as the current value relative to the peak
    drawdown = (cumulative_returns - peak) / peak
    
    # Max Drawdown is the minimum drawdown value (largest loss)
    return drawdown.min()

def calculate_historical_var(daily_returns, confidence_level=0.95):
    """Calculates the Historical Value at Risk (VaR) at a given confidence level."""
    if daily_returns.empty:
        return 0
    
    # VaR is the negative of the specified percentile of the daily returns distribution
    # For 95% confidence, we use the 5th percentile (1 - 0.95 = 0.05)
    var_percentile = 1 - confidence_level
    var_daily = -daily_returns.quantile(var_percentile)
    
    # VaR is typically annualized or shown for the daily period. We'll show the daily loss %
    return var_daily

# ------------------------ Page config ------------------------#

st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded",
    page_title="NSE & BSE Portfolio"
    
)

st.title("📊 TradeScope – NSE & BSE")


# ------------------------ TITLE & CUSTOM CSS ------------------------#

# Centering the Title
st.markdown("""
    <style>
    /* 1. Center the Main Title */
    .stApp > header {
        display: none; /* Hides the default Streamlit header bar */
    }
    .stApp > div:nth-child(1) > div:nth-child(1) > div:nth-child(1) > div:nth-child(1) > div:nth-child(1) {
        text-align: center; /* Centers the st.title content */
    }

    /* 2. Style the Tab Buttons */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px; /* Space between tabs */
        justify-content: center; /* Center the tabs themselves */
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: nowrap;
        border-radius: 12px; /* Rounded corners */
        border: 1px solid #87CEEB; /* Border color for definition */
        padding: 10px 20px;
        font-weight: bold;
        background-color: #87CEEB; /* Default tab background (Light Blue) */
        transition: background-color 0.3s, border-color 0.3s;
    }

    /* Style for the SELECTED/HIGHLIGHTED Tab */
    .stTabs [aria-selected="true"] {
        background-color: #e4f7f6; /* Primary color for highlight */
        color: #121212 !important; /* Dark text on the bright background */
        border-color: #e4f7f6;
        box-shadow: 0 4px 8px rgba(0, 196, 154, 0.4); /* Subtle shadow for 3D effect */
        font-weight: 800;
    }
    
    /* Style for Hover effect */
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #b8fafc; /* Slightly darker shade on hover */
        cursor: pointer;
    }

    </style>
""", unsafe_allow_html=True)


# ------------------------ Tickers & Benchmarks ------------------------#
nse_tickers = {
    "RELIANCE.NS": "Reliance Industries",
    "TCS.NS": "Tata Consultancy Services",
    "INFY.NS": "Infosys",
    "HDFCBANK.NS": "HDFC Bank",
    "ICICIBANK.NS": "ICICI Bank",
    "^NSEI": "Nifty 50 Index"
}

bse_tickers = {
    "500325.BO": "Reliance Industries BSE",
    "532540.BO": "TCS BSE",
    "500209.BO": "Infosys BSE",
    "500180.BO": "HDFC Bank BSE",
    "532174.BO": "ICICI Bank BSE",
    "^BSESN": "Sensex Index"
}

benchmark_tickers = {
    "^NSEI": "Nifty 50 (Benchmark)",
    "^BSESN": "Sensex (Benchmark)"
}

ticker_list = {**nse_tickers, **bse_tickers}
ticker_names = [f"{k} - {v}" for k, v in ticker_list.items()]


# ------------------------ STOCK LOGO IMAGES (LOCAL) ------------------------#
stock_images = {
    "RELIANCE.NS": "images/reliance.png",
    "TCS.NS": "images/tcs.jpg",
    "INFY.NS": "images/infosys.png",
    "HDFCBANK.NS": "images/hdfc.jpg",
    "ICICIBANK.NS": "images/icici.jpg",

    "500325.BO": "images/reliance.png",
    "532540.BO": "images/tcs.jpg",
    "500209.BO": "images/infosys.png",
    "500180.BO": "images/hdfc.jpg",
    "532174.BO": "images/icici.jpg",

    "^NSEI": "images/nifty.png",
    "^BSESN": "images/sensex.png"
}

# ------------------------ Sidebar ------------------------#
with st.sidebar:
    # Changed label to include Benchmarks
    sel_tickers = st.multiselect("Select Stocks & Benchmarks", options=ticker_names)
    sel_tickers_list = [x.split(" - ")[0] for x in sel_tickers]
    
    # Identify which selected tickers are stocks and which are benchmarks
    benchmark_list = [t for t in sel_tickers_list if t in benchmark_tickers] 
    stock_list = [t for t in sel_tickers_list if t not in benchmark_list] 

    st.subheader("Portfolio Weighting")
    weights = {}
    
    # Iterate only over actual stocks for weighting
    if stock_list:
        total_weight = 0
        for ticker in stock_list:
            default_weight = 100 // len(stock_list) if len(stock_list) > 0 else 0
            w = st.slider(f"Weight for {ticker} (%)", 0, 100, default_weight, key=f"weight_{ticker}")
            weights[ticker] = w / 100
            total_weight += w
        
        if total_weight != 100:
            st.warning(f"Total weight is {total_weight}%. Consider adjusting to 100%.")

    # If no stocks are selected, ensure total_w is 0 for later checks
    if not stock_list:
        total_weight = 0

    st.subheader("Chart Options")
    # Added Candlestick option for better Usability/Interactive Analysis in Tab 1
    chart_type = st.radio("Chart Type", ["Line", "Area", "Candlestick"]) 
    show_pct = st.checkbox("Show % Change", value=True)
    show_ma50 = st.checkbox("Show 50-Day Moving Average", value=True)

# ------------------------ Date selector ------------------------#
col1, col2 = st.columns(2)
sel_dt1 = col1.date_input("Start Date", value=dt.datetime(2024, 1, 1))
sel_dt2 = col2.date_input("End Date", value=dt.datetime.today())


# ------------------------ Fetch Historical Data ------------------------#
@st.cache_data
def get_data(tickers, start, end):
    if len(tickers) == 0:
        return pd.DataFrame(), pd.DataFrame()

    data_all = yf.download(tickers, start=start, end=end)
    
    # --- Portfolio Data (Melted Close Price) ---
    # Handle the single ticker vs multiple ticker column structure from yfinance
    if len(tickers) == 1:
        data_all = data_all.loc[:, (slice(None), tickers[0])]
        data_all.columns = data_all.columns.droplevel(1)
        data_close = data_all['Close'].reset_index().rename(columns={'Close': tickers[0]})
    else:
        data_close = data_all['Close'].reset_index()
        
    data = data_close.melt(id_vars=['Date'], var_name='ticker', value_name='price')
    data['price_start'] = data.groupby('ticker').price.transform('first')
    data['price_pct_daily'] = data.groupby('ticker').price.pct_change()
    data['price_pct'] = (data.price - data.price_start) / data.price_start
    
    # --- OHLC Data for Candlestick Chart (Robust Extraction) ---
    ohlc_data_list = []
    
    for ticker in tickers:
        # Check if the ticker exists in the MultiIndex columns
        if len(tickers) > 1 and ticker in data_all.columns.get_level_values(1):
            df_t = data_all.loc[:, (['Open', 'High', 'Low', 'Close'], ticker)].copy()
            df_t.columns = df_t.columns.droplevel(1)
            df_t['Date'] = df_t.index
            df_t['ticker'] = ticker
            ohlc_data_list.append(df_t[['Date', 'ticker', 'Open', 'High', 'Low', 'Close']].reset_index(drop=True))
        elif len(tickers) == 1: # Single ticker case handled above
             df_t = data_all.loc[:, ['Open', 'High', 'Low', 'Close']].copy()
             df_t['Date'] = df_t.index
             df_t['ticker'] = ticker
             ohlc_data_list.append(df_t[['Date', 'ticker', 'Open', 'High', 'Low', 'Close']].reset_index(drop=True))
        
    ohlc_df = pd.concat(ohlc_data_list, ignore_index=True) if ohlc_data_list else pd.DataFrame()
    
    return data, ohlc_df

# ------------------------ Data Assignment (Robust) ------------------------#
try:
    yfdata, ohlc_data = get_data(sel_tickers_list, sel_dt1, sel_dt2) 
except Exception as e:
    st.error(f"An error occurred during data fetching from Yahoo Finance. Please try again. Error: {e}")
    yfdata = pd.DataFrame()
    ohlc_data = pd.DataFrame() 

# --- Define the chart function dynamically based on sidebar input ---
chart_func = px.area if chart_type == "Area" else px.line

# ------------------------ Portfolio Return Calculation (REQUIRED for Tabs 2, 3, 4) ------------------------#
if stock_list and yfdata is not None and not yfdata.empty and total_weight > 0:
    
    norm_weights = {k: weights[k] / total_weight for k in stock_list}
    
    # Pivot the data to get daily returns for *only* the weighted stocks
    stock_returns_df = yfdata[yfdata.ticker.isin(stock_list)].pivot(index='Date', columns='ticker', values='price_pct_daily').fillna(0)
    
    # Calculate the weighted daily portfolio return
    weighted_daily_returns_df = stock_returns_df.multiply(pd.Series(norm_weights), axis='columns')
    portfolio_daily_returns = weighted_daily_returns_df.sum(axis=1)
    
    # Calculate the cumulative portfolio return
    portfolio_cumulative_returns = (1 + portfolio_daily_returns).cumprod()
    
else:
    portfolio_daily_returns = pd.Series(dtype=float)
    portfolio_cumulative_returns = pd.Series(dtype=float)

# Define chart functions for cumulative tabs (excluding Candlestick)
calc_analysis_chart_func = px.area if chart_type == "Area" else px.line

# ------------------------ Tabs (UPDATED FOR NEW TAB 4) ------------------------#
tab1, tab2, tab3, tab4 = st.tabs(["Portfolio", "Calculator", "Analysis", "Diversification"]) 

# ------------------------ Portfolio Tab ------------------------#
with tab1:
    if len(sel_tickers_list) == 0:
        st.info("Select stocks and dates to view charts")
    elif yfdata.empty:
        st.warning("No historical data found for the selected stocks and date range.")
    else:
        # --- Overall Portfolio Performance ---
        st.subheader("Overall Portfolio Performance")
        
        if chart_type == "Candlestick": # Handle Candlestick for overall chart
            st.info("Overall portfolio performance is shown as a Line chart as Candlestick charts are for individual assets.")
            current_chart_func = px.line
        else:
            current_chart_func = chart_func

        y_overall = 'price_pct' if show_pct else 'price'
        y_title_overall = '% Change' if show_pct else 'Price'
        
        fig = current_chart_func(
            yfdata, x='Date', y=y_overall, color='ticker',
            title=f"Daily {y_title_overall} of Selected Assets",
            markers=(chart_type == "Line") 
        )

        if show_pct:
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            fig.update_yaxes(tickformat=",.2%")
        
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        
        # ------------------------ Individual Stock Performance ------------------------#
        st.subheader("Individual Asset Performance")

        for ticker in sel_tickers_list:
            data_t = yfdata[yfdata.ticker == ticker].copy()
            data_ohlc = ohlc_data[ohlc_data.ticker == ticker].copy()
            
            if data_t.empty:
                continue

            # Calculate 50-Day Moving Average 
            data_t['MA50'] = data_t['price'].rolling(window=50).mean()
            
            latest_price = data_t.price.iloc[-1]
            price_change = data_t.price_pct_daily.iloc[-1] if not data_t.price_pct_daily.empty else 0
            
            y_data = 'price_pct' if show_pct else 'price'
            y_title = '% Change' if show_pct else 'Price'

            with st.expander(f"📈 **{ticker}** – {ticker_list.get(ticker, 'Unknown Stock')}", expanded=False):
                col_met1, col_met2, col_met3, col_img = st.columns([1, 1, 1, 0.5])

                # Metrics
                col_met1.metric(
                    "Latest Price",
                    f"₹{round(latest_price, 2)}",
                    delta=f"{round(price_change * 100, 2)}%" if price_change else "0.00%",
                )
                
                col_met2.metric("50-Day Avg", round(data_t.MA50.iloc[-1], 2) if not data_t.MA50.empty else "N/A")
                col_met3.metric("1-Year Low", round(data_t.price.tail(365).min(), 2))
                col_met3.metric("1-Year High", round(data_t.price.tail(365).max(), 2))
                
                # Logo (Local)
                if ticker in stock_images:
                    col_img.image(stock_images[ticker], width=60)
                
                # --- Charting logic (Candlestick or Line/Area) ---
                if chart_type == "Candlestick":
                    if show_pct:
                        st.warning("Candlestick charts display Price data, not % Change. Showing Price.")
                    
                    fig_ind = go.Figure(data=[go.Candlestick(
                        x=data_ohlc['Date'],
                        open=data_ohlc['Open'],
                        high=data_ohlc['High'],
                        low=data_ohlc['Low'],
                        close=data_ohlc['Close'],
                        name='Price'
                    )])
                    fig_ind.update_layout(title=f"{ticker} – Candlestick Chart", xaxis_rangeslider_visible=False)
                    
                    if show_ma50:
                        fig_ind.add_trace(go.Scatter(
                            x=data_t['Date'], y=data_t['MA50'], mode='lines', 
                            name='50-Day MA', line=dict(color='red', dash='dash')
                        ))
                else:
                    fig_ind = chart_func(data_t, x='Date', y=y_data, title=f"{ticker} – {y_title} Movement")
                    
                    if show_ma50 and not show_pct:
                        fig_ind.add_trace(go.Scatter(
                            x=data_t['Date'], y=data_t['MA50'], mode='lines', 
                            name='50-Day MA', line=dict(color='red', dash='dash')
                        ))
                
                if show_pct:
                    fig_ind.update_yaxes(tickformat=",.2%")
                
                st.plotly_chart(fig_ind, use_container_width=True)


# ------------------------ Portfolio Calculator Tab ------------------------#
with tab2:
    if len(sel_tickers_list) == 0:
        st.info("Select tickers to use calculator")
    elif yfdata.empty:
        st.warning("No historical data found for the selected stocks and date range.")
    elif not stock_list:
        st.warning("Please select at least one *stock* and set its weight to use the calculator. Benchmarks are excluded.")
    elif portfolio_cumulative_returns.empty: # Check if portfolio returns were calculated
        st.warning("Cannot calculate portfolio value. Ensure stocks are selected and weights sum to a positive number.")
    else:
        st.subheader("Portfolio Investment Calculator")

        total_inv = 0
        amounts = {}

        # Iterate over only STOCKS for investment calculation
        for ticker in stock_list: 
            st.write("---")
            col_img, col_input = st.columns([1, 3])

            if ticker in stock_images:
                col_img.image(stock_images[ticker], width=60)
            else:
                col_img.subheader(ticker)

            # Use the calculated weight to suggest a default amount
            default_amount = (weights.get(ticker, 0.0) * 10000)
            amt = col_input.number_input(
                f"Investment in {ticker} (₹)", 
                min_value=0, 
                step=50, 
                key=f"calc_input_{ticker}",
                value=int(default_amount) 
            )
            amounts[ticker] = amt
            total_inv += amt

        st.subheader(f"Total Investment: ₹{total_inv:,.2f}")

        st.markdown("---")
        goal = st.number_input("Portfolio Goal (₹)", min_value=0, step=50, key="portfolio_goal_input")
        st.markdown("---")

        # P&L Calculation using the pre-calculated portfolio_cumulative_returns
        if total_inv > 0 and not portfolio_cumulative_returns.empty:
            final_value = total_inv * portfolio_cumulative_returns.iloc[-1]
            profit_loss = final_value - total_inv
        else:
            final_value = total_inv
            profit_loss = 0
        
        # Create a DataFrame for the chart (Investment * Cumulative Growth)
        df_sum_temp = (portfolio_cumulative_returns * total_inv).reset_index(name='amount')

        # Use the appropriate chart function (Line or Area)
        fig_calc = calc_analysis_chart_func(df_sum_temp, x='Date', y='amount', title="Portfolio Value Over Time (Cumulative)")
        fig_calc.add_hline(y=goal, line_color='green', line_dash='dash', line_width=3, name="Goal")
        fig_calc.update_yaxes(tickprefix="₹")

        if not df_sum_temp[df_sum_temp.amount >= goal].empty:
            goal_date = df_sum_temp[df_sum_temp.amount >= goal].Date.iloc[0]
            fig_calc.add_vline(x=goal_date, line_color='green', line_dash='dash', line_width=3)
            fig_calc.add_annotation(
                x=goal_date, y=goal * 1.05,
                text=f"Goal Achieved: {goal_date.date()}",
                showarrow=True, arrowcolor='green', bgcolor="rgba(255, 255, 255, 0.7)"
            )
        else:
            if goal > 0:
                st.warning("Goal can't be reached in this time frame with the current investment and historical performance.")

        st.plotly_chart(fig_calc, use_container_width=True)

        st.markdown("---")
        st.subheader("Historical Performance Summary (Results)")
        
        pnl_col1, pnl_col2 = st.columns(2)
        
        pnl_col1.metric("Final Portfolio Value", f"₹{final_value:,.2f}")
        
        pnl_col2.metric(
            "Total Profit/Loss",
            f"₹{profit_loss:,.2f}",
            delta=f"{profit_loss/total_inv * 100:,.2f}%" if total_inv > 0 else "0.00%"
        )
        st.markdown("---")
        
# ------------------------ Analysis Tab (Comparisons & Benchmarking) ------------------------#
with tab3:
    st.header("🔬 Portfolio Risk & Return Analysis")

    # Define Annualization Factor and Risk-Free Rate
    annualization_factor = np.sqrt(252)
    risk_free_rate = 0.05
    analysis_df = pd.DataFrame(columns=['CAGR', 'Volatility', 'Sharpe Ratio', 'Max Drawdown', '95% VaR (Daily)'])

    if not sel_tickers_list:
        st.info("Select stocks and/or benchmarks in the sidebar to view analysis.")
    elif yfdata.empty:
        st.warning("No historical data found for the selected assets or date range.")
    else:
        # --- 1. Metric Calculation ---
        
        benchmark_returns = {}
        
        # Portfolio Metrics
        if not portfolio_daily_returns.empty:
            trading_days = len(portfolio_daily_returns)
            total_return = portfolio_cumulative_returns.iloc[-1] - 1
            years = trading_days / 252.0
        
            cagr = ((1 + total_return) ** (1 / years)) - 1 if years > 0 else 0
            portfolio_volatility = portfolio_daily_returns.std() * annualization_factor
            sharpe_ratio = (cagr - risk_free_rate) / portfolio_volatility if portfolio_volatility != 0 else np.inf
        
            # --- NEW RISK METRICS ---
            mdd = calculate_max_drawdown(portfolio_cumulative_returns)
            var = calculate_historical_var(portfolio_daily_returns)
        
            analysis_df.loc['Portfolio'] = [cagr, portfolio_volatility, sharpe_ratio, mdd, var]
            
        
        # Benchmark Metrics
        for b_ticker in benchmark_list:
            if b_ticker in yfdata['ticker'].unique():
                b_daily_returns = yfdata[yfdata.ticker == b_ticker].set_index('Date')['price_pct_daily'].fillna(0)
                b_cumulative_returns = (1 + b_daily_returns).cumprod()
                
                b_trading_days = len(b_daily_returns)
                b_total_return = b_cumulative_returns.iloc[-1] - 1
                b_years = b_trading_days / 252.0
                
                b_cagr = ((1 + b_total_return) ** (1 / b_years)) - 1 if b_years > 0 else 0
                b_volatility = b_daily_returns.std() * annualization_factor
                b_sharpe_ratio = (b_cagr - risk_free_rate) / b_volatility if b_volatility != 0 else np.inf
                
                # --- NEW RISK METRICS ---
                b_mdd = calculate_max_drawdown(b_cumulative_returns)
                b_var = calculate_historical_var(b_daily_returns)
                
                analysis_df.loc[ticker_list[b_ticker]] = [b_cagr, b_volatility, b_sharpe_ratio, b_mdd, b_var]
                benchmark_returns[b_ticker] = b_cumulative_returns
                
        # --- 2. Display Metrics (Table) --- 
        st.markdown("---")
        st.subheader("Performance Comparison Table")
        
        if not analysis_df.empty:
            st.dataframe(
                analysis_df.style.format({
                'CAGR': "{:.2%}",
                'Volatility': "{:.2%}",
                'Sharpe Ratio': "{:.2f}",
                'Max Drawdown': "{:.2%}",      
                '95% VaR (Daily)': "{:.2%}"     
                    }),
                use_container_width=True
            )
            if 'trading_days' in locals():
                st.caption(f"*Calculations based on {trading_days} trading days. Risk-Free Rate assumed to be {risk_free_rate*100}%.*")
        else:
            st.info("No data for comparison. Select stocks and/or benchmarks.")

        st.markdown("---")
        
        # --- 3. Interactive Chart: Cumulative Returns Comparison ---
        if not portfolio_cumulative_returns.empty or benchmark_returns:
            st.subheader("Portfolio vs. Benchmark Cumulative Returns")
            
            chart_df = pd.DataFrame()
            if not portfolio_cumulative_returns.empty:
                chart_df['Portfolio'] = portfolio_cumulative_returns
                
            for name, returns in benchmark_returns.items():
                chart_df[ticker_list[name]] = returns
                
            fig_comp = calc_analysis_chart_func( # Uses dynamic Line/Area function
                chart_df,
                title="Portfolio vs. Benchmarks (Cumulative Growth)",
                labels={'value': 'Cumulative Return (Starting at 1.0)', 'Date': 'Date'}
            )
            fig_comp.update_yaxes(tickformat=".2f")
            fig_comp.add_hline(y=1.0, line_dash="dash", line_color="grey") # Normalize start at 1.0
            st.plotly_chart(fig_comp, use_container_width=True)
            st.markdown("---")

        # --- 4. Interactive Chart: Rolling Volatility (Risk Over Time) ---
        if not portfolio_daily_returns.empty or benchmark_list:
            st.subheader("Rolling Volatility (Risk Over Time)")
            
            rolling_vol_window = st.slider("Rolling Volatility Window (Trading Days)", 10, 120, 30, key='analysis_roll_vol')
            
            rolling_vol_df = pd.DataFrame()
            
            # Calculate rolling volatility for the portfolio
            if not portfolio_daily_returns.empty:
                rolling_vol_df['Portfolio'] = portfolio_daily_returns.rolling(window=rolling_vol_window).std() * annualization_factor
            
            # Calculate rolling volatility for benchmarks
            for b_ticker in benchmark_list:
                if b_ticker in yfdata['ticker'].unique():
                    b_daily_returns = yfdata[yfdata.ticker == b_ticker].set_index('Date')['price_pct_daily'].fillna(0)
                    rolling_vol_df[ticker_list[b_ticker]] = b_daily_returns.rolling(window=rolling_vol_window).std() * annualization_factor

            if not rolling_vol_df.empty:
                fig_vol = px.line(rolling_vol_df, title=f"{rolling_vol_window}-Day Rolling Annualized Volatility")
                fig_vol.update_yaxes(tickformat=",.2%")
                st.plotly_chart(fig_vol, use_container_width=True)
            # Remove the markdown separator here so the tab doesn't end abruptly
            # st.markdown("---") # REMOVED 
            
# ------------------------ Diversification Tab (NEW TAB 4) ------------------------#
with tab4:
    st.header("🔗 Asset Correlation & Diversification")
    
    if not sel_tickers_list or len(sel_tickers_list) < 2:
         st.info("Select at least two stocks/benchmarks in the sidebar to calculate and display correlation.")
    elif yfdata.empty:
        st.warning("No historical data found for the selected assets or date range.")
    else:
        # Create a pivoted DataFrame of all selected daily returns (stocks + benchmarks)
        correlation_df = yfdata[yfdata.ticker.isin(sel_tickers_list)].pivot(index='Date', columns='ticker', values='price_pct_daily')
        
        # Calculate the Correlation Matrix
        corr_matrix = correlation_df.corr()
        
        if not corr_matrix.empty:
            st.subheader("Asset Correlation Heatmap")
            st.caption("Lower correlation values (closer to 0) indicate better diversification. Values near 1 or -1 suggest assets move closely together.")
            
            
            # Rename columns for better readability on the chart (e.g., replace Ticker with Name)
            corr_matrix.columns = [ticker_list.get(t, t) for t in corr_matrix.columns]
            corr_matrix.index = corr_matrix.columns
            
            fig_corr = px.imshow(
                corr_matrix,
                text_auto=".2f", # Format text to two decimal places
                aspect="auto",
                color_continuous_scale=px.colors.diverging.RdBu, # Red-Blue scale
                zmin=-1, zmax=1, # Fix the range from -1 to 1
                title="Daily Return Correlation Matrix (Lower is better for diversification)"
            )
            fig_corr.update_layout(
                xaxis_showgrid=False,
                yaxis_showgrid=False,
                xaxis_nticks=len(corr_matrix.columns),
                yaxis_nticks=len(corr_matrix.index),
                height=600 
            )
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("No daily return data available for correlation calculation.")