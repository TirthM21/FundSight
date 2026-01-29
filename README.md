# FundSight 📈 
### *Professional-Grade Mutual Fund Analytics & Strategy Dashboard*

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**FundSight** is an advanced detailed analytics platform for Indian Mutual Funds. Unlike basic trackers, FundSight focuses on **Rolling Returns**, **Risk-Adjusted Metrics**, and **Quantitative Strategies** to help investors make data-driven decisions. 

It fetches live data from AMFI (Association of Mutual Funds in India) and combines it with `yfinance` market data for benchmark comparisons and sector analysis.

---

## 🌟 Why FundSight?

Most platforms show "Point-to-Point" returns (e.g., "5 Year Return: 15%"). This is misleading because it depends heavily on the start and end dates. 
**FundSight** calculates **Rolling Returns**—calculating the return for *every* 5-year period in the fund's history—giving you a true picture of consistency and probability of returns.

---

## 🚀 Key Features

### 📊 Deep Performance Analysis
*   **Rolling Returns Engine**: Analyze 1Y, 3Y, 5Y, 7Y, 10Y, 15Y, and 20Y rolling CAGRs.
*   **Distribution Charts**: See the probability of negative vs. positive returns.
*   **Advanced Risk Metrics**: Sharpe Ratio, Sortino Ratio, Calmar Ratio, Max Drawdown, and Beta.
*   **Alpha Generation**: Calculate Jensen's Alpha and Information Ratio against customizable benchmarks.

### 🧬 Genetic Fund Ranking System
*   **Customizable Scoring**: Don't rely on generic ratings. Create your own ranking model by adjusting weights for:
    *   Consistency (Long/Medium/Short term)
    *   Downside Risk (Sortino)
    *   Recovery Ability
    *   Expense Ratio
*   **Smart Filtering**: Automatically strips out IDCW (Dividend) plans, showing only **Growth** options for pure analysis.

### 🔄 Quantitative Strategy Backtesting
*   **Sector Rotation Model**: Test institutional-grade momentum strategies.
    *   *Strategy*: Buy top N performing sectors (e.g., Bank, IT, Pharma).
    *   *Parameters*: Customize Lookback Period (3-12m) and Rebalance Frequency.
    *   *Result*: View Equity Curve, CAGR, and Drawdowns vs. Buy & Hold.
*   **Stress Testing**: Simulate how your portfolio would have performed during historic crashes (2008 Financial Crisis, 2020 Covid Crash).

### 🛠️ Portfolio Tools
*   **Auto-Portfolio Construction**: Get suggestions based on risk profile (Conservative, Moderate, Aggressive).
*   **Portfolio Builder**: Allocate weights and analyze aggregate stats.
*   **Correlation Matrix**: Check overlapping risks in your portfolio.

---

## 📦 Installation & Setup

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/yourusername/fundsight.git
    cd fundsight
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```
    *Core libraries: `streamlit`, `pandas`, `numpy`, `mftool`, `yfinance`, `plotly`, `scipy`*

3.  **Run the Dashboard**
    ```bash
    streamlit run app.py
    ```
    The app will open automatically in your browser at `http://localhost:8501`.

---

## 📖 User Guide

### 1. 🔍 Search & Filter
Go to **Search Funds**. You can search by name or browse by category (Large Cap, Flexi Cap, etc.).
> **Note**: The app automatically hides "Direct" vs "Regular" duplicates if configured, and strictly removes Dividend/IDCW plans to reduce noise.

### 2. 📈 Analyze a Fund
Select a fund and go to **Rolling Returns**.
*   Check the **Mean w/ Std Dev** to understand volatility.
*   Look at the **Win Rate** (percentage of periods with positive returns).

### 3. ⭐ Rank Funds
Go to **Fund Rankings**.
*   Select a category (e.g., "Mid Cap").
*   Open the **"⚙️ Customize Ranking Weights"** panel.
*   Slide "Sortino Ratio" to high if you hate losing money.
*   See the live updated table with top-scored funds.

### 4. 🔄 Backtest Sectors
Go to **Advanced Analysis > Sector Rotation**.
*   Pick sectors (Auto, Bank, IT, etc.).
*   Set Lookback to **6 Months** and Rebalance to **Monthly**.
*   Click **Run Backtest** to see if Momentum beats the market.

---

## 🛠️ Built With

*   **[Streamlit](https://streamlit.io/)** - For the interactive UI
*   **[mftool](https://github.com/NayakwadiS/mftool)** - For AMFI Mutual Fund API
*   **[Plotly](https://plotly.com/)** - For financial charting
*   **[yfinance](https://pypi.org/project/yfinance/)** - For NSE Sector Indices data

---

## ⚠️ Disclaimer

*This tool is for educational purposes only. Mutual Fund investments are subject to market risks. Past performance (even rolling returns) is not indicative of future results. Please consult a SEBI-registered financial advisor before investing.*

---
*Created with ❤️ by the FundSight Team*
