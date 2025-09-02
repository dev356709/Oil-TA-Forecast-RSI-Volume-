# Oil-TA-Forecast-RSI-Volume-


A Streamlit app for **intraday traders** and **swing traders** to explore oil price forecasts using **technical analysis**. The app predicts the next 7 trading days’ close, high, and low based on RSI, volume momentum, and ATR bands. It also provides backtesting, parameter auto-tuning, and intraday 15-minute analysis.

---

 Features

* **7-Day Forecast**

  * Daily predictions of Open, High, Low, Close.
  * Forecast bands using ATR.
Backtest Panel

  * Walk-forward 1-step ahead testing.
  * Metrics: MAE, MAPE, Band Coverage, Bias, StdDev.
  * Residual diagnostics: Histogram, QQ Plot, ACF
 
Auto-Tune Parameters

  * Grid search for RSI/ATR/EMA/Vol-Z.
  * Apply best parameters with one click.

Intraday 15-Minute View

  * Candlestick with Volume and RSI overlays.
  * Intraday notes (RSI oversold/overbought counts, volume spikes, drift).
Downloadable Data


 Forecast and backtest results as CSV.



Installation

Clone this repo and install dependencies:

```bash
git clone https://github.com/your-username/oil-ta-forecast.git
cd oil-ta-forecast
pip install -r requirements.txt
```

Or install manually:

```bash
pip install streamlit yfinance pandas numpy plotly pytz scipy
```

---

 Usage

Run the Streamlit app:

```bash
streamlit run oil-ta-forecast_streamlit_app.py
```

Then open the provided local URL in your browser.



Sidebar Controls

* **RSI Period**: Relative Strength Index window.
* **ATR Period / Multiplier**: Controls forecast High/Low bands.
* **EMA Drift Span**: Controls trend persistence.
* **Volume Z-score Lookback**: Captures abnormal volume.
* **Forecast Horizon**: Number of trading days (default: 7).
* **Intraday Window**: Number of past days for 15-minute chart.
* **Backtest Window**: Days to evaluate past predictions.
* **Auto-tune (Grid Search)**: Try different parameter combinations.



 Outputs

* **Forecast Table & Chart** — Next 7 trading days with predicted ranges.
* **Backtest Metrics** — Accuracy statistics.
* **Residual Diagnostics** — Check bias and error patterns.
* **Intraday Candlestick** — 15m granularity with RSI & Volume.

Disclaimer

This tool is for **educational purposes only**. It uses heuristic technical analysis and Yahoo Finance data. It is **not financial advice**. Trading oil futures involves substantial risk.



 License

MIT License — free to use and modify.



Contributing

Pull requests are welcome! Please open issues for bug reports or feature requests.

---

 Project Structure

```
oil-ta-forecast/
├── oil-ta-forecast_streamlit_app.py   # Main Streamlit app
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Ignore cache/venv/CSV etc.
```



 Future Improvements

* Export charts as PNG (via `kaleido`).
* Persist best parameter set between sessions.
* Add more TA indicators (MACD, Bollinger Bands).
