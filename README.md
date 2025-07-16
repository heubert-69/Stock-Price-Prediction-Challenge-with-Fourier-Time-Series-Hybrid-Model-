# 📈 Stock Price Prediction with Fourier Time Series (Hybrid Model)

This project applies advanced **time series feature engineering** and **machine learning** to predict next-day stock returns. Using a hybrid approach, we integrate **Fourier Transform (FFT)** with traditional financial indicators to capture both frequency-domain and time-domain signals.

## 🧠 Model
- **Algorithm**: Random Forest Regressor
- **Data**: Multi-ticker OHLCV stock data
- **Target**: Next-day return (`Returns.shift(-1)`)

## ⚙️ Hybrid Features Engineered
- 🔁 **Fourier Transform Features** (`fft_amp_1`, `fft_dom_freq`, etc.)
- 📉 **Volatility & Risk**: rolling std, drawdown, Sharpe proxy, volatility ratios
- 📊 **Statistics**: rolling mean, skew, kurtosis
- 🔁 **Momentum**: lagged returns, 10-day return
- 📈 **Trend**: log slope over time
- 💰 **Volume Metrics**: VWAP, log volume, dollar volume
- 🧠 **Others**: RelToHigh/Low, entropy, autocorrelation

## 📊 Evaluation Metrics

| Metric                   | Value      |
|--------------------------|------------|
| R² Score                 | 0.0234     |
| MAE                      | 0.011691   |
| RMSE                     | 0.019550   |
| Directional Accuracy     | 53.09%     |
| Sharpe Ratio             | 0.0084     |

### 📉 Quantile Losses

| Quantile | Loss     |
|----------|----------|
| 0.10     | 0.005859 |
| 0.25     | 0.005855 |
| 0.50     | 0.005847 |
| 0.75     | 0.005840 |
| 0.90     | 0.005836 |

> ⚠️ Although the model’s R² is low, it shows consistent predictive behavior across quantiles and a slightly above-random directional accuracy.

## 📦 Dependencies

- Python 3.11+
- `pandas`, `numpy`, `scikit-learn`, `scipy`, `matplotlib`, `seaborn`

## 🚀 How to Run

```bash
pip install -r requirements.txt
python train_model.py
```

📁 Project Structure
```bash
├── combine_test_data.py
├── data
│   └── StockData.csv
├── DataPreprocessing.ipynb
├── raw_data
│   ├── AAPL.csv
│   ├── ABBV.csv
│   ├── AXP.csv
│   ├── BA.csv
│   ├── BOOT.csv
│   ├── CALM.csv
│   ├── CAT.csv
│   ├── CL.csv
│   ├── CSCO.csv
│   ├── CVX.csv
│   ├── DD.csv
│   ├── DENN.csv
│   ├── DIS.csv
│   ├── F.csv
│   ├── GE.csv
│   ├── GM.csv
│   ├── GS.csv
│   ├── HON.csv
│   ├── IBM.csv
│   ├── INTC.csv
│   ├── IP.csv
│   ├── JNJ.csv
│   ├── JPM.csv
│   ├── KO.csv
│   ├── LMT.csv
│   ├── MA.csv
│   ├── MCD.csv
│   └── MG.csv
├── raw_test_data
│   ├── test_1.csv
│   ├── test_2.csv
│   ├── test_3.csv
│   ├── test_4.csv
│   └── test_5.csv
├── StockPriceCommunityPrediction.ipynb
├── submission.csv
└── Visualizations
    ├── Correlation Analysis.png
    ├── Live Prediction Tracking.png
    ├── Most Dominant Ticker.png
    ├── Outliers.png
    ├── Price Distribution.png
    ├── Price Imbalance.png
    ├── Residuals vs Predicted.png
    └── Volume Analysis.png
```
