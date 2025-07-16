#separate script btw
import pandas as pd
import glob
import joblib
from pandas.tseries.offsets import BDay

# === Load your 5 test stocks ===
test_files = glob.glob("test/*.csv")
test_dfs = {f.split('/')[-1].replace('.csv', ''): pd.read_csv(f) for f in test_files}

# === Initialize result holder ===
returns_forecast = {}

# === Loop over each stock ===
for i, (ticker, df) in enumerate(test_dfs.items(), start=1):
    df['Ticker'] = ticker
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index(['Date', 'Ticker'], inplace=True)
    df.sort_index(inplace=True)

    # ✅ Transform features using trained transformer
    X_test_transformed = search.best_estimator_.named_steps['features'].transform(df)
    last_features = X_test_transformed.iloc[-1:]

    # ✅ Predict next 10 returns by repeating last known features
    preds = [search.predict(last_features)[0] for _ in range(10)]

    # ✅ Save predictions for current stock
    returns_forecast[f'Returns_{i}'] = preds

    # ✅ Save date range from last available date
    if i == 1:
        last_date = df.index.get_level_values('Date').max()
        forecast_dates = pd.bdate_range(start=last_date + BDay(1), periods=10)

# === Combine all predictions into submission DataFrame ===
submission_df = pd.DataFrame({'Date': forecast_dates})

for col, values in returns_forecast.items():
    submission_df[col] = values

submission_df['Date'] = submission_df['Date'].dt.strftime('%-m/%-d/%Y')  # use %-m on Linux/Mac; %m if on Windows

submission_df.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
print(submission_df)

