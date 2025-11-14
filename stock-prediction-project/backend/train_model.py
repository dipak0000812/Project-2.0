"""
===============================================================================
STOCK PRICE PREDICTION - WITH SAMPLE DATA
===============================================================================
Use this if yfinance is not working
This generates realistic sample data for training
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("STOCK PREDICTION - TRAINING WITH SAMPLE DATA")
print("="*70)
print("NOTE: Using generated sample data for demonstration\n")

# Generate realistic stock data
print("Generating sample stock data...")
np.random.seed(42)

# Create date range (2 years)
end_date = datetime.now()
start_date = end_date - timedelta(days=730)
dates = pd.date_range(start=start_date, end=end_date, freq='D')

# Generate realistic price movements
initial_price = 150.0
daily_returns = np.random.normal(0.0005, 0.015, len(dates))
prices = initial_price * np.cumprod(1 + daily_returns)

# Add some trend
trend = np.linspace(0, 20, len(dates))
prices = prices + trend

# Create DataFrame
df = pd.DataFrame({
    'Open': prices * (1 + np.random.uniform(-0.01, 0.01, len(dates))),
    'High': prices * (1 + np.random.uniform(0.005, 0.02, len(dates))),
    'Low': prices * (1 - np.random.uniform(0.005, 0.02, len(dates))),
    'Close': prices,
    'Adj Close': prices,
    'Volume': np.random.randint(50000000, 150000000, len(dates))
}, index=dates)

print(f"✓ Generated {len(df)} days of sample data\n")

# Feature Engineering
print("Creating features...")
df['MA_5'] = df['Close'].rolling(window=5).mean()
df['MA_10'] = df['Close'].rolling(window=10).mean()
df['MA_20'] = df['Close'].rolling(window=20).mean()
df['Price_Change'] = df['Close'].diff()
df['Price_Change_Pct'] = df['Close'].pct_change() * 100
df['Volatility'] = df['Close'].rolling(window=10).std()
df['High_Low_Diff'] = df['High'] - df['Low']
df['Volume_Change'] = df['Volume'].pct_change() * 100
df['Close_Lag_1'] = df['Close'].shift(1)
df['Close_Lag_2'] = df['Close'].shift(2)
df['Close_Lag_3'] = df['Close'].shift(3)
df['Target'] = df['Close'].shift(-1)

df = df.dropna()
print(f"✓ Created features. Dataset: {len(df)} rows\n")

# Prepare training data
feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 
                   'MA_5', 'MA_10', 'MA_20', 'Price_Change_Pct',
                   'Volatility', 'High_Low_Diff', 'Volume_Change',
                   'Close_Lag_1', 'Close_Lag_2', 'Close_Lag_3']

X = df[feature_columns]
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

print(f"Training set: {len(X_train)} samples")
print(f"Testing set: {len(X_test)} samples\n")

# Train model
print("Training Random Forest model...")
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
print("✓ Model trained!\n")

# Evaluate
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)
mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100

print("Model Performance:")
print(f"  • MAE: ${mae:.2f}")
print(f"  • RMSE: ${rmse:.2f}")
print(f"  • R² Score: {r2:.4f} ({r2*100:.2f}% accuracy)")
print(f"  • MAPE: {mape:.2f}%\n")

# Save model
joblib.dump(model, 'stock_prediction_model.pkl')
joblib.dump(feature_columns, 'feature_names.pkl')
joblib.dump({'r2_score': r2, 'mae': mae, 'mape': mape}, 'model_metrics.pkl')

print("✓ Model saved successfully!")
print("\nFiles created:")
print("  📁 stock_prediction_model.pkl")
print("  📁 feature_names.pkl")
print("  📁 model_metrics.pkl")

print("\n" + "="*70)
print("✅ READY! Run 'python app.py' to start the server")
print("="*70)