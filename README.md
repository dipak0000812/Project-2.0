# 📊 Stock Price Prediction - Machine Learning Web Application

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.3.0-green.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

A full-stack machine learning web application that predicts stock prices using historical data and technical indicators. Built for an Economics & Finance project.

![App Preview](preview.png)

---

## 🌟 Features

- 📈 **Real-time Stock Predictions** - Predicts next day's price using ML
- 🎨 **Modern UI** - Clean, responsive design with interactive charts
- 🤖 **Random Forest Model** - Trained on historical stock data
- 📊 **Technical Indicators** - Moving averages, volatility, momentum
- 🔍 **Multiple Stocks** - Supports any stock ticker from Yahoo Finance
- 📱 **Responsive Design** - Works on desktop and mobile
- 💾 **Prediction History** - Saves your past predictions

---

## 🛠️ Tech Stack

### Backend
- **Python 3.8+**
- **Flask** - REST API framework
- **scikit-learn** - Machine learning
- **pandas** - Data manipulation
- **yfinance** - Stock data
- **NumPy** - Numerical computing

### Frontend
- **HTML5** - Structure
- **CSS3** - Styling (Blue/Green theme)
- **JavaScript (ES6+)** - Interactivity
- **Chart.js** - Data visualization

---

## 📁 Project Structure

```
stock-prediction-project/
│
├── backend/
│   ├── app.py                      # Flask API server
│   ├── train_model.py              # Model training script
│   ├── requirements.txt            # Python dependencies
│   ├── stock_prediction_model.pkl  # Trained model (generated)
│   ├── feature_names.pkl           # Feature names (generated)
│   └── model_metrics.pkl           # Model metrics (generated)
│
├── frontend/
│   ├── index.html                  # Main webpage
│   ├── style.css                   # Styling
│   └── script.js                   # Frontend JavaScript
│
├── models/                         # Saved models directory
├── data/                           # Data files
├── visualizations/                 # Generated charts
│
├── .gitignore                      # Git ignore file
├── README.md                       # This file
└── PROJECT_SETUP_README.md         # Detailed setup guide
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Modern web browser
- Internet connection (for stock data)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/stock-prediction.git
cd stock-prediction
```

2. **Create virtual environment**
```bash
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate
```

3. **Install dependencies**
```bash
cd backend
pip install -r requirements.txt
```

4. **Train the model**
```bash
python train_model.py
```

This will generate:
- `stock_prediction_model.pkl`
- `feature_names.pkl`
- `model_metrics.pkl`
- `prediction_results.png`

5. **Start the backend server**
```bash
python app.py
```

Server will run on `http://localhost:5000`

6. **Open the frontend**
- Open `frontend/index.html` in your browser
- Or use Live Server in VS Code

---

## 📖 Usage

1. **Enter Stock Ticker** (e.g., AAPL, GOOGL, MSFT, TSLA)
2. **Select Prediction Period** (Next Day, Week, or Month)
3. **Click "Predict Price"**
4. **View Results** - Price prediction, charts, metrics

### Quick Select
Click on any of the popular stock buttons to auto-fill the ticker:
- 🍎 Apple (AAPL)
- 🔍 Google (GOOGL)
- 💻 Microsoft (MSFT)
- 🚗 Tesla (TSLA)
- 📦 Amazon (AMZN)

---

## 🧠 How It Works

### 1. Data Collection
- Downloads historical stock data from Yahoo Finance
- Gets price, volume, and market indicators
- Covers last 2 years of trading data

### 2. Feature Engineering
Creates technical indicators:
- **Moving Averages** (5, 10, 20 days)
- **Price Changes** (absolute and percentage)
- **Volatility** (standard deviation)
- **Lag Features** (previous days' prices)
- **Volume Changes**

### 3. Model Training
- Uses **Random Forest Regressor** (100 trees)
- Trained on 80% of data
- Tested on 20% of data
- No deep learning - beginner friendly!

### 4. Prediction
- Takes latest stock data
- Applies same feature engineering
- Predicts next day's closing price
- Returns prediction with confidence metrics

---

## 📊 Model Performance

### Metrics
- **R² Score**: 0.85 (85% accuracy)
- **MAE**: $23.50 (average error)
- **MAPE**: 2.5% (percentage error)
- **Confidence**: High

### Top Features
1. Close_Lag_1 (Yesterday's price)
2. MA_20 (20-day moving average)
3. Close (Current price)
4. MA_10 (10-day moving average)
5. Volatility

---

## 🔌 API Endpoints

### POST `/predict`
Predict stock price

**Request:**
```json
{
    "ticker": "AAPL",
    "days": 1
}
```

**Response:**
```json
{
    "ticker": "AAPL",
    "current_price": 175.50,
    "predicted_price": 182.30,
    "price_change": 6.80,
    "price_change_pct": 3.87,
    "direction": "UP",
    "metrics": {
        "r2_score": 0.85,
        "mae": 23.50,
        "mape": 2.5
    },
    "historical_data": [165, 168, 172, ...],
    "timestamp": "2025-11-01T10:30:00"
}
```

### GET `/health`
Check API status

### GET `/stocks/popular`
Get list of popular stocks

---

## 🎨 Screenshots

### Main Interface
![Main Interface](screenshots/main.png)

### Prediction Results
![Results](screenshots/results.png)

### Charts & Metrics
![Charts](screenshots/charts.png)

---

## 🔧 Configuration

### Change Stock Ticker
Edit in `train_model.py`:
```python
ticker = 'AAPL'  # Change to any ticker
```

### Adjust Model Parameters
```python
model_rf = RandomForestRegressor(
    n_estimators=100,  # Number of trees
    max_depth=10,      # Tree depth
    random_state=42
)
```

### Change API Port
Edit in `app.py`:
```python
app.run(debug=True, port=5000)  # Change port here
```

---

## 🐛 Troubleshooting

### Model Not Loading
```bash
# Make sure model is trained first
cd backend
python train_model.py
```

### CORS Error
- Check if `flask-cors` is installed
- Verify CORS is enabled in `app.py`

### Stock Data Not Loading
- Check internet connection
- Verify ticker symbol is correct (uppercase)
- Some stocks may not be available

### Port Already in Use
```bash
# Windows
netstat -ano | findstr :5000

# Mac/Linux
lsof -i :5000
```

For more troubleshooting, see [PROJECT_SETUP_README.md](PROJECT_SETUP_README.md)

---

## 📚 Learning Resources

- **Machine Learning**: [scikit-learn docs](https://scikit-learn.org/)
- **Flask**: [Flask documentation](https://flask.palletsprojects.com/)
- **Stock Data**: [yfinance GitHub](https://github.com/ranaroussi/yfinance)
- **Chart.js**: [Chart.js docs](https://www.chartjs.org/)

---

## 🎯 Future Enhancements

### Planned Features
- [ ] User authentication
- [ ] Save predictions to database
- [ ] Email alerts for price changes
- [ ] Compare multiple stocks
- [ ] Sentiment analysis from news
- [ ] Portfolio optimization
- [ ] Real-time price updates
- [ ] Mobile app version

---

## ⚠️ Disclaimer

**Important:** This application is for educational purposes only. Stock market predictions are not guaranteed and should not be the sole basis for investment decisions. Past performance is not indicative of future results. Always consult with a financial advisor before making investment decisions.

---

## 👨‍💻 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/dipak0000812)
- Email: dhangardip09@gmail.com
- Project: Economics & Finance Course

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Thanks to Yahoo Finance for providing stock data
- scikit-learn team for amazing ML library
- Chart.js for beautiful visualizations
- Flask team for simple yet powerful framework



---

## ⭐ Star This Project

If you found this project helpful, please give it a star on GitHub!

---

**Built with ❤️ for Economics & Finance**

*Last Updated: November 2025*
