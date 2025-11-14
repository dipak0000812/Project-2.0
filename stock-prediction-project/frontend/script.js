// ============================================
// STOCK PREDICTION APP - FIXED JAVASCRIPT
// Save as: frontend/script.js
// ============================================

// Configuration
const API_URL = 'http://localhost:5000';  // Change this if backend runs on different port

// Global variables
let predictionChart = null;
let predictionHistory = [];

// ============================================
// INITIALIZATION
// ============================================
document.addEventListener('DOMContentLoaded', function() {
    console.log('%c Stock Prediction App Loaded! ', 'background: #1e3c72; color: white; font-size: 16px; padding: 10px;');
    console.log(`API URL: ${API_URL}`);
    
    initializeEventListeners();
    loadPredictionHistory();
    checkBackendConnection();
});

// ============================================
// CHECK BACKEND CONNECTION
// ============================================
async function checkBackendConnection() {
    try {
        const response = await fetch(`${API_URL}/health`);
        if (response.ok) {
            console.log('%c ✓ Backend Connected ', 'background: green; color: white; padding: 5px;');
        } else {
            console.warn('%c ⚠ Backend Not Responding ', 'background: orange; color: white; padding: 5px;');
            showAlert('Backend server might not be running', 'warning');
        }
    } catch (error) {
        console.error('%c ✗ Backend Not Available ', 'background: red; color: white; padding: 5px;');
        console.error('Make sure backend is running: python app.py');
        showAlert('Backend not connected. Start server with: python app.py', 'error');
    }
}

// ============================================
// EVENT LISTENERS
// ============================================
function initializeEventListeners() {
    // Form submission
    const form = document.getElementById('predictionForm');
    form.addEventListener('submit', handlePrediction);
    
    // Quick select chips
    const chips = document.querySelectorAll('.chip');
    chips.forEach(chip => {
        chip.addEventListener('click', function() {
            const ticker = this.getAttribute('data-ticker');
            document.getElementById('stockTicker').value = ticker;
        });
    });
}

// ============================================
// MAIN PREDICTION FUNCTION
// ============================================
async function handlePrediction(e) {
    e.preventDefault();
    
    // Get form values
    const ticker = document.getElementById('stockTicker').value.trim().toUpperCase();
    const days = document.getElementById('predictionDays').value;
    
    // Validate input
    if (!ticker) {
        showAlert('Please enter a stock ticker symbol', 'error');
        return;
    }
    
    console.log(`Making prediction for ${ticker}...`);
    
    // Show loading state
    showLoading(true);
    
    try {
        // Call Python backend API
        console.log(`Fetching: ${API_URL}/predict`);
        const prediction = await fetchPrediction(ticker, days);
        
        console.log('Prediction received:', prediction);
        
        // Display results
        displayResults(prediction, ticker);
        
        // Save to history
        savePredictionToHistory(prediction, ticker);
        
        // Show success message
        showAlert('Prediction completed successfully!', 'success');
        
    } catch (error) {
        console.error('Prediction error:', error);
        
        // Check if it's a network error
        if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {
            showAlert('Cannot connect to backend. Make sure Flask server is running on port 5000', 'error');
            console.error('Backend Connection Error:');
            console.error('1. Check if backend is running: python app.py');
            console.error('2. Check if port 5000 is available');
            console.error(`3. Try accessing: ${API_URL}/health`);
        } else {
            showAlert(`Error: ${error.message}`, 'error');
        }
        
        // Show demo data for testing
        console.log('Showing demo data for testing...');
        showDemoData(ticker);
        
    } finally {
        showLoading(false);
    }
}

// ============================================
// API CALL TO PYTHON BACKEND
// ============================================
async function fetchPrediction(ticker, days) {
    const url = `${API_URL}/predict`;
    
    console.log('Request URL:', url);
    console.log('Request Body:', { ticker, days: parseInt(days) });
    
    const response = await fetch(url, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            ticker: ticker,
            days: parseInt(days)
        })
    });
    
    console.log('Response status:', response.status);
    
    if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
    }
    
    const data = await response.json();
    return data;
}

// ============================================
// DISPLAY RESULTS
// ============================================
function displayResults(data, ticker) {
    console.log('Displaying results for', ticker);
    
    // Show results card
    const resultsCard = document.getElementById('resultsCard');
    resultsCard.style.display = 'block';
    
    // Scroll to results
    setTimeout(() => {
        resultsCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);
    
    // Update price information
    document.getElementById('currentPrice').textContent = `$${data.current_price.toFixed(2)}`;
    document.getElementById('predictedPrice').textContent = `$${data.predicted_price.toFixed(2)}`;
    
    const changeElement = document.getElementById('priceChange');
    const changeText = `$${Math.abs(data.price_change).toFixed(2)} (${data.price_change_pct.toFixed(2)}%)`;
    changeElement.textContent = changeText;
    changeElement.className = data.price_change >= 0 ? 'value text-success' : 'value text-danger';
    
    // Update trend indicator
    updateTrendIndicator(data);
    
    // Update metrics
    updateMetrics(data.metrics);
    
    // Update chart
    updateChart(data, ticker);
}

// ============================================
// UPDATE TREND INDICATOR
// ============================================
function updateTrendIndicator(data) {
    const indicator = document.getElementById('trendIndicator');
    const icon = document.getElementById('trendIcon');
    const title = document.getElementById('trendTitle');
    const description = document.getElementById('trendDescription');
    
    if (data.direction === 'UP') {
        indicator.classList.remove('down');
        icon.textContent = '📈';
        title.textContent = 'Bullish Trend Detected';
        description.textContent = `Our model predicts the stock price will increase by ${data.price_change_pct.toFixed(2)}% in the next period.`;
    } else {
        indicator.classList.add('down');
        icon.textContent = '📉';
        title.textContent = 'Bearish Trend Detected';
        description.textContent = `Our model predicts the stock price will decrease by ${Math.abs(data.price_change_pct).toFixed(2)}% in the next period.`;
    }
}

// ============================================
// UPDATE METRICS
// ============================================
function updateMetrics(metrics) {
    document.getElementById('r2Score').textContent = metrics.r2_score.toFixed(3);
    document.getElementById('maeScore').textContent = `$${metrics.mae.toFixed(2)}`;
    document.getElementById('mapeScore').textContent = `${metrics.mape.toFixed(2)}%`;
    
    // Calculate confidence level
    const confidence = metrics.r2_score >= 0.8 ? 'High' : 
                      metrics.r2_score >= 0.6 ? 'Medium' : 'Low';
    document.getElementById('confidence').textContent = confidence;
}

// ============================================
// UPDATE CHART
// ============================================
function updateChart(data, ticker) {
    const ctx = document.getElementById('priceChart').getContext('2d');
    
    // Destroy existing chart if exists
    if (predictionChart) {
        predictionChart.destroy();
    }
    
    // Prepare chart data
    const labels = data.historical_data.map((_, i) => `Day ${i + 1}`);
    labels.push('Prediction');
    
    // Create new chart
    predictionChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Historical Price',
                data: data.historical_data,
                borderColor: '#1e3c72',
                backgroundColor: 'rgba(30, 60, 114, 0.1)',
                borderWidth: 3,
                tension: 0.4,
                pointRadius: 4,
                pointHoverRadius: 6
            }, {
                label: 'Predicted Price',
                data: [...Array(data.historical_data.length).fill(null), data.predicted_price],
                borderColor: '#10b981',
                backgroundColor: 'rgba(16, 185, 129, 0.1)',
                borderWidth: 3,
                borderDash: [10, 5],
                pointRadius: 6,
                pointHoverRadius: 8,
                pointStyle: 'star'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                title: {
                    display: true,
                    text: `${ticker} Stock Price Prediction`,
                    font: {
                        size: 16,
                        weight: 'bold'
                    }
                },
                legend: {
                    display: true,
                    position: 'top'
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `$${context.parsed.y.toFixed(2)}`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    ticks: {
                        callback: function(value) {
                            return '$' + value.toFixed(2);
                        }
                    }
                }
            }
        }
    });
}

// ============================================
// PREDICTION HISTORY
// ============================================
function savePredictionToHistory(data, ticker) {
    const historyItem = {
        date: new Date().toLocaleString(),
        ticker: ticker,
        currentPrice: data.current_price,
        predictedPrice: data.predicted_price,
        change: data.price_change,
        changePct: data.price_change_pct,
        status: data.direction
    };
    
    predictionHistory.unshift(historyItem);
    
    if (predictionHistory.length > 10) {
        predictionHistory = predictionHistory.slice(0, 10);
    }
    
    localStorage.setItem('predictionHistory', JSON.stringify(predictionHistory));
    updateHistoryTable();
}

function loadPredictionHistory() {
    const saved = localStorage.getItem('predictionHistory');
    if (saved) {
        predictionHistory = JSON.parse(saved);
        updateHistoryTable();
    }
}

function updateHistoryTable() {
    if (predictionHistory.length === 0) return;
    
    const historyCard = document.getElementById('historyCard');
    const tbody = document.getElementById('historyTableBody');
    
    historyCard.style.display = 'block';
    tbody.innerHTML = '';
    
    predictionHistory.forEach(item => {
        const row = document.createElement('tr');
        
        const changeClass = item.change >= 0 ? 'text-success' : 'text-danger';
        const statusIcon = item.status === 'UP' ? '📈' : '📉';
        
        row.innerHTML = `
            <td>${item.date}</td>
            <td><strong>${item.ticker}</strong></td>
            <td>$${item.currentPrice.toFixed(2)}</td>
            <td>$${item.predictedPrice.toFixed(2)}</td>
            <td class="${changeClass}">
                ${item.change >= 0 ? '+' : ''}$${item.change.toFixed(2)} 
                (${item.changePct.toFixed(2)}%)
            </td>
            <td>${statusIcon} ${item.status}</td>
        `;
        
        tbody.appendChild(row);
    });
}

// ============================================
// DEMO DATA
// ============================================
function showDemoData(ticker) {
    console.log('Showing demo data for:', ticker);
    
    const demoData = {
        ticker: ticker,
        current_price: 175.50,
        predicted_price: 182.30,
        price_change: 6.80,
        price_change_pct: 3.87,
        direction: 'UP',
        metrics: {
            r2_score: 0.85,
            mae: 23.50,
            mape: 2.5
        },
        historical_data: [165, 168, 172, 170, 173, 175.50],
        predicted_data: [182.30]
    };
    
    displayResults(demoData, ticker);
    savePredictionToHistory(demoData, ticker);
}

// ============================================
// UI HELPERS
// ============================================
function showLoading(isLoading) {
    const btnText = document.querySelector('.btn-text');
    const btnLoader = document.querySelector('.btn-loader');
    const btn = document.getElementById('predictBtn');
    
    if (isLoading) {
        btnText.style.display = 'none';
        btnLoader.style.display = 'inline-flex';
        btn.disabled = true;
    } else {
        btnText.style.display = 'inline';
        btnLoader.style.display = 'none';
        btn.disabled = false;
    }
}

function showAlert(message, type) {
    const alert = document.createElement('div');
    alert.className = `alert alert-${type}`;
    
    let bgColor = '#10b981';
    if (type === 'error') bgColor = '#ef4444';
    if (type === 'warning') bgColor = '#f59e0b';
    
    alert.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 15px 25px;
        background: ${bgColor};
        color: white;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        z-index: 1000;
        animation: slideIn 0.4s ease;
        font-weight: 500;
        max-width: 400px;
    `;
    alert.textContent = message;
    
    const style = document.createElement('style');
    style.textContent = `
        @keyframes slideIn {
            from { transform: translateX(400px); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
    `;
    document.head.appendChild(style);
    
    document.body.appendChild(alert);
    
    setTimeout(() => {
        alert.remove();
    }, 5000);
}

// ============================================
// CONSOLE INFO
// ============================================
console.log('%c Backend URL: ' + API_URL, 'color: #10b981; font-size: 12px;');
console.log('%c To change backend URL, edit API_URL in script.js ', 'color: #f59e0b; font-size: 12px;');