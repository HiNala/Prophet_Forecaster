# Prophet Forecaster 📈

A sophisticated time series forecasting tool that combines Facebook's Prophet model with technical analysis for stock market predictions. This tool is designed to be both powerful for experts and accessible for beginners.

## Getting Started in 30 Seconds 🚀

We provide two ways to use Prophet Forecaster:

1. **Interactive Mode** (Recommended for Beginners)
   ```bash
   python run_interactive.py
   ```
   We created this interface to make it super easy to get started - just follow the prompts and you'll be forecasting in minutes! No configuration needed.

2. **Command Line Interface** (For Advanced Users)
   ```bash
   python main.py --symbol AAPL --interval 1d
   ```
   Perfect for automation and scripting. See [QUICKSTART.md](QUICKSTART.md) for all available options.

## What Does It Do? 🎯

Think of Prophet Forecaster as a crystal ball for stock prices - but instead of magic, it uses data science! Here's what it does in simple terms:

1. **Gets Stock Data**: Downloads stock information (like prices) from Yahoo Finance
2. **Cleans the Data**: Makes sure the information is accurate and complete
3. **Analyzes Patterns**: Looks for trends and patterns in the stock's history
4. **Makes Predictions**: Forecasts where the stock price might go in the future
5. **Shows Results**: Creates nice graphs and reports to help you understand the predictions

## Key Features 🌟

### For Everyone
- 📊 Interactive command-line interface - just answer simple questions!
- 📈 Beautiful visualizations of predictions
- 🔄 Automatic data updates from Yahoo Finance
- 📋 Easy-to-read reports and summaries

### For Technical Users
- 🧪 Advanced ensemble learning combining multiple models
- 🛠️ Hyperparameter tuning for optimal performance
- 📐 Cross-validation for robust model evaluation
- 📊 Technical indicators (SMA, EMA, RSI, etc.)
- 🔍 Outlier detection and handling
- 📉 Missing value imputation

## How It Works 🔨

### The Learning Process

1. **Supervised Learning**
   - Uses historical stock data to learn patterns
   - Example: If a stock usually goes up after good earnings reports, the model learns this pattern

2. **Ensemble Learning**
   - Combines multiple prediction methods
   - Like getting opinions from different experts and combining their wisdom
   - Methods include:
     * Prophet (Facebook's forecasting tool)
     * Technical indicators (market analysis tools)
     * Statistical analysis

3. **Technical Analysis**
   - Calculates popular market indicators
   - Examples:
     * Moving Averages: Smooths out price data to show trends
     * RSI: Shows if a stock might be overbought or oversold
     * Bollinger Bands: Shows if prices are unusually high or low

### The Pipeline

1. **Getting Data**
   ```python
   # Example: Getting Apple stock data
   symbol = "AAPL"
   data = fetch_stock_data(symbol)
   ```

2. **Cleaning Data**
   - Removes bad data points
   - Fills in missing values
   - Makes sure dates are in order

3. **Feature Engineering**
   - Creates new insights from the data
   - Example: Calculating a 20-day moving average
   ```python
   data['SMA_20'] = data['Close'].rolling(window=20).mean()
   ```

4. **Model Training**
   - Teaches the model using historical data
   - Uses cross-validation to ensure reliability
   - Example periods:
     * Training: 60 days
     * Validation: 7 days
     * Testing: 7 days

5. **Making Predictions**
   - Forecasts future values
   - Provides confidence intervals
   - Example: 30-day forecast with 95% confidence bounds

## Quick Start 🚀

1. **Install**
   ```bash
   git clone https://github.com/yourusername/prophet-forecaster.git
   cd prophet-forecaster
   pip install -r requirements.txt
   ```

2. **Set Up Environment**
   - Copy `.env.example` to `.env`
   - Add any required API keys
   - NOTE: API keys are not required for the default configuration.

3. **Run**
   ```bash
   python run_interactive.py
   ```

4. **Follow the Prompts**
   ```
   ? Select data source: Yahoo Finance
   ? Enter stock symbol: AAPL
   ? Select data interval: 1d
   ```

## Understanding the Results 📊

### What You Get
1. **Forecast Plot**
   - Blue line: Actual stock prices
   - Red line: Predicted prices
   - Shaded area: Uncertainty range

2. **Performance Metrics**
   - RMSE (Root Mean Square Error): How accurate the predictions are
   - MAE (Mean Absolute Error): Average size of prediction errors

3. **Technical Analysis**
   - Trend indicators
   - Momentum signals
   - Volatility measures

### Example Interpretation
```
Forecast for AAPL:
- Current Price: $150
- 30-day Forecast: $165 ±10
- Trend: Upward
- Confidence: High
```

## Advanced Usage 🔬

For technical users who want to dive deeper:

1. **Custom Indicators**
   ```python
   # Add your own technical indicators
   def custom_indicator(data):
       # Your calculation here
       return result
   ```

2. **Model Tuning**
   ```python
   # Adjust hyperparameters
   params = {
       'changepoint_prior_scale': [0.001, 0.01, 0.1],
       'seasonality_prior_scale': [0.01, 0.1, 1.0]
   }
   ```

3. **Cross-Validation Settings**
   ```python
   # Modify validation periods
   cv_config = {
       'initial': '60 days',
       'period': '7 days',
       'horizon': '7 days'
   }
   ```

## Contributing 🤝

Contributions are welcome!
## License 📄

This project is licensed under the MIT License

## Acknowledgments 🙏

- Facebook's Prophet team for their amazing forecasting tool
- Yahoo Finance for providing stock data
- The open-source community for various technical analysis tools

---

Remember: Stock forecasting involves risk, and no prediction tool is perfect. Always do your own research and consider multiple sources before making investment decisions! 📈 