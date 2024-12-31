# Quick Start Guide 🚀

This guide will help you get Prophet Forecaster up and running quickly.

## Prerequisites 📋

1. **Python 3.8+**
   - Download from [python.org](https://www.python.org/downloads/)
   - Make sure to check "Add Python to PATH" during installation

2. **Git** (optional, for cloning)
   - Download from [git-scm.com](https://git-scm.com/downloads)

## Installation 💻

1. **Get the Code**
   ```bash
   # Option 1: Clone with Git
   git clone https://github.com/yourusername/prophet-forecaster.git
   cd prophet-forecaster

   # Option 2: Download ZIP
   # Download and extract the ZIP file, then open a terminal in the extracted folder
   ```

2. **Set Up Virtual Environment** (recommended)
   ```bash
   # Create virtual environment
   python -m venv venv

   # Activate virtual environment
   # Windows:
   venv\Scripts\activate
   # Unix/MacOS:
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   # Install required packages
   pip install -r requirements.txt
   ```

## Configuration ⚙️

1. **Environment Setup**
   ```bash
   # Copy example environment file
   cp .env.example .env
   ```

2. **Edit Environment Variables**
   - Open `.env` in your text editor
   - Add any required API keys
   - Adjust configuration settings if needed
   - NOTE: API keys are not required for the default configuration.

## Running the Program 🏃

The Prophet Forecaster offers two interfaces to suit different needs:

### 1. Interactive Mode (Recommended for Beginners)
Using `run_interactive.py`:
```bash
python run_interactive.py
```

This interface will:
- Guide you through each step with clear prompts
- Provide sensible defaults
- Show progress and explanations
- Handle all configuration automatically

Example session:
```
? Select data source: Yahoo Finance
? Enter stock symbol: AAPL
? Select data interval: 1d
? Use default date range? Yes
? Select cleaning method: impute
? Select outlier detection: zscore
? Select technical indicators: [trend, momentum]
? Use hyperparameter tuning? Yes
```

### 2. Command Line Interface (For Advanced Users)
Using `main.py` with flags:
```bash
# Basic usage
python main.py --symbol AAPL --interval 1d

# Full options
python main.py --symbol AAPL \
               --interval 1d \
               --start-date 2023-01-01 \
               --end-date 2023-12-31 \
               --cleaning-method impute \
               --outlier-method zscore \
               --indicators trend momentum \
               --tune-hyperparameters \
               --forecast-periods 30
```

Available flags:
- `--symbol`: Stock symbol (e.g., AAPL, MSFT)
- `--interval`: Data interval (1d, 1h, 1wk, 1mo)
- `--start-date`: Start date (YYYY-MM-DD)
- `--end-date`: End date (YYYY-MM-DD)
- `--cleaning-method`: Data cleaning method (impute, drop)
- `--outlier-method`: Outlier detection (zscore, iqr)
- `--indicators`: Technical indicator categories (trend, momentum, volatility, volume)
- `--tune-hyperparameters`: Enable hyperparameter tuning
- `--forecast-periods`: Number of periods to forecast
- `--output-dir`: Custom output directory
- `--config-file`: Custom configuration file

### Useful Commands

1. **Get Help**
   ```bash
   python main.py --help
   ```

2. **Quick Start with Defaults**
   ```bash
   python main.py --symbol AAPL
   ```

3. **Custom Date Range**
   ```bash
   python main.py --symbol AAPL --start-date 2023-01-01 --end-date 2023-12-31
   ```

4. **Full Analysis**
   ```bash
   python main.py --symbol AAPL \
                  --interval 1d \
                  --indicators trend momentum volatility \
                  --tune-hyperparameters \
                  --forecast-periods 30
   ```

### Tips
- Use `run_interactive.py` if you're new to the program
- Use `main.py` for automation or scripting
- The interactive mode is more beginner-friendly and explains each step
- The command-line interface is faster for repeated analyses
- Both interfaces produce the same high-quality results

## Output Files 📁

The program generates several files in the following directories:
- `data/raw/`: Raw downloaded data
- `data/interim/`: Processed and cleaned data
- `data/features/`: Generated features and indicators
- `models/`: Trained models and metadata
- `visualizations/`: Plots and charts

## Common Issues & Solutions 🔧

1. **Missing Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Data Download Issues**
   - Check your internet connection
   - Verify the stock symbol exists
   - Try a different time period

3. **Memory Issues**
   - Reduce the date range
   - Close other applications
   - Use a smaller number of indicators

## Next Steps 🎯

1. **Explore Advanced Features**
   - Try different technical indicators
   - Adjust model parameters
   - Experiment with ensemble methods

2. **Customize Configuration**
   - Modify `.env` settings
   - Add custom indicators
   - Adjust validation periods

For more detailed information, see the full [README.md](README.md).