Yahoo Finance (**yfinance**) provides a wide range of data types for free through its API. Below is a breakdown of the types of data you can retrieve, the granularity available, and relevant details for each.

---

## **1. Historical Price Data**
Yahoo Finance provides historical pricing data for various financial instruments, including **stocks**, **ETFs**, **indices**, **mutual funds**, **cryptocurrencies**, **commodities**, and **forex**.

### **Data Available**
- **Open, High, Low, Close (OHLC)**
- **Adjusted Close**: Price adjusted for splits and dividends.
- **Volume**

### **Granularity**
- **Daily**: Standard for most trading strategies.
- **Weekly**: Aggregated weekly OHLC.
- **Monthly**: Aggregated monthly OHLC.
- **Intraday** (limited):
  - **1-minute, 2-minute, 5-minute, 15-minute, 30-minute, and 60-minute intervals**.
  - Limited to the past 60 days (for free users).
  - Available only for actively traded securities.

### **Usage Example**
```python
import yfinance as yf

# Fetch historical daily data for Apple (AAPL)
data = yf.download("AAPL", start="2022-01-01", end="2023-01-01", interval="1d")
print(data.head())
```

---

## **2. Real-Time or Recent Data**
Yahoo Finance provides real-time or recent pricing data for free.

### **Data Available**
- **Last Trade Price**
- **Bid/Ask Price**
- **Pre-Market and After-Hours Prices** (if applicable)

### **Granularity**
- Limited to the most recent quote (no history available for this type of data).

### **Usage Example**
```python
# Fetch real-time data for Tesla (TSLA)
ticker = yf.Ticker("TSLA")
print(ticker.info['regularMarketPrice'])  # Most recent trade price
```

---

## **3. Fundamental Data**
Yahoo Finance offers a comprehensive set of fundamental financial data.

### **Data Available**
- **Income Statement**:
  - Revenue, Gross Profit, Operating Income, Net Income
- **Balance Sheet**:
  - Assets, Liabilities, Equity
- **Cash Flow Statement**:
  - Operating Cash Flow, Investing Cash Flow, Financing Cash Flow
- **Key Ratios**:
  - PE Ratio, Price-to-Book, Dividend Yield, Earnings Per Share (EPS)
- **Valuation Metrics**:
  - Market Cap, Enterprise Value, PEG Ratio
- **Company Info**:
  - Sector, Industry, Full Description

### **Granularity**
- **Quarterly** or **Annual** data, depending on the report type.

### **Usage Example**
```python
# Fetch fundamental data for Microsoft (MSFT)
ticker = yf.Ticker("MSFT")
print(ticker.financials)  # Income statement
print(ticker.balance_sheet)  # Balance sheet
print(ticker.cashflow)  # Cash flow statement
```

---

## **4. Dividends & Stock Splits**
Yahoo Finance tracks corporate actions for securities.

### **Data Available**
- **Dividends**:
  - Dates (Declaration, Ex-Dividend, Payment)
  - Amounts (per share)
- **Stock Splits**:
  - Split ratio and dates.

### **Granularity**
- Historical records available (date-specific).

### **Usage Example**
```python
# Fetch dividends and stock splits for Google (GOOG)
ticker = yf.Ticker("GOOG")
print(ticker.dividends)  # Historical dividend data
print(ticker.splits)  # Historical stock splits
```

---

## **5. Options Data**
Yahoo Finance provides data on options chains for stocks.

### **Data Available**
- **Strike Prices**
- **Expiration Dates**
- **Call and Put Prices**:
  - Last price, bid, ask, open interest, implied volatility

### **Granularity**
- Current options chains for all available expiration dates.

### **Usage Example**
```python
# Fetch options chain for NVIDIA (NVDA)
ticker = yf.Ticker("NVDA")
options = ticker.option_chain('2024-01-19')  # Specify expiration date
print(options.calls)
print(options.puts)
```

---

## **6. Market Data**
Yahoo Finance offers aggregated market data for indices, commodities, and forex.

### **Data Available**
- **Major Indices**: Dow Jones, S&P 500, NASDAQ, etc.
- **Commodities**: Gold, Crude Oil, Natural Gas, etc.
- **Currencies/Forex**: USD/EUR, USD/JPY, etc.
- **Cryptocurrencies**: Bitcoin, Ethereum, etc.

### **Granularity**
- Daily, Weekly, Monthly (similar to stock price granularity).

### **Usage Example**
```python
# Fetch data for the S&P 500 Index (^GSPC)
data = yf.download("^GSPC", start="2022-01-01", end="2023-01-01", interval="1d")
print(data.head())
```

---

## **7. Analyst Recommendations**
Yahoo Finance tracks analyst ratings and price targets.

### **Data Available**
- **Consensus Rating** (e.g., Buy, Hold, Sell)
- **Target Prices** (High, Low, Median, Mean)

### **Granularity**
- Current recommendations (historical data not available for free).

### **Usage Example**
```python
# Fetch analyst recommendations for Netflix (NFLX)
ticker = yf.Ticker("NFLX")
print(ticker.recommendations)
```

---

## **8. News & Sentiment**
Yahoo Finance provides aggregated news articles related to securities.

### **Data Available**
- **Headlines**
- **Source**: Publisher of the article
- **Timestamp**
- **Link**: URL to the full article

### **Granularity**
- Current news articles (limited historical coverage).

### **Usage Example**
```python
# Fetch recent news for Amazon (AMZN)
ticker = yf.Ticker("AMZN")
print(ticker.news)
```

---

## **9. Calendar Data**
Yahoo Finance tracks key events such as earnings, dividends, and splits.

### **Data Available**
- **Earnings Dates** (upcoming and historical)
- **Dividends** (upcoming and historical)
- **Splits** (upcoming and historical)

### **Granularity**
- Upcoming and historical event data.

### **Usage Example**
```python
# Fetch earnings calendar for Apple (AAPL)
ticker = yf.Ticker("AAPL")
print(ticker.calendar)
```

---

## **10. Other Data**
- **Sustainability Scores**: ESG (Environmental, Social, and Governance) scores.
- **Institutional Ownership**: Percentage of shares owned by institutions.
- **Insider Transactions**: Insider buying/selling activity (if available).

---

## **Considerations When Using Yahoo Finance**
1. **Rate Limiting**: The free API has rate limits. Avoid excessive requests or use caching to minimize repeated data fetching.
2. **Data Completeness**: Intraday and fundamental data are limited in depth compared to paid services.
3. **Accuracy**: While reliable for general use, always cross-check critical data for financial decisions.

By combining multiple data types (e.g., historical prices, options, and fundamentals), **yfinance** can provide a solid foundation for building trading strategies, financial models, or forecasting systems.