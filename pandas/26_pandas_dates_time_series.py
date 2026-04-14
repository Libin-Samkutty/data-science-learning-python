"""
LESSON 26: WORKING WITH DATES AND TIME SERIES
================================================================================

What You Will Learn:
- Creating and parsing datetime columns
- Extracting date components (year, month, day, weekday, hour)
- Date arithmetic and time deltas
- Resampling time series data (daily to monthly, hourly to daily, etc.)
- Rolling windows and moving averages
- Handling time zones correctly
- Shifting and lagging time series data
- Generating date ranges
- Real world time series analysis examples

Real World Usage:
- Stock price analysis and trend calculation
- Sales forecasting by time period
- Sensor data aggregation over time windows
- Financial reporting periods (quarterly, yearly)
- Customer activity tracking over time
- IoT device monitoring dashboards

Dataset Used:
Stock prices or sales data generated synthetically with realistic patterns
(Fallback uses Titanic dataset if external data unavailable)

================================================================================
"""

import numpy as np
import pandas as pd
import re

print("=" * 70)
print("LESSON 26: WORKING WITH DATES AND TIME SERIES")
print("=" * 70)


# ==============================================================================
# SECTION 1: CREATE REALISTIC TIME SERIES DATA
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 1: CREATE REALISTIC TIME SERIES DATA")
print("=" * 70)

np.random.seed(42)

def standardize_columns(df):
    """Standardize column names to snake_case."""
    def to_snake(name):
        name = str(name).strip()
        name = re.sub(r"[\s\-\.]+", "_", name)
        name = re.sub(r"[^\w]", "", name)
        name = re.sub(r"_+", "_", name)
        return name.strip("_").lower()
    df.columns = [to_snake(col) for col in df.columns]
    return df


# Create a realistic stock price time series
print("Creating realistic daily stock price time series...")

# Generate 2 years of trading days (~252 trading days per year)
start_date = "2022-01-03"
end_date = "2023-12-31"

# Create business day frequency (monday-friday only)
dates = pd.date_range(start=start_date, end=end_date, freq="B")

n_days = len(dates)
print(f"Total trading days: {n_days}")

# Simulate stock price with realistic patterns
base_price = 100.0
price_changes = np.random.normal(0.001, 0.02, n_days)  # Slight upward drift
cumulative_returns = np.cumprod(1 + price_changes)
open_prices = base_price * cumulative_returns * (1 + np.random.uniform(-0.005, 0.005, n_days))
close_prices = open_prices * (1 + price_changes * np.random.uniform(0.9, 1.1, n_days))
high_prices = np.maximum(open_prices, close_prices) * (1 + np.abs(np.random.normal(0, 0.01, n_days)))
low_prices = np.minimum(open_prices, close_prices) * (1 - np.abs(np.random.normal(0, 0.01, n_days)))
volume = np.random.randint(100000, 1000000, n_days).astype(int)

# Add some missing data (simulates holidays or errors)
missing_indices = np.random.choice(n_days, size=10, replace=False)
close_prices_copy = close_prices.copy()
close_prices_copy[missing_indices] = np.nan
high_prices[missing_indices] = np.nan
low_prices[missing_indices] = np.nan

# Create DataFrame
stock_data = pd.DataFrame({
    "date": dates,
    "ticker": "XYZ",
    "open": np.round(open_prices, 2),
    "high": np.round(high_prices, 2),
    "low": np.round(low_prices, 2),
    "close": np.round(close_prices_copy, 2),
    "volume": volume
})

# Set date as index
stock_data = stock_data.set_index("date")

print("\nDaily Stock Price Data:")
print(f"Date range: {stock_data.index.min()} to {stock_data.index.max()}")
print(f"Shape: {stock_data.shape}")
print("\nFirst 5 rows:")
print(stock_data.head())
print("\nMissing values:")
print(stock_data.isnull().sum())


# Create sales transaction data
print("\n\nCreating hourly sales transaction data...")

sales_start = "2023-01-01 00:00:00"
sales_end = "2023-01-10 23:00:00"
sales_dates = pd.date_range(start=sales_start, end=sales_end, freq="h")

n_sales = len(sales_dates)

# Hourly sales with realistic patterns
# Higher during business hours (9am-6pm), lower at night
hour_of_day = sales_dates.hour
business_hours_factor = np.where((hour_of_day >= 9) & (hour_of_day <= 18), 2.0, 0.5)
day_of_week_factor = np.where(sales_dates.dayofweek < 5, 1.2, 0.8)  # Weekends lower

sales_amount = (
    100 + np.random.randn(n_sales) * 50  # Base sales
    + 50 * business_hours_factor          # Business hours boost
    + 30 * day_of_week_factor             # Weekend adjustment
    + np.random.rand(n_sales) * 100       # Random variation
)
sales_amount = np.maximum(sales_amount, 0)  # No negative sales

# Customer IDs
customer_ids = np.random.choice(range(1, 1001), n_sales)

# Product categories
categories = ["electronics", "clothing", "home", "sports"]
category_probs = [0.35, 0.30, 0.20, 0.15]
products = np.random.choice(categories, n_sales, p=category_probs)

sales_data = pd.DataFrame({
    "timestamp": sales_dates,
    "sale_id": range(1, n_sales + 1),
    "amount": np.round(sales_amount, 2),
    "customer_id": customer_ids,
    "category": products
})

sales_data = standardize_columns(sales_data)
sales_data = sales_data.rename(columns={"timestamp": "transaction_time"})

print(f"\nHourly Sales Transaction Data:")
print(f"Date range: {sales_data['transaction_time'].min()} to {sales_data['transaction_time'].max()}")
print(f"Total transactions: {len(sales_data)}")
print("\nFirst 5 rows:")
print(sales_data.head())
print("\nCategories distribution:")
print(sales_data["category"].value_counts())


# ==============================================================================
# SECTION 2: CREATING AND PARSING DATETIME COLUMNS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 2: CREATING AND PARSING DATETIME COLUMNS")
print("=" * 70)

explanation = """
DatetimeIndex is the key to working with time series in Pandas.
You can create it directly or convert string/date columns using pd.to_datetime().
This unlocks all time-based functionality (.dt accessor, resampling, rolling, etc.).
"""
print(explanation)

# ------------------------------------------------------------------------------
# 2.1 Converting Strings to Datetime
# ------------------------------------------------------------------------------
print("\n--- 2.1 Converting Strings to Datetime ---")

# Example: dates stored as strings in various formats
string_dates = pd.Series([
    "2023-01-15",           # ISO format
    "01/15/2023",           # US format (MM/DD/YYYY)
    "15-Jan-2023",          # European style
    "Jan 15, 2023",         # Another common format
    "20230115",             # Numeric compact format
    "15/01/2023 10:30:00",  # With time
])

print("Original string dates:")
print(string_dates.tolist())
print(f"Dtype before conversion: {string_dates.dtype}")

# Auto-parsing (Pandas infers format automatically)
parsed_auto = pd.to_datetime(string_dates, format="mixed")
print(f"\nAuto-parsed dates:")
print(parsed_auto.tolist())
print(f"Dtype after conversion: {parsed_auto.dtype}")

# Specific format for performance (faster on large datasets)
date_strings = pd.Series(["2023-01-15", "2023-02-20", "2023-03-10"])
parsed_format = pd.to_datetime(date_strings, format="%Y-%m-%d")
print(f"\nWith explicit format='%Y-%m-%d':")
print(parsed_format.tolist())

# Handle errors gracefully
mixed_dates = pd.Series(["2023-01-15", "not_a_date", "2023-03-10"])
parsed_coerce = pd.to_datetime(mixed_dates, errors="coerce")
print(f"\nWith errors='coerce' (bad dates become NaT):")
print(parsed_coerce.tolist())
print(f"NaT count: {parsed_coerce.isna().sum()}")

# In real data
if "date" not in sales_data.columns:
    sales_data["date_only"] = pd.to_datetime(
        sales_data["transaction_time"], 
        errors="coerce"
    )
else:
    print("\nConverting sales_data timestamp to datetime...")
    sales_data["transaction_time"] = pd.to_datetime(
        sales_data["transaction_time"],
        errors="coerce"
    )
    print(f"Dtype: {sales_data['transaction_time'].dtype}")


# ------------------------------------------------------------------------------
# 2.2 Creating Datetime Ranges
# ------------------------------------------------------------------------------
print("\n--- 2.2 Creating Datetime Ranges ---")

# Common frequencies for date ranges
frequencies = {
    "D":     "Daily",
    "B":     "Business days (Mon-Fri)",
    "W-MON": "Weekly, starting Monday",
    "ME":    "Month-end",
    "QE":    "Quarter-end",
    "YE":    "Year-end",
    "h":     "Hourly",
    "min":   "Minute-level",
    "s":     "Second-level"
}

for freq_code, description in frequencies.items():
    try:
        sample = pd.date_range(start="2023-01-01", periods=3, freq=freq_code)
        print(f"{freq_code:<8} ({description:>20}): {sample.tolist()}")
    except Exception as e:
        print(f"{freq_code:<8} Error: {e}")


# ------------------------------------------------------------------------------
# 2.3 Setting Datetime as Index
# ------------------------------------------------------------------------------
print("\n--- 2.3 Setting Datetime as Index ---")

print(f"Before: Index type = {type(stock_data.index)}")
print(f"After:  Index type = {type(stock_data.index)}")
print(f"Is DatetimeIndex: {isinstance(stock_data.index, pd.DatetimeIndex)}")

# If we had a DateTime column instead of index
if "date" in stock_data.columns:
    stock_data = stock_data.set_index("date")

print("\nAccess time slices easily with index:")
print(f"  First week of January: {len(stock_data.loc['2023-01-01':'2023-01-07'])} rows")
print(f"  All of March 2023:     {len(stock_data.loc['2023-03'])} rows")
print(f"  Last quarter 2023:     {len(stock_data.loc['2023-Q4'])} rows")


# ==============================================================================
# SECTION 3: EXTRACTING DATE COMPONENTS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 3: EXTRACTING DATE COMPONENTS")
print("=" * 70)

explanation = """
Use .dt accessor to extract components from DatetimeIndex or datetime columns.
This enables powerful feature engineering for time-based analysis and ML models.
"""
print(explanation)

# Work with stock data (datetime index)
df_demo = stock_data.reset_index().copy()
df_demo = df_demo.rename(columns={"index": "date"})
df_demo["date"] = pd.to_datetime(df_demo["date"])

# Extract all available components
print("Extractable components from datetime:")
components = {
    "year":              df_demo["date"].dt.year,
    "month":             df_demo["date"].dt.month,
    "day":               df_demo["date"].dt.day,
    "dayofweek":         df_demo["date"].dt.dayofweek,  # 0=Monday, 6=Sunday
    "weekday_name":      df_demo["date"].dt.day_name(),
    "month_name":        df_demo["date"].dt.month_name(),
    "quarter":           df_demo["date"].dt.quarter,
    "week":              df_demo["date"].dt.isocalendar().week,
    "is_month_start":    df_demo["date"].dt.is_month_start,
    "is_month_end":      df_demo["date"].dt.is_month_end,
    "is_quarter_start":  df_demo["date"].dt.is_quarter_start,
    "is_quarter_end":    df_demo["date"].dt.is_quarter_end,
    "is_year_end":       df_demo["date"].dt.is_year_end,
    "is_leap_year":      df_demo["date"].dt.is_leap_year,
}

print("Sample extracted features (first 3 rows):")
components_df = pd.DataFrame(components, index=df_demo.index)
print(df_demo[["date"]].join(components_df)[["date", "year", "month", "day",
                                              "dayofweek", "weekday_name", "quarter"]].head(3))

# Practical use cases
print("\n=== PRACTICAL USE CASES ===")

# Count transactions by day of week
df_demo["dayofweek_name"] = df_demo["date"].dt.day_name()
print("\nAverage close price by day of week:")
avg_by_dow = df_demo.groupby("dayofweek_name")["close"].mean()
# Reorder by actual day order
day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
avg_by_dow = avg_by_dow.reindex(day_order)
print(avg_by_dow.round(2))

# Identify end-of-month trading behavior
print("\nEnd-of-month vs other days average closing price:")
eom_mask = df_demo["date"].dt.is_month_end
eom_avg = df_demo[eom_mask]["close"].mean()
non_eom_avg = df_demo[~eom_mask]["close"].mean()
print(f"  End-of-month avg:   ${eom_avg:.2f}")
print(f"  Other days avg:     ${non_eom_avg:.2f}")
print(f"  Difference:         ${eom_avg - non_eom_avg:.2f}")


# ==============================================================================
# SECTION 4: DATE ARITHMETIC AND TIME DELTAS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 4: DATE ARITHMETIC AND TIME DELTAS")
print("=" * 70)

# ------------------------------------------------------------------------------
# 4.1 Basic Date Arithmetic
# ------------------------------------------------------------------------------
print("\n--- 4.1 Basic Date Arithmetic ---")

today = pd.Timestamp("2023-06-15")
tomorrow = today + pd.Timedelta(days=1)
yesterday = today - pd.Timedelta(days=1)
next_month = today + pd.offsets.MonthBegin(1)

print(f"Today:              {today}")
print(f"Tomorrow (+1 day):  {tomorrow}")
print(f"Yesterday (-1 day): {yesterday}")
print(f"Next month start:   {next_month}")

# Add/subtract time components
print(f"\nAdd 2 weeks:      {today + pd.Timedelta(weeks=2)}")
print(f"Subtract 6 hours:  {today - pd.Timedelta(hours=6)}")
print(f"Add 30 minutes:    {today + pd.Timedelta(minutes=30)}")

# Business day arithmetic (skips weekends)
print(f"\nBusiness day arithmetic:")
print(f"+5 calendar days:  {today + pd.Timedelta(days=5)}")
print(f"+5 business days:  {today + pd.tseries.offsets.BDay(5)}")
print(f"(using BDay offset)")
print(f"+5 BDay:           {today + pd.tseries.offsets.BDay(5)}")

# ------------------------------------------------------------------------------
# 4.2 Calculating Duration Between Dates
# ------------------------------------------------------------------------------
print("\n--- 4.2 Calculating Duration Between Dates ---")

start_date = pd.Timestamp("2023-01-15 09:00:00")
end_date = pd.Timestamp("2023-01-17 17:30:00")

duration = end_date - start_date
print(f"Start: {start_date}")
print(f"End:   {end_date}")
print(f"\nDuration object: {duration}")
print(f"Dtype:           {type(duration)}")

# Extract duration components
print(f"\nDuration breakdown:")
print(f"  Total days:            {duration.days}")
print(f"  Total seconds:         {duration.total_seconds()}")
print(f"  Days component:        {duration.days}")
print(f"  Seconds component:     {duration.seconds}")
print(f"  Microseconds component:{duration.microseconds}")

# Convert to convenient units
hours = duration.total_seconds() / 3600
minutes = duration.total_seconds() / 60
print(f"  Hours:                 {hours:.2f}")
print(f"  Minutes:               {minutes:.0f}")

# Using timedelta components directly
print(f"\nDirect timedelta access:")
print(f"  days attribute:        {duration.days}")
print(f"  seconds attribute:     {duration.seconds}")

# Calculate tenure/duration in dataset
if "signup_date" in df_demo.columns:
    last_date = df_demo["date"].max()
    df_demo["days_active"] = (last_date - df_demo["date"]).dt.days
    print(f"\nDays active calculated (sample):")
    print(df_demo[["date", "days_active"]].head(3))

# Age calculation example
print("\nAge calculation from birth date:")
birth_dates = pd.Series([
    "1990-05-15",
    "1985-12-03",
    "2000-08-22"
])
birth_df = pd.DataFrame({"name": ["Alice", "Bob", "Charlie"], 
                          "birthdate": pd.to_datetime(birth_dates)})
birth_df["age_years"] = (
    (pd.Timestamp("today") - birth_df["birthdate"]) / np.timedelta64(365, 'D')
).round(1)
print(birth_df)


# ==============================================================================
# SECTION 5: RESAMPLING TIME SERIES DATA
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 5: RESAMPLING TIME SERIES DATA")
print("=" * 70)

explanation = """
Resampling changes the frequency of time series data.
It aggregates multiple periods into one (downsampling) or splits periods 
(upscaling via forward/backward fill or interpolation).

Common operations:
  Daily to Weekly -> sum sales per week
  Hourly to Daily -> aggregate hourly metrics
  Monthly to Quarterly -> financial reporting

resample() returns a Resampler object that acts like groupby().
"""
print(explanation)

# Work with stock data
stock_resampled = stock_data.copy()

# ------------------------------------------------------------------------------
# 5.1 Downsampling: Daily to Weekly
# ------------------------------------------------------------------------------
print("\n--- 5.1 Downsampling: Daily to Weekly ---")

weekly_stock = stock_resampled.resample('W').agg({
    'open':   'first',   # First day's open
    'high':   'max',     # Highest high
    'low':    'min',     # Lowest low
    'close':  'last',    # Last day's close
    'volume': 'sum'      # Total weekly volume
}).dropna(how='all')

print("Weekly OHLCV from daily data:")
print(f"Original shape: {stock_resampled.shape}")
print(f"Weekly shape:   {weekly_stock.shape}")
print("\nFirst 5 weeks:")
print(weekly_stock.head())

# ------------------------------------------------------------------------------
# 5.2 Daily to Monthly
# ------------------------------------------------------------------------------
print("\n--- 5.2 Daily to Monthly ---")

monthly_stock = stock_resampled.resample('ME').agg({
    'open':   'first',
    'high':   'max',
    'low':    'min',
    'close':  ['first', 'last', 'mean', 'std'],
    'volume': 'sum'
})

# Flatten multi-level column names
monthly_stock.columns = [f"{col[0]}_{col[1]}" if isinstance(col, tuple) else col for col in monthly_stock.columns]
monthly_stock = monthly_stock.dropna(how='all')

print("Monthly aggregations:")
print(f"Monthly shape: {monthly_stock.shape}")
print("\nMonthly summary:")
display_cols = ['open', 'close_first', 'close_last', 'close_mean', 'close_std', 'volume']
available_cols = [c for c in display_cols if c in monthly_stock.columns]
print(monthly_stock[available_cols].round(2))

# ------------------------------------------------------------------------------
# 5.3 Upsampling with Forward Fill
# ------------------------------------------------------------------------------
print("\n--- 5.3 Upsampling (Weekly to Daily) ---")

# Create sparse weekly data
weekly_sparse = weekly_stock[['close']].copy()
print(f"Weekly sparse data points: {len(weekly_sparse)}")

# Upsample to daily and forward-fill (carry last known value)
daily_upsampled = weekly_sparse.resample('D').ffill()
print(f"Daily upsampled: {len(daily_upsampled)}")
print("\nUpsampled (shows daily with ffill):")
print(daily_upsampled.head(10))

# Alternative: interpolate linearly between known points
daily_interpolated = weekly_sparse.resample('D').interpolate(method='linear')
print("\nInterpolated (linear between points):")
print(daily_interpolated.head(10))

# ------------------------------------------------------------------------------
# 5.4 Custom Aggregation Functions
# ------------------------------------------------------------------------------
print("\n--- 5.4 Custom Aggregations ---")

# Calculate daily return and volatility
def calculate_daily_return(group):
    """Calculate daily returns from price series."""
    return group['close'].pct_change()

daily_returns = stock_resampled.resample('D')['close'].apply(lambda x: x.pct_change()).fillna(0)

print("Daily returns calculation:")
print(f"Mean daily return: {daily_returns.mean():.4f}")
print(f"Std daily return:  {daily_returns.std():.4f}")
print(f"Max daily return:  {daily_returns.max():.4f}")
print(f"Min daily return:  {daily_returns.min():.4f}")

# Weekly aggregated returns (compound)
weekly_compound = stock_resampled.resample('W')['close'].apply(
    lambda x: (1 + x.pct_change().dropna()).prod() - 1
)
print(f"\nWeekly compound returns (first 3 weeks):")
print(weekly_compound[:3])

# Rolling statistics within resample
print("\nRolling statistics within each period:")
rolling_weekly = stock_resampled.resample('W').agg({
    'close': lambda x: pd.Series(x.values).rolling(window=5, min_periods=1).mean().iloc[-1]
}).dropna()
print(rolling_weekly.head(5))


# ==============================================================================
# SECTION 6: ROLLING WINDOWS AND MOVING AVERAGES
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 6: ROLLING WINDOWS AND MOVING AVERAGES")
print("=" * 70)

explanation = """
Rolling windows compute statistics over a sliding window of observations.
Essential for smoothing noisy data, detecting trends, and identifying turning points.

Common applications:
  - Moving averages for trend identification
  - Bollinger bands for volatility
  - Detecting regime changes
  - Technical indicators in finance
"""
print(explanation)

# ------------------------------------------------------------------------------
# 6.1 Simple Moving Average (SMA)
# ------------------------------------------------------------------------------
print("\n--- 6.1 Simple Moving Average (SMA) ---")

stock_with_ma = stock_data.reset_index().copy()
stock_with_ma["close"] = stock_with_ma["close"].ffill()

# Different window sizes
windows = [5, 10, 20, 50]
print("Simple Moving Averages of closing price:")
for window in windows:
    sma_col = f"sma_{window}"
    stock_with_ma[sma_col] = stock_with_ma["close"].rolling(window=window).mean()

print(f"\nLast 10 rows with SMAs:")
print(stock_with_ma[["date", "close"] + [f"sma_{w}" for w in windows]].tail(10).round(2))

# Visualization hint (text-based comparison)
print("\nSMA crossover detection (5-day crossing above 20-day):")
crossover = (
    (stock_with_ma["sma_5"] > stock_with_ma["sma_20"]) &
    (stock_with_ma["sma_5"].shift(1) <= stock_with_ma["sma_20"].shift(1))
)
print(f"Bullish crossovers detected: {crossover.sum()}")

# ------------------------------------------------------------------------------
# 6.2 Exponential Moving Average (EMA)
# ------------------------------------------------------------------------------
print("\n--- 6.2 Exponential Moving Average (EMA) ---")

ema_windows = [10, 20, 50]
print("Exponential Moving Averages (more weight to recent data):")
for span in ema_windows:
    ema_col = f"ema_{span}"
    stock_with_ma[ema_col] = stock_with_ma["close"].ewm(span=span, adjust=False).mean()

print(f"\nLast 10 rows with EMAs:")
print(stock_with_ma[["date", "close"] + [f"ema_{w}" for w in ema_windows]].tail(10).round(2))

# Compare SMA vs EMA responsiveness
recent_close = stock_with_ma["close"].tail(20).values
recent_sma = stock_with_ma["sma_10"].tail(20).values
recent_ema = stock_with_ma["ema_10"].tail(20).values

print("\nResponsiveness comparison (last 20 days):")
print(f"Close range:      {recent_close.min():.2f} - {recent_close.max():.2f}")
print(f"SMA10 range:      {recent_sma.min():.2f} - {recent_sma.max():.2f}")
print(f"EMA10 range:      {recent_ema.min():.2f} - {recent_ema.max():.2f}")
print("EMA follows price more closely than SMA")

# ------------------------------------------------------------------------------
# 6.3 Rolling Statistics Beyond Mean
# ------------------------------------------------------------------------------
print("\n--- 6.3 Rolling Variance, Std, Min, Max ---")

stock_with_stats = stock_data.reset_index().copy()
stock_with_stats["close"] = stock_with_stats["close"].ffill()

window = 20
stock_with_stats[f"roll_var_{window}"] = stock_with_stats["close"].rolling(window=window).var()
stock_with_stats[f"roll_std_{window}"] = stock_with_stats["close"].rolling(window=window).std()
stock_with_stats[f"roll_min_{window}"] = stock_with_stats["close"].rolling(window=window).min()
stock_with_stats[f"roll_max_{window}"] = stock_with_stats["close"].rolling(window=window).max()

print(f"Rolling statistics (window={window}) last 5 days:")
cols_to_show = ["date", "close", f"roll_std_{window}", f"roll_min_{window}", f"roll_max_{window}"]
print(stock_with_stats[cols_to_show].tail(5).round(4))

# Bollinger Bands calculation
middle_band = stock_with_stats["sma_20"] if "sma_20" in stock_with_stats.columns else stock_with_stats["close"].rolling(window=20).mean()
std_dev = stock_with_stats.get(f"roll_std_{window}", stock_with_stats["close"].rolling(window=20).std())
upper_band = middle_band + 2 * std_dev
lower_band = middle_band - 2 * std_dev

print("\nBollinger Bands (±2 standard deviations):")
bollinger = pd.DataFrame({
    "date": stock_with_stats["date"].tail(5),
    "upper": upper_band.tail(5),
    "middle": middle_band.tail(5),
    "lower": lower_band.tail(5),
    "width": (upper_band - lower_band).tail(5)
})
print(bollinger.round(4))

# Price touching bands (potential reversal signals)
touches_upper = stock_with_stats["close"] >= upper_band
touches_lower = stock_with_stats["close"] <= lower_band
print(f"\nTouches upper band: {touches_upper.sum()} times")
print(f"Touches lower band: {touches_lower.sum()} times")

# ------------------------------------------------------------------------------
# 6.4 Expanding Window (Cumulative)
# ------------------------------------------------------------------------------
print("\n--- 6.4 Expanding Window (Cumulative Statistics) ---")

stock_with_expanding = stock_data.reset_index().copy()
stock_with_expanding["expanding_mean"] = stock_with_expanding["close"].expanding().mean()
stock_with_expanding["expanding_min"] = stock_with_expanding["close"].expanding().min()
stock_with_expanding["expanding_max"] = stock_with_expanding["close"].expanding().max()
stock_with_expanding["expanding_std"] = stock_with_expanding["close"].expanding().std()

print("Expanding statistics (cumulative from start):")
print(stock_with_expanding[["date", "close", "expanding_mean", "expanding_min", "expanding_max"]].head(10).round(2))

# Drawdown calculation (from peak)
peak = stock_with_expanding["expanding_max"]
drawdown = (stock_with_expanding["close"] - peak) / peak * 100
print(f"\nMaximum drawdown: {drawdown.min():.2f}%")
print(f"Current drawdown:  {drawdown.iloc[-1]:.2f}%")


# ==============================================================================
# SECTION 7: SHIFTING AND LAGGING DATA
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 7: SHIFTING AND LAGGING DATA")
print("=" * 70)

explanation = """
shift() moves data forward or backward in time.
Essential for creating lagged features, calculating changes, and detecting patterns.

Positive shift = lag (data moves down, NaN appears at top)
Negative shift = lead (data moves up, NaN appears at bottom)

Common use cases:
  - Lagged prices for predictive modeling
  - Calculate daily returns
  - Detect consecutive events
  - Feature engineering for ML
"""
print(explanation)

# ------------------------------------------------------------------------------
# 7.1 Shift for Lag Features
# ------------------------------------------------------------------------------
print("\n--- 7.1 Shift for Lag Features ---")

lag_data = stock_data.reset_index().copy()
lag_data["close"] = lag_data["close"].ffill()

# Create lagged features
lag_data["close_lag1"] = lag_data["close"].shift(1)
lag_data["close_lag2"] = lag_data["close"].shift(2)
lag_data["close_lag5"] = lag_data["close"].shift(5)

print("Lagged closing prices:")
print(lag_data[["date", "close", "close_lag1", "close_lag2", "close_lag5"]].head(7).round(2))

# Verify lag relationship
print(f"\nVerification: lag1 row 5 should equal close row 4")
print(f"  lag1[5]:  {lag_data['close_lag1'].iloc[5]:.2f}")
print(f"  close[4]: {lag_data['close'].iloc[4]:.2f}")
print(f"  Match:    {abs(lag_data['close_lag1'].iloc[5] - lag_data['close'].iloc[4]) < 0.01}")

# ------------------------------------------------------------------------------
# 7.2 Calculating Changes (Returns, Differences)
# ------------------------------------------------------------------------------
print("\n--- 7.2 Calculating Changes ---")

change_data = stock_data.reset_index().copy()
change_data["close"] = change_data["close"].ffill()

# Absolute change
change_data["absolute_change"] = change_data["close"] - change_data["close"].shift(1)

# Percentage change
change_data["percent_change"] = change_data["close"].pct_change() * 100

# pct_change() is equivalent to: (x - x.shift(1)) / x.shift(1)
print("Daily price changes:")
print(change_data[["date", "close", "absolute_change", "percent_change"]].head(10).round(4))

# N-period change
change_data["change_5d"] = change_data["close"].pct_change(periods=5) * 100
print(f"\n5-day percentage change (last 5 days):")
print(change_data[["date", "change_5d"]].tail(5).round(2))

# Cumulative return
cumulative_return = (1 + change_data["percent_change"].fillna(0)/100).cumprod() - 1
print(f"\nCumulative return from start: {(cumulative_return.iloc[-1])*100:.2f}%")

# ------------------------------------------------------------------------------
# 7.3 Multiple Period Shifts
# ------------------------------------------------------------------------------
print("\n--- 7.3 Multiple Period Comparison ---")

multi_period = stock_data.reset_index().copy()
multi_period["close"] = multi_period["close"].ffill()

periods_to_compare = [1, 7, 30]  # 1 day, 1 week, 1 month
for period in periods_to_compare:
    col_name = f"return_{period}d"
    multi_period[col_name] = (
        multi_period["close"] / multi_period["close"].shift(period) - 1
    ) * 100

print(f"Return comparisons (last 5 days):")
display_cols = ["date"] + [f"return_{p}d" for p in periods_to_compare]
print(multi_period[display_cols].tail(5).round(2))

# ------------------------------------------------------------------------------
# 7.4 Detecting Consecutive Patterns
# ------------------------------------------------------------------------------
print("\n--- 7.4 Detecting Consecutive Events ---")

consecutive_data = stock_data.reset_index().copy()
consecutive_data["is_positive"] = (consecutive_data["close"] > consecutive_data["close"].shift(1)).astype(int)

# Consecutive positive days
consecutive_data["consecutive_positive"] = (
    consecutive_data["is_positive"]
    .groupby((consecutive_data["is_positive"] != consecutive_data["is_positive"].shift()).cumsum())
    .cumcount()
    + 1
).where(consecutive_data["is_positive"].astype(bool), 0)

print("Consecutive positive day streaks:")
streak_rows = consecutive_data[consecutive_data["consecutive_positive"] >= 3].head(10)
print(streak_rows[["date", "is_positive", "consecutive_positive"]].to_string(index=False))

# Maximum streak length
max_streak = consecutive_data["consecutive_positive"].max()
print(f"\nLongest consecutive positive day streak: {int(max_streak)} days")

# Also find losing streaks
consecutive_data["is_negative"] = (consecutive_data["close"] < consecutive_data["close"].shift(1)).astype(int)
consecutive_data["consecutive_negative"] = (
    consecutive_data["is_negative"]
    .groupby((consecutive_data["is_negative"] != consecutive_data["is_negative"].shift()).cumsum())
    .cumcount()
    + 1
).where(consecutive_data["is_negative"].astype(bool), 0)

losing_streak = consecutive_data["consecutive_negative"].max()
print(f"Longest consecutive negative day streak: {int(losing_streak)} days")


# ==============================================================================
# SECTION 8: HANDLING TIME ZONES
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 8: HANDLING TIME ZONES")
print("=" * 70)

explanation = """
Time zones are critical when working with global data or timestamps from different sources.
Best practice: store ALL data in UTC, convert to local timezone only for display.

Key operations:
  tz_localize()   : Assign timezone to naive datetime
  tz_convert()    : Convert between timezones
  utc.normalize() : Get UTC equivalent
"""
print(explanation)

# ------------------------------------------------------------------------------
# 8.1 Creating Timezone-Aware Datetimes
# ------------------------------------------------------------------------------
print("\n--- 8.1 Creating Timezone-Aware Datetimes ---")

# Start with timezone-naive datetime
naive_dt = pd.Series(pd.date_range("2023-06-15 10:00:00", periods=3, freq="h"))
print(f"Timezone-naive: {naive_dt.tolist()}")
print(f"Has timezone:   {naive_dt.dt.tz is not None}")

# Localize to UTC (assign timezone, don't shift time)
utc_aware = naive_dt.dt.tz_localize("UTC")
print(f"\nAfter tz_localize('UTC'): {utc_aware.tolist()}")
print(f"Has timezone: {utc_aware.dt.tz is not None}")

# Convert to another timezone
ny_aware = utc_aware.dt.tz_convert("America/New_York")
print(f"\nAfter tz_convert('America/New_York'): {ny_aware.tolist()}")

# The clock time changed but it represents the same moment
print(f"\nVerification: Both represent same instant")
print(f"  UTC: {utc_aware[0]}")
print(f"  NY:  {ny_aware[0]}")

# ------------------------------------------------------------------------------
# 8.2 Best Practice: Store in UTC, Display in Local
# ------------------------------------------------------------------------------
print("\n--- 8.2 Best Practice Pattern ---")

# Scenario: Global transaction log
transactions_utc = pd.DataFrame({
    "transaction_id": range(1, 6),
    "timestamp_utc": pd.to_datetime([
        "2023-06-15 14:00:00",
        "2023-06-15 15:30:00",
        "2023-06-15 16:45:00",
        "2023-06-15 17:15:00",
        "2023-06-15 18:00:00"
    ]).tz_localize("UTC"),
    "amount": [100, 250, 175, 320, 95],
    "region": ["NA", "EU", "APAC", "NA", "EU"]
})

print("Transactions stored in UTC:")
print(transactions_utc[["transaction_id", "timestamp_utc", "region", "amount"]])

# For display to regional managers
timezone_mapping = {
    "NA":   "America/New_York",
    "EU":   "Europe/London",
    "APAC": "Asia/Tokyo"
}

print("\nConvert to local time for regional reports:")
for region, tz_name in timezone_mapping.items():
    region_data = transactions_utc[transactions_utc["region"] == region].copy()
    if len(region_data) > 0:
        region_data["local_time"] = region_data["timestamp_utc"].dt.tz_convert(tz_name)
        print(f"\n{region} Region ({tz_name}):")
        print(region_data[["transaction_id", "local_time", "amount"]])

# ------------------------------------------------------------------------------
# 8.3 Mixing Timezones Correctly
# ------------------------------------------------------------------------------
print("\n--- 8.3 Handling Mixed Timezones ---")

# Simulate receiving data from multiple sources with different timezones
source1 = pd.Series(pd.to_datetime(["2023-06-15 10:00:00", "2023-06-15 11:00:00"]).tz_localize("UTC"))
source2 = pd.Series(pd.to_datetime(["2023-06-15 09:00:00", "2023-06-15 10:00:00"]).tz_localize("US/Pacific").tz_convert("UTC"))

# Combine them safely
combined = pd.concat([source1, source2], ignore_index=True)

# Normalize all to UTC first
combined_utc = combined.dt.tz_convert("UTC")

print("Data from multiple timezone sources:")
print("After normalization to UTC:")
print(combined_utc.tolist())

# Sort by actual time
sorted_events = combined_utc.sort_values()
print("\nSorted by actual chronological time:")
print(sorted_events.tolist())


# ==============================================================================
# SECTION 9: REAL WORLD TIME SERIES ANALYSIS PIPELINE
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 9: REAL WORLD TIME SERIES ANALYSIS PIPELINE")
print("=" * 70)

def analyze_stock_performance(data, ticker="XYZ"):
    """
    Complete time series analysis pipeline for stock data.

    Parameters:
        data : pd.DataFrame with datetime index and OHLCV columns
        ticker : Stock symbol identifier

    Returns:
        dict with analysis results
    """
    print(f"\n{'='*60}")
    print(f"STOCK PERFORMANCE ANALYSIS: {ticker}")
    print('='*60)

    result = {}

    # Step 1: Validate data
    print("\n[Step 1] Validating data quality...")
    assert isinstance(data.index, pd.DatetimeIndex), "Index must be DatetimeIndex"
    missing_close = data["close"].isna().sum() if "close" in data.columns else 0
    missing_volume = data["volume"].isna().sum() if "volume" in data.columns else 0
    print(f"  Missing close prices: {missing_close}")
    print(f"  Missing volume data:  {missing_volume}")
    result["missing_data"] = {"close": missing_close, "volume": missing_volume}

    # Step 2: Calculate returns
    print("\n[Step 2] Calculating returns...")
    if "close" in data.columns:
        clean_close = data["close"].ffill().bfill()
        daily_returns = clean_close.pct_change()
        
        result["returns"] = {
            "mean_daily": daily_returns.mean(),
            "std_daily": daily_returns.std(),
            "max_daily": daily_returns.max(),
            "min_daily": daily_returns.min(),
            "annualized_return": (1 + daily_returns.mean())**252 - 1,
            "annualized_volatility": daily_returns.std() * (252 ** 0.5)
        }
        print(f"  Mean daily return:   {result['returns']['mean_daily']*100:.3f}%")
        print(f"  Annualized return:   {result['returns']['annualized_return']*100:.2f}%")
        print(f"  Annualized volatilty: {result['returns']['annualized_volatility']*100:.2f}%")

    # Step 3: Trend analysis with moving averages
    print("\n[Step 3] Trend analysis (moving averages)...")
    if "close" in data.columns:
        data_copy = data.copy()
        data_copy["SMA_20"] = clean_close.rolling(window=20).mean()
        data_copy["SMA_50"] = clean_close.rolling(window=50).mean()
        data_copy["EMA_12"] = clean_close.ewm(span=12, adjust=False).mean()
        data_copy["EMA_26"] = clean_close.ewm(span=26, adjust=False).mean()
        
        # MACD signal
        macd_line = data_copy["EMA_12"] - data_copy["EMA_26"]
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        macd_histogram = macd_line - signal_line
        
        current_macd = macd_line.iloc[-1]
        current_signal = signal_line.iloc[-1]
        
        result["trend"] = {
            "current_price": float(clean_close.iloc[-1]),
            "above_sma20": bool(clean_close.iloc[-1] > data_copy["SMA_20"].iloc[-1]),
            "above_sma50": bool(clean_close.iloc[-1] > data_copy["SMA_50"].iloc[-1]),
            "macd_value": float(current_macd),
            "signal_value": float(current_signal),
            "bullish_signal": bool(current_macd > current_signal)
        }
        print(f"  Current price:        ${clean_close.iloc[-1]:.2f}")
        print(f"  Above 20-day SMA:     {'Yes' if result['trend']['above_sma20'] else 'No'}")
        print(f"  MACD Signal:          {'Bullish' if result['trend']['bullish_signal'] else 'Bearish'}")

    # Step 4: Volatility analysis
    print("\n[Step 4] Volatility analysis...")
    if "close" in data.columns:
        rolling_std = clean_close.rolling(window=20).std()
        
        result["volatility"] = {
            "avg_20d_volatility": float(rolling_std.mean()),
            "current_20d_volatility": float(rolling_std.iloc[-1]),
            "max_20d_volatility": float(rolling_std.max()),
            "volatility_trend": "increasing" if rolling_std.iloc[-1] > rolling_std.mean() else "decreasing"
        }
        print(f"  Avg 20-day vol:    {result['volatility']['avg_20d_volatility']*100:.2f}%")
        print(f"  Current 20-day vol:{result['volatility']['current_20d_volatility']*100:.2f}%")
        print(f"  Trend:             {result['volatility']['volatility_trend']}")

    # Step 5: Price extremes
    print("\n[Step 5] Price range analysis...")
    if "close" in data.columns:
        result["price_range"] = {
            "highest": float(clean_close.max()),
            "lowest": float(clean_close.min()),
            "current": float(clean_close.iloc[-1]),
            "distance_from_high": ((clean_close.iloc[-1] / clean_close.max() - 1) * 100),
            "distance_from_low": ((clean_close.iloc[-1] / clean_close.min() - 1) * 100)
        }
        print(f"  Year-to-date high: ${result['price_range']['highest']:.2f}")
        print(f"  Year-to-date low:  ${result['price_range']['lowest']:.2f}")
        print(f"  Distance from high: {result['price_range']['distance_from_high']:.1f}%")
        print(f"  Distance from low:  {result['price_range']['distance_from_low']:.1f}%")

    # Step 6: Volume analysis
    print("\n[Step 6] Volume analysis...")
    if "volume" in data.columns:
        clean_volume = data["volume"].fillna(0)
        result["volume"] = {
            "avg_daily_volume": int(clean_volume.mean()),
            "total_volume": int(clean_volume.sum()),
            "current_vs_avg": float(clean_volume.iloc[-1] / clean_volume.mean() if clean_volume.mean() > 0 else 0)
        }
        print(f"  Avg daily volume: {result['volume']['avg_daily_volume']:,}")
        print(f"  Today vs avg:     {result['volume']['current_vs_avg']:.1f}x normal")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    
    return result


# Run the analysis pipeline
analysis_results = analyze_stock_performance(stock_data.reset_index().set_index("date"))


# ==============================================================================
# SECTION 10: SALES AGGREGATION PIPELINE
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 10: SALES AGGREGATION PIPELINE")
print("=" * 70)

print("Aggregating hourly sales data into useful time periods...")

# Set transaction_time as index
sales_indexed = sales_data.set_index("transaction_time")

# Ensure index is properly datetime (should already be)
if not isinstance(sales_indexed.index, pd.DatetimeIndex):
    sales_indexed.index = pd.to_datetime(sales_indexed.index)

# ------------------------------------------------------------------------------
# 10.1 Hourly to Daily Aggregation
# ------------------------------------------------------------------------------
print("\n--- 10.1 Hourly to Daily Aggregation ---")

daily_sales = sales_indexed.resample('D').agg({
    'amount': ['sum', 'mean', 'count', 'max', 'min'],
    'customer_id': 'nunique'
})

# Flatten column names
daily_sales.columns = ['total_sales', 'avg_sale', 'transaction_count', 
                       'max_transaction', 'min_transaction', 'unique_customers']
daily_sales = daily_sales.dropna(how='all')

print("Daily sales summary:")
print(f"Analysis period: {daily_sales.index.min().strftime('%Y-%m-%d')} to {daily_sales.index.max().strftime('%Y-%m-%d')}")
print(f"Total days: {len(daily_sales)}")
print("\nFirst 7 days:")
print(daily_sales.head(7).round(2))

# Day of week analysis
daily_sales_temp = sales_data.copy()
daily_sales_temp["day_of_week"] = daily_sales_temp["transaction_time"].dt.day_name()
daily_sales_temp["day_num"] = daily_sales_temp["transaction_time"].dt.dayofweek

daily_summary = daily_sales_temp.groupby("day_num").agg({
    "amount": ["sum", "mean", "count"],
    "customer_id": "nunique"
}).reset_index()
daily_summary.columns = ["day_num", "total_sales", "avg_sale", "tx_count", "customers"]

# Reorder to Mon-Sun
day_order = list(range(7))
day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
daily_summary = daily_summary.set_index("day_num").reindex(day_order)
daily_summary["day_name"] = daily_summary.index.map(dict(zip(day_order, day_names)))

print("\nSales by day of week:")
print(daily_summary.round(2))

# ------------------------------------------------------------------------------
# 10.2 Rolling Daily Trends
# ------------------------------------------------------------------------------
print("\n--- 10.2 Rolling Trends ---")

daily_sales["rolling_7d_sales"] = daily_sales["total_sales"].rolling(window=7).sum()
daily_sales["rolling_7d_avg"] = daily_sales["total_sales"].rolling(window=7).mean()
daily_sales["growth_rate"] = daily_sales["total_sales"].pct_change(periods=7) * 100

print("7-day rolling sales (last 10 days):")
print(daily_sales[["total_sales", "rolling_7d_sales", "rolling_7d_avg"]].tail(10).round(2))

# ------------------------------------------------------------------------------
# 10.3 Category Performance Over Time
# ------------------------------------------------------------------------------
print("\n--- 10.3 Category Performance ---")

category_daily = sales_data.groupby(
    [sales_data["transaction_time"].dt.date, "category"]
)["amount"].sum().reset_index()

pivot_table = category_daily.pivot_table(
    index="transaction_time",
    columns="category",
    values="amount",
    aggfunc="sum"
)

print("Category sales evolution (first 5 days):")
print(pivot_table.head(5).round(2))

# Share of total by category
category_totals = sales_data.groupby("category")["amount"].sum()
category_share = category_totals / category_totals.sum() * 100

print("\nCategory revenue share:")
print(category_share.round(2))

# Category growth trend
category_daily_for_trend = sales_data.groupby(
    [sales_data["transaction_time"].dt.to_period("D"), "category"]
)["amount"].sum().reset_index()

category_growth = category_daily_for_trend.groupby("category").agg({
    "amount": ["first", "last", "mean"]
}).reset_index()

print("\nCategory performance (first day vs last day):")
for idx, row in category_growth.iterrows():
    category = row["category"]
    first_day = row[("amount", "first")]
    last_day = row[("amount", "last")]
    growth_pct = ((last_day - first_day) / first_day * 100) if first_day > 0 else 0
    print(f"  {category}: ${first_day:.2f} -> ${last_day:.2f} ({growth_pct:+.1f}%)")


# ==============================================================================
# SECTION 11: COMMON PITFALLS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 11: COMMON PITFALLS")
print("=" * 70)

pitfalls = """
Pitfall 1: Not converting strings to datetime before time operations
   - pd.to_datetime() MUST be called before using .dt accessor
   - Otherwise you cannot resample, slice by time, or extract components

Pitfall 2: Mixing timezone-aware and naive datetimes
   - Raises TypeError when combining
   - Always normalize to single timezone (prefer UTC) before operations

Pitfall 3: Assuming resample fills gaps automatically
   - resample('D') does NOT fill missing dates
   - Use .asfreq() or .ffill() after resample to fill gaps

Pitfall 4: Confusing shift direction
   - Positive shift (shift(1)) = lag (moves data DOWN, old becomes current)
   - Negative shift (shift(-1)) = lead (moves data UP)
   - Check with .head() after shifting!

Pitfall 5: Rolling window with insufficient data
   - Window size > data length gives all NaN
   - Use min_periods parameter to control this

Pitfall 6: Datetime slicing syntax
   - WRONG: df.loc["2023-01":"2023-03"] works ONLY with DatetimeIndex
   - RIGHT: df[df["date"] >= "2023-01"] if using column

Pitfall 7: Timezone confusion between localize and convert
   - tz_localize(): ASSIGN timezone (doesn't shift time)
   - tz_convert(): CHANGE timezone (preserves absolute moment)
   - Only call localize ONCE on naive datetime

Pitfall 8: Resampling doesn't work with MultiIndex
   - Must reset index to get datetime column first
   - Then set as index for resample to work

Pitfall 9: Forget NaN handling in calculations
   - Many operations propagate NaN silently
   - Use .fillna() or dropna() explicitly where needed

Pitfall 10: Performance on large datasets
   - Avoid looping through timestamps
   - Use vectorized operations (.dt accessor, resample, rolling)
"""
print(pitfalls)


# ==============================================================================
# SECTION 12: SUMMARY TABLE
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 12: SUMMARY TABLE")
print("=" * 70)

summary = """
Operation                              | Syntax
---------------------------------------|-------------------------------------------------
Convert string to datetime             | pd.to_datetime(series, format=..., errors=...)
Create datetime range                  | pd.date_range(start, end, periods=n, freq='D')
Set datetime index                     | df.set_index('date_column')
Extract year                           | df['date'].dt.year
Extract month                          | df['date'].dt.month
Extract day of week                    | df['date'].dt.dayofweek (0=Mon, 6=Sun)
Get day name                           | df['date'].dt.day_name()
Check if month start/end               | df['date'].dt.is_month_start
Add time delta                         | timestamp + pd.Timedelta(days=7)
Duration between two dates             | end_timestamp - start_timestamp
Resample daily to weekly               | df.resample('W').agg({'col': 'sum'})
Resample with multiple functions       | df.resample('M').agg(['mean', 'std'])
Simple moving average                  | df['col'].rolling(window=20).mean()
Exponential moving average             | df['col'].ewm(span=10, adjust=False).mean()
Shift data (create lag)                | df['col'].shift(1)
Calculate percentage change            | df['col'].pct_change()
Localize to timezone                   | df['date'].dt.tz_localize('UTC')
Convert between timezones              | df['date'].dt.tz_convert('America/New_York')
Fill missing dates in series           | series.resample('D').asfreq().ffill()
Detect consecutive conditions          | group cumcount trick shown in lesson
Filter by time period                  | df.loc['2023-01':'2023-03'] (with DatetimeIndex)
"""
print(summary)


# ==============================================================================
# SECTION 13: PRACTICE EXERCISES
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 13: PRACTICE EXERCISES")
print("=" * 70)

print("Exercise 1: Create quarterly report from daily data")
exercise1 = stock_data.reset_index().set_index("date").resample('QE').agg({
    'close': ['first', 'last', 'mean'],
    'volume': 'sum'
})
print("Quarterly summary:")
print(exercise1.round(2))

print("\nExercise 2: Find all Fridays in the dataset")
friday_data = stock_data[stock_data.index.weekday == 4]
print(f"Fridays count: {len(friday_data)} out of {len(stock_data)} total days")
print(f"Friday average close: ${friday_data['close'].mean():.2f}")

print("\nExercise 3: Calculate 30-day momentum indicator")
momentum_data = stock_data.reset_index().copy()
momentum_data["close"] = momentum_data["close"].ffill()
momentum_data["momentum_30d"] = momentum_data["close"] / momentum_data["close"].shift(30) - 1
print("30-day momentum (positive = upward momentum):")
print(momentum_data[["date", "momentum_30d"]].tail(10).round(4))

print("\nExercise 4: Identify highest and lowest volume days")
if "volume" in stock_data.columns:
    volumes = stock_data["volume"].fillna(0)
    max_vol_row = stock_data.loc[volumes.idxmax()]
    min_vol_row = stock_data.loc[volumes.idxmin()]
    print(f"Highest volume: {max_vol_row.name.strftime('%Y-%m-%d')} - {int(max_vol_row['volume']):,}")
    print(f"Lowest volume:  {min_vol_row.name.strftime('%Y-%m-%d')} - {int(min_vol_row['volume']):,}")

print("\nExercise 5: Calculate weekend vs weekday sales difference")
if len(sales_data) > 0:
    sales_data["is_weekend"] = sales_data["transaction_time"].dt.weekday >= 5
    weekend_avg = sales_data[sales_data["is_weekend"]]["amount"].mean()
    weekday_avg = sales_data[~sales_data["is_weekend"]]["amount"].mean()
    diff_pct = ((weekend_avg - weekday_avg) / weekday_avg) * 100
    print(f"Weekend avg sale: ${weekend_avg:.2f}")
    print(f"Weekday avg sale: ${weekday_avg:.2f}")
    print(f"Relative difference: {diff_pct:+.1f}%")


# ==============================================================================
# SECTION 14: KEY TAKEAWAYS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 14: KEY TAKEAWAYS")
print("=" * 70)

takeaways = """
1.  ALWAYS convert string dates to datetime using pd.to_datetime() first
2.  Use DatetimeIndex for efficient time-based operations and slicing
3.  .dt accessor provides all date extraction methods (year, month, etc.)
4.  Resampling requires DatetimeIndex and allows flexible aggregations
5.  Rolling windows smooth noisy data; choose appropriate window size
6.  shift() creates lag features for predictive modeling
7.  pct_change() is cleaner than manual return calculations
8.  Store all timestamps in UTC internally, convert only for display
9.  tz_localize assigns timezone to naive datetime (once only!)
10. tz_convert shifts between timezones preserving absolute moment
11. Validate missing data before time series calculations
12. Use expand() for cumulative statistics from start of series
13. Resample with agg dictionary for multiple aggregations at once
14. Always check that datetime parsing succeeded (count NaT values)
15. Timezone mismatches cause silent failures - validate early
"""
print(takeaways)


# ==============================================================================
# NEXT LESSON PREVIEW
# ==============================================================================
print("\n" + "=" * 70)
print("NEXT LESSON PREVIEW")
print("=" * 70)

preview = """
Lesson 27: Applying Functions (apply, map, transform, vectorize)

You will learn:
- When to use apply vs map vs vectorized operations
- Row-wise vs column-wise function application
- Custom functions with multiple parameters
- Lambda functions for quick transformations
- Performance considerations and optimization
- Building reusable transformation pipelines
- Common patterns in production code
"""
print(preview)

print("\n" + "=" * 70)
print("END OF LESSON 26")
print("=" * 70)
print("\nYou now have complete mastery over datetime operations in Pandas.")
print("You can parse dates, extract components, resample time series,")
print("compute rolling statistics, handle timezones correctly, and build")
print("production-ready time series analysis pipelines.")
