import numpy as np
import warnings

# ============================================================
# PART 1: Understanding NaN and inf
# ============================================================

print("="*60)
print("PART 1: UNDERSTANDING NaN AND inf")
print("="*60)

print("""
NaN (Not a Number):
- Represents missing or undefined values
- Result of invalid operations (0/0, inf-inf, sqrt(-1))
- Propagates through calculations (any operation with NaN → NaN)
- Special comparison behavior (NaN != NaN)

inf (Infinity):
- Represents infinite values
- Result of overflow or division by zero (1/0 → inf)
- Can be positive or negative
- Participates in arithmetic (inf + 1 = inf)
""")

# Creating NaN and inf
print("--- Creating Special Values ---")
nan_value = np.nan
inf_value = np.inf
neg_inf = -np.inf

print(f"NaN: {nan_value}, type: {type(nan_value)}")
print(f"inf: {inf_value}, type: {type(inf_value)}")
print(f"-inf: {neg_inf}")

# Operations that produce NaN
print("\n--- Operations Producing NaN ---")
print(f"0 / 0 = {np.float64(0.0) / np.float64(0.0)}")
print(f"inf - inf = {np.inf - np.inf}")
print(f"sqrt(-1) = {np.sqrt(-1)}")
print(f"log(-1) = {np.log(-1)}")

# Operations that produce inf
print("\n--- Operations Producing inf ---")
print(f"1 / 0 = {np.float64(1.0) / np.float64(0.0)}")
print(f"-1 / 0 = {np.float64(-1.0) / np.float64(0.0)}")
print(f"exp(1000) = {np.exp(1000)}")

# NaN propagation
print("\n--- NaN Propagation ---")
arr = np.array([1, 2, np.nan, 4, 5])
print(f"Array: {arr}")
print(f"Sum: {arr.sum()}")  # NaN!
print(f"Mean: {arr.mean()}")  # NaN!
print(f"Max: {arr.max()}")  # NaN!

print("\n⚠️  Single NaN contaminates entire calculation!")

# ============================================================
# PART 2: Detecting NaN and inf
# ============================================================

print("\n" + "="*60)
print("PART 2: DETECTING NaN AND inf")
print("="*60)

data = np.array([1.0, 2.0, np.nan, 4.0, np.inf, -np.inf, 7.0])
print(f"Data: {data}")

# Detecting NaN
print("\n--- Detecting NaN ---")
print(f"np.isnan(data): {np.isnan(data)}")
print(f"Number of NaNs: {np.isnan(data).sum()}")

# CRITICAL: NaN != NaN !!!
print("\n⚠️  CRITICAL: NaN Comparison Behavior")
print(f"np.nan == np.nan: {np.nan == np.nan}")  # False!
print(f"np.nan != np.nan: {np.nan != np.nan}")  # True!
print("Always use np.isnan(), never == to check for NaN")

# Detecting inf
print("\n--- Detecting inf ---")
print(f"np.isinf(data): {np.isinf(data)}")
print(f"np.isposinf(data): {np.isposinf(data)}")
print(f"np.isneginf(data): {np.isneginf(data)}")

# Detecting finite values
print("\n--- Detecting Finite Values ---")
print(f"np.isfinite(data): {np.isfinite(data)}")
print(f"Number of valid values: {np.isfinite(data).sum()}")

# Combined detection
print("\n--- Combined Detection ---")
has_nan = np.any(np.isnan(data))
has_inf = np.any(np.isinf(data))
print(f"Contains NaN? {has_nan}")
print(f"Contains inf? {has_inf}")
print(f"All finite? {np.all(np.isfinite(data))}")

# ============================================================
# PART 3: Real-World Example - Weather Station Data
# ============================================================

print("\n" + "="*60)
print("PART 3: REAL-WORLD - WEATHER STATION DATA")
print("="*60)

# Simulated temperature readings with sensor failures
np.random.seed(42)
num_readings = 24  # 24 hours

temperatures = np.random.normal(25, 3, num_readings)

# Introduce sensor failures (NaN) and errors (inf)
temperatures[5] = np.nan   # Sensor offline
temperatures[12] = np.nan  # Sensor malfunction
temperatures[18] = np.inf  # Sensor error
temperatures[20] = -999    # Error code (common in real data)

print("Temperature Readings (°C) for 24 hours:")
print(temperatures)

# Data quality assessment
print("\n--- Data Quality Assessment ---")
total_readings = len(temperatures)
nan_count = np.isnan(temperatures).sum()
inf_count = np.isinf(temperatures).sum()
error_code_count = (temperatures == -999).sum()
valid_count = np.isfinite(temperatures).sum() - error_code_count

print(f"Total readings: {total_readings}")
print(f"Missing (NaN): {nan_count} ({nan_count/total_readings*100:.1f}%)")
print(f"Invalid (inf): {inf_count} ({inf_count/total_readings*100:.1f}%)")
print(f"Error codes (-999): {error_code_count}")
print(f"Valid readings: {valid_count} ({valid_count/total_readings*100:.1f}%)")

# Problem: Can't calculate statistics with NaN/inf
print("\n--- Statistics (Incorrect - includes NaN/inf) ---")
print(f"Mean: {temperatures.mean():.2f}")  # NaN!
print(f"Std: {temperatures.std():.2f}")    # NaN!

# Solution 1: Use nan-safe functions
print("\n--- Solution 1: NaN-Safe Functions ---")
print(f"Mean (nanmean): {np.nanmean(temperatures):.2f}")
print(f"Std (nanstd): {np.nanstd(temperatures):.2f}")
print(f"Min (nanmin): {np.nanmin(temperatures):.2f}")
print(f"Max (nanmax): {np.nanmax(temperatures):.2f}")

print("\n⚠️  But still includes inf and error codes!")

# Solution 2: Clean data first
print("\n--- Solution 2: Clean Data First ---")
# Replace error codes with NaN
temps_clean = temperatures.copy()
temps_clean[temps_clean == -999] = np.nan

# Filter to finite values only
valid_mask = np.isfinite(temps_clean)
temps_valid = temps_clean[valid_mask]

print(f"Valid temperatures: {temps_valid}")
print(f"Mean: {temps_valid.mean():.2f}°C")
print(f"Std: {temps_valid.std():.2f}°C")
print(f"Min: {temps_valid.min():.2f}°C")
print(f"Max: {temps_valid.max():.2f}°C")

# ============================================================
# PART 4: Handling Missing Data - Strategies
# ============================================================

print("\n" + "="*60)
print("PART 4: HANDLING MISSING DATA STRATEGIES")
print("="*60)

# Sample data with missing values
data = np.array([10.0, 15.0, np.nan, 20.0, np.nan, 25.0, 30.0])
print(f"Original data: {data}")
print(f"Missing: {np.isnan(data).sum()} values")

# Strategy 1: Drop missing values
print("\n--- Strategy 1: Drop Missing Values ---")
data_dropped = data[~np.isnan(data)]
print(f"After dropping NaN: {data_dropped}")
print(f"Remaining: {len(data_dropped)}/{len(data)} values")

# Strategy 2: Fill with constant
print("\n--- Strategy 2: Fill with Constant ---")
data_filled_zero = np.nan_to_num(data, nan=0.0)
print(f"Fill with 0: {data_filled_zero}")

# Strategy 3: Fill with mean
print("\n--- Strategy 3: Fill with Mean ---")
mean_value = np.nanmean(data)
data_filled_mean = data.copy()
data_filled_mean[np.isnan(data_filled_mean)] = mean_value
print(f"Fill with mean ({mean_value:.2f}): {data_filled_mean}")

# Strategy 4: Fill with median (robust to outliers)
print("\n--- Strategy 4: Fill with Median ---")
median_value = np.nanmedian(data)
data_filled_median = data.copy()
data_filled_median[np.isnan(data_filled_median)] = median_value
print(f"Fill with median ({median_value:.2f}): {data_filled_median}")

# Strategy 5: Forward fill (use last valid value)
print("\n--- Strategy 5: Forward Fill ---")
data_ffill = data.copy()
last_valid = None
for i in range(len(data_ffill)):
    if np.isnan(data_ffill[i]):
        if last_valid is not None:
            data_ffill[i] = last_valid
    else:
        last_valid = data_ffill[i]
print(f"Forward fill: {data_ffill}")

# Strategy 6: Linear interpolation
print("\n--- Strategy 6: Linear Interpolation ---")
# Find valid indices
valid_indices = np.where(~np.isnan(data))[0]
valid_values = data[valid_indices]

# Interpolate
data_interp = data.copy()
for i in range(len(data)):
    if np.isnan(data[i]):
        # Find surrounding valid values
        left_idx = valid_indices[valid_indices < i]
        right_idx = valid_indices[valid_indices > i]
        
        if len(left_idx) > 0 and len(right_idx) > 0:
            left_idx = left_idx[-1]
            right_idx = right_idx[0]
            
            # Linear interpolation
            weight = (i - left_idx) / (right_idx - left_idx)
            data_interp[i] = data[left_idx] + weight * (data[right_idx] - data[left_idx])

print(f"Linear interpolation: {data_interp}")

# ============================================================
# PART 5: Real-World Example - Sales Data Cleaning
# ============================================================

print("\n" + "="*60)
print("PART 5: REAL-WORLD - SALES DATA CLEANING")
print("="*60)

# Simulated daily sales with missing data
np.random.seed(42)
days = 30
daily_sales = np.random.gamma(100, 2, days)

# Introduce missing data (various reasons)
daily_sales[5] = np.nan    # System downtime
daily_sales[6] = np.nan    # Continued downtime
daily_sales[15] = np.nan   # Holiday (store closed)
daily_sales[22] = np.nan   # Data entry error
daily_sales[10] = -1       # Error code
daily_sales[25] = 999999   # Obvious outlier/error

print("Daily Sales ($):")
print(daily_sales.round(2))

# Step 1: Identify issues
print("\n--- Step 1: Identify Issues ---")
print(f"Total days: {len(daily_sales)}")
print(f"Missing (NaN): {np.isnan(daily_sales).sum()}")
print(f"Negative values: {(daily_sales < 0).sum()}")
print(f"Suspicious high values (>10000): {(daily_sales > 10000).sum()}")

# Step 2: Replace error codes with NaN
print("\n--- Step 2: Replace Error Codes ---")
sales_clean = daily_sales.copy()
sales_clean[sales_clean < 0] = np.nan
sales_clean[sales_clean > 10000] = np.nan  # Outlier threshold

print(f"Missing after cleanup: {np.isnan(sales_clean).sum()}")

# Step 3: Calculate statistics on valid data
print("\n--- Step 3: Statistics (Valid Data Only) ---")
valid_sales = sales_clean[~np.isnan(sales_clean)]
print(f"Valid days: {len(valid_sales)}")
print(f"Mean daily sales: ${valid_sales.mean():.2f}")
print(f"Median daily sales: ${np.median(valid_sales):.2f}")
print(f"Std deviation: ${valid_sales.std():.2f}")
print(f"Total (valid days): ${valid_sales.sum():,.2f}")

# Step 4: Impute missing values for forecasting
print("\n--- Step 4: Imputation for Analysis ---")
# Use median for robustness
median_sales = np.median(valid_sales)
sales_imputed = sales_clean.copy()
sales_imputed[np.isnan(sales_imputed)] = median_sales

print(f"Imputed with median (${median_sales:.2f}):")
print(f"Total revenue estimate: ${sales_imputed.sum():,.2f}")
print(f"Average daily (imputed): ${sales_imputed.mean():.2f}")

# Step 5: Flag imputed values for transparency
print("\n--- Step 5: Track Imputed Values ---")
imputed_mask = np.isnan(sales_clean)
print(f"Imputed days: {np.where(imputed_mask)[0]}")
print(f"Percentage imputed: {imputed_mask.sum()/len(sales_clean)*100:.1f}%")

# ============================================================
# PART 6: Masked Arrays - Alternative Approach
# ============================================================

print("\n" + "="*60)
print("PART 6: MASKED ARRAYS")
print("="*60)

print("""
Masked arrays are NumPy's built-in solution for missing data.
They keep track of which values are valid vs. invalid.
""")

# Create masked array
data = np.array([1.0, 2.0, -999, 4.0, -999, 6.0, 7.0])
print(f"Raw data (with error codes): {data}")

# Create mask (True = invalid/masked)
mask = (data == -999)
masked_data = np.ma.masked_array(data, mask=mask)

print(f"\nMasked array: {masked_data}")
print(f"Mask: {masked_data.mask}")

# Operations automatically ignore masked values
print("\n--- Operations on Masked Arrays ---")
print(f"Mean: {masked_data.mean():.2f}")
print(f"Sum: {masked_data.sum():.2f}")
print(f"Std: {masked_data.std():.2f}")

# Create masked array from NaN
print("\n--- Creating from NaN ---")
data_with_nan = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
masked_from_nan = np.ma.masked_invalid(data_with_nan)
print(f"Data with NaN: {data_with_nan}")
print(f"Masked array: {masked_from_nan}")
print(f"Mean: {masked_from_nan.mean():.2f}")

# Accessing valid data only
print("\n--- Accessing Valid Data ---")
print(f"Valid data (compressed): {masked_from_nan.compressed()}")

# Filling masked values
print("\n--- Filling Masked Values ---")
filled = masked_from_nan.filled(0)  # Fill with 0
print(f"Filled with 0: {filled}")

filled_mean = masked_from_nan.filled(masked_from_nan.mean())
print(f"Filled with mean: {filled_mean}")

# ============================================================
# PART 7: Handling inf Values
# ============================================================

print("\n" + "="*60)
print("PART 7: HANDLING inf VALUES")
print("="*60)

# Data with inf values
data = np.array([1.0, 2.0, np.inf, 4.0, -np.inf, 6.0])
print(f"Data with inf: {data}")

# Detection
print("\n--- Detection ---")
print(f"Has inf: {np.any(np.isinf(data))}")
print(f"Positive inf mask: {np.isposinf(data)}")
print(f"Negative inf mask: {np.isneginf(data)}")

# Replacement
print("\n--- Replacement Strategies ---")

# Replace with NaN
data_nan = data.copy()
data_nan[np.isinf(data_nan)] = np.nan
print(f"Replace inf with NaN: {data_nan}")

# Replace with large finite values
data_capped = data.copy()
data_capped[np.isposinf(data_capped)] = 1e10
data_capped[np.isneginf(data_capped)] = -1e10
print(f"Cap inf values: {data_capped}")

# Remove inf entirely
data_finite = data[np.isfinite(data)]
print(f"Keep only finite: {data_finite}")

# Using np.nan_to_num (handles both NaN and inf)
print("\n--- Using np.nan_to_num ---")
data_mixed = np.array([1.0, np.nan, np.inf, 4.0, -np.inf])
print(f"Original: {data_mixed}")

cleaned = np.nan_to_num(data_mixed, nan=0.0, posinf=999.0, neginf=-999.0)
print(f"Cleaned: {cleaned}")

# ============================================================
# PART 8: Data Validation Pipeline
# ============================================================

print("\n" + "="*60)
print("PART 8: DATA VALIDATION PIPELINE")
print("="*60)

def validate_and_clean(data, name="Data", 
                       error_codes=None, 
                       valid_range=None,
                       impute_strategy='median'):
    """
    Comprehensive data validation and cleaning pipeline.
    
    Parameters:
    - data: array to clean
    - name: descriptive name for reporting
    - error_codes: list of values to treat as errors
    - valid_range: tuple (min, max) for valid values
    - impute_strategy: 'drop', 'mean', 'median', 'zero'
    """
    print(f"\n{'='*50}")
    print(f"Validating: {name}")
    print(f"{'='*50}")
    
    cleaned = data.copy()
    original_size = len(data)
    
    # Report 1: Initial state
    print(f"\n1. Initial Assessment:")
    print(f"   Total values: {original_size}")
    print(f"   NaN count: {np.isnan(cleaned).sum()}")
    print(f"   inf count: {np.isinf(cleaned).sum()}")
    
    # Step 1: Replace error codes with NaN
    if error_codes:
        print(f"\n2. Replacing error codes: {error_codes}")
        for code in error_codes:
            error_count = (cleaned == code).sum()
            if error_count > 0:
                print(f"   Found {error_count} instances of {code}")
                cleaned[cleaned == code] = np.nan
    
    # Step 2: Handle inf
    print(f"\n3. Handling inf values:")
    inf_count = np.isinf(cleaned).sum()
    if inf_count > 0:
        print(f"   Found {inf_count} inf values → replacing with NaN")
        cleaned[np.isinf(cleaned)] = np.nan
    else:
        print(f"   No inf values found ✓")
    
    # Step 3: Range validation
    if valid_range:
        print(f"\n4. Range validation: {valid_range}")
        min_val, max_val = valid_range
        out_of_range = ((cleaned < min_val) | (cleaned > max_val)) & ~np.isnan(cleaned)
        oor_count = out_of_range.sum()
        if oor_count > 0:
            print(f"   Found {oor_count} out-of-range values → replacing with NaN")
            cleaned[out_of_range] = np.nan
        else:
            print(f"   All values in range ✓")
    
    # Report 2: After cleaning
    nan_count = np.isnan(cleaned).sum()
    valid_count = original_size - nan_count
    
    print(f"\n5. After Cleaning:")
    print(f"   Valid values: {valid_count} ({valid_count/original_size*100:.1f}%)")
    print(f"   Missing values: {nan_count} ({nan_count/original_size*100:.1f}%)")
    
    # Step 4: Imputation
    print(f"\n6. Imputation Strategy: {impute_strategy}")
    if nan_count > 0:
        if impute_strategy == 'drop':
            result = cleaned[~np.isnan(cleaned)]
            print(f"   Dropped {nan_count} rows")
            print(f"   Final size: {len(result)}")
        
        elif impute_strategy == 'mean':
            fill_value = np.nanmean(cleaned)
            result = cleaned.copy()
            result[np.isnan(result)] = fill_value
            print(f"   Filled {nan_count} values with mean: {fill_value:.2f}")
        
        elif impute_strategy == 'median':
            fill_value = np.nanmedian(cleaned)
            result = cleaned.copy()
            result[np.isnan(result)] = fill_value
            print(f"   Filled {nan_count} values with median: {fill_value:.2f}")
        
        elif impute_strategy == 'zero':
            result = cleaned.copy()
            result[np.isnan(result)] = 0
            print(f"   Filled {nan_count} values with zero")
        
        else:
            result = cleaned
            print(f"   No imputation performed")
    else:
        result = cleaned
        print(f"   No missing values to impute ✓")
    
    # Final report
    print(f"\n7. Final Statistics:")
    print(f"   Mean: {np.mean(result):.2f}")
    print(f"   Median: {np.median(result):.2f}")
    print(f"   Std: {np.std(result):.2f}")
    print(f"   Min: {np.min(result):.2f}")
    print(f"   Max: {np.max(result):.2f}")
    
    return result

# Example usage
print("\n" + "="*60)
print("EXAMPLE: CLEANING SENSOR DATA")
print("="*60)

# Messy sensor data
np.random.seed(42)
sensor_data = np.random.normal(25, 3, 30)
sensor_data[5] = np.nan
sensor_data[10] = -999  # Error code
sensor_data[15] = np.inf
sensor_data[20] = 100  # Out of range
sensor_data[25] = -50  # Out of range

print("Raw sensor data (first 30 values):")
print(sensor_data)

# Clean it
cleaned_sensor = validate_and_clean(
    sensor_data,
    name="Temperature Sensor",
    error_codes=[-999],
    valid_range=(-50, 50),
    impute_strategy='median'
)

print("\n" + "="*60)
print("EXAMPLE: CLEANING SALES DATA")
print("="*60)

# Messy sales data
np.random.seed(123)
sales_data = np.random.gamma(100, 2, 50)
sales_data[10] = np.nan
sales_data[20] = -1  # Error code
sales_data[30] = 0   # Error code
sales_data[40] = -5  # Negative

print("Raw sales data (50 days):")
print(sales_data[:20])  # Show first 20

cleaned_sales = validate_and_clean(
    sales_data,
    name="Daily Sales",
    error_codes=[-1, 0],
    valid_range=(1, 100000),
    impute_strategy='mean'
)

# ============================================================
# PART 9: Common Pitfalls and Solutions
# ============================================================

print("\n" + "="*60)
print("PART 9: COMMON PITFALLS WITH NaN")
print("="*60)

print("\n--- Pitfall 1: Comparing NaN with == ---")
data = np.array([1, 2, np.nan, 4])
print(f"Data: {data}")
print(f"data == np.nan: {data == np.nan}")  # All False!
print(f"np.isnan(data): {np.isnan(data)}")  # ✓ Correct

print("\n--- Pitfall 2: Sorting with NaN ---")
data = np.array([3, 1, np.nan, 2, np.nan, 5])
print(f"Original: {data}")
sorted_data = np.sort(data)
print(f"Sorted: {sorted_data}")
print("⚠️  NaN values moved to end")

print("\n--- Pitfall 3: Unique with NaN ---")
data = np.array([1, 2, np.nan, 2, np.nan, 1])
unique = np.unique(data)
print(f"Data: {data}")
print(f"Unique: {unique}")
print("⚠️  Multiple NaNs still appear as 'unique'")

print("\n--- Pitfall 4: Aggregations ---")
data = np.array([1, 2, np.nan, 4, 5])
print(f"Data: {data}")
print(f"np.sum(data): {np.sum(data)}")      # NaN!
print(f"np.nansum(data): {np.nansum(data)}")  # ✓ Correct

print("\n--- Pitfall 5: Boolean Operations ---")
data = np.array([1, 2, np.nan, 4])
print(f"Data: {data}")
print(f"data > 2: {data > 2}")  # NaN becomes False!
print("⚠️  NaN in comparisons evaluates to False")

# ============================================================
# PART 10: Best Practices Summary
# ============================================================

print("\n" + "="*60)
print("PART 10: BEST PRACTICES SUMMARY")
print("="*60)

print("""
✅ DETECTION:
1. Use np.isnan() to detect NaN (never use ==)
2. Use np.isinf() to detect inf
3. Use np.isfinite() to detect valid values
4. Always check for missing data before analysis

✅ HANDLING NaN:
1. Use nan-safe functions: np.nanmean(), np.nansum(), etc.
2. Decide on strategy: drop, impute, or keep
3. Document imputation strategy for reproducibility
4. Track which values were imputed

✅ IMPUTATION STRATEGIES:
- Drop: When data is abundant and loss is acceptable
- Mean: For normally distributed data
- Median: For skewed data or with outliers
- Forward/backward fill: For time series
- Interpolation: For continuous data with gaps
- Domain-specific: Use business logic

❌ COMMON MISTAKES:
1. Using data == np.nan (always False!)
2. Forgetting NaN propagation in calculations
3. Not validating data before analysis
4. Using mean when median is more robust
5. Silently imputing without tracking
6. Not handling inf values separately
7. Using regular functions instead of nan-safe ones

⚠️  CRITICAL RULES:
- NaN != NaN (always returns False)
- Any operation with NaN returns NaN
- NaN in boolean context is False
- Always validate before aggregating
- Document your cleaning decisions
""")

print("\n" + "="*60)
print("FUNCTION REFERENCE")
print("="*60)

print("""
DETECTION:
  np.isnan(arr)      - Detect NaN
  np.isinf(arr)      - Detect inf
  np.isfinite(arr)   - Detect finite values
  np.isposinf(arr)   - Detect +inf
  np.isneginf(arr)   - Detect -inf

NaN-SAFE OPERATIONS:
  np.nansum(arr)     - Sum ignoring NaN
  np.nanmean(arr)    - Mean ignoring NaN
  np.nanmedian(arr)  - Median ignoring NaN
  np.nanstd(arr)     - Std dev ignoring NaN
  np.nanvar(arr)     - Variance ignoring NaN
  np.nanmin(arr)     - Minimum ignoring NaN
  np.nanmax(arr)     - Maximum ignoring NaN
  np.nanargmin(arr)  - Index of min ignoring NaN
  np.nanargmax(arr)  - Index of max ignoring NaN

REPLACEMENT:
  np.nan_to_num(arr) - Replace NaN/inf with numbers
  arr[mask] = value  - Manual replacement
  np.ma.masked_invalid() - Create masked array

MASKED ARRAYS:
  np.ma.masked_array()   - Create masked array
  np.ma.masked_invalid() - Mask NaN/inf
  arr.compressed()       - Get valid values only
  arr.filled(value)      - Fill masked with value
""")
