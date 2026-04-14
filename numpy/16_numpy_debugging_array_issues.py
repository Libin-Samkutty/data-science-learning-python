"""
================================================================================
LESSON 16: DEBUGGING ARRAY ISSUES
================================================================================

What You Will Learn:
- Common array bugs and how to identify them
- Shape mismatch debugging
- Data type issues and silent errors
- NaN/Inf propagation problems
- Memory issues (views vs copies confusion)
- Techniques for inspecting array state
- Building defensive code with assertions
- A systematic debugging workflow

Real-World Usage:
- Catching data corruption early in pipelines
- Debugging ML feature engineering errors
- Validating sensor data processing
- Ensuring financial calculations are correct
- Preventing silent errors in production

Dataset Used:
- Publicly available Air Quality dataset
- URL: https://raw.githubusercontent.com/dsrscientist/dataset1/master/air_quality.csv
- Fallback: Synthetically generated realistic data (same structure)

================================================================================
"""

import numpy as np
import urllib.request
import sys

print("=" * 70)
print("LESSON 16: DEBUGGING ARRAY ISSUES")
print("=" * 70)


# ==============================================================================
# SECTION 1: LOAD REAL-WORLD DATASET
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 1: LOAD REAL-WORLD DATASET")
print("=" * 70)

# We will use air quality data (PM2.5, PM10, NO2, CO, temperature, humidity)
# This mimics real sensor data which is full of issues to debug

def load_air_quality_data():
    """
    Try to load real air quality data.
    Fall back to synthetic data with same structure if unavailable.
    """
    url = (
        "https://raw.githubusercontent.com/"
        "dsrscientist/dataset1/master/air_quality.csv"
    )

    try:
        print(f"\nAttempting to load data from:\n  {url}")
        response = urllib.request.urlopen(url, timeout=10)
        raw = response.read().decode("utf-8")
        lines = raw.strip().split("\n")

        # Parse numeric columns (skip header)
        data_rows = []
        for line in lines[1:]:
            values = line.split(",")
            try:
                row = [float(v.strip()) for v in values[1:7]]
                data_rows.append(row)
            except ValueError:
                continue  # Skip malformed rows

        data = np.array(data_rows)
        print(f"Successfully loaded real data: {data.shape}")
        return data

    except Exception as e:
        print(f"Could not load real data ({e})")
        print("Generating synthetic air quality data (same structure)...")

        # Reproducible synthetic data
        np.random.seed(42)
        n = 500

        # Realistic air quality sensor readings
        # Columns: PM2.5, PM10, NO2, CO, Temperature, Humidity
        pm25      = np.random.exponential(35, n)               # ?g/m?
        pm10      = pm25 * 1.5 + np.random.normal(10, 5, n)   # ?g/m?
        no2       = np.random.normal(40, 15, n)                # ?g/m?
        co        = np.random.normal(1.2, 0.4, n)              # mg/m?
        temp      = np.random.normal(22, 8, n)                 # ?C
        humidity  = np.random.normal(60, 15, n)                # %

        # Inject realistic issues into data (sensors malfunction)
        # Missing readings
        pm25[np.random.choice(n, 20, replace=False)]     = np.nan
        pm10[np.random.choice(n, 15, replace=False)]     = np.nan
        no2[np.random.choice(n, 10, replace=False)]      = np.nan

        # Sensor spikes (outliers)
        pm25[np.random.choice(n, 5, replace=False)]      = 9999.0
        co[np.random.choice(n, 3, replace=False)]        = -999.0

        # Physically impossible values
        humidity[np.random.choice(n, 8, replace=False)]  = 150.0
        temp[np.random.choice(n, 5, replace=False)]      = -99.0

        data = np.column_stack([pm25, pm10, no2, co, temp, humidity])
        print(f"Synthetic data generated: {data.shape}")
        return data


# Column names for reference throughout the lesson
COL_NAMES = ["PM2.5", "PM10", "NO2", "CO", "Temperature", "Humidity"]

raw_data = load_air_quality_data()

print(f"\nDataset shape: {raw_data.shape}")
print(f"Columns: {COL_NAMES}")
print(f"\nFirst 5 rows:")
print(raw_data[:5])


# ==============================================================================
# SECTION 2: THE DEBUGGING MINDSET
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 2: THE DEBUGGING MINDSET")
print("=" * 70)

mindset = """
When debugging NumPy code, always ask:

1. SHAPE   ? Is the array the shape I expect?
2. DTYPE   ? Is the data type correct?
3. VALUES  ? Are the values in a valid range?
4. NaN/Inf ? Is there any missing or infinite data?
5. MEMORY  ? Is this a view or a copy?
6. AXIS    ? Am I operating along the correct axis?
7. BROADCAST ? Are shapes compatible for broadcasting?

GOLDEN RULE: Check early, check often.
A bug caught at step 1 is cheaper than one caught at step 10.
"""
print(mindset)


# ==============================================================================
# SECTION 3: INSPECTING ARRAY STATE
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 3: INSPECTING ARRAY STATE")
print("=" * 70)

# ------------------------------------------------------------------------------
# 3.1 The Inspection Toolkit
# ------------------------------------------------------------------------------
print("\n--- 3.1 The Inspection Toolkit ---")

def inspect_array(arr, name="array"):
    """
    Comprehensive array inspection function.
    Use this at every stage of a pipeline to catch issues early.
    """
    print(f"\n{'='*50}")
    print(f"Inspecting: {name}")
    print(f"{'='*50}")

    # Basic structure
    print(f"  Shape:       {arr.shape}")
    print(f"  Dimensions:  {arr.ndim}")
    print(f"  Dtype:       {arr.dtype}")
    print(f"  Size:        {arr.size} elements")
    print(f"  Memory:      {arr.nbytes / 1024:.2f} KB")

    # Memory layout
    print(f"  C-contiguous: {arr.flags['C_CONTIGUOUS']}")
    print(f"  Is view:      {arr.base is not None}")

    # Value statistics (only for numeric types)
    if np.issubdtype(arr.dtype, np.number):
        nan_count  = np.isnan(arr).sum()
        inf_count  = np.isinf(arr).sum()
        zero_count = (arr == 0).sum()

        print(f"\n  NaN count:   {nan_count}")
        print(f"  Inf count:   {inf_count}")
        print(f"  Zero count:  {zero_count}")

        # Stats ignoring NaN
        if nan_count < arr.size:
            print(f"\n  Min:         {np.nanmin(arr):.4f}")
            print(f"  Max:         {np.nanmax(arr):.4f}")
            print(f"  Mean:        {np.nanmean(arr):.4f}")
            print(f"  Std:         {np.nanstd(arr):.4f}")

    print(f"{'='*50}")

# Inspect our raw dataset
inspect_array(raw_data, "raw_data")

# ------------------------------------------------------------------------------
# 3.2 Per-Column Inspection
# ------------------------------------------------------------------------------
print("\n--- 3.2 Per-Column Inspection ---")

print("\nPer-column summary:")
print(f"{'Column':<14} {'Min':>10} {'Max':>10} {'Mean':>10} {'NaNs':>6} {'Infs':>6}")
print("-" * 58)

for i, col_name in enumerate(COL_NAMES):
    col = raw_data[:, i]
    nan_c = np.isnan(col).sum()
    inf_c = np.isinf(col).sum()

    # Use nanmin/nanmax to avoid NaN contamination in stats
    col_min  = np.nanmin(col)
    col_max  = np.nanmax(col)
    col_mean = np.nanmean(col)

    print(
        f"{col_name:<14} {col_min:>10.2f} {col_max:>10.2f} "
        f"{col_mean:>10.2f} {nan_c:>6} {inf_c:>6}"
    )


# ==============================================================================
# SECTION 4: SHAPE MISMATCH BUGS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 4: SHAPE MISMATCH BUGS")
print("=" * 70)

# ------------------------------------------------------------------------------
# 4.1 The Classic Shape Mismatch
# ------------------------------------------------------------------------------
print("\n--- 4.1 The Classic Shape Mismatch ---")

# Simulated scenario: normalizing each column with precomputed stats
col_means = raw_data.mean(axis=0)   # Shape: (6,)
col_stds  = raw_data.std(axis=0)    # Shape: (6,)

print(f"raw_data shape: {raw_data.shape}")
print(f"col_means shape: {col_means.shape}")
print(f"col_stds shape: {col_stds.shape}")

# This works due to broadcasting - (500, 6) - (6,) = (500, 6)
normalized = (raw_data - col_means) / col_stds
print(f"\nNormalization result shape: {normalized.shape}")
print("Broadcasting worked correctly!")

# Now a common mistake: wrong reshape
wrong_means = col_means.reshape(6, 1)  # Shape: (6, 1) instead of (1, 6)
print(f"\nWrong reshape - col_means.reshape(6,1): {wrong_means.shape}")

print("\nAttempting subtraction with wrong shape:")
try:
    wrong_result = raw_data - wrong_means
    print(f"Result shape: {wrong_result.shape}")
    print("WARNING: No error but shape is wrong!")
    print(f"Expected (500, 6), got {wrong_result.shape}")
except ValueError as e:
    print(f"Error caught: {e}")

# ------------------------------------------------------------------------------
# 4.2 Diagnosing Shape Bugs
# ------------------------------------------------------------------------------
print("\n--- 4.2 Diagnosing Shape Bugs ---")

def safe_normalize(data, means, stds):
    """
    Normalize with explicit shape validation.
    This is how you write defensive production code.
    """
    print(f"\n[safe_normalize] Input shapes:")
    print(f"  data:  {data.shape}")
    print(f"  means: {means.shape}")
    print(f"  stds:  {stds.shape}")

    # Explicit shape validation
    n_rows, n_cols = data.shape

    assert means.shape == (n_cols,), (
        f"means shape mismatch: expected ({n_cols},), got {means.shape}"
    )
    assert stds.shape == (n_cols,), (
        f"stds shape mismatch: expected ({n_cols},), got {stds.shape}"
    )
    assert not np.any(stds == 0), (
        "Division by zero: some stds are 0"
    )

    result = (data - means) / stds
    print(f"  result: {result.shape}")
    return result

# Call with correct shapes
clean_data = np.nan_to_num(raw_data, nan=0.0, posinf=0.0, neginf=0.0)
result = safe_normalize(clean_data, col_means, col_stds)

# ------------------------------------------------------------------------------
# 4.3 Common Shape Bugs Reference
# ------------------------------------------------------------------------------
print("\n--- 4.3 Common Shape Bugs Reference ---")

# Bug 1: Forgetting axis produces scalar instead of array
data_2d = np.array([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]], dtype=float)

scalar_mean = data_2d.mean()       # Scalar - OFTEN NOT WHAT YOU WANT
row_means   = data_2d.mean(axis=1) # Per-row means - usually intended
col_means_2 = data_2d.mean(axis=0) # Per-column means

print("2D array:")
print(data_2d)
print(f"\ndata.mean()        -> {scalar_mean}  (scalar - axis forgotten)")
print(f"data.mean(axis=1)  -> {row_means}  (per-row, usually intended)")
print(f"data.mean(axis=0)  -> {col_means_2}  (per-column)")

# Bug 2: Extra dimension causing silent broadcast
arr_a = np.array([[1, 2, 3]])  # Shape: (1, 3)
arr_b = np.array([10, 20, 30]) # Shape: (3,)

result_ok = arr_a + arr_b      # OK - (1,3) + (3,) = (1,3)
print(f"\n(1,3) + (3,) = {result_ok.shape}  <- correct but has extra dimension")

# Squeeze to remove size-1 dimensions
result_squeezed = result_ok.squeeze()
print(f"After squeeze: {result_squeezed.shape}  <- cleaner")


# ==============================================================================
# SECTION 5: DATA TYPE BUGS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 5: DATA TYPE BUGS")
print("=" * 70)

# ------------------------------------------------------------------------------
# 5.1 Integer Division Truncation
# ------------------------------------------------------------------------------
print("\n--- 5.1 Integer Division Truncation (Silent Bug) ---")

# Simulated: count sensors and compute ratio
total_sensors  = np.array([100, 200, 150], dtype=np.int32)
active_sensors = np.array([75, 180, 90],  dtype=np.int32)

# Bug: integer division truncates result
wrong_ratio = active_sensors / total_sensors
# In Python 3 this is actually fine (true division)
# But the DTYPE issue appears when you do it differently

wrong_pct = (active_sensors * 100) // total_sensors  # Floor division
right_pct = (active_sensors.astype(np.float64) * 100) / total_sensors

print("Total sensors:  ", total_sensors)
print("Active sensors: ", active_sensors)
print(f"\nWrong (floor division %): {wrong_pct}")
print(f"Correct (%):              {right_pct}")

# ------------------------------------------------------------------------------
# 5.2 Integer Overflow
# ------------------------------------------------------------------------------
print("\n--- 5.2 Integer Overflow (Dangerous Silent Bug) ---")

# Sensor reading accumulator
reading = np.array([32767], dtype=np.int16)  # Max int16 value

print(f"Max int16 value: {np.iinfo(np.int16).max}")
print(f"Current reading: {reading[0]}")

# Adding 1 to max int16 silently overflows!
overflow_result = reading + np.int16(1)
print(f"After +1 (int16): {overflow_result[0]}  <- SILENT OVERFLOW!")

# Fix: use appropriate dtype
safe_reading = reading.astype(np.int32)
safe_result  = safe_reading + 1
print(f"After +1 (int32): {safe_result[0]}   <- correct")

# Check before operating
def safe_add(arr, value):
    """Add value to array with overflow check."""
    if np.issubdtype(arr.dtype, np.integer):
        info     = np.iinfo(arr.dtype)
        max_safe = info.max - value
        if np.any(arr > max_safe):
            print(f"  WARNING: Overflow risk detected. Upcasting dtype.")
            arr = arr.astype(np.int64)
    return arr + value

print("\nUsing safe_add:")
result = safe_add(np.array([32767], dtype=np.int16), 1)
print(f"Result: {result[0]}  dtype: {result.dtype}")

# ------------------------------------------------------------------------------
# 5.3 Float Precision Issues
# ------------------------------------------------------------------------------
print("\n--- 5.3 Float Precision Issues ---")

# Common in financial and scientific calculations
a = np.float32(0.1)
b = np.float32(0.2)
c = np.float64(0.1)
d = np.float64(0.2)

print(f"float32: 0.1 + 0.2 = {a + b}")
print(f"float64: 0.1 + 0.2 = {c + d}")
print(f"Is 0.1 + 0.2 == 0.3 (float32)? {a + b == np.float32(0.3)}")
print(f"Is 0.1 + 0.2 == 0.3 (float64)? {c + d == 0.3}")

# Correct way to compare floats
print(f"\nCorrect comparison with np.isclose:")
print(f"  np.isclose(0.1+0.2, 0.3): {np.isclose(0.1 + 0.2, 0.3)}")

# Real-world scenario: comparing sensor thresholds
threshold = 2.5   # CO threshold in mg/m?
co_readings = np.array([2.499999, 2.5, 2.500001], dtype=np.float32)

print(f"\nCO readings: {co_readings}")
print(f"Exact == 2.5: {co_readings == threshold}")
print(f"np.isclose:   {np.isclose(co_readings, threshold, rtol=1e-4)}")

# ------------------------------------------------------------------------------
# 5.4 Type Coercion in Mixed Operations
# ------------------------------------------------------------------------------
print("\n--- 5.4 Type Coercion in Mixed Operations ---")

int_arr   = np.array([1, 2, 3], dtype=np.int32)
float_arr = np.array([1.5, 2.5, 3.5], dtype=np.float64)
bool_arr  = np.array([True, False, True])

print("Dtypes before operations:")
print(f"  int_arr:   {int_arr.dtype}")
print(f"  float_arr: {float_arr.dtype}")
print(f"  bool_arr:  {bool_arr.dtype}")

result_if = int_arr + float_arr
result_ib = int_arr + bool_arr
result_fb = float_arr * bool_arr

print("\nDtypes after operations:")
print(f"  int32 + float64 -> {result_if.dtype}: {result_if}")
print(f"  int32 + bool    -> {result_ib.dtype}: {result_ib}")
print(f"  float64 * bool  -> {result_fb.dtype}: {result_fb}")

print("\nRule: NumPy upcasts to the 'safer' type automatically")
print("Bool=0/1, int < float in the hierarchy")


# ==============================================================================
# SECTION 6: NaN AND INF PROPAGATION BUGS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 6: NaN AND INF PROPAGATION BUGS")
print("=" * 70)

# ------------------------------------------------------------------------------
# 6.1 NaN Contamination
# ------------------------------------------------------------------------------
print("\n--- 6.1 NaN Contamination - The Silent Killer ---")

# Extract PM2.5 column with NaN values
pm25 = raw_data[:, 0].copy()

print(f"PM2.5 column - NaN count: {np.isnan(pm25).sum()}")
print(f"\nNaN contaminates all downstream calculations:")
print(f"  pm25.sum()   = {pm25.sum()}")
print(f"  pm25.mean()  = {pm25.mean()}")
print(f"  pm25.max()   = {pm25.max()}")
print(f"  pm25.min()   = {pm25.min()}")

print(f"\nNaN-safe versions:")
print(f"  np.nansum()   = {np.nansum(pm25):.2f}")
print(f"  np.nanmean()  = {np.nanmean(pm25):.2f}")
print(f"  np.nanmax()   = {np.nanmax(pm25):.2f}")
print(f"  np.nanmin()   = {np.nanmin(pm25):.2f}")

# ------------------------------------------------------------------------------
# 6.2 Tracking NaN Origin
# ------------------------------------------------------------------------------
print("\n--- 6.2 Tracking Where NaN Comes From ---")

# NaN can appear from various sources
print("NaN sources in NumPy:")

# Source 1: Direct assignment
arr1 = np.array([1.0, np.nan, 3.0])
print(f"\n1. Direct:    {arr1}")

# Source 2: Division by zero (float)
arr2 = np.array([1.0, 0.0, 3.0])
result_div = np.array([1.0, 2.0, 3.0]) / arr2
print(f"2. Div by 0:  {result_div}")  # [1, inf, 1] - inf not nan here

# Source 3: 0/0 produces NaN
arr3 = np.array([0.0]) / np.array([0.0])
print(f"3. 0/0:       {arr3}")

# Source 4: sqrt of negative number
with np.errstate(invalid='ignore'):  # Suppress warning
    arr4 = np.sqrt(np.array([-1.0, 4.0, -9.0]))
print(f"4. sqrt(-x):  {arr4}")

# Source 5: log of negative or zero
with np.errstate(divide='ignore', invalid='ignore'):
    arr5 = np.log(np.array([-1.0, 0.0, 10.0]))
print(f"5. log(<=0):  {arr5}")

# How to detect NaN propagation chain
print("\nDebugging NaN propagation:")
data_pipeline = np.array([10.0, -5.0, 20.0, 0.0, 15.0])

step1 = np.sqrt(data_pipeline)            # -5 produces NaN
step2 = step1 * 2
step3 = step2 + 1

print(f"  Input:  {data_pipeline}")
print(f"  Step 1 (sqrt): {step1}")
print(f"  Step 2 (*2):   {step2}")
print(f"  Step 3 (+1):   {step3}")
print(f"  NaN originated at Step 1, index 1 (sqrt of -5)")

# Locate NaN
nan_positions = np.where(np.isnan(step1))[0]
print(f"  NaN positions: {nan_positions}")
print(f"  Original values at those positions: {data_pipeline[nan_positions]}")

# ------------------------------------------------------------------------------
# 6.3 NaN Handling Strategies
# ------------------------------------------------------------------------------
print("\n--- 6.3 NaN Handling Strategies ---")

pm25_raw = raw_data[:, 0].copy()
nan_mask = np.isnan(pm25_raw)

print(f"Total readings: {len(pm25_raw)}")
print(f"NaN count: {nan_mask.sum()}")

# Strategy 1: Drop NaN values
pm25_dropped = pm25_raw[~nan_mask]
print(f"\nStrategy 1 - Drop NaN:")
print(f"  Remaining: {len(pm25_dropped)} readings")
print(f"  Loss: {nan_mask.sum()} readings")

# Strategy 2: Replace with mean
pm25_mean_fill = pm25_raw.copy()
pm25_mean_fill[nan_mask] = np.nanmean(pm25_raw)
print(f"\nStrategy 2 - Replace with mean ({np.nanmean(pm25_raw):.2f}):")
print(f"  NaN count after: {np.isnan(pm25_mean_fill).sum()}")

# Strategy 3: Forward fill (carry last valid value)
def forward_fill(arr):
    """Replace NaN with last valid (non-NaN) value."""
    result = arr.copy()
    mask   = np.isnan(result)
    indices = np.where(~mask)[0]  # Indices of valid values

    if len(indices) == 0:
        return result  # All NaN - can't fill

    # For each NaN position, find nearest previous valid index
    # np.searchsorted finds where each nan index would fit
    for i in np.where(mask)[0]:
        pos = np.searchsorted(indices, i, side='left')
        if pos > 0:
            result[i] = arr[indices[pos - 1]]
        # If pos == 0, no previous valid value - leave as NaN
    return result

pm25_ffill = forward_fill(pm25_raw)
remaining_nan = np.isnan(pm25_ffill).sum()
print(f"\nStrategy 3 - Forward fill:")
print(f"  NaN count after: {remaining_nan}")

# Strategy 4: Clip and fill (for sensor spikes)
# First handle physical impossibilities
pm25_clipped = np.clip(pm25_raw, a_min=0, a_max=500)  # Valid PM2.5 range
pm25_clipped[nan_mask] = np.nanmedian(pm25_raw)
print(f"\nStrategy 4 - Clip to valid range + median fill:")
print(f"  Min: {pm25_clipped.min():.2f}, Max: {pm25_clipped.max():.2f}")
print(f"  NaN count after: {np.isnan(pm25_clipped).sum()}")

# ------------------------------------------------------------------------------
# 6.4 Infinity Handling
# ------------------------------------------------------------------------------
print("\n--- 6.4 Infinity Handling ---")

# Inf commonly appears in division or log operations
data_with_inf = np.array([1.0, 2.0, 0.0, 4.0, 0.0, 6.0])
result_with_inf = 10.0 / data_with_inf

print(f"10 / {data_with_inf} = {result_with_inf}")
print(f"Inf count: {np.isinf(result_with_inf).sum()}")
print(f"Is finite: {np.isfinite(result_with_inf)}")

# Replace inf with large but finite value or NaN
result_no_inf = np.where(np.isinf(result_with_inf), np.nan, result_with_inf)
print(f"\nAfter replacing inf with NaN: {result_no_inf}")

# Or use np.nan_to_num for comprehensive replacement
result_clean = np.nan_to_num(
    result_with_inf,
    nan=0.0,
    posinf=999.0,
    neginf=-999.0
)
print(f"After nan_to_num:            {result_clean}")


# ==============================================================================
# SECTION 7: MEMORY BUGS (VIEWS VS COPIES)
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 7: MEMORY BUGS (VIEWS VS COPIES)")
print("=" * 70)

# ------------------------------------------------------------------------------
# 7.1 Accidental View Modification
# ------------------------------------------------------------------------------
print("\n--- 7.1 Accidental View Modification ---")

# Common real-world scenario: extracting a subset to process
original_data = raw_data.copy()

# This looks like a copy but it's a VIEW
subset = original_data[:100, :]  # First 100 rows - THIS IS A VIEW

print(f"subset is a view: {subset.base is not None}")

# Modifying subset accidentally modifies original!
subset[0, 0] = -9999.0  # Intentional modification for demo

print(f"Original row 0, col 0 after modifying subset: {original_data[0, 0]}")
print("WARNING: Original data was modified through view!")

# Fix: explicitly copy when you need independence
original_data = raw_data.copy()  # Restore
subset_safe = original_data[:100, :].copy()  # Force copy

subset_safe[0, 0] = -9999.0  # Now modifies only the copy

print(f"\nAfter using .copy():")
print(f"Original row 0, col 0: {original_data[0, 0]}  <- unchanged")
print(f"Subset row 0, col 0:   {subset_safe[0, 0]}    <- changed independently")

# ------------------------------------------------------------------------------
# 7.2 Detecting Views Programmatically
# ------------------------------------------------------------------------------
print("\n--- 7.2 Detecting Views Programmatically ---")

def is_view_of(arr, base_arr):
    """Check if arr is a view of base_arr."""
    return arr.base is not None and np.shares_memory(arr, base_arr)

original = np.arange(100, dtype=float)

slice_view  = original[10:50]      # View
step_view   = original[::2]        # View
reshaped    = original.reshape(10, 10)  # View
forced_copy = original[10:50].copy()   # Copy

print("View detection:")
print(f"  original[10:50]       - is view: {is_view_of(slice_view, original)}")
print(f"  original[::2]         - is view: {is_view_of(step_view, original)}")
print(f"  original.reshape()    - is view: {is_view_of(reshaped, original)}")
print(f"  original[10:50].copy()- is view: {is_view_of(forced_copy, original)}")

# ------------------------------------------------------------------------------
# 7.3 View vs Copy Performance Trade-off
# ------------------------------------------------------------------------------
print("\n--- 7.3 When to Use View vs Copy ---")

guidance = """
USE VIEWS when:
  [OK] You only need to READ the data
  [OK] Performance is critical (no allocation)
  [OK] You're iterating over large array sections
  [OK] You WANT changes to propagate to original

USE COPIES when:
  [OK] You need to MODIFY the data independently
  [OK] You're passing data to external functions
  [OK] You want to preserve original data
  [OK] You're unsure if modification will occur
"""
print(guidance)


# ==============================================================================
# SECTION 8: AXIS BUGS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 8: AXIS BUGS")
print("=" * 70)

# ------------------------------------------------------------------------------
# 8.1 Wrong Axis in Aggregation
# ------------------------------------------------------------------------------
print("\n--- 8.1 Wrong Axis in Aggregation ---")

# Use clean data for this section
clean = np.nan_to_num(raw_data, nan=0.0, posinf=0.0, neginf=0.0)

print(f"clean shape: {clean.shape}  (rows=readings, cols=pollutants)")

# Common mistake: computing mean without specifying axis
wrong_mean = clean.mean()
print(f"\nclean.mean()        = {wrong_mean:.4f}")
print("  <- WRONG: This is the mean of ALL 3000 values (6 cols ? 500 rows)")

col_means_correct = clean.mean(axis=0)
row_means_correct = clean.mean(axis=1)

print(f"\nclean.mean(axis=0)  = {col_means_correct.round(2)}")
print("  <- Mean per pollutant (correct for column-wise stats)")
print(f"\nclean.mean(axis=1)  shape = {row_means_correct.shape}")
print("  <- Mean per reading (correct for row-wise stats)")

# Visualize the axis direction
print("\nAxis direction reminder:")
print("  axis=0 -> collapses ROWS    -> result has shape (n_cols,)")
print("  axis=1 -> collapses COLUMNS -> result has shape (n_rows,)")

# ------------------------------------------------------------------------------
# 8.2 Axis Bug in Normalization
# ------------------------------------------------------------------------------
print("\n--- 8.2 Common Axis Bug in Normalization ---")

sample = np.array([[10.0, 100.0, 1.0],
                   [20.0, 200.0, 2.0],
                   [30.0, 300.0, 3.0]])

print("Sample data:")
print(sample)

# Bug: normalizing with wrong axis
wrong_norm = (sample - sample.mean(axis=1, keepdims=True)) / \
              sample.std(axis=1, keepdims=True)
correct_norm = (sample - sample.mean(axis=0)) / sample.std(axis=0)

print("\nWrong normalization (axis=1 - normalizes rows):")
print(wrong_norm.round(4))
print("  Each row sums to ~0, but columns have different scales")

print("\nCorrect normalization (axis=0 - normalizes columns):")
print(correct_norm.round(4))
print("  Each column is independently zero-mean, unit-variance")


# ==============================================================================
# SECTION 9: BROADCASTING BUGS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 9: BROADCASTING BUGS")
print("=" * 70)

# ------------------------------------------------------------------------------
# 9.1 Silent Broadcasting Error
# ------------------------------------------------------------------------------
print("\n--- 9.1 Silent Broadcasting Error ---")

# Scenario: Adding thresholds to a dataset
data_3x3 = np.ones((3, 3))

# Intended: add different value to each row
row_offsets = np.array([10, 20, 30])  # Shape: (3,)

# This broadcasts along axis=1 (columns) NOT axis=0 (rows)!
result_wrong = data_3x3 + row_offsets
print("data (3x3 of ones):")
print(data_3x3)
print(f"\nrow_offsets shape: {row_offsets.shape}")
print("\ndata + row_offsets (WRONG - adds to COLUMNS not ROWS):")
print(result_wrong)

# Fix: reshape to (3, 1) to broadcast along rows
row_offsets_col = row_offsets.reshape(-1, 1)  # Shape: (3, 1)
result_correct = data_3x3 + row_offsets_col
print(f"\nrow_offsets.reshape(-1,1) shape: {row_offsets_col.shape}")
print("\ndata + row_offsets.reshape(-1,1) (CORRECT - adds to ROWS):")
print(result_correct)

# ------------------------------------------------------------------------------
# 9.2 Broadcasting Shape Compatibility Checker
# ------------------------------------------------------------------------------
print("\n--- 9.2 Broadcasting Shape Compatibility Checker ---")

def check_broadcast_compatibility(shape_a, shape_b):
    """
    Check if two shapes are broadcast-compatible and show result shape.
    Useful for debugging before running expensive operations.
    """
    # Pad shorter shape with 1s on the left
    max_dims = max(len(shape_a), len(shape_b))
    a_padded = (1,) * (max_dims - len(shape_a)) + tuple(shape_a)
    b_padded = (1,) * (max_dims - len(shape_b)) + tuple(shape_b)

    result_shape = []
    compatible   = True

    for a_dim, b_dim in zip(a_padded, b_padded):
        if a_dim == b_dim:
            result_shape.append(a_dim)
        elif a_dim == 1:
            result_shape.append(b_dim)
        elif b_dim == 1:
            result_shape.append(a_dim)
        else:
            compatible = False
            break

    if compatible:
        print(f"  {shape_a} + {shape_b} -> {tuple(result_shape)}  [OK]")
    else:
        print(f"  {shape_a} + {shape_b} -> INCOMPATIBLE  [ERR]")

print("Broadcasting compatibility check:")
check_broadcast_compatibility((500, 6), (6,))
check_broadcast_compatibility((500, 6), (500, 1))
check_broadcast_compatibility((500, 6), (6, 1))
check_broadcast_compatibility((500, 6), (500,))
check_broadcast_compatibility((3, 1, 4), (3, 4))


# ==============================================================================
# SECTION 10: BUILDING A DEFENSIVE DATA PIPELINE
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 10: BUILDING A DEFENSIVE DATA PIPELINE")
print("=" * 70)

print("\nApplying all debugging concepts to clean the air quality dataset...")

# ------------------------------------------------------------------------------
# Step 1: Validate input
# ------------------------------------------------------------------------------
def validate_input(data, col_names):
    """Validate raw data before processing."""
    print("\n[Step 1] Validating input...")

    assert isinstance(data, np.ndarray), "Data must be a NumPy array"
    assert data.ndim == 2, f"Expected 2D array, got {data.ndim}D"
    assert data.shape[1] == len(col_names), (
        f"Expected {len(col_names)} columns, got {data.shape[1]}"
    )
    assert np.issubdtype(data.dtype, np.floating), (
        f"Expected float dtype, got {data.dtype}"
    )

    print(f"  Shape: {data.shape} [OK]")
    print(f"  Dtype: {data.dtype} [OK]")
    print(f"  Columns: {col_names} [OK]")
    return True


# ------------------------------------------------------------------------------
# Step 2: Report data quality
# ------------------------------------------------------------------------------
def report_data_quality(data, col_names):
    """Report issues found in data."""
    print("\n[Step 2] Data quality report...")

    n_rows, n_cols = data.shape
    issues_found   = 0

    for i, col in enumerate(col_names):
        col_data  = data[:, i]
        nan_count = np.isnan(col_data).sum()
        inf_count = np.isinf(col_data).sum()

        if nan_count > 0 or inf_count > 0:
            print(f"  {col}: {nan_count} NaNs, {inf_count} Infs")
            issues_found += nan_count + inf_count

    if issues_found == 0:
        print("  No NaN or Inf values found [OK]")
    else:
        print(f"  Total issues found: {issues_found}")

    return issues_found


# ------------------------------------------------------------------------------
# Step 3: Remove physically impossible values
# ------------------------------------------------------------------------------
def remove_impossible_values(data):
    """
    Replace physically impossible sensor readings with NaN.
    Each sensor has known valid ranges.
    """
    print("\n[Step 3] Removing impossible values...")

    result = data.copy()  # Always work on a copy

    # Valid ranges per column
    # [PM2.5, PM10, NO2, CO, Temperature, Humidity]
    valid_ranges = [
        (0, 500),     # PM2.5: ?g/m?
        (0, 600),     # PM10: ?g/m?
        (0, 300),     # NO2: ?g/m?
        (0, 50),      # CO: mg/m?
        (-50, 60),    # Temperature: ?C
        (0, 100),     # Humidity: %
    ]

    total_replaced = 0

    for i, (low, high) in enumerate(valid_ranges):
        col         = result[:, i]
        invalid     = (col < low) | (col > high)
        count       = invalid.sum()
        if count > 0:
            result[invalid, i] = np.nan
            print(f"  Col {COL_NAMES[i]}: {count} values outside [{low}, {high}] -> NaN")
            total_replaced += count

    print(f"  Total replaced with NaN: {total_replaced}")
    return result


# ------------------------------------------------------------------------------
# Step 4: Fill remaining NaN values
# ------------------------------------------------------------------------------
def fill_nan_values(data):
    """Fill NaN values with column medians."""
    print("\n[Step 4] Filling NaN values with column medians...")

    result = data.copy()

    for i in range(data.shape[1]):
        col      = result[:, i]
        nan_mask = np.isnan(col)
        n_nan    = nan_mask.sum()

        if n_nan > 0:
            median_val    = np.nanmedian(col)
            col[nan_mask] = median_val
            print(
                f"  Col {COL_NAMES[i]}: filled {n_nan} NaNs "
                f"with median {median_val:.2f}"
            )

    return result


# ------------------------------------------------------------------------------
# Step 5: Validate output
# ------------------------------------------------------------------------------
def validate_output(data, original_shape):
    """Validate cleaned data meets all requirements."""
    print("\n[Step 5] Validating output...")

    # Shape preserved
    assert data.shape == original_shape, (
        f"Shape changed: {original_shape} -> {data.shape}"
    )
    print(f"  Shape preserved: {data.shape} [OK]")

    # No NaN or Inf
    assert not np.any(np.isnan(data)), "NaN values remain!"
    assert not np.any(np.isinf(data)), "Inf values remain!"
    print("  No NaN values [OK]")
    print("  No Inf values [OK]")

    # Valid ranges
    valid_ranges = [
        (0, 500), (0, 600), (0, 300),
        (0, 50), (-50, 60), (0, 100)
    ]
    for i, (low, high) in enumerate(valid_ranges):
        col = data[:, i]
        assert np.all(col >= low) and np.all(col <= high), (
            f"Column {COL_NAMES[i]} has out-of-range values!"
        )
    print("  All values within valid ranges [OK]")

    return True


# ------------------------------------------------------------------------------
# Run the full pipeline
# ------------------------------------------------------------------------------
print("\n" + "-" * 50)
print("Running defensive data pipeline...")
print("-" * 50)

original_shape = raw_data.shape

validate_input(raw_data, COL_NAMES)
issues = report_data_quality(raw_data, COL_NAMES)
step3  = remove_impossible_values(raw_data)
step4  = fill_nan_values(step3)
validate_output(step4, original_shape)

clean_data = step4

print("\n" + "-" * 50)
print("Pipeline complete!")
print(f"Input shape:  {raw_data.shape}")
print(f"Output shape: {clean_data.shape}")
print(f"Issues fixed: {issues + int(np.isnan(step3).sum())}")
print("-" * 50)


# ==============================================================================
# SECTION 11: DEBUGGING CHEAT SHEET
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 11: DEBUGGING CHEAT SHEET")
print("=" * 70)

cheat_sheet = """
INSPECT:
  arr.shape          -> dimensions
  arr.dtype          -> data type
  arr.ndim           -> number of dimensions
  arr.size           -> total elements
  arr.flags          -> memory layout info
  arr.base           -> None if owns data, else parent array

DETECT ISSUES:
  np.isnan(arr)      -> boolean mask of NaN locations
  np.isinf(arr)      -> boolean mask of Inf locations
  np.isfinite(arr)   -> boolean mask of finite values
  np.any(np.isnan(arr))     -> True if any NaN exists
  np.where(np.isnan(arr))   -> indices of NaN values

NaN-SAFE OPERATIONS:
  np.nansum()   np.nanmean()   np.nanstd()
  np.nanmin()   np.nanmax()    np.nanmedian()

FIX VALUES:
  np.nan_to_num(arr, nan=0, posinf=0, neginf=0)
  np.clip(arr, a_min=0, a_max=100)
  arr[np.isnan(arr)] = fill_value

SHAPE TOOLS:
  arr.reshape(rows, cols)     -> reshape (error if size mismatch)
  arr.reshape(rows, -1)       -> auto-infer last dim
  arr.squeeze()               -> remove size-1 dims
  arr[:, np.newaxis]          -> add new axis
  np.broadcast_shapes(s1, s2) -> check broadcast result

COMPARE FLOATS:
  np.isclose(a, b)              -> element-wise
  np.allclose(arr1, arr2)       -> all elements
  np.array_equal(arr1, arr2)    -> exact equality

MEMORY:
  arr.copy()                    -> force independent copy
  np.shares_memory(a, b)        -> check if sharing memory
  np.ascontiguousarray(arr)     -> force C-contiguous layout

ASSERTIONS:
  assert arr.shape == (100, 6), f"Got {arr.shape}"
  assert not np.any(np.isnan(arr)), "NaN found!"
  assert arr.dtype == np.float64, f"Wrong dtype: {arr.dtype}"
"""
print(cheat_sheet)


# ==============================================================================
# SECTION 12: PRACTICE EXERCISES
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 12: PRACTICE EXERCISES")
print("=" * 70)

print("\n--- Exercise 1: Find and Fix the Bug ---")

buggy_data = np.array([
    [1.0, 2.0, 3.0],
    [4.0, np.nan, 6.0],
    [7.0, 8.0, 9.0]
])

# Bug: using .mean() without axis or nan handling
buggy_col_means = buggy_data.mean()          # WRONG
correct_col_means = np.nanmean(buggy_data, axis=0)  # CORRECT

print(f"Buggy col means:   {buggy_col_means}")
print(f"Correct col means: {correct_col_means}")

print("\n--- Exercise 2: Diagnose Shape Issue ---")

matrix = np.random.rand(50, 4)
scales = np.array([10, 100, 1000, 10000])

try:
    wrong = matrix * scales.reshape(4, 1)   # Shape (4,1) ? crashes: (50,4)*(4,1) not broadcastable
    print(f"wrong result shape:   {wrong.shape}   <- silent broadcast error")
except ValueError as e:
    print(f"Shape mismatch error (expected): {e}")

correct = matrix * scales                  # Broadcasting (50,4)*(4,)

print(f"matrix shape: {matrix.shape}")
print(f"correct result shape: {correct.shape} <- intended")

print("\n--- Exercise 3: Overflow Detection ---")

# Newer NumPy raises OverflowError instead of silently wrapping,
# so we cast first to show the overflow concept, then use safe int32.
counts_raw = np.array([100, 200, 255, 300])              # Python int, no dtype yet
counts = counts_raw.astype(np.uint8)                     # wraps 300 -> 44
wrapped_300 = counts[-1]  # what 300 becomes after uint8 wrapping
print(f"uint8 array (300 wraps to {wrapped_300}): {counts}")
print(f"After +10: {counts + np.uint8(10)}")             # 255+10 wraps silently
safe = counts_raw.astype(np.int32) + 10
print(f"Safe (int32): {safe}")


# ==============================================================================
# NEXT LESSON PREVIEW
# ==============================================================================
print("\n" + "=" * 70)
print("NEXT LESSON PREVIEW")
print("=" * 70)

preview = """
Lesson 17 ? Transition to Pandas: Why Pandas Over NumPy for Real Data

You will learn:
- Where NumPy stops being convenient
- What Pandas adds on top of NumPy
- Series and DataFrame as labeled arrays
- Why labels matter in real-world data
- Loading a real CSV dataset into Pandas
- Comparing NumPy array operations vs Pandas equivalents

This is the bridge lesson into the Pandas world!
"""
print(preview)

print("\n" + "=" * 70)
print("END OF LESSON 16")
print("=" * 70)
print("\nYou can now IDENTIFY, DIAGNOSE, and FIX the most common NumPy bugs!")
print("Always validate shapes, dtypes, and NaN presence at every pipeline step! [rocket]")