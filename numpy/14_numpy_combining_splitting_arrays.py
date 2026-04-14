"""
================================================================================
LESSON 14: COMBINING & SPLITTING ARRAYS
================================================================================

What You Will Learn:
- How to combine multiple arrays into one (concatenate, stack, join)
- How to split arrays into multiple parts
- When to use each combining/splitting method
- Real-world data merging scenarios
- Memory and performance implications

Real-World Usage:
- Batch Processing: Combining results from parallel computations
- Data Collection: Merging sensor readings from different time periods
- Feature Engineering: Combining different feature sets for machine learning
- Report Generation: Splitting data for different departments/regions
- Time Series: Merging historical and new data

================================================================================
"""

import numpy as np
import time

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 70)
print("LESSON 14: COMBINING & SPLITTING ARRAYS")
print("=" * 70)


# ==============================================================================
# SECTION 1: DATASET PREPARATION
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 1: DATASET PREPARATION")
print("=" * 70)

# Real climate data pattern: Monthly temperature readings from weather stations
# Station 1: Coastal city (moderate temperatures)
station1_temps = np.array([15.2, 16.1, 18.3, 21.2, 24.5, 27.8, 
                           29.1, 28.9, 26.4, 22.7, 18.9, 16.3])

# Station 2: Inland city (more extreme temperatures)
station2_temps = np.array([8.1, 10.2, 14.5, 19.8, 25.3, 30.2, 
                           33.4, 32.8, 27.9, 21.3, 14.7, 9.8])

# Station 3: Mountain region (cooler temperatures)
station3_temps = np.array([2.3, 3.8, 6.9, 11.2, 15.8, 19.4, 
                           21.7, 21.2, 17.8, 13.1, 7.4, 3.6])

print("\nWeather Station Temperature Data (°C):")
print("-" * 50)
print(f"Station 1 (Coastal):  {station1_temps}")
print(f"Station 2 (Inland):   {station2_temps}")
print(f"Station 3 (Mountain): {station3_temps}")
print(f"\nEach station has {len(station1_temps)} monthly readings")


# ==============================================================================
# SECTION 2: CONCATENATION - JOINING ARRAYS END-TO-END
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 2: CONCATENATION - JOINING ARRAYS END-TO-END")
print("=" * 70)

# ------------------------------------------------------------------------------
# 2.1 Concatenating 1D Arrays
# ------------------------------------------------------------------------------
print("\n--- 2.1 Concatenating 1D Arrays ---")

# Scenario: Merge temperature data from Q1 and Q2
q1_temps = np.array([15.2, 16.1, 18.3])  # Jan, Feb, Mar
q2_temps = np.array([21.2, 24.5, 27.8])  # Apr, May, Jun

# Combine into half-year data using concatenate
half_year = np.concatenate([q1_temps, q2_temps])

print(f"Q1 temperatures: {q1_temps}")
print(f"Q2 temperatures: {q2_temps}")
print(f"Half-year combined: {half_year}")
print(f"Shape: {half_year.shape}")
print("\nKey Point: For 1D arrays, concatenation simply extends the array")

# ------------------------------------------------------------------------------
# 2.2 Concatenating 2D Arrays - Along Rows (axis=0)
# ------------------------------------------------------------------------------
print("\n--- 2.2 Concatenating 2D Arrays - Along Rows (axis=0) ---")

# Each station has 12 monthly readings
# Create a dataset with all stations (rows = stations, columns = months)
station1_2d = station1_temps.reshape(1, 12)  # Shape: (1, 12)
station2_2d = station2_temps.reshape(1, 12)
station3_2d = station3_temps.reshape(1, 12)

# Concatenate along axis=0 (stack stations as rows)
all_stations = np.concatenate([station1_2d, station2_2d, station3_2d], axis=0)

print(f"Individual station shape: {station1_2d.shape}")
print(f"\nAll stations combined (axis=0):")
print(all_stations)
print(f"Shape: {all_stations.shape}")
print(f"\nInterpretation: {all_stations.shape[0]} stations × {all_stations.shape[1]} months")
print("\nReal-World Use: Combining data from multiple sensors/sources into a single matrix")

# ------------------------------------------------------------------------------
# 2.3 Concatenating 2D Arrays - Along Columns (axis=1)
# ------------------------------------------------------------------------------
print("\n--- 2.3 Concatenating 2D Arrays - Along Columns (axis=1) ---")

# Scenario: Add next year's data as additional columns
year1 = all_stations.copy()  # Shape: (3, 12)

# Simulate next year data (slight warming trend)
np.random.seed(42)
year2 = year1 + np.random.normal(0.5, 1.0, year1.shape)

# Concatenate along axis=1 (add months horizontally)
two_years = np.concatenate([year1, year2], axis=1)

print(f"Year 1 shape: {year1.shape}")
print(f"Year 2 shape: {year2.shape}")
print(f"\nTwo years combined (axis=1):")
print(f"Shape: {two_years.shape}")
print(f"\nInterpretation: {two_years.shape[0]} stations × {two_years.shape[1]} months (2 years)")
print("\nReal-World Use: Appending new time periods to existing datasets")

# ------------------------------------------------------------------------------
# 2.4 Concatenating Multiple Arrays at Once
# ------------------------------------------------------------------------------
print("\n--- 2.4 Concatenating Multiple Arrays at Once ---")

# Concatenate more than 2 arrays in single operation
q3_temps = np.array([29.1, 28.9, 26.4])  # Jul, Aug, Sep
q4_temps = np.array([22.7, 18.9, 16.3])  # Oct, Nov, Dec

# All quarters in one operation
full_year = np.concatenate([q1_temps, q2_temps, q3_temps, q4_temps])

print(f"Q1: {q1_temps}")
print(f"Q2: {q2_temps}")
print(f"Q3: {q3_temps}")
print(f"Q4: {q4_temps}")
print(f"\nFull year (single concatenation): {full_year}")
print(f"Shape: {full_year.shape}")


# ==============================================================================
# SECTION 3: STACKING - CREATING NEW DIMENSIONS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 3: STACKING - CREATING NEW DIMENSIONS")
print("=" * 70)

# ------------------------------------------------------------------------------
# 3.1 Vertical Stack (vstack) - Stack Rows
# ------------------------------------------------------------------------------
print("\n--- 3.1 Vertical Stack (vstack) - Stack Rows ---")

# vstack is shorthand for stacking 1D arrays as rows in 2D array
temps_vstack = np.vstack([station1_temps, station2_temps, station3_temps])

print("Using vstack on three 1D arrays:")
print(temps_vstack)
print(f"Shape: {temps_vstack.shape}")
print("\nWhen to Use: Stacking 1D arrays into a 2D matrix (common in data collection)")

# Verify equivalence with concatenate
temps_concat = np.concatenate([station1_temps.reshape(1, -1), 
                                station2_temps.reshape(1, -1), 
                                station3_temps.reshape(1, -1)], axis=0)
print(f"\nEquivalent to concatenate with reshape: {np.array_equal(temps_vstack, temps_concat)}")

# ------------------------------------------------------------------------------
# 3.2 Horizontal Stack (hstack) - Stack Columns
# ------------------------------------------------------------------------------
print("\n--- 3.2 Horizontal Stack (hstack) - Stack Columns ---")

# Scenario: Combine temperature and humidity readings side-by-side
jan_temps = np.array([15.2, 8.1, 2.3])      # 3 stations
jan_humidity = np.array([72, 45, 88])        # % humidity

# Stack horizontally - need to reshape to column vectors first
jan_data = np.hstack([jan_temps.reshape(-1, 1), 
                      jan_humidity.reshape(-1, 1)])

print(f"January temperatures (3 stations): {jan_temps}")
print(f"January humidity (3 stations): {jan_humidity}")
print(f"\nCombined (temp, humidity) for each station:")
print(jan_data)
print(f"Shape: {jan_data.shape}")
print(f"\nInterpretation: {jan_data.shape[0]} stations × {jan_data.shape[1]} variables")
print("\nReal-World Use: Combining different measurements for the same entities")

# ------------------------------------------------------------------------------
# 3.3 Depth Stack (dstack) - Stack Along 3rd Dimension
# ------------------------------------------------------------------------------
print("\n--- 3.3 Depth Stack (dstack) - Stack Along 3rd Dimension ---")

# Scenario: Stack daily readings over multiple days
# Day 1 readings (3 stations, 4 hourly readings)
day1 = np.array([[15, 17, 19, 18],
                 [8, 11, 14, 12],
                 [2, 4, 6, 5]])

# Day 2 readings
day2 = np.array([[16, 18, 20, 19],
                 [9, 12, 15, 13],
                 [3, 5, 7, 6]])

# Stack along depth (3rd dimension)
multi_day = np.dstack([day1, day2])

print(f"Day 1 shape: {day1.shape}")
print(f"Day 2 shape: {day2.shape}")
print(f"\nStacked shape: {multi_day.shape}")
print(f"Interpretation: {multi_day.shape[0]} stations × {multi_day.shape[1]} hours × {multi_day.shape[2]} days")
print(f"\nAccess Station 1, Hour 3, Day 2: {multi_day[0, 2, 1]}")
print("\nReal-World Use: Time series with multiple dimensions (sensors × time × days)")

# ------------------------------------------------------------------------------
# 3.4 General Stack (np.stack) - Stack Along Any Axis
# ------------------------------------------------------------------------------
print("\n--- 3.4 General Stack (np.stack) - Stack Along Any Axis ---")

# Most flexible stacking function - creates NEW axis at specified position

print(f"Original array shape: {station1_temps.shape}")

# New axis at position 0 (stations become 1st dimension)
stack_axis0 = np.stack([station1_temps, station2_temps, station3_temps], axis=0)
print(f"\nStack with axis=0:")
print(f"Shape: {stack_axis0.shape}")
print("Interpretation: (stations, months)")

# New axis at position 1 (months become 1st dimension, stations 2nd)
stack_axis1 = np.stack([station1_temps, station2_temps, station3_temps], axis=1)
print(f"\nStack with axis=1:")
print(f"Shape: {stack_axis1.shape}")
print("Interpretation: (months, stations)")
print(f"\nFirst month across all stations: {stack_axis1[0]}")

print("\n" + "-" * 50)
print("KEY DIFFERENCE:")
print("  np.stack: Creates a NEW axis")
print("  np.concatenate: Uses an EXISTING axis")
print("-" * 50)


# ==============================================================================
# SECTION 4: SPLITTING ARRAYS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 4: SPLITTING ARRAYS")
print("=" * 70)

# ------------------------------------------------------------------------------
# 4.1 Split into Equal Parts (np.split)
# ------------------------------------------------------------------------------
print("\n--- 4.1 Split into Equal Parts (np.split) ---")

# Scenario: Split year into quarters for quarterly reports
full_year_temps = station1_temps.copy()  # 12 months

# Split into 4 equal parts
quarters = np.split(full_year_temps, 4)

print(f"Full year: {full_year_temps}")
print(f"\nQuarterly splits:")
for i, q in enumerate(quarters, 1):
    print(f"  Q{i}: {q} | Mean: {q.mean():.1f}°C")

print("\nRequirement: Array must be evenly divisible by number of splits")

# ------------------------------------------------------------------------------
# 4.2 Split at Specific Indices
# ------------------------------------------------------------------------------
print("\n--- 4.2 Split at Specific Indices ---")

# Scenario: Split into unequal periods (winter, spring/summer, fall)
# Split at indices 3 and 9 (creates 3 parts)
periods = np.split(full_year_temps, [3, 9])

print(f"Full year: {full_year_temps}")
print(f"Split at indices [3, 9]:")
print(f"  Winter (Jan-Mar):        {periods[0]}")
print(f"  Spring/Summer (Apr-Sep): {periods[1]}")
print(f"  Fall (Oct-Dec):          {periods[2]}")
print("\nKey Point: Indices specify WHERE to cut, not how many pieces")

# ------------------------------------------------------------------------------
# 4.3 Array Split (np.array_split) - Handles Uneven Divisions
# ------------------------------------------------------------------------------
print("\n--- 4.3 Array Split (np.array_split) - Handles Uneven Divisions ---")

# Scenario: Split 12 months into 5 parts (not evenly divisible)
# np.split would fail here, np.array_split handles it gracefully

parts = np.array_split(full_year_temps, 5)

print(f"Split 12 months into 5 parts:")
for i, part in enumerate(parts, 1):
    print(f"  Part {i}: {part} | Size: {len(part)}")

# Demonstrate that np.split fails with uneven division
print("\nDemonstrating np.split failure with uneven division:")
try:
    wrong = np.split(full_year_temps, 5)  # 12 not divisible by 5
except ValueError as e:
    print(f"  Error: {e}")

print("\nAdvantage of array_split: Doesn't raise errors for uneven splits")

# ------------------------------------------------------------------------------
# 4.4 Splitting 2D Arrays
# ------------------------------------------------------------------------------
print("\n--- 4.4 Splitting 2D Arrays ---")

print(f"Original all_stations shape: {all_stations.shape}")
print(all_stations)

# Split by rows (separate each station)
station_splits = np.split(all_stations, 3, axis=0)

print(f"\nSplit by stations (axis=0):")
for i, station in enumerate(station_splits, 1):
    print(f"  Station {i}: {station.flatten()}")

# Split by columns (split year into halves)
half_year_splits = np.split(all_stations, 2, axis=1)

print(f"\nSplit by time (axis=1):")
print(f"  First half (Jan-Jun) shape: {half_year_splits[0].shape}")
print(f"  Second half (Jul-Dec) shape: {half_year_splits[1].shape}")

# ------------------------------------------------------------------------------
# 4.5 Vertical and Horizontal Split
# ------------------------------------------------------------------------------
print("\n--- 4.5 Vertical and Horizontal Split ---")

# Shorthand functions for axis-specific splits

# vsplit: split along axis=0 (rows)
two_stations = all_stations[:2]  # First 2 stations
top_bottom = np.vsplit(two_stations, 2)

print("Vertical split (stations):")
print(f"  Top: {top_bottom[0].flatten()[:6]}...")  # Show first 6 values
print(f"  Bottom: {top_bottom[1].flatten()[:6]}...")

# hsplit: split along axis=1 (columns)
left_right = np.hsplit(all_stations, 2)

print(f"\nHorizontal split (months):")
print(f"  Left half shape: {left_right[0].shape}")
print(f"  Right half shape: {left_right[1].shape}")


# ==============================================================================
# SECTION 5: REAL-WORLD EXAMPLE - BATCH PROCESSING SENSOR DATA
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 5: REAL-WORLD EXAMPLE - BATCH PROCESSING SENSOR DATA")
print("=" * 70)

# Scenario: Process temperature data from 10 sensors in batches
np.random.seed(100)

# 10 sensors, each with 1000 hourly readings
sensor_data = np.random.normal(20, 5, (10, 1000))  # Mean 20°C, std 5°C

print(f"Original data shape: {sensor_data.shape}")
print(f"Total readings: {sensor_data.size}")

# Split into 5 batches for parallel processing
batches = np.array_split(sensor_data, 5, axis=1)

print(f"\nSplit into {len(batches)} batches:")
for i, batch in enumerate(batches, 1):
    print(f"  Batch {i}: {batch.shape}")

# Process each batch (simulate: calculate anomalies)
print("\nProcessing batches (detecting anomalies):")
processed_batches = []
total_anomalies = 0

for i, batch in enumerate(batches):
    # Flag values beyond 2 standard deviations
    mean = batch.mean()
    std = batch.std()
    
    anomalies = np.abs(batch - mean) > 2 * std
    anomaly_count = anomalies.sum()
    total_anomalies += anomaly_count
    
    print(f"  Batch {i+1}: {anomaly_count} anomalies detected")
    processed_batches.append(batch)  # In real scenario, might be cleaned data

# Recombine processed batches
final_data = np.concatenate(processed_batches, axis=1)

print(f"\nTotal anomalies across all batches: {total_anomalies}")
print("\n=== VALIDATION ===")
print(f"Original shape: {sensor_data.shape}")
print(f"Final shape: {final_data.shape}")
print(f"Shapes match: {sensor_data.shape == final_data.shape}")
print(f"Data preserved: {np.allclose(sensor_data, final_data)}")

print("\nReal-World Use: Processing large datasets in chunks to avoid memory issues")


# ==============================================================================
# SECTION 6: REAL-WORLD EXAMPLE - MERGING REGIONAL SALES DATA
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 6: REAL-WORLD EXAMPLE - MERGING REGIONAL SALES DATA")
print("=" * 70)

# Scenario: Combine quarterly sales from different regions
np.random.seed(50)

# Sales data: Regions × Products (5 products, 4 quarters)
north_sales = np.random.randint(100, 500, (5, 4))
south_sales = np.random.randint(80, 450, (5, 4))
east_sales = np.random.randint(90, 480, (5, 4))
west_sales = np.random.randint(110, 520, (5, 4))

print("North region sales (5 products × 4 quarters):")
print(north_sales)

# Combine all regions (stack as new dimension)
all_regions = np.stack([north_sales, south_sales, east_sales, west_sales], axis=0)

print(f"\nCombined shape: {all_regions.shape}")
print("Interpretation: (4 regions, 5 products, 4 quarters)")

# Calculate total sales per region
region_totals = all_regions.sum(axis=(1, 2))  # Sum over products and quarters
regions = ['North', 'South', 'East', 'West']

print("\n=== REGIONAL TOTALS ===")
for region, total in zip(regions, region_totals):
    print(f"  {region}: ${total:,}")

# Calculate product performance across all regions
product_totals = all_regions.sum(axis=(0, 2))  # Sum over regions and quarters

print("\n=== PRODUCT PERFORMANCE ===")
for i, total in enumerate(product_totals, 1):
    print(f"  Product {i}: ${total:,}")

# Extract Q4 data across all regions for year-end report
q4_data = all_regions[:, :, 3]  # All regions, all products, quarter 4

print("\n=== Q4 SALES BY REGION ===")
print(f"Shape: {q4_data.shape}")
for i, region in enumerate(regions):
    print(f"  {region}: {q4_data[i]} | Total: ${q4_data[i].sum():,}")

# Find best performing region-product combination
max_idx = np.unravel_index(all_regions.argmax(), all_regions.shape)
print(f"\nBest single quarter sale:")
print(f"  Region: {regions[max_idx[0]]}, Product: {max_idx[1]+1}, Quarter: Q{max_idx[2]+1}")
print(f"  Value: ${all_regions[max_idx]:,}")

print("\nReal-World Application: Aggregating business data from different sources/timeframes")


# ==============================================================================
# SECTION 7: PERFORMANCE CONSIDERATIONS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 7: PERFORMANCE CONSIDERATIONS")
print("=" * 70)

# Compare concatenation methods for different array sizes
print("\nComparing concatenation methods:")
print("-" * 50)

sizes = [1000, 10000, 50000]

for size in sizes:
    # Generate test data: 100 arrays of given size
    arrays = [np.random.rand(size) for _ in range(100)]
    
    # Method 1: Repeated concatenation (SLOW - creates intermediate copies)
    start = time.time()
    result1 = arrays[0].copy()
    for arr in arrays[1:]:
        result1 = np.concatenate([result1, arr])
    time1 = time.time() - start
    
    # Method 2: Single concatenation (FAST - one operation)
    start = time.time()
    result2 = np.concatenate(arrays)
    time2 = time.time() - start
    
    # Validate both methods produce same result
    assert np.allclose(result1, result2), "Results don't match!"
    
    print(f"\nArray size: {size:,}")
    print(f"  Repeated concatenation: {time1:.4f}s")
    print(f"  Single concatenation:   {time2:.4f}s")
    print(f"  Speedup: {time1/time2:.1f}x")

print("\n" + "-" * 50)
print("PERFORMANCE RULE: Always prefer single concatenation over repeated operations")
print("-" * 50)

# Memory consideration example
print("\n\nMemory Considerations:")
print("-" * 50)

# Show memory usage
large_array = np.random.rand(1000, 1000)
print(f"Large array shape: {large_array.shape}")
print(f"Memory usage: {large_array.nbytes / 1024 / 1024:.2f} MB")

# Splitting creates views (memory efficient)
split_arrays = np.split(large_array, 4, axis=0)
print(f"\nAfter splitting into 4 parts:")
for i, arr in enumerate(split_arrays):
    # Check if it's a view (shares memory with original)
    is_view = arr.base is not None
    print(f"  Part {i+1}: shape {arr.shape}, is_view: {is_view}")


# ==============================================================================
# SECTION 8: COMMON MISTAKES AND HOW TO AVOID THEM
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 8: COMMON MISTAKES AND HOW TO AVOID THEM")
print("=" * 70)

# ------------------------------------------------------------------------------
# Mistake 1: Wrong Axis for Concatenation
# ------------------------------------------------------------------------------
print("\n--- Mistake 1: Wrong Axis for Concatenation ---")

arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5, 6]])

print(f"arr1 shape: {arr1.shape}")
print(f"arr2 shape: {arr2.shape}")

# This will fail - incompatible shapes for axis=1
print("\nAttempting concatenate with axis=1:")
try:
    wrong = np.concatenate([arr1, arr2], axis=1)
except ValueError as e:
    print(f"  Error: {e}")

# Correct: use axis=0 for row-wise concatenation
correct = np.concatenate([arr1, arr2], axis=0)
print(f"\nCorrect result (axis=0):")
print(correct)

# ------------------------------------------------------------------------------
# Mistake 2: Confusing Stack and Concatenate
# ------------------------------------------------------------------------------
print("\n--- Mistake 2: Confusing Stack and Concatenate ---")

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Concatenate: extends along existing axis
concat_result = np.concatenate([a, b])
print(f"Concatenate: {concat_result} | Shape: {concat_result.shape}")

# Stack: creates new axis
stack_result = np.stack([a, b])
print(f"Stack:\n{stack_result}")
print(f"Shape: {stack_result.shape}")

print("\nRule: Use concatenate for same-dimension joining, stack for adding dimensions")

# ------------------------------------------------------------------------------
# Mistake 3: Uneven Split with np.split
# ------------------------------------------------------------------------------
print("\n--- Mistake 3: Uneven Split with np.split ---")

arr = np.arange(10)
print(f"Array: {arr}")

# This will fail
print("\nAttempting np.split(arr, 3):")
try:
    wrong = np.split(arr, 3)  # 10 not divisible by 3
except ValueError as e:
    print(f"  Error: {e}")

# Use array_split instead
correct = np.array_split(arr, 3)
print("\nUsing array_split instead:")
for i, part in enumerate(correct):
    print(f"  Part {i}: {part}")

# ------------------------------------------------------------------------------
# Mistake 4: Forgetting to Reshape Before hstack/vstack
# ------------------------------------------------------------------------------
print("\n--- Mistake 4: Forgetting to Reshape Before hstack ---")

temps = np.array([20, 25, 30])
humidity = np.array([50, 60, 70])

# Wrong: hstack with 1D arrays just concatenates them
wrong_hstack = np.hstack([temps, humidity])
print(f"Wrong (1D hstack): {wrong_hstack} | Shape: {wrong_hstack.shape}")

# Correct: reshape to column vectors first
correct_hstack = np.hstack([temps.reshape(-1, 1), humidity.reshape(-1, 1)])
print(f"Correct (reshaped):\n{correct_hstack}")
print(f"Shape: {correct_hstack.shape}")


# ==============================================================================
# SECTION 9: SUMMARY TABLE
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 9: SUMMARY TABLE")
print("=" * 70)

summary = """
| Operation         | Function            | Use Case                        | Creates New Axis? |
|-------------------|---------------------|----------------------------------|-------------------|
| Concatenate       | np.concatenate()    | Join along existing axis         | No                |
| Vertical Stack    | np.vstack()         | Stack rows (1D → 2D)             | No (just axis=0)  |
| Horizontal Stack  | np.hstack()         | Stack columns                    | No (just axis=1)  |
| Depth Stack       | np.dstack()         | Stack along 3rd dimension        | Yes               |
| General Stack     | np.stack()          | Stack along any new axis         | Yes               |
| Split             | np.split()          | Equal divisions (strict)         | No                |
| Array Split       | np.array_split()    | Uneven divisions allowed         | No                |
| Vertical Split    | np.vsplit()         | Split rows                       | No                |
| Horizontal Split  | np.hsplit()         | Split columns                    | No                |
"""
print(summary)


# ==============================================================================
# SECTION 10: KEY TAKEAWAYS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 10: KEY TAKEAWAYS")
print("=" * 70)

takeaways = """
✅ Use CONCATENATE for merging existing data along same dimension
✅ Use STACK for creating new dimensional structure  
✅ Use SPLIT for dividing data into parts
✅ Use ARRAY_SPLIT for uneven divisions (more flexible)
✅ Always VALIDATE shapes before/after operations
✅ Single concatenation >> repeated concatenations (performance)
✅ Choose correct AXIS based on data organization
✅ Splits often create VIEWS (memory efficient)
✅ RESHAPE arrays before hstack if combining 1D as columns
"""
print(takeaways)


# ==============================================================================
# SECTION 11: PRACTICE EXERCISES (WITH SOLUTIONS)
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 11: PRACTICE EXERCISES")
print("=" * 70)

print("\n--- Exercise 1: Combine Weekly Sales Data ---")
# Three stores, each with weekly sales for 4 weeks
store_a = np.array([1200, 1350, 1100, 1400])
store_b = np.array([980, 1050, 1200, 1100])
store_c = np.array([1500, 1600, 1450, 1550])

# Task: Create a 2D array with stores as rows, weeks as columns
sales_matrix = np.vstack([store_a, store_b, store_c])
print("Sales matrix (stores × weeks):")
print(sales_matrix)
print(f"Total sales per store: {sales_matrix.sum(axis=1)}")

print("\n--- Exercise 2: Split Data for Train/Test ---")
# 100 samples of data
np.random.seed(42)
data = np.random.rand(100, 5)  # 100 samples, 5 features

# Task: Split into 80% train, 20% test
train, test = np.split(data, [80], axis=0)
print(f"Original data shape: {data.shape}")
print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

print("\n--- Exercise 3: Combine and Validate ---")
# Recombine and verify
recombined = np.concatenate([train, test], axis=0)
print(f"Recombined shape: {recombined.shape}")
print(f"Data preserved: {np.array_equal(data, recombined)}")


# ==============================================================================
# NEXT LESSON PREVIEW
# ==============================================================================
print("\n" + "=" * 70)
print("NEXT LESSON PREVIEW")
print("=" * 70)

preview = """
Lesson 15 — Why NumPy is Fast (Performance Deep Dive)

You will learn:
- Memory layout and contiguity (C-order vs F-order)
- Vectorization internals
- Cache efficiency and memory access patterns
- When NumPy is slower than Python
- Profiling NumPy code
- Optimization strategies for real-world applications

This lesson will help you write faster, more efficient NumPy code!
"""
print(preview)

print("\n" + "=" * 70)
print("END OF LESSON 14")
print("=" * 70)
print("\nYou now understand how to COMBINE and SPLIT arrays — essential for")
print("data merging, batch processing, and pipeline construction! 🚀")