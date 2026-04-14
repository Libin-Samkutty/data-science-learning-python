import numpy as np

# ============================================================
# PART 1: Basic Indexing (1D Arrays)
# ============================================================

print("="*60)
print("PART 1: BASIC INDEXING (1D Arrays)")
print("="*60)

# Create sample data: daily temperatures for a week (°C)
temperatures = np.array([22, 24, 19, 25, 23, 26, 21])
print("Weekly temperatures (°C):", temperatures)

# Access single elements (zero-indexed)
print(f"\nMonday (index 0): {temperatures[0]}°C")
print(f"Wednesday (index 2): {temperatures[2]}°C")
print(f"Sunday (index 6): {temperatures[6]}°C")

# Negative indexing (count from end)
print(f"\nLast day (index -1): {temperatures[-1]}°C")
print(f"Second-to-last (index -2): {temperatures[-2]}°C")

# Modify elements through indexing
temperatures[0] = 23  # Update Monday's temperature
print(f"\nAfter updating Monday: {temperatures}")

# ============================================================
# PART 2: Slicing (1D Arrays)
# ============================================================

print("\n" + "="*60)
print("PART 2: SLICING (1D Arrays)")
print("="*60)

# Create sample data: daily sales for 10 days
np.random.seed(42)
daily_sales = np.random.randint(100, 500, 10)
print("Daily sales (10 days):", daily_sales)

# Basic slicing: array[start:stop:step]
# Remember: start is INCLUSIVE, stop is EXCLUSIVE

print("\nFirst 3 days:", daily_sales[0:3])  # Indices 0, 1, 2
print("First 3 days (shorthand):", daily_sales[:3])  # Same as above

print("\nLast 3 days:", daily_sales[-3:])  # Last 3 elements
print("Days 4-7:", daily_sales[3:7])  # Indices 3, 4, 5, 6

# Step parameter
print("\nEvery other day:", daily_sales[::2])  # Indices 0, 2, 4, 6, 8
print("Every 3rd day:", daily_sales[::3])  # Indices 0, 3, 6, 9

# Reverse array
print("\nReversed:", daily_sales[::-1])

# Real-world example: Weekly aggregation
week1 = daily_sales[0:7]
week2 = daily_sales[7:10]  # Remaining days
print(f"\nWeek 1 average sales: ${week1.mean():.2f}")
print(f"Week 2 average sales: ${week2.mean():.2f}")

# ============================================================
# PART 3: Multi-Dimensional Indexing (2D Arrays)
# ============================================================

print("\n" + "="*60)
print("PART 3: MULTI-DIMENSIONAL INDEXING (2D Arrays)")
print("="*60)

# Student exam scores: 5 students × 4 subjects
# Rows = students, Columns = subjects (Math, Physics, Chemistry, Biology)
scores = np.array([
    [85, 90, 78, 92],  # Student 0
    [88, 85, 91, 87],  # Student 1
    [92, 88, 85, 90],  # Student 2
    [75, 80, 70, 85],  # Student 3
    [95, 92, 88, 94]   # Student 4
])

print("Exam Scores (5 students × 4 subjects):")
print(scores)
print(f"Shape: {scores.shape}")

# Access single element: array[row, column]
print(f"\nStudent 0, Math (row 0, col 0): {scores[0, 0]}")
print(f"Student 2, Chemistry (row 2, col 2): {scores[2, 2]}")
print(f"Last student, last subject: {scores[-1, -1]}")

# Access entire rows
print(f"\nStudent 1's all scores: {scores[1, :]}")  # Row 1, all columns
print(f"Student 1's all scores (shorthand): {scores[1]}")  # Same

# Access entire columns
print(f"\nAll students' Math scores: {scores[:, 0]}")  # All rows, column 0
print(f"All students' Biology scores: {scores[:, 3]}")  # All rows, column 3

# Calculate column statistics
math_avg = scores[:, 0].mean()
physics_avg = scores[:, 1].mean()
print(f"\nClass average in Math: {math_avg:.1f}")
print(f"Class average in Physics: {physics_avg:.1f}")

# ============================================================
# PART 4: Slicing 2D Arrays
# ============================================================

print("\n" + "="*60)
print("PART 4: SLICING 2D ARRAYS")
print("="*60)

# Regional sales: 6 months × 4 regions
sales_data = np.array([
    [50, 55, 48, 52],  # Jan
    [52, 58, 50, 54],  # Feb
    [55, 60, 53, 57],  # Mar
    [58, 62, 55, 60],  # Apr
    [60, 65, 58, 63],  # May
    [62, 68, 60, 65]   # Jun
])

print("Regional Sales (6 months × 4 regions, in $1000s):")
print(sales_data)

# First quarter (Jan-Mar), all regions
q1_sales = sales_data[0:3, :]
print("\nQ1 Sales (Jan-Mar):")
print(q1_sales)
print(f"Q1 Total: ${q1_sales.sum()}k")

# Last 3 months, first 2 regions
subset = sales_data[-3:, :2]
print("\nLast 3 months, first 2 regions:")
print(subset)

# Every other month, all regions
bimonthly = sales_data[::2, :]
print("\nBi-monthly view (Jan, Mar, May):")
print(bimonthly)

# Region 2 and 3, all months
regions_2_3 = sales_data[:, 1:3]
print("\nRegions 2 and 3 (all months):")
print(regions_2_3)

# ============================================================
# PART 5: Boolean Indexing (Conditional Selection)
# ============================================================

print("\n" + "="*60)
print("PART 5: BOOLEAN INDEXING")
print("="*60)

# Temperature sensor data for 12 hours
np.random.seed(42)
hourly_temps = np.random.randint(15, 35, 12)
print("Hourly temperatures (°C):", hourly_temps)

# Create boolean mask
above_25 = hourly_temps > 25
print(f"\nMask (temp > 25°C): {above_25}")
print(f"Type of mask: {type(above_25)}, dtype: {above_25.dtype}")

# Use mask to filter data
hot_temps = hourly_temps[above_25]
print(f"Temperatures above 25°C: {hot_temps}")
print(f"Number of hot hours: {len(hot_temps)}")

# Direct filtering (more common in practice)
cold_temps = hourly_temps[hourly_temps < 20]
print(f"\nTemperatures below 20°C: {cold_temps}")

comfortable_temps = hourly_temps[(hourly_temps >= 20) & (hourly_temps <= 25)]
print(f"Comfortable temps (20-25°C): {comfortable_temps}")

# Real-world example: Stock price analysis
stock_prices = np.array([150, 155, 148, 162, 158, 170, 165, 172, 168, 175])
print(f"\nStock prices: {stock_prices}")

# Find days when price increased from previous day
price_changes = np.diff(stock_prices)  # Calculate differences
print(f"Daily changes: {price_changes}")

# Days with gains (excluding first day since no previous day)
gain_days = price_changes > 0
print(f"Gain days mask: {gain_days}")
print(f"Number of gain days: {gain_days.sum()}")  # True=1, False=0

# ============================================================
# PART 6: Boolean Indexing on 2D Arrays
# ============================================================

print("\n" + "="*60)
print("PART 6: BOOLEAN INDEXING ON 2D ARRAYS")
print("="*60)

# Product sales: 5 products × 4 quarters
product_sales = np.array([
    [120, 135, 128, 140],  # Product A
    [95, 88, 92, 85],      # Product B
    [150, 162, 158, 170],  # Product C
    [88, 90, 85, 87],      # Product D
    [200, 215, 208, 220]   # Product E
])

print("Product Sales (5 products × 4 quarters, in $1000s):")
print(product_sales)

# Find all sales above $150k
high_sales = product_sales > 150
print(f"\nSales above $150k (boolean mask):")
print(high_sales)

# Get actual values (returns 1D array)
high_values = product_sales[high_sales]
print(f"Actual high sales values: {high_values}")

# Count how many sales exceed threshold
print(f"Number of quarters with sales > $150k: {high_sales.sum()}")

# Find which products (rows) had ANY quarter above $150k
products_with_high_sales = np.any(high_sales, axis=1)
print(f"\nProducts with at least one quarter > $150k: {products_with_high_sales}")
print(f"Product indices: {np.where(products_with_high_sales)[0]}")  # Indices

# Find products where ALL quarters exceeded $100k
all_quarters_good = np.all(product_sales > 100, axis=1)
print(f"\nProducts with all quarters > $100k: {all_quarters_good}")
print(f"Product indices: {np.where(all_quarters_good)[0]}")

# ============================================================
# PART 7: Fancy Indexing (Using Arrays as Indices)
# ============================================================

print("\n" + "="*60)
print("PART 7: FANCY INDEXING")
print("="*60)

# Employee productivity scores (arbitrary order)
employees = np.array(['Alice', 'Bob', 'Charlie', 'David', 'Eve'])
scores = np.array([92, 85, 78, 88, 95])

print("Employees:", employees)
print("Scores:", scores)

# Select specific employees by index positions
selected_indices = np.array([0, 2, 4])  # Alice, Charlie, Eve
selected_employees = employees[selected_indices]
selected_scores = scores[selected_indices]

print(f"\nSelected employees: {selected_employees}")
print(f"Their scores: {selected_scores}")

# Get top 3 performers (using argsort)
top3_indices = np.argsort(scores)[-3:][::-1]  # Descending order
print(f"\nTop 3 indices (sorted by score): {top3_indices}")
print(f"Top 3 employees: {employees[top3_indices]}")
print(f"Top 3 scores: {scores[top3_indices]}")

# 2D fancy indexing
data = np.array([
    [10, 20, 30],
    [40, 50, 60],
    [70, 80, 90]
])
print("\nData matrix:")
print(data)

# Select specific elements: (row_indices, col_indices)
row_indices = np.array([0, 1, 2])
col_indices = np.array([0, 1, 2])  # Diagonal elements
diagonal = data[row_indices, col_indices]
print(f"Diagonal elements: {diagonal}")

# Select arbitrary positions
rows = np.array([0, 2, 1])
cols = np.array([2, 0, 1])
selected = data[rows, cols]  # Elements at (0,2), (2,0), (1,1)
print(f"Selected positions: {selected}")

# ============================================================
# PART 8: Combining Indexing Techniques
# ============================================================

print("\n" + "="*60)
print("PART 8: COMBINING INDEXING TECHNIQUES")
print("="*60)

# Sales data: 10 days × 3 products
np.random.seed(42)
daily_product_sales = np.random.randint(50, 200, (10, 3))
print("Daily Sales (10 days × 3 products):")
print(daily_product_sales)

# Get last 5 days for product 2 (column 1)
recent_product2 = daily_product_sales[-5:, 1]
print(f"\nLast 5 days, Product 2: {recent_product2}")

# Find days where product 1 (column 0) sales exceeded 150
high_sales_days = daily_product_sales[:, 0] > 150
print(f"\nDays with Product 1 > 150: mask = {high_sales_days}")

# Get all product sales on those high days
high_days_data = daily_product_sales[high_sales_days, :]
print(f"All products on high sales days:")
print(high_days_data)

# Get product 3 sales only on high days for product 1
product3_on_high_days = daily_product_sales[high_sales_days, 2]
print(f"\nProduct 3 sales on Product 1's high days: {product3_on_high_days}")

# ============================================================
# PART 9: Modifying Arrays Through Indexing
# ============================================================

print("\n" + "="*60)
print("PART 9: MODIFYING ARRAYS THROUGH INDEXING")
print("="*60)

# Sensor readings with some errors
readings = np.array([22.5, 23.1, 999.0, 24.3, 22.8, -999.0, 23.5])
print("Original readings (with error codes):", readings)

# Fix error values (999.0 and -999.0 are error codes)
# Replace with mean of valid readings

valid_mask = (readings != 999.0) & (readings != -999.0)
valid_readings = readings[valid_mask]
mean_valid = valid_readings.mean()

print(f"Valid readings: {valid_readings}")
print(f"Mean of valid readings: {mean_valid:.2f}")

# Replace error values
readings[readings == 999.0] = mean_valid
readings[readings == -999.0] = mean_valid
print(f"Corrected readings: {readings}")

# Clip values to range (another modification technique)
values = np.array([5, 15, 25, 35, 45])
print(f"\nOriginal values: {values}")

# Cap at maximum 30, minimum 10
values_clipped = np.clip(values, 10, 30)
print(f"Clipped [10, 30]: {values_clipped}")

# ============================================================
# PART 10: Real-World Example - Data Quality Check
# ============================================================

print("\n" + "="*60)
print("PART 10: REAL-WORLD DATA QUALITY CHECK")
print("="*60)

# Simulated customer purchase data (amount in $)
np.random.seed(42)
purchases = np.random.uniform(5, 500, 20)
# Introduce some anomalies
purchases[5] = 9999  # Data entry error
purchases[12] = -50  # Negative value (refund coded incorrectly)
purchases[18] = 0    # Zero value

print("Purchase amounts:")
print(purchases.round(2))

# Quality checks
print("\n--- Data Quality Checks ---")

# Check 1: Negative values
negative_mask = purchases < 0
print(f"Negative values found: {negative_mask.sum()}")
if negative_mask.any():
    print(f"  Positions: {np.where(negative_mask)[0]}")
    print(f"  Values: {purchases[negative_mask]}")

# Check 2: Suspiciously high values (> 1000)
high_mask = purchases > 1000
print(f"\nSuspiciously high values (>$1000): {high_mask.sum()}")
if high_mask.any():
    print(f"  Positions: {np.where(high_mask)[0]}")
    print(f"  Values: {purchases[high_mask]}")

# Check 3: Zero values
zero_mask = purchases == 0
print(f"\nZero values: {zero_mask.sum()}")

# Clean data: keep only valid range ($5 - $500)
valid_mask = (purchases >= 5) & (purchases <= 500)
clean_purchases = purchases[valid_mask]

print(f"\n--- Results ---")
print(f"Original count: {len(purchases)}")
print(f"Clean count: {len(clean_purchases)}")
print(f"Removed: {len(purchases) - len(clean_purchases)}")
print(f"Clean data mean: ${clean_purchases.mean():.2f}")
print(f"Clean data median: ${np.median(clean_purchases):.2f}")

# ============================================================
# PART 11: Performance Comparison - Indexing Methods
# ============================================================

print("\n" + "="*60)
print("PART 11: INDEXING PERFORMANCE NOTES")
print("="*60)

import time

large_array = np.random.randint(0, 100, 1_000_000)

# Method 1: Boolean indexing (vectorized)
start = time.time()
result1 = large_array[large_array > 50]
time1 = time.time() - start

# Method 2: Loop (slow - DON'T DO THIS)
start = time.time()
result2 = []
for val in large_array:
    if val > 50:
        result2.append(val)
result2 = np.array(result2)
time2 = time.time() - start

print(f"Boolean indexing: {time1:.4f} seconds")
print(f"Python loop: {time2:.4f} seconds")
print(f"Speedup: {time2/time1:.1f}x faster")
print("\n✅ Always use boolean indexing over loops!")
