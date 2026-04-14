import numpy as np

# ============================================================
# PART 1: Basic Aggregations - 1D Arrays
# ============================================================

print("="*60)
print("PART 1: BASIC AGGREGATIONS (1D Arrays)")
print("="*60)

# Daily sales for a week (in $1000s)
daily_sales = np.array([120, 135, 98, 142, 156, 189, 175])

print("Daily Sales ($1000s):", daily_sales)
print()

# Basic aggregation functions
print("--- Common Aggregations ---")
print(f"Total (sum):        ${daily_sales.sum():,}")
print(f"Average (mean):     ${daily_sales.mean():.2f}")
print(f"Median:             ${np.median(daily_sales):.2f}")
print(f"Minimum:            ${daily_sales.min()}")
print(f"Maximum:            ${daily_sales.max()}")
print(f"Range (max - min):  ${daily_sales.max() - daily_sales.min()}")

# Statistical measures
print("\n--- Statistical Measures ---")
print(f"Standard Deviation: ${daily_sales.std():.2f}")
print(f"Variance:           ${daily_sales.var():.2f}")
print(f"Sum of Squares:     {(daily_sales ** 2).sum():,}")

# Cumulative operations
print("\n--- Cumulative Operations ---")
print(f"Cumulative Sum:     {daily_sales.cumsum()}")
print(f"Cumulative Product: {daily_sales.cumprod()}")

# Counting
print("\n--- Counting ---")
high_sales = daily_sales > 150
print(f"Days with sales > $150k: {high_sales.sum()}")  # True=1, False=0
print(f"Percentage of high days: {high_sales.mean() * 100:.1f}%")

# ============================================================
# PART 2: Understanding the Axis Parameter
# ============================================================

print("\n" + "="*60)
print("PART 2: UNDERSTANDING AXIS PARAMETER")
print("="*60)

print("""
The 'axis' parameter is CRITICAL for multi-dimensional arrays.

Key concept:
- axis=0 → aggregate DOWN the rows (collapse rows)
- axis=1 → aggregate ACROSS the columns (collapse columns)
- axis=None (default) → aggregate entire array to single value

Visual guide for 2D array:
    
         axis=1 (across columns) →
    ┌─────────────────────────────┐
    │  col0  col1  col2  col3     │
a   │ ┌────┬────┬────┬────┐       │
x   │ │ 10 │ 20 │ 30 │ 40 │ → 100│  
i   │ ├────┼────┼────┼────┤       │
s   │ │ 50 │ 60 │ 70 │ 80 │ → 260│
=   │ └────┴────┴────┴────┘       │
0   │   ↓    ↓    ↓    ↓          │
↓   │   60   80  100  120         │
    └─────────────────────────────┘

Mnemonic: "axis is what DISAPPEARS"
- axis=0: rows disappear → left with columns
- axis=1: columns disappear → left with rows
""")

# Sales data: 4 weeks × 7 days
np.random.seed(42)
weekly_sales = np.array([
    [120, 135, 98, 142, 156, 189, 175],  # Week 1
    [110, 125, 105, 138, 148, 192, 168], # Week 2
    [115, 140, 102, 145, 152, 185, 172], # Week 3
    [125, 130, 95, 140, 160, 195, 180]   # Week 4
])

print("\nWeekly Sales (4 weeks × 7 days, in $1000s):")
print("     Mon  Tue  Wed  Thu  Fri  Sat  Sun")
print(weekly_sales)
print(f"Shape: {weekly_sales.shape} (4 weeks, 7 days)")

# axis=None (default) - entire array
print("\n--- axis=None (entire array) ---")
total_all = weekly_sales.sum()
avg_all = weekly_sales.mean()
print(f"Total sales (all weeks, all days): ${total_all:,}")
print(f"Average daily sales: ${avg_all:.2f}")

# axis=0 - down the rows (per day across weeks)
print("\n--- axis=0 (down rows → per DAY across weeks) ---")
daily_totals = weekly_sales.sum(axis=0)
daily_averages = weekly_sales.mean(axis=0)

print(f"Sum per day (across 4 weeks): {daily_totals}")
print(f"Avg per day (across 4 weeks): {daily_averages.round(1)}")
print(f"\nResult shape: {daily_totals.shape} - one value per day")

days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
print("\nInterpretation:")
for day, total, avg in zip(days, daily_totals, daily_averages):
    print(f"  {day}: Total=${total}, Avg=${avg:.1f}")

# axis=1 - across columns (per week across days)
print("\n--- axis=1 (across columns → per WEEK across days) ---")
weekly_totals = weekly_sales.sum(axis=1)
weekly_averages = weekly_sales.mean(axis=1)

print(f"Sum per week (across 7 days): {weekly_totals}")
print(f"Avg per week (across 7 days): {weekly_averages.round(1)}")
print(f"\nResult shape: {weekly_totals.shape} - one value per week")

print("\nInterpretation:")
for i, (total, avg) in enumerate(zip(weekly_totals, weekly_averages), 1):
    print(f"  Week {i}: Total=${total}, Avg=${avg:.1f}")

# ============================================================
# PART 3: Real-World Example - Product Sales Analysis
# ============================================================

print("\n" + "="*60)
print("PART 3: REAL-WORLD - PRODUCT SALES ANALYSIS")
print("="*60)

# Sales data: 5 products × 4 quarters
product_sales = np.array([
    [120, 135, 145, 160],  # Product A
    [80, 85, 90, 95],      # Product B
    [200, 210, 220, 230],  # Product C
    [50, 55, 58, 62],      # Product D
    [150, 160, 165, 170]   # Product E
])

products = ['Product A', 'Product B', 'Product C', 'Product D', 'Product E']
quarters = ['Q1', 'Q2', 'Q3', 'Q4']

print("Product Sales (5 products × 4 quarters, in $1000s):")
print("           Q1   Q2   Q3   Q4")
for product, sales in zip(products, product_sales):
    print(f"{product:10} {sales[0]:3} {sales[1]:3} {sales[2]:3} {sales[3]:3}")

# Question 1: Which product had highest total sales?
print("\n--- Question 1: Total Sales by Product ---")
product_totals = product_sales.sum(axis=1)  # Sum across quarters
print("Total sales per product:")
for product, total in zip(products, product_totals):
    print(f"  {product}: ${total:,}")

best_product_idx = product_totals.argmax()
print(f"\n✅ Best product: {products[best_product_idx]} (${product_totals[best_product_idx]:,})")

# Question 2: Which quarter had highest total sales?
print("\n--- Question 2: Total Sales by Quarter ---")
quarter_totals = product_sales.sum(axis=0)  # Sum across products
print("Total sales per quarter:")
for quarter, total in zip(quarters, quarter_totals):
    print(f"  {quarter}: ${total:,}")

best_quarter_idx = quarter_totals.argmax()
print(f"\n✅ Best quarter: {quarters[best_quarter_idx]} (${quarter_totals[best_quarter_idx]:,})")

# Question 3: Average performance by product
print("\n--- Question 3: Average Quarterly Performance ---")
product_averages = product_sales.mean(axis=1)
product_stds = product_sales.std(axis=1)

print("Product performance (average ± std):")
for product, avg, std in zip(products, product_averages, product_stds):
    print(f"  {product}: ${avg:.1f} ± ${std:.2f}")

# Question 4: Quarter-over-quarter growth
print("\n--- Question 4: Quarter-over-Quarter Growth ---")
# Calculate growth rate for each product
qoq_growth = np.diff(product_sales, axis=1) / product_sales[:, :-1] * 100

print("QoQ Growth Rate (%):")
print("           Q1→Q2  Q2→Q3  Q3→Q4")
for product, growth in zip(products, qoq_growth):
    print(f"{product:10} {growth[0]:5.1f}% {growth[1]:5.1f}% {growth[2]:5.1f}%")

avg_growth = qoq_growth.mean(axis=1)
print("\nAverage QoQ growth per product:")
for product, growth in zip(products, avg_growth):
    print(f"  {product}: {growth:.2f}%")

# ============================================================
# PART 4: Advanced Aggregations - Percentiles and Quantiles
# ============================================================

print("\n" + "="*60)
print("PART 4: PERCENTILES AND QUANTILES")
print("="*60)

# Simulated employee salaries (in $1000s)
np.random.seed(42)
salaries = np.random.gamma(50, 2, 1000)  # Skewed distribution

print(f"Employee Salaries (n={len(salaries)}):")
print(f"  Mean:   ${salaries.mean():.2f}k")
print(f"  Median: ${np.median(salaries):.2f}k")
print(f"  Std:    ${salaries.std():.2f}k")

# Percentiles
print("\n--- Salary Percentiles ---")
percentiles = [10, 25, 50, 75, 90, 95, 99]
values = np.percentile(salaries, percentiles)

for p, v in zip(percentiles, values):
    print(f"  {p}th percentile: ${v:.2f}k")

# Quartiles (25%, 50%, 75%)
q1, q2, q3 = np.percentile(salaries, [25, 50, 75])
iqr = q3 - q1  # Interquartile range

print(f"\n--- Quartile Analysis ---")
print(f"  Q1 (25%): ${q1:.2f}k")
print(f"  Q2 (50%, median): ${q2:.2f}k")
print(f"  Q3 (75%): ${q3:.2f}k")
print(f"  IQR (Q3-Q1): ${iqr:.2f}k")

# Using quantile (0 to 1 scale)
print("\n--- Using quantile() ---")
q_values = np.quantile(salaries, [0.25, 0.5, 0.75])
print(f"Quantiles [0.25, 0.5, 0.75]: {q_values.round(2)}")

# Multi-dimensional percentiles
print("\n--- Percentiles with axis ---")
test_scores = np.array([
    [85, 90, 78],  # Student 1
    [92, 88, 95],  # Student 2
    [76, 82, 80],  # Student 3
    [88, 85, 87]   # Student 4
])

print("Test scores (4 students × 3 tests):")
print(test_scores)

# 75th percentile per test (across students)
test_p75 = np.percentile(test_scores, 75, axis=0)
print(f"\n75th percentile per test: {test_p75}")

# 75th percentile per student (across tests)
student_p75 = np.percentile(test_scores, 75, axis=1)
print(f"75th percentile per student: {student_p75}")

# ============================================================
# PART 5: Conditional Aggregations
# ============================================================

print("\n" + "="*60)
print("PART 5: CONDITIONAL AGGREGATIONS")
print("="*60)

# Temperature data: 30 days × 24 hours
np.random.seed(42)
temperatures = np.random.normal(25, 5, (30, 24))  # Mean 25°C, std 5°C

print(f"Temperature Data: {temperatures.shape} (30 days × 24 hours)")
print(f"Sample (Day 1, first 12 hours):")
print(temperatures[0, :12].round(1))

# Question: Average temperature during daytime (8am-6pm)
daytime_temps = temperatures[:, 8:18]  # Hours 8-17
daytime_avg = daytime_temps.mean()

print(f"\n--- Daytime Analysis (8am-6pm) ---")
print(f"Average daytime temperature: {daytime_avg:.2f}°C")

# Question: How many hours exceeded 30°C per day?
hot_hours = (temperatures > 30).sum(axis=1)  # Sum per day
print(f"\n--- Hot Hours per Day (>30°C) ---")
print(f"Hours above 30°C per day: {hot_hours[:10]} (first 10 days)")
print(f"Average hot hours per day: {hot_hours.mean():.1f}")
print(f"Max hot hours in a day: {hot_hours.max()}")

# Question: Days with extreme temperatures
extreme_days = ((temperatures < 15) | (temperatures > 35)).any(axis=1)
print(f"\n--- Extreme Temperature Days ---")
print(f"Days with temps <15°C or >35°C: {extreme_days.sum()}")

# Average temperature only for non-extreme days
normal_temps = temperatures[~extreme_days]
print(f"Average temp on normal days: {normal_temps.mean():.2f}°C")

# ============================================================
# PART 6: Weighted Aggregations
# ============================================================

print("\n" + "="*60)
print("PART 6: WEIGHTED AGGREGATIONS")
print("="*60)

# Student grades with different weights
assignments = np.array([85, 90, 78, 92])
weights = np.array([0.20, 0.20, 0.30, 0.30])  # 20%, 20%, 30%, 30%

print("Assignment scores:", assignments)
print("Weights:", weights)
print(f"Sum of weights: {weights.sum()}")  # Should be 1.0

# Simple average (wrong for weighted grades)
simple_avg = assignments.mean()
print(f"\nSimple average: {simple_avg:.2f}")

# Weighted average
weighted_avg = np.average(assignments, weights=weights)
print(f"Weighted average: {weighted_avg:.2f}")

# Manual calculation (for understanding)
manual_weighted = (assignments * weights).sum()
print(f"Manual calculation: {manual_weighted:.2f}")

# Multiple students
print("\n--- Multiple Students ---")
student_scores = np.array([
    [85, 90, 78, 92],  # Student 1
    [75, 80, 85, 88],  # Student 2
    [95, 92, 90, 94],  # Student 3
])

print("Student scores (3 students × 4 assignments):")
print(student_scores)

# Weighted average per student
weighted_grades = np.average(student_scores, axis=1, weights=weights)
print(f"\nWeighted final grades: {weighted_grades.round(2)}")

# ============================================================
# PART 7: Multiple Axis Aggregations (3D Arrays)
# ============================================================

print("\n" + "="*60)
print("PART 7: MULTI-DIMENSIONAL AGGREGATIONS (3D)")
print("="*60)

# Sales data: 3 stores × 4 quarters × 5 products
np.random.seed(42)
store_sales = np.random.randint(50, 200, (3, 4, 5))

print(f"Sales Data: {store_sales.shape} (3 stores × 4 quarters × 5 products)")
print("\nSample - Store 1, Q1:")
print(store_sales[0, 0, :])

# Total sales per store (aggregate over quarters and products)
store_totals = store_sales.sum(axis=(1, 2))  # Aggregate axes 1 and 2
print(f"\n--- Total Sales per Store ---")
print(f"Store totals: {store_totals}")
for i, total in enumerate(store_totals, 1):
    print(f"  Store {i}: ${total:,}")

# Total sales per quarter (aggregate over stores and products)
quarter_totals = store_sales.sum(axis=(0, 2))  # Aggregate axes 0 and 2
print(f"\n--- Total Sales per Quarter ---")
print(f"Quarter totals: {quarter_totals}")
for i, total in enumerate(quarter_totals, 1):
    print(f"  Q{i}: ${total:,}")

# Total sales per product (aggregate over stores and quarters)
product_totals = store_sales.sum(axis=(0, 1))  # Aggregate axes 0 and 1
print(f"\n--- Total Sales per Product ---")
print(f"Product totals: {product_totals}")
for i, total in enumerate(product_totals, 1):
    print(f"  Product {i}: ${total:,}")

# Average sales per product per store (aggregate over quarters only)
store_product_avg = store_sales.mean(axis=1)  # Average over quarters
print(f"\n--- Average Product Sales per Store ---")
print("Shape:", store_product_avg.shape, "(3 stores × 5 products)")
print(store_product_avg.round(0))

# ============================================================
# PART 8: Useful Aggregation Functions
# ============================================================

print("\n" + "="*60)
print("PART 8: USEFUL AGGREGATION FUNCTIONS")
print("="*60)

data = np.array([45, 67, 89, 23, 91, 78, 34, 56])
print("Data:", data)

print("\n--- Finding Extremes ---")
print(f"Minimum value: {data.min()}")
print(f"Maximum value: {data.max()}")
print(f"Index of minimum: {data.argmin()}")
print(f"Index of maximum: {data.argmax()}")

print("\n--- Statistics ---")
print(f"Mean: {data.mean():.2f}")
print(f"Median: {np.median(data):.2f}")
print(f"Std deviation: {data.std():.2f}")
print(f"Variance: {data.var():.2f}")

print("\n--- Special Aggregations ---")
print(f"Product of all values: {data.prod()}")
print(f"Any value > 90?: {np.any(data > 90)}")
print(f"All values > 20?: {np.all(data > 20)}")

# 2D example
matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

print("\nMatrix:")
print(matrix)

print("\n--- argmin/argmax on 2D (flattened index) ---")
print(f"Argmin (flattened): {matrix.argmin()} → value: {matrix.flat[matrix.argmin()]}")
print(f"Argmax (flattened): {matrix.argmax()} → value: {matrix.flat[matrix.argmax()]}")

print("\n--- argmin/argmax with axis ---")
print(f"Argmin per column (axis=0): {matrix.argmin(axis=0)}")
print(f"Argmax per row (axis=1): {matrix.argmax(axis=1)}")

# ============================================================
# PART 9: Real-World Example - E-commerce Analytics
# ============================================================

print("\n" + "="*60)
print("PART 9: REAL-WORLD - E-COMMERCE ANALYTICS")
print("="*60)

# Transaction data: 100 transactions
np.random.seed(42)
num_transactions = 100

transaction_amounts = np.random.gamma(30, 2, num_transactions)  # Skewed
transaction_categories = np.random.choice(['Electronics', 'Clothing', 'Books', 'Home'], 
                                          num_transactions)
transaction_regions = np.random.choice(['North', 'South', 'East', 'West'], 
                                       num_transactions)

print(f"Analyzing {num_transactions} transactions...")

# Overall statistics
print("\n--- Overall Metrics ---")
print(f"Total revenue: ${transaction_amounts.sum():,.2f}")
print(f"Average order value: ${transaction_amounts.mean():.2f}")
print(f"Median order value: ${np.median(transaction_amounts):.2f}")
print(f"Std deviation: ${transaction_amounts.std():.2f}")

print("\n--- Order Value Distribution ---")
print(f"Min order: ${transaction_amounts.min():.2f}")
print(f"25th percentile: ${np.percentile(transaction_amounts, 25):.2f}")
print(f"50th percentile (median): ${np.percentile(transaction_amounts, 50):.2f}")
print(f"75th percentile: ${np.percentile(transaction_amounts, 75):.2f}")
print(f"Max order: ${transaction_amounts.max():.2f}")

# High-value transactions
high_value_threshold = np.percentile(transaction_amounts, 90)
high_value_count = (transaction_amounts >= high_value_threshold).sum()
high_value_revenue = transaction_amounts[transaction_amounts >= high_value_threshold].sum()

print("\n--- High-Value Transactions (Top 10%) ---")
print(f"Threshold: ${high_value_threshold:.2f}")
print(f"Count: {high_value_count}")
print(f"Revenue from top 10%: ${high_value_revenue:,.2f}")
print(f"Percentage of total revenue: {high_value_revenue/transaction_amounts.sum()*100:.1f}%")

# Category analysis
print("\n--- Revenue by Category ---")
categories = np.unique(transaction_categories)
for category in categories:
    mask = transaction_categories == category
    cat_revenue = transaction_amounts[mask].sum()
    cat_count = mask.sum()
    cat_avg = transaction_amounts[mask].mean()
    print(f"{category:12} - Count: {cat_count:2}, Revenue: ${cat_revenue:7.2f}, Avg: ${cat_avg:.2f}")

# Region analysis
print("\n--- Revenue by Region ---")
regions = np.unique(transaction_regions)
for region in regions:
    mask = transaction_regions == region
    reg_revenue = transaction_amounts[mask].sum()
    reg_count = mask.sum()
    reg_avg = transaction_amounts[mask].mean()
    print(f"{region:5} - Count: {reg_count:2}, Revenue: ${reg_revenue:7.2f}, Avg: ${reg_avg:.2f}")

# ============================================================
# PART 10: Custom Aggregations with apply_along_axis
# ============================================================

print("\n" + "="*60)
print("PART 10: CUSTOM AGGREGATIONS")
print("="*60)

# Sometimes you need custom aggregation logic
data = np.array([
    [10, 20, 30, 40],
    [15, 25, 35, 45],
    [12, 22, 32, 42]
])

print("Data (3 rows × 4 columns):")
print(data)

# Custom function: range (max - min)
def range_func(arr):
    return arr.max() - arr.min()

# Apply to each row
row_ranges = np.apply_along_axis(range_func, axis=1, arr=data)
print(f"\nRange per row: {row_ranges}")

# Apply to each column
col_ranges = np.apply_along_axis(range_func, axis=0, arr=data)
print(f"Range per column: {col_ranges}")

# Custom function: coefficient of variation (std/mean)
def coef_variation(arr):
    return arr.std() / arr.mean() if arr.mean() != 0 else 0

cv_rows = np.apply_along_axis(coef_variation, axis=1, arr=data)
print(f"\nCoefficient of variation per row: {cv_rows.round(3)}")

# ============================================================
# PART 11: Aggregation Performance Tips
# ============================================================

print("\n" + "="*60)
print("PART 11: PERFORMANCE COMPARISON")
print("="*60)

import time

large_array = np.random.randn(10000, 1000)
print(f"Large array: {large_array.shape}")

# Method 1: NumPy built-in
start = time.time()
result1 = large_array.mean(axis=1)
time1 = time.time() - start

# Method 2: Manual loop (SLOW)
start = time.time()
result2 = np.array([row.mean() for row in large_array])
time2 = time.time() - start

print(f"\nNumPy built-in:  {time1:.4f} seconds")
print(f"Manual loop:     {time2:.4f} seconds")
print(f"Speedup: {time2/time1:.1f}x")
print(f"\n✅ Always use built-in aggregation functions!")

print("\n" + "="*60)
print("AGGREGATION FUNCTIONS SUMMARY")
print("="*60)

print("""
BASIC AGGREGATIONS:
  sum()       - Sum of elements
  mean()      - Average value
  median()    - Middle value
  min()       - Minimum value
  max()       - Maximum value
  std()       - Standard deviation
  var()       - Variance
  
POSITIONAL:
  argmin()    - Index of minimum
  argmax()    - Index of maximum
  
CUMULATIVE:
  cumsum()    - Cumulative sum
  cumprod()   - Cumulative product
  
PERCENTILES:
  percentile() - Percentile (0-100)
  quantile()   - Quantile (0-1)
  
LOGICAL:
  any()       - True if any element is True
  all()       - True if all elements are True
  
WEIGHTED:
  average()   - Weighted average
  
CUSTOM:
  apply_along_axis() - Apply custom function

AXIS PARAMETER:
  axis=None   - Aggregate entire array
  axis=0      - Down rows (result: columns)
  axis=1      - Across columns (result: rows)
  axis=(0,1)  - Multiple axes (3D+)
""")
