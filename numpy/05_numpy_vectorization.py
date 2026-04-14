import numpy as np
import time

# ============================================================
# PART 1: Understanding Vectorization - The Basics
# ============================================================

print("="*60)
print("PART 1: VECTORIZATION BASICS")
print("="*60)

# Example: Convert temperatures from Celsius to Fahrenheit
# Formula: F = C * 9/5 + 32

temperatures_c = np.array([0, 10, 20, 30, 40])
print("Temperatures (°C):", temperatures_c)

# METHOD 1: Python loop (SLOW - how beginners often write it)
print("\n--- Loop Approach (AVOID) ---")
temps_f_loop = []
for temp in temperatures_c:
    temp_f = temp * 9/5 + 32
    temps_f_loop.append(temp_f)
temps_f_loop = np.array(temps_f_loop)
print("Result (loop):", temps_f_loop)

# METHOD 2: Vectorized (FAST - the NumPy way)
print("\n--- Vectorized Approach (PREFERRED) ---")
temps_f_vectorized = temperatures_c * 9/5 + 32
print("Result (vectorized):", temps_f_vectorized)

print("\n✅ Both give same result, but vectorization is:")
print("   - Cleaner (one line vs 4 lines)")
print("   - Faster (operations in C vs Python)")
print("   - More readable (expresses intent clearly)")

# ============================================================
# PART 2: Performance Comparison - Small Scale
# ============================================================

print("\n" + "="*60)
print("PART 2: PERFORMANCE COMPARISON")
print("="*60)

# Create larger dataset
np.random.seed(42)
data = np.random.randn(100_000)  # 100k random numbers

print(f"Dataset size: {len(data):,} elements")

# METHOD 1: Loop with list
print("\n--- Method 1: Python Loop + List ---")
start = time.time()
result_loop = []
for x in data:
    result_loop.append(x ** 2 + 2 * x + 1)  # Polynomial: x² + 2x + 1
result_loop = np.array(result_loop)
time_loop = time.time() - start
print(f"Time: {time_loop:.4f} seconds")

# METHOD 2: Loop with pre-allocated array
print("\n--- Method 2: Python Loop + Pre-allocated Array ---")
start = time.time()
result_prealloc = np.empty(len(data))
for i in range(len(data)):
    result_prealloc[i] = data[i] ** 2 + 2 * data[i] + 1
time_prealloc = time.time() - start
print(f"Time: {time_prealloc:.4f} seconds")
print(f"Speedup vs Method 1: {time_loop/time_prealloc:.1f}x")

# METHOD 3: Vectorized
print("\n--- Method 3: Vectorized (NumPy) ---")
start = time.time()
result_vectorized = data ** 2 + 2 * data + 1
time_vectorized = time.time() - start
print(f"Time: {time_vectorized:.4f} seconds")
print(f"Speedup vs Method 1: {time_loop/time_vectorized:.1f}x")
print(f"Speedup vs Method 2: {time_prealloc/time_vectorized:.1f}x")

# Verify all methods give same result
print("\n--- Verification ---")
print(f"Loop == Vectorized: {np.allclose(result_loop, result_vectorized)}")
print(f"Max difference: {np.abs(result_loop - result_vectorized).max()}")

# ============================================================
# PART 3: Common Vectorization Patterns
# ============================================================

print("\n" + "="*60)
print("PART 3: COMMON VECTORIZATION PATTERNS")
print("="*60)

# Pattern 1: Element-wise operations
print("\n--- Pattern 1: Element-wise Operations ---")
prices = np.array([100, 200, 150, 300, 250])
tax_rate = 0.18  # 18% tax

# ❌ Loop version
prices_with_tax_loop = []
for price in prices:
    prices_with_tax_loop.append(price * (1 + tax_rate))

# ✅ Vectorized version
prices_with_tax = prices * (1 + tax_rate)

print(f"Original prices: {prices}")
print(f"With tax (vectorized): {prices_with_tax}")

# Pattern 2: Conditional operations
print("\n--- Pattern 2: Conditional Operations ---")
sales = np.array([120, 80, 150, 45, 200, 30])
print(f"Sales: {sales}")

# ❌ Loop version
commissions_loop = []
for sale in sales:
    if sale >= 100:
        commissions_loop.append(sale * 0.15)  # 15% for high sales
    else:
        commissions_loop.append(sale * 0.10)  # 10% for low sales

# ✅ Vectorized version using np.where
commissions = np.where(sales >= 100, sales * 0.15, sales * 0.10)

print(f"Commissions (vectorized): {commissions}")

# Pattern 3: Multiple conditions
print("\n--- Pattern 3: Multiple Conditions ---")
scores = np.array([45, 72, 88, 91, 55, 67, 95, 38])
print(f"Scores: {scores}")

# Assign grades: A (>=90), B (>=75), C (>=60), D (>=50), F (<50)

# ❌ Loop version (nested ifs)
grades_loop = []
for score in scores:
    if score >= 90:
        grades_loop.append('A')
    elif score >= 75:
        grades_loop.append('B')
    elif score >= 60:
        grades_loop.append('C')
    elif score >= 50:
        grades_loop.append('D')
    else:
        grades_loop.append('F')

# ✅ Vectorized version (nested np.where)
grades = np.where(scores >= 90, 'A',
         np.where(scores >= 75, 'B',
         np.where(scores >= 60, 'C',
         np.where(scores >= 50, 'D', 'F'))))

print(f"Grades (vectorized): {grades}")

# Alternative: Using np.select (cleaner for many conditions)
conditions = [
    scores >= 90,
    scores >= 75,
    scores >= 60,
    scores >= 50
]
choices = ['A', 'B', 'C', 'D']
grades_select = np.select(conditions, choices, default='F')
print(f"Grades (np.select): {grades_select}")

# Pattern 4: Aggregations
print("\n--- Pattern 4: Aggregations ---")
daily_revenue = np.array([1200, 1500, 980, 1800, 1650, 1100, 1350])

# ❌ Loop version
total_loop = 0
for revenue in daily_revenue:
    total_loop += revenue
avg_loop = total_loop / len(daily_revenue)

# ✅ Vectorized version
total = daily_revenue.sum()
avg = daily_revenue.mean()

print(f"Total revenue: ${total:,} (loop: ${total_loop:,})")
print(f"Average revenue: ${avg:,.2f} (loop: ${avg_loop:,.2f})")

# ============================================================
# PART 4: Real-World Example - Employee Salary Processing
# ============================================================

print("\n" + "="*60)
print("PART 4: REAL-WORLD - EMPLOYEE SALARY PROCESSING")
print("="*60)

# Simulated employee data
np.random.seed(42)
num_employees = 10_000

# Base salaries ($40k - $120k)
base_salaries = np.random.uniform(40000, 120000, num_employees)

# Years of service (0-20 years)
years_service = np.random.randint(0, 21, num_employees)

# Performance ratings (1-5)
performance = np.random.randint(1, 6, num_employees)

print(f"Processing salaries for {num_employees:,} employees...")

# Calculate raises based on:
# - Base raise: 3% for all
# - Experience bonus: 0.5% per year of service (max 10%)
# - Performance bonus: 0%, 2%, 5%, 8%, 12% for ratings 1-5

# ❌ Loop approach
print("\n--- Loop Approach ---")
start = time.time()
new_salaries_loop = np.empty(num_employees)
for i in range(num_employees):
    base_raise = 0.03
    exp_bonus = min(years_service[i] * 0.005, 0.10)
    
    if performance[i] == 1:
        perf_bonus = 0.00
    elif performance[i] == 2:
        perf_bonus = 0.02
    elif performance[i] == 3:
        perf_bonus = 0.05
    elif performance[i] == 4:
        perf_bonus = 0.08
    else:  # 5
        perf_bonus = 0.12
    
    total_raise = base_raise + exp_bonus + perf_bonus
    new_salaries_loop[i] = base_salaries[i] * (1 + total_raise)

time_loop = time.time() - start
print(f"Time: {time_loop:.4f} seconds")

# ✅ Vectorized approach
print("\n--- Vectorized Approach ---")
start = time.time()

# Base raise (3% for all)
base_raise = 0.03

# Experience bonus (capped at 10%)
exp_bonus = np.minimum(years_service * 0.005, 0.10)

# Performance bonus using np.select
perf_conditions = [
    performance == 1,
    performance == 2,
    performance == 3,
    performance == 4,
    performance == 5
]
perf_choices = [0.00, 0.02, 0.05, 0.08, 0.12]
perf_bonus = np.select(perf_conditions, perf_choices)

# Calculate new salaries
total_raise = base_raise + exp_bonus + perf_bonus
new_salaries = base_salaries * (1 + total_raise)

time_vectorized = time.time() - start
print(f"Time: {time_vectorized:.4f} seconds")
print(f"Speedup: {time_loop/time_vectorized:.1f}x faster")

# Verify results match
print(f"\nResults match: {np.allclose(new_salaries_loop, new_salaries)}")

# Analysis
print("\n--- Salary Analysis ---")
print(f"Total payroll increase: ${(new_salaries.sum() - base_salaries.sum()):,.2f}")
print(f"Average salary: ${base_salaries.mean():,.2f} → ${new_salaries.mean():,.2f}")
print(f"Average raise: {((new_salaries.mean() / base_salaries.mean()) - 1) * 100:.2f}%")

# ============================================================
# PART 5: Vectorizing with Functions - Universal Functions
# ============================================================

print("\n" + "="*60)
print("PART 5: UNIVERSAL FUNCTIONS (ufuncs)")
print("="*60)

# NumPy provides vectorized versions of math functions

data = np.array([1, 4, 9, 16, 25])
print(f"Data: {data}")

# Mathematical ufuncs
print(f"\nSquare root: {np.sqrt(data)}")
print(f"Logarithm: {np.log(data)}")
print(f"Exponential: {np.exp([1, 2, 3])}")

# Trigonometric ufuncs
angles = np.array([0, 30, 45, 60, 90])  # degrees
angles_rad = np.deg2rad(angles)  # Convert to radians
print(f"\nAngles: {angles}°")
print(f"Sin: {np.sin(angles_rad).round(2)}")
print(f"Cos: {np.cos(angles_rad).round(2)}")

# Statistical ufuncs
readings = np.random.randn(1000)
print(f"\n1000 random readings:")
print(f"Mean: {np.mean(readings):.3f}")
print(f"Std Dev: {np.std(readings):.3f}")
print(f"Median: {np.median(readings):.3f}")
print(f"Min: {np.min(readings):.3f}")
print(f"Max: {np.max(readings):.3f}")

# ============================================================
# PART 6: Custom Vectorization with np.vectorize
# ============================================================

print("\n" + "="*60)
print("PART 6: CUSTOM VECTORIZATION (np.vectorize)")
print("="*60)

# Sometimes you have a complex function that isn't built-in
def calculate_shipping(weight_kg, distance_km):
    """Calculate shipping cost based on weight and distance"""
    base_cost = 5.0
    weight_cost = weight_kg * 0.5
    distance_cost = distance_km * 0.02
    
    # Bulk discount
    if weight_kg > 100:
        discount = 0.15
    elif weight_kg > 50:
        discount = 0.10
    elif weight_kg > 20:
        discount = 0.05
    else:
        discount = 0.0
    
    total = (base_cost + weight_cost + distance_cost) * (1 - discount)
    return round(total, 2)

# Sample data
weights = np.array([10, 25, 55, 120, 15])
distances = np.array([100, 250, 500, 1000, 150])

print("Package weights (kg):", weights)
print("Distances (km):", distances)

# ❌ Loop version
print("\n--- Loop Approach ---")
costs_loop = []
for w, d in zip(weights, distances):
    costs_loop.append(calculate_shipping(w, d))
costs_loop = np.array(costs_loop)
print(f"Shipping costs: ${costs_loop}")

# ✅ Vectorized version using np.vectorize
print("\n--- Vectorized Approach (np.vectorize) ---")
vectorized_shipping = np.vectorize(calculate_shipping)
costs_vectorized = vectorized_shipping(weights, distances)
print(f"Shipping costs: ${costs_vectorized}")

print("\n⚠️ Note: np.vectorize is convenient but NOT always faster!")
print("   It's essentially a loop under the hood.")
print("   Use for convenience, not performance critical code.")

# ============================================================
# PART 7: Broadcasting - Advanced Vectorization
# ============================================================

print("\n" + "="*60)
print("PART 7: BROADCASTING")
print("="*60)

# Broadcasting allows operations on arrays of different shapes

# Example: Apply discounts to all products
products = np.array([100, 200, 150, 300])  # Prices
discounts = np.array([0.10, 0.15, 0.20])   # 10%, 15%, 20% off

print("Product prices:", products)
print("Discount rates:", discounts)

# Create a discount matrix: each row is a discount applied to all products
# Shape: (3, 4) - 3 discounts × 4 products

# ❌ Loop version
print("\n--- Loop Approach ---")
discount_matrix_loop = np.empty((len(discounts), len(products)))
for i, discount in enumerate(discounts):
    for j, price in enumerate(products):
        discount_matrix_loop[i, j] = price * (1 - discount)

print("Discounted prices (loop):")
print(discount_matrix_loop)

# ✅ Broadcasting version
print("\n--- Broadcasting Approach ---")
# Reshape discounts to (3, 1) so it broadcasts with (4,) products
discounts_col = discounts[:, np.newaxis]  # Shape: (3, 1)
discount_matrix = products * (1 - discounts_col)  # Broadcasts to (3, 4)

print("Discounted prices (broadcasting):")
print(discount_matrix)

print("\n✅ Broadcasting eliminates nested loops!")

# ============================================================
# PART 8: When Loops ARE Necessary
# ============================================================

print("\n" + "="*60)
print("PART 8: WHEN LOOPS ARE NECESSARY")
print("="*60)

print("""
Vectorization is NOT always possible. Use loops when:

1. **Operations depend on previous iterations**
   Example: Cumulative calculations where each step needs the last result
   (Though NumPy has np.cumsum, np.cumprod for common cases)

2. **Complex conditional logic with state**
   Example: State machines, sequential decision trees

3. **Breaking early based on conditions**
   Example: Finding first occurrence that meets complex criteria

4. **Interacting with external systems**
   Example: API calls, database queries per item

5. **Mixed data types or objects**
   Example: Processing strings with complex rules
""")

# Example where loop is appropriate: Finding first valid value
print("--- Example: Early Exit ---")
data_with_errors = np.array([np.nan, -999, np.nan, 42, 100, 200])
print(f"Data: {data_with_errors}")

# Find first valid value (not NaN, not -999)
print("\n❌ Vectorization can't easily handle early exit")
print("✅ Loop is appropriate here:")

first_valid = None
for i, val in enumerate(data_with_errors):
    if not np.isnan(val) and val != -999:
        first_valid = val
        print(f"   Found first valid value: {val} at index {i}")
        break

# ============================================================
# PART 9: Performance Best Practices Summary
# ============================================================

print("\n" + "="*60)
print("PART 9: PERFORMANCE BEST PRACTICES")
print("="*60)

print("""
✅ DO:
1. Use vectorized operations whenever possible
2. Use built-in NumPy functions (sum, mean, std, etc.)
3. Use boolean indexing instead of filtering loops
4. Use np.where for conditional operations
5. Pre-allocate arrays if loops are unavoidable

❌ DON'T:
1. Loop through NumPy arrays for element-wise operations
2. Build arrays by appending in loops
3. Use Python math functions (use np.sqrt, not math.sqrt)
4. Write nested loops for operations that can broadcast
5. Use np.vectorize for performance (only for convenience)

⚡ PERFORMANCE HIERARCHY (fastest → slowest):
1. Built-in NumPy ufuncs (np.sum, np.sqrt, etc.)
2. Vectorized expressions (arr * 2 + 1)
3. Broadcasting operations
4. Pre-allocated array + loop
5. List comprehension + convert to array
6. Loop with list.append() + convert to array
7. Pure Python loops with math operations
""")

# ============================================================
# PART 10: Final Performance Benchmark
# ============================================================

print("\n" + "="*60)
print("PART 10: COMPREHENSIVE BENCHMARK")
print("="*60)

# Task: Calculate compound interest for 1 million accounts
# Formula: A = P(1 + r/n)^(nt)
# P = principal, r = rate, n = compounds per year, t = years

np.random.seed(42)
num_accounts = 1_000_000

principals = np.random.uniform(1000, 100000, num_accounts)
rate = 0.05  # 5% annual
compounds = 12  # Monthly
years = 10

print(f"Calculating compound interest for {num_accounts:,} accounts...")
print(f"Rate: {rate*100}%, Compounding: {compounds}x/year, Years: {years}")

# Method 1: Pure Python loop
print("\n--- Method 1: Python Loop ---")
start = time.time()
results_loop = []
for p in principals:
    amount = p * (1 + rate/compounds) ** (compounds * years)
    results_loop.append(amount)
results_loop = np.array(results_loop)
time1 = time.time() - start
print(f"Time: {time1:.4f} seconds")

# Method 2: Vectorized
print("\n--- Method 2: Vectorized ---")
start = time.time()
results_vectorized = principals * (1 + rate/compounds) ** (compounds * years)
time2 = time.time() - start
print(f"Time: {time2:.4f} seconds")

print(f"\n🚀 Speedup: {time1/time2:.1f}x faster!")
print(f"Results match: {np.allclose(results_loop, results_vectorized)}")

print(f"\n--- Financial Summary ---")
total_invested = principals.sum()
total_final = results_vectorized.sum()
total_interest = total_final - total_invested

print(f"Total invested: ${total_invested:,.2f}")
print(f"Total after {years} years: ${total_final:,.2f}")
print(f"Total interest earned: ${total_interest:,.2f}")
print(f"Average return: {((total_final/total_invested - 1) * 100):.2f}%")