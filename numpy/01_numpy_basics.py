import numpy as np
import time

# ============================================================
# PART 1: Python Lists vs NumPy Arrays
# ============================================================

# Python list of numbers
python_list = [1, 2, 3, 4, 5]
print("Python List:", python_list)
print("Type:", type(python_list))

# NumPy array from the same data
numpy_array = np.array([1, 2, 3, 4, 5])
print("\nNumPy Array:", numpy_array)
print("Type:", type(numpy_array))

# ============================================================
# PART 2: Key Difference - Operations
# ============================================================

# With Python lists, operations require loops or list comprehensions
python_list_doubled = [x * 2 for x in python_list]
print("\nPython list doubled:", python_list_doubled)

# With NumPy, operations are vectorized (no explicit loop needed)
numpy_array_doubled = numpy_array * 2
print("NumPy array doubled:", numpy_array_doubled)

# ============================================================
# PART 3: Performance Comparison (The Real Reason NumPy Exists)
# ============================================================

# Create large datasets
size = 1_000_000
large_python_list = list(range(size))
large_numpy_array = np.arange(size)  # NumPy's efficient range

# Time Python list operation
start = time.time()
result_list = [x * 2 for x in large_python_list]
python_time = time.time() - start

# Time NumPy array operation
start = time.time()
result_numpy = large_numpy_array * 2
numpy_time = time.time() - start

print(f"\n--- Performance Test (1 million numbers) ---")
print(f"Python list time: {python_time:.4f} seconds")
print(f"NumPy array time: {numpy_time:.4f} seconds")
print(f"NumPy is {python_time/numpy_time:.1f}x faster!")

# ============================================================
# PART 4: Memory Efficiency
# ============================================================

import sys

# Memory used by Python list
python_list_memory = sys.getsizeof(large_python_list)

# Memory used by NumPy array
numpy_array_memory = large_numpy_array.nbytes

print(f"\n--- Memory Usage ---")
print(f"Python list: {python_list_memory / 1_000_000:.2f} MB")
print(f"NumPy array: {numpy_array_memory / 1_000_000:.2f} MB")
print(f"NumPy uses {python_list_memory / numpy_array_memory:.1f}x less memory!")

# ============================================================
# PART 5: Why This Matters - Real Data Scenario
# ============================================================

# Simulate daily temperature readings for a year (365 days)
# In Celsius, ranging from -10 to 35 degrees
np.random.seed(42)  # For reproducibility
daily_temps = np.random.uniform(-10, 35, 365)

# Convert all temperatures to Fahrenheit: F = C * 9/5 + 32
# With NumPy, this is ONE line, no loop needed
temps_fahrenheit = daily_temps * 9/5 + 32

print(f"\n--- Temperature Conversion (365 days) ---")
print(f"First 5 days (Celsius): {daily_temps[:5]}")
print(f"First 5 days (Fahrenheit): {temps_fahrenheit[:5]}")

# Calculate statistics instantly
print(f"\nAverage temperature: {daily_temps.mean():.1f}°C")
print(f"Hottest day: {daily_temps.max():.1f}°C")
print(f"Coldest day: {daily_temps.min():.1f}°C")