"""
================================================================================
LESSON 15: WHY NUMPY IS FAST (PERFORMANCE DEEP DIVE)
================================================================================

What You Will Learn:
- How NumPy achieves its speed (memory layout, C extensions, vectorization)
- Memory contiguity and cache efficiency
- Comparing NumPy vs Python loops with real benchmarks
- When NumPy is actually SLOWER than Python
- Profiling and optimizing NumPy code
- Practical optimization strategies

Real-World Usage:
- Processing millions of data points efficiently
- Real-time data analysis
- Machine learning feature computation
- Financial calculations at scale
- Scientific simulations

================================================================================
"""

import numpy as np
import time
import sys

print("=" * 70)
print("LESSON 15: WHY NUMPY IS FAST (PERFORMANCE DEEP DIVE)")
print("=" * 70)


# ==============================================================================
# SECTION 1: THE FUNDAMENTAL REASON - COMPILED C CODE
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 1: THE FUNDAMENTAL REASON - COMPILED C CODE")
print("=" * 70)

explanation = """
Why is NumPy fast? Three main reasons:

1. COMPILED C/FORTRAN CODE
   - NumPy operations run in pre-compiled C code
   - Python loops run in interpreted Python (much slower)
   - C is ~10-100x faster than interpreted Python

2. CONTIGUOUS MEMORY LAYOUT
   - NumPy arrays store data in continuous memory blocks
   - Python lists store pointers to scattered objects
   - Continuous memory = better CPU cache utilization

3. VECTORIZATION (SIMD)
   - Single Instruction, Multiple Data
   - CPU processes multiple array elements simultaneously
   - Modern CPUs have special instructions for this

Let's prove each of these with benchmarks!
"""
print(explanation)


# ==============================================================================
# SECTION 2: BENCHMARK - NUMPY VS PYTHON LOOPS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 2: BENCHMARK - NUMPY VS PYTHON LOOPS")
print("=" * 70)

def benchmark(func, *args, runs=5):
    """Run a function multiple times and return average time."""
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        result = func(*args)
        end = time.perf_counter()
        times.append(end - start)
    return np.mean(times), result

# ------------------------------------------------------------------------------
# 2.1 Simple Sum Operation
# ------------------------------------------------------------------------------
print("\n--- 2.1 Simple Sum Operation ---")

sizes = [10_000, 100_000, 1_000_000]

for size in sizes:
    # Create test data
    python_list = list(range(size))
    numpy_array = np.arange(size)
    
    # Python loop sum
    def python_sum(lst):
        total = 0
        for x in lst:
            total += x
        return total
    
    # Python built-in sum
    def builtin_sum(lst):
        return sum(lst)
    
    # NumPy sum
    def numpy_sum(arr):
        return np.sum(arr)
    
    # Benchmark
    py_time, py_result = benchmark(python_sum, python_list)
    builtin_time, builtin_result = benchmark(builtin_sum, python_list)
    np_time, np_result = benchmark(numpy_sum, numpy_array)
    
    print(f"\nSize: {size:,} elements")
    print(f"  Python loop:    {py_time*1000:.3f} ms")
    print(f"  Python sum():   {builtin_time*1000:.3f} ms")
    print(f"  NumPy sum():    {np_time*1000:.3f} ms")
    print(f"  Speedup (vs loop): {py_time/np_time:.1f}x")
    print(f"  Results match: {py_result == np_result}")

# ------------------------------------------------------------------------------
# 2.2 Element-wise Operations
# ------------------------------------------------------------------------------
print("\n--- 2.2 Element-wise Operations (Squaring) ---")

size = 1_000_000
python_list = list(range(size))
numpy_array = np.arange(size, dtype=np.float64)

# Python loop - square each element
def python_square(lst):
    result = []
    for x in lst:
        result.append(x * x)
    return result

# Python list comprehension
def python_comprehension(lst):
    return [x * x for x in lst]

# NumPy vectorized
def numpy_square(arr):
    return arr * arr

# Benchmark
py_time, _ = benchmark(python_square, python_list, runs=3)
comp_time, _ = benchmark(python_comprehension, python_list, runs=3)
np_time, _ = benchmark(numpy_square, numpy_array, runs=3)

print(f"\nSquaring {size:,} elements:")
print(f"  Python loop:          {py_time*1000:.2f} ms")
print(f"  List comprehension:   {comp_time*1000:.2f} ms")
print(f"  NumPy vectorized:     {np_time*1000:.2f} ms")
print(f"  Speedup (vs loop):    {py_time/np_time:.1f}x")

# ------------------------------------------------------------------------------
# 2.3 Complex Mathematical Operations
# ------------------------------------------------------------------------------
print("\n--- 2.3 Complex Mathematical Operations ---")

size = 500_000
numpy_array = np.random.rand(size) * 100

# Python: sin(x) + cos(x) * sqrt(x)
def python_math(lst):
    import math
    result = []
    for x in lst:
        result.append(math.sin(x) + math.cos(x) * math.sqrt(x))
    return result

# NumPy: same operation vectorized
def numpy_math(arr):
    return np.sin(arr) + np.cos(arr) * np.sqrt(arr)

python_list = numpy_array.tolist()

py_time, _ = benchmark(python_math, python_list, runs=3)
np_time, _ = benchmark(numpy_math, numpy_array, runs=3)

print(f"\nComplex math on {size:,} elements (sin + cos * sqrt):")
print(f"  Python loop:      {py_time*1000:.2f} ms")
print(f"  NumPy vectorized: {np_time*1000:.2f} ms")
print(f"  Speedup:          {py_time/np_time:.1f}x")


# ==============================================================================
# SECTION 3: MEMORY LAYOUT AND CONTIGUITY
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 3: MEMORY LAYOUT AND CONTIGUITY")
print("=" * 70)

# ------------------------------------------------------------------------------
# 3.1 Understanding Memory Layout
# ------------------------------------------------------------------------------
print("\n--- 3.1 Understanding Memory Layout ---")

# Python list vs NumPy array memory
python_list = [1.0, 2.0, 3.0, 4.0, 5.0]
numpy_array = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

print("Python List Memory Structure:")
print("  Each element is a separate Python object")
print("  List stores POINTERS to these objects")
print("  Objects scattered across memory")
print(f"  Size of list: {sys.getsizeof(python_list)} bytes")
print(f"  Total (including objects): ~{sys.getsizeof(python_list) + sum(sys.getsizeof(x) for x in python_list)} bytes")

print("\nNumPy Array Memory Structure:")
print("  All elements stored CONTIGUOUSLY")
print("  Single block of memory")
print("  No Python object overhead per element")
print(f"  Size of array: {numpy_array.nbytes} bytes (data only)")
print(f"  Element size: {numpy_array.itemsize} bytes")

# ------------------------------------------------------------------------------
# 3.2 C-Order vs Fortran-Order
# ------------------------------------------------------------------------------
print("\n--- 3.2 C-Order vs Fortran-Order ---")

# Create a 2D array
arr_2d = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

print("Original 2D array:")
print(arr_2d)

# C-order (row-major): rows are contiguous in memory
c_order = np.array(arr_2d, order='C')
print(f"\nC-order (row-major) - default:")
print(f"  Memory layout: {c_order.flatten('C')}")
print(f"  Is C-contiguous: {c_order.flags['C_CONTIGUOUS']}")
print(f"  Is F-contiguous: {c_order.flags['F_CONTIGUOUS']}")
print("  Rows are stored contiguously [1,2,3], [4,5,6], [7,8,9]")

# Fortran-order (column-major): columns are contiguous in memory
f_order = np.array(arr_2d, order='F')
print(f"\nFortran-order (column-major):")
print(f"  Memory layout: {f_order.flatten('F')}")
print(f"  Is C-contiguous: {f_order.flags['C_CONTIGUOUS']}")
print(f"  Is F-contiguous: {f_order.flags['F_CONTIGUOUS']}")
print("  Columns are stored contiguously [1,4,7], [2,5,8], [3,6,9]")

# ------------------------------------------------------------------------------
# 3.3 Why Contiguity Matters - Performance Impact
# ------------------------------------------------------------------------------
print("\n--- 3.3 Why Contiguity Matters - Performance Impact ---")

# Create large arrays in different orders
size = 5000
c_array = np.random.rand(size, size).astype(np.float64)  # C-order by default
f_array = np.asfortranarray(c_array)  # Convert to Fortran order

print(f"Array shape: {c_array.shape}")
print(f"C-order contiguous: {c_array.flags['C_CONTIGUOUS']}")
print(f"F-order contiguous: {f_array.flags['F_CONTIGUOUS']}")

# Sum along rows (axis=1) - should be faster for C-order
def sum_rows(arr):
    return arr.sum(axis=1)

# Sum along columns (axis=0) - should be faster for F-order  
def sum_cols(arr):
    return arr.sum(axis=0)

# Benchmark row sums
c_row_time, _ = benchmark(sum_rows, c_array, runs=10)
f_row_time, _ = benchmark(sum_rows, f_array, runs=10)

print(f"\nRow sums (axis=1) - {size}x{size} array:")
print(f"  C-order array: {c_row_time*1000:.2f} ms")
print(f"  F-order array: {f_row_time*1000:.2f} ms")
print(f"  C-order is {f_row_time/c_row_time:.2f}x faster (rows contiguous)")

# Benchmark column sums
c_col_time, _ = benchmark(sum_cols, c_array, runs=10)
f_col_time, _ = benchmark(sum_cols, f_array, runs=10)

print(f"\nColumn sums (axis=0) - {size}x{size} array:")
print(f"  C-order array: {c_col_time*1000:.2f} ms")
print(f"  F-order array: {f_col_time*1000:.2f} ms")
print(f"  F-order is {c_col_time/f_col_time:.2f}x faster (columns contiguous)")

# ------------------------------------------------------------------------------
# 3.4 Non-Contiguous Arrays (Views)
# ------------------------------------------------------------------------------
print("\n--- 3.4 Non-Contiguous Arrays (Views) ---")

# Original contiguous array
original = np.arange(1_000_000)

# Slice creates a view (may not be contiguous)
every_other = original[::2]  # Every other element

print(f"Original array:")
print(f"  Shape: {original.shape}")
print(f"  Contiguous: {original.flags['C_CONTIGUOUS']}")

print(f"\nEvery other element (view):")
print(f"  Shape: {every_other.shape}")
print(f"  Contiguous: {every_other.flags['C_CONTIGUOUS']}")
print(f"  Strides: {every_other.strides}")

# Performance comparison
def array_sum(arr):
    return arr.sum()

orig_time, _ = benchmark(array_sum, original, runs=10)
view_time, _ = benchmark(array_sum, every_other, runs=10)

# Make contiguous copy
contiguous_copy = np.ascontiguousarray(every_other)
copy_time, _ = benchmark(array_sum, contiguous_copy, runs=10)

print(f"\nSum performance:")
print(f"  Original (1M, contiguous):     {orig_time*1000:.3f} ms")
print(f"  View (500K, non-contiguous):   {view_time*1000:.3f} ms")
print(f"  Copy (500K, contiguous):       {copy_time*1000:.3f} ms")
print(f"  Contiguous copy speedup: {view_time/copy_time:.2f}x")


# ==============================================================================
# SECTION 4: STRIDES - HOW NUMPY NAVIGATES MEMORY
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 4: STRIDES - HOW NUMPY NAVIGATES MEMORY")
print("=" * 70)

# ------------------------------------------------------------------------------
# 4.1 Understanding Strides
# ------------------------------------------------------------------------------
print("\n--- 4.1 Understanding Strides ---")

arr = np.array([[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]], dtype=np.int32)

print("Array:")
print(arr)
print(f"\nShape: {arr.shape}")
print(f"Strides: {arr.strides}")
print(f"Item size: {arr.itemsize} bytes")

print(f"""
Stride explanation:
- Shape: (3, 4) means 3 rows, 4 columns
- Strides: {arr.strides} means:
  - To move to next ROW: skip {arr.strides[0]} bytes ({arr.strides[0]//arr.itemsize} elements)
  - To move to next COLUMN: skip {arr.strides[1]} bytes ({arr.strides[1]//arr.itemsize} element)
""")

# ------------------------------------------------------------------------------
# 4.2 Strides Enable Zero-Copy Operations
# ------------------------------------------------------------------------------
print("\n--- 4.2 Strides Enable Zero-Copy Operations ---")

original = np.arange(12).reshape(3, 4)
print("Original array:")
print(original)
print(f"Strides: {original.strides}")

# Transpose - just changes strides, no data copy!
transposed = original.T
print("\nTransposed array:")
print(transposed)
print(f"Strides: {transposed.strides}")
print(f"Shares memory: {np.shares_memory(original, transposed)}")

# Slicing - also just changes strides
every_other_col = original[:, ::2]
print("\nEvery other column:")
print(every_other_col)
print(f"Strides: {every_other_col.strides}")
print(f"Shares memory: {np.shares_memory(original, every_other_col)}")

print("\nKey insight: These operations are O(1) - instant regardless of array size!")

# ------------------------------------------------------------------------------
# 4.3 Stride Tricks (Advanced)
# ------------------------------------------------------------------------------
print("\n--- 4.3 Stride Tricks - Creating Views Without Copying ---")

# Create sliding windows using stride tricks
from numpy.lib.stride_tricks import sliding_window_view

# Time series data
data = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
print(f"Original data: {data}")

# Create sliding windows of size 3
windows = sliding_window_view(data, window_shape=3)
print(f"\nSliding windows (size 3):")
print(windows)
print(f"Shape: {windows.shape}")
print(f"Shares memory with original: {np.shares_memory(data, windows)}")

# Calculate rolling mean without any loops or copies
rolling_mean = windows.mean(axis=1)
print(f"\nRolling mean (window=3): {rolling_mean}")


# ==============================================================================
# SECTION 5: WHEN NUMPY IS SLOWER THAN PYTHON
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 5: WHEN NUMPY IS SLOWER THAN PYTHON")
print("=" * 70)

# ------------------------------------------------------------------------------
# 5.1 Small Arrays - Overhead Dominates
# ------------------------------------------------------------------------------
print("\n--- 5.1 Small Arrays - Overhead Dominates ---")

small_sizes = [5, 10, 50, 100, 500, 1000]

print("Sum operation - Python list vs NumPy array:")
print("-" * 50)

for size in small_sizes:
    python_list = list(range(size))
    numpy_array = np.arange(size)
    
    # Python sum
    py_time, _ = benchmark(sum, python_list, runs=100)
    
    # NumPy sum
    np_time, _ = benchmark(np.sum, numpy_array, runs=100)
    
    winner = "NumPy" if np_time < py_time else "Python"
    ratio = py_time/np_time if np_time < py_time else np_time/py_time
    
    print(f"Size {size:4d}: Python {py_time*1e6:6.2f}µs | NumPy {np_time*1e6:6.2f}µs | Winner: {winner} ({ratio:.1f}x)")

print("\nConclusion: For very small arrays (<100), Python can be faster due to NumPy overhead")

# ------------------------------------------------------------------------------
# 5.2 Creating Arrays Repeatedly
# ------------------------------------------------------------------------------
print("\n--- 5.2 Creating Arrays Repeatedly (Avoid This!) ---")

# Bad pattern: creating NumPy arrays in a loop
def bad_pattern():
    result = 0
    for i in range(1000):
        arr = np.array([i, i+1, i+2])  # Creating array each iteration - SLOW
        result += arr.sum()
    return result

# Good pattern: create array once, operate on it
def good_pattern():
    arr = np.arange(1000)
    # Operations on single array
    result = np.sum(arr + arr + 1 + arr + 2)
    return result

bad_time, _ = benchmark(bad_pattern, runs=10)
good_time, _ = benchmark(good_pattern, runs=10)

print(f"Bad pattern (create array in loop):  {bad_time*1000:.2f} ms")
print(f"Good pattern (single array):         {good_time*1000:.2f} ms")
print(f"Good pattern is {bad_time/good_time:.1f}x faster")

# ------------------------------------------------------------------------------
# 5.3 Non-Vectorizable Operations
# ------------------------------------------------------------------------------
print("\n--- 5.3 Non-Vectorizable Operations ---")

# Some operations can't be easily vectorized
# Example: Cumulative operation with complex dependency

def cumulative_complex_python(n):
    """Each element depends on previous in complex way."""
    result = [1.0]
    for i in range(1, n):
        # Next value depends on previous value in non-trivial way
        result.append(result[-1] * 1.01 + np.sin(result[-1]))
    return result

def cumulative_complex_numpy(n):
    """Same operation - NumPy can't really help here."""
    result = np.zeros(n)
    result[0] = 1.0
    for i in range(1, n):
        result[i] = result[i-1] * 1.01 + np.sin(result[i-1])
    return result

n = 10000
py_time, _ = benchmark(cumulative_complex_python, n, runs=5)
np_time, _ = benchmark(cumulative_complex_numpy, n, runs=5)

print(f"Sequential dependency operation ({n:,} elements):")
print(f"  Python list:  {py_time*1000:.2f} ms")
print(f"  NumPy array:  {np_time*1000:.2f} ms")
print(f"  Difference:   {abs(py_time-np_time)/min(py_time,np_time)*100:.1f}%")
print("\nConclusion: When operations have sequential dependencies,")
print("NumPy can't parallelize - both are similarly slow")

# ------------------------------------------------------------------------------
# 5.4 Object Arrays
# ------------------------------------------------------------------------------
print("\n--- 5.4 Object Arrays (Avoid!) ---")

# NumPy with object dtype loses all performance benefits
size = 100_000

# Numeric array (fast)
numeric_array = np.arange(size, dtype=np.int64)

# Object array (slow - stores Python objects)
object_array = np.array([i for i in range(size)], dtype=object)

# Python list
python_list = list(range(size))

num_time, _ = benchmark(np.sum, numeric_array, runs=10)
# For object array, we need to convert to work with sum
def obj_sum(arr):
    return sum(arr)
obj_time, _ = benchmark(obj_sum, object_array, runs=10)
py_time, _ = benchmark(sum, python_list, runs=10)

print(f"Sum of {size:,} integers:")
print(f"  NumPy int64:      {num_time*1000:.3f} ms")
print(f"  NumPy object:     {obj_time*1000:.3f} ms")
print(f"  Python list:      {py_time*1000:.3f} ms")
print(f"\nNumPy int64 is {obj_time/num_time:.0f}x faster than object array")
print("Conclusion: Object arrays eliminate NumPy's speed advantage!")


# ==============================================================================
# SECTION 6: OPTIMIZATION STRATEGIES
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 6: OPTIMIZATION STRATEGIES")
print("=" * 70)

# ------------------------------------------------------------------------------
# 6.1 Use Appropriate Data Types
# ------------------------------------------------------------------------------
print("\n--- 6.1 Use Appropriate Data Types ---")

# Create arrays with different dtypes
size = 10_000_000

float64_arr = np.random.rand(size).astype(np.float64)
float32_arr = float64_arr.astype(np.float32)
int64_arr = (float64_arr * 100).astype(np.int64)
int32_arr = int64_arr.astype(np.int32)

arrays = [
    ("float64", float64_arr),
    ("float32", float32_arr),
    ("int64", int64_arr),
    ("int32", int32_arr)
]

print(f"Sum of {size:,} elements with different dtypes:")
print("-" * 50)

for dtype_name, arr in arrays:
    time_taken, result = benchmark(np.sum, arr, runs=10)
    print(f"  {dtype_name}: {time_taken*1000:.2f} ms | Memory: {arr.nbytes/1e6:.1f} MB")

print("\nConclusion: Smaller dtypes = less memory = faster operations")

# ------------------------------------------------------------------------------
# 6.2 Avoid Temporary Arrays
# ------------------------------------------------------------------------------
print("\n--- 6.2 Avoid Temporary Arrays ---")

size = 5_000_000
a = np.random.rand(size)
b = np.random.rand(size)
c = np.random.rand(size)

# Bad: Creates multiple temporary arrays
def with_temps(a, b, c):
    temp1 = a * 2        # temporary array 1
    temp2 = b + 3        # temporary array 2
    temp3 = temp1 + temp2  # temporary array 3
    return temp3 + c     # temporary array 4

# Better: Use output parameter
def with_output(a, b, c, out=None):
    if out is None:
        out = np.empty_like(a)
    np.multiply(a, 2, out=out)
    np.add(out, b, out=out)
    np.add(out, 3, out=out)
    np.add(out, c, out=out)
    return out

# Even better: Use numexpr-style evaluation (single pass)
# Note: For this simple case, NumPy is smart enough to optimize somewhat
def single_expression(a, b, c):
    return a * 2 + b + 3 + c

out_buffer = np.empty_like(a)

temp_time, _ = benchmark(with_temps, a, b, c, runs=10)
output_time, _ = benchmark(with_output, a, b, c, out_buffer, runs=10)
single_time, _ = benchmark(single_expression, a, b, c, runs=10)

print(f"Computing: a*2 + b + 3 + c ({size:,} elements)")
print(f"  With temporaries:    {temp_time*1000:.2f} ms")
print(f"  With output param:   {output_time*1000:.2f} ms")
print(f"  Single expression:   {single_time*1000:.2f} ms")

# ------------------------------------------------------------------------------
# 6.3 Use In-Place Operations
# ------------------------------------------------------------------------------
print("\n--- 6.3 Use In-Place Operations ---")

size = 5_000_000

# Creating new array
def create_new(arr):
    return arr * 2

# In-place modification
def in_place(arr):
    arr *= 2
    return arr

arr1 = np.random.rand(size)
arr2 = arr1.copy()

new_time, _ = benchmark(lambda: np.random.rand(size) * 2, runs=10)
inplace_time, _ = benchmark(lambda: np.random.rand(size).__imul__(2), runs=10)

print(f"Multiplying {size:,} elements by 2:")
print(f"  Create new array:  {new_time*1000:.2f} ms")
print(f"  In-place (*=):     {inplace_time*1000:.2f} ms")
print("\nIn-place operations save memory allocation time")

# ------------------------------------------------------------------------------
# 6.4 Use Views Instead of Copies
# ------------------------------------------------------------------------------
print("\n--- 6.4 Use Views Instead of Copies ---")

large_array = np.random.rand(10_000, 10_000)

# Getting a slice - view (fast, no copy)
def get_view():
    return large_array[:5000, :5000]

# Making a copy (slow, allocates memory)
def get_copy():
    return large_array[:5000, :5000].copy()

view_time, view_result = benchmark(get_view, runs=100)
copy_time, copy_result = benchmark(get_copy, runs=100)

print(f"Accessing 5000x5000 subarray from 10000x10000 array:")
print(f"  View (no copy):  {view_time*1e6:.2f} µs")
print(f"  Copy:            {copy_time*1000:.2f} ms")
print(f"  View is {copy_time/view_time:.0f}x faster")
print(f"\nView shares memory: {np.shares_memory(large_array, view_result)}")
print(f"Copy shares memory: {np.shares_memory(large_array, copy_result)}")


# ==============================================================================
# SECTION 7: PROFILING NUMPY CODE
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 7: PROFILING NUMPY CODE")
print("=" * 70)

# ------------------------------------------------------------------------------
# 7.1 Simple Timing Decorator
# ------------------------------------------------------------------------------
print("\n--- 7.1 Simple Timing Decorator ---")

def timeit(func):
    """Decorator to time function execution."""
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"  {func.__name__}: {(end-start)*1000:.3f} ms")
        return result
    return wrapper

@timeit
def step1_load_data(n):
    """Simulate loading data."""
    return np.random.rand(n, 100)

@timeit
def step2_normalize(data):
    """Normalize data."""
    return (data - data.mean(axis=0)) / data.std(axis=0)

@timeit
def step3_compute_features(data):
    """Compute features."""
    return np.column_stack([
        data.mean(axis=1),
        data.std(axis=1),
        data.max(axis=1),
        data.min(axis=1)
    ])

@timeit
def step4_filter(data, features):
    """Filter based on features."""
    mask = features[:, 0] > 0  # Mean > 0
    return data[mask], features[mask]

print("Profiling a data pipeline (100,000 samples):")
data = step1_load_data(100_000)
normalized = step2_normalize(data)
features = step3_compute_features(normalized)
filtered_data, filtered_features = step4_filter(normalized, features)

print(f"\nPipeline complete: {len(filtered_data):,} samples after filtering")

# ------------------------------------------------------------------------------
# 7.2 Memory Profiling
# ------------------------------------------------------------------------------
print("\n--- 7.2 Memory Usage Analysis ---")

def memory_usage_example():
    """Demonstrate memory tracking."""
    
    # Track memory at each step
    print("Memory usage at each step:")
    
    # Step 1: Create base array
    arr = np.zeros((5000, 5000), dtype=np.float64)
    print(f"  1. Initial array:      {arr.nbytes / 1e6:.1f} MB")
    
    # Step 2: Operation that creates copy
    arr2 = arr * 2
    total = arr.nbytes + arr2.nbytes
    print(f"  2. After arr * 2:      {total / 1e6:.1f} MB (new array created)")
    
    # Step 3: In-place operation (no new memory)
    arr *= 2
    print(f"  3. After arr *= 2:     {arr.nbytes / 1e6:.1f} MB (in-place, no extra)")
    
    # Step 4: View operation (no new memory for data)
    view = arr[:2500, :2500]
    print(f"  4. After slice view:   {view.nbytes / 1e6:.1f} MB (view data)")
    print(f"     But actual new memory: ~0 MB (shares data)")
    
    return arr

result = memory_usage_example()


# ==============================================================================
# SECTION 8: REAL-WORLD OPTIMIZATION EXAMPLE
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 8: REAL-WORLD OPTIMIZATION EXAMPLE")
print("=" * 70)

# Scenario: Calculate pairwise distances between points
print("\n--- Computing Pairwise Distances ---")

np.random.seed(42)
n_points = 2000
n_dims = 3

# Generate random 3D points
points = np.random.rand(n_points, n_dims) * 100

print(f"Dataset: {n_points} points in {n_dims}D space")

# Method 1: Pure Python nested loops (SLOW)
def pairwise_distances_python(points):
    n = len(points)
    distances = []
    for i in range(n):
        row = []
        for j in range(n):
            dist = 0
            for k in range(len(points[0])):
                dist += (points[i][k] - points[j][k]) ** 2
            row.append(dist ** 0.5)
        distances.append(row)
    return distances

# Method 2: NumPy with explicit loops (MEDIUM)
def pairwise_distances_numpy_loops(points):
    n = len(points)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            distances[i, j] = np.sqrt(np.sum((points[i] - points[j]) ** 2))
    return distances

# Method 3: NumPy partially vectorized (FASTER)
def pairwise_distances_partial_vec(points):
    n = len(points)
    distances = np.zeros((n, n))
    for i in range(n):
        # Vectorize inner loop
        diff = points - points[i]  # Broadcasting
        distances[i] = np.sqrt(np.sum(diff ** 2, axis=1))
    return distances

# Method 4: Fully vectorized with broadcasting (FASTEST)
def pairwise_distances_full_vec(points):
    # Shape: (n, 1, dims) - (1, n, dims) = (n, n, dims)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    # Sum over dims, sqrt
    return np.sqrt(np.sum(diff ** 2, axis=2))

# Benchmark (use smaller size for slow methods)
small_points = points[:200].tolist()  # For Python version
small_points_np = points[:200]

print("\nBenchmarking with 200 points:")
print("-" * 50)

# Python (only 200 points - too slow for 2000)
py_time, _ = benchmark(pairwise_distances_python, small_points, runs=1)
print(f"1. Pure Python loops:        {py_time*1000:.1f} ms")

# NumPy with loops (only 200 points)
np_loop_time, _ = benchmark(pairwise_distances_numpy_loops, small_points_np, runs=1)
print(f"2. NumPy with loops:         {np_loop_time*1000:.1f} ms")

# Partial vectorization (200 points)
partial_time, _ = benchmark(pairwise_distances_partial_vec, small_points_np, runs=3)
print(f"3. Partial vectorization:    {partial_time*1000:.2f} ms")

# Full vectorization (200 points)
full_time, _ = benchmark(pairwise_distances_full_vec, small_points_np, runs=3)
print(f"4. Full vectorization:       {full_time*1000:.2f} ms")

print(f"\nSpeedup: Python → Full vectorization: {py_time/full_time:.0f}x")

# Now benchmark full vectorization on full dataset
print(f"\nFull vectorization on {n_points} points:")
full_time_large, distances = benchmark(pairwise_distances_full_vec, points, runs=3)
print(f"  Time: {full_time_large*1000:.1f} ms")
print(f"  Result shape: {distances.shape}")
print(f"  Memory: {distances.nbytes / 1e6:.1f} MB")


# ==============================================================================
# SECTION 9: COMMON PERFORMANCE PITFALLS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 9: COMMON PERFORMANCE PITFALLS")
print("=" * 70)

pitfalls = """
1. USING PYTHON LOOPS OVER ARRAY ELEMENTS
   ❌ for i in range(len(arr)): result[i] = arr[i] * 2
   ✅ result = arr * 2

2. CREATING ARRAYS INSIDE LOOPS
   ❌ for x in data: arr = np.array([x]); process(arr)
   ✅ arr = np.array(data); process(arr)

3. USING OBJECT DTYPE
   ❌ np.array([obj1, obj2], dtype=object)
   ✅ Use numeric dtypes whenever possible

4. IGNORING CONTIGUITY
   ❌ Operating on highly strided arrays
   ✅ Use np.ascontiguousarray() when needed

5. UNNECESSARY COPIES
   ❌ data_copy = data[:1000].copy()  # If not modifying
   ✅ data_view = data[:1000]  # View is sufficient

6. WRONG AXIS OPERATIONS
   ❌ for row in data: row.sum()  # Loop over rows
   ✅ data.sum(axis=1)  # Vectorized row sums

7. NOT USING BROADCASTING
   ❌ for i, row in enumerate(data): data[i] = row - mean
   ✅ data = data - mean  # Broadcasting

8. GROWING ARRAYS IN LOOPS
   ❌ arr = np.array([]); for x in data: arr = np.append(arr, x)
   ✅ arr = np.array(data)  # Or pre-allocate with np.zeros
"""
print(pitfalls)


# ==============================================================================
# SECTION 10: SUMMARY AND KEY TAKEAWAYS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 10: SUMMARY AND KEY TAKEAWAYS")
print("=" * 70)

summary = """
WHY NUMPY IS FAST:
==================
1. Compiled C code (vs interpreted Python)
2. Contiguous memory layout (cache-friendly)
3. Vectorization/SIMD (parallel processing)
4. Strides (zero-copy operations)

WHEN NUMPY IS SLOWER:
=====================
1. Very small arrays (< 100 elements)
2. Creating arrays repeatedly in loops
3. Non-vectorizable sequential operations
4. Object dtype arrays

OPTIMIZATION STRATEGIES:
========================
1. Use appropriate dtypes (smaller = faster)
2. Avoid temporary arrays (use output=)
3. Use in-place operations (arr *= 2)
4. Use views instead of copies
5. Ensure contiguous memory when needed
6. Vectorize everything possible
7. Use broadcasting instead of loops

MEMORY HIERARCHY:
=================
- L1 Cache: ~4 cycles access (32KB)
- L2 Cache: ~10 cycles access (256KB)
- L3 Cache: ~40 cycles access (8MB)
- RAM: ~100+ cycles access

Contiguous NumPy arrays fit in cache better than scattered Python objects!

PERFORMANCE RULES OF THUMB:
===========================
- Vectorized NumPy: 10-100x faster than Python loops
- Contiguous vs non-contiguous: 2-5x difference
- float32 vs float64: ~2x memory, similar speed
- In-place vs new array: 10-30% faster
"""
print(summary)


# ==============================================================================
# SECTION 11: PRACTICE EXERCISES
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 11: PRACTICE EXERCISES")
print("=" * 70)

print("\n--- Exercise 1: Optimize This Code ---")
print("Original (slow) code:")
print("""
def slow_normalize(data):
    result = []
    for row in data:
        row_mean = sum(row) / len(row)
        row_std = (sum((x - row_mean)**2 for x in row) / len(row)) ** 0.5
        normalized_row = [(x - row_mean) / row_std for x in row]
        result.append(normalized_row)
    return result
""")

# Solution
def slow_normalize(data):
    result = []
    for row in data:
        row_mean = sum(row) / len(row)
        row_std = (sum((x - row_mean)**2 for x in row) / len(row)) ** 0.5
        normalized_row = [(x - row_mean) / row_std for x in row]
        result.append(normalized_row)
    return result

def fast_normalize(data):
    """Optimized version using NumPy vectorization."""
    mean = data.mean(axis=1, keepdims=True)
    std = data.std(axis=1, keepdims=True)
    return (data - mean) / std

# Benchmark
test_data_list = np.random.rand(1000, 100).tolist()
test_data_np = np.random.rand(1000, 100)

slow_time, _ = benchmark(slow_normalize, test_data_list, runs=3)
fast_time, _ = benchmark(fast_normalize, test_data_np, runs=10)

print("Solution - Optimized code:")
print("""
def fast_normalize(data):
    mean = data.mean(axis=1, keepdims=True)
    std = data.std(axis=1, keepdims=True)
    return (data - mean) / std
""")
print(f"Slow version: {slow_time*1000:.1f} ms")
print(f"Fast version: {fast_time*1000:.2f} ms")
print(f"Speedup: {slow_time/fast_time:.0f}x")

print("\n--- Exercise 2: Memory-Efficient Moving Average ---")

def moving_average_efficient(arr, window):
    """Memory-efficient moving average using cumsum."""
    cumsum = np.cumsum(arr)
    cumsum[window:] = cumsum[window:] - cumsum[:-window]
    return cumsum[window-1:] / window

# Test
data = np.random.rand(1_000_000)
window = 10

ma_time, result = benchmark(moving_average_efficient, data, window, runs=10)
print(f"Moving average of 1M elements (window={window}):")
print(f"  Time: {ma_time*1000:.2f} ms")
print(f"  Result length: {len(result):,}")


# ==============================================================================
# NEXT LESSON PREVIEW
# ==============================================================================
print("\n" + "=" * 70)
print("NEXT LESSON PREVIEW")
print("=" * 70)

preview = """
Lesson 16 — Debugging Array Issues

You will learn:
- Common array bugs and how to identify them
- Shape mismatch debugging
- Data type issues
- NaN/Inf propagation problems
- Memory issues (views vs copies confusion)
- Techniques for inspecting array state
- Building defensive code with assertions

This lesson will help you quickly identify and fix NumPy issues!
"""
print(preview)

print("\n" + "=" * 70)
print("END OF LESSON 15")
print("=" * 70)
print("\nYou now understand WHY NumPy is fast and HOW to optimize your code!")
print("Remember: Vectorize, use appropriate dtypes, avoid copies when possible! 🚀")