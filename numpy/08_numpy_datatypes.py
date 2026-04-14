import numpy as np
import sys

# ============================================================
# PART 1: NumPy Data Types Overview
# ============================================================

print("="*60)
print("PART 1: NumPy DATA TYPES OVERVIEW")
print("="*60)

print("""
NumPy supports many data types optimized for different use cases:

INTEGERS:
  int8    - 8-bit signed integer (-128 to 127)
  int16   - 16-bit signed integer (-32,768 to 32,767)
  int32   - 32-bit signed integer (-2.1B to 2.1B)
  int64   - 64-bit signed integer (default on 64-bit systems)
  uint8   - 8-bit unsigned (0 to 255) - images!
  uint16  - 16-bit unsigned (0 to 65,535)
  uint32  - 32-bit unsigned (0 to 4.3B)
  uint64  - 64-bit unsigned

FLOATS:
  float16 - 16-bit float (half precision) - ML, graphics
  float32 - 32-bit float (single precision)
  float64 - 64-bit float (double precision) - default
  
COMPLEX:
  complex64  - 2×32-bit floats (real + imaginary)
  complex128 - 2×64-bit floats (default)

BOOLEAN:
  bool    - True/False (1 byte per value)

STRINGS:
  <U10    - Unicode string, 10 characters max
  S10     - Byte string, 10 bytes max

OTHER:
  datetime64 - Date/time representation
  object     - Python objects (AVOID for performance)
""")

# ============================================================
# PART 2: Checking and Setting Data Types
# ============================================================

print("="*60)
print("PART 2: CHECKING AND SETTING DATA TYPES")
print("="*60)

# Default type inference
int_array = np.array([1, 2, 3, 4, 5])
float_array = np.array([1.0, 2.0, 3.0])
mixed_array = np.array([1, 2, 3.5])  # Will upcast to float

print("--- Type Inference ---")
print(f"int_array: {int_array}, dtype: {int_array.dtype}")
print(f"float_array: {float_array}, dtype: {float_array.dtype}")
print(f"mixed_array: {mixed_array}, dtype: {mixed_array.dtype}")

# Explicit type specification
print("\n--- Explicit Type Specification ---")
int8_array = np.array([1, 2, 3], dtype=np.int8)
float32_array = np.array([1.0, 2.0, 3.0], dtype=np.float32)
bool_array = np.array([True, False, True], dtype=bool)

print(f"int8:    {int8_array}, dtype: {int8_array.dtype}")
print(f"float32: {float32_array}, dtype: {float32_array.dtype}")
print(f"bool:    {bool_array}, dtype: {bool_array.dtype}")

# Using dtype objects
print("\n--- Different Ways to Specify dtype ---")
a = np.array([1, 2, 3], dtype='int16')
b = np.array([1, 2, 3], dtype=np.int16)
c = np.array([1, 2, 3], dtype='i2')  # Short code

print(f"dtype='int16':  {a.dtype}")
print(f"dtype=np.int16: {b.dtype}")
print(f"dtype='i2':     {c.dtype}")

# Checking type properties
print("\n--- Type Properties ---")
arr = np.array([1, 2, 3], dtype=np.int32)
print(f"Array: {arr}")
print(f"dtype: {arr.dtype}")
print(f"dtype name: {arr.dtype.name}")
print(f"dtype itemsize (bytes): {arr.dtype.itemsize}")
print(f"Total memory: {arr.nbytes} bytes")

# ============================================================
# PART 3: Memory Impact of Data Types
# ============================================================

print("\n" + "="*60)
print("PART 3: MEMORY IMPACT OF DATA TYPES")
print("="*60)

# Create same data with different types
data = np.arange(1_000_000)

print(f"1 million integers in different types:")
print()

for dtype in [np.int8, np.int16, np.int32, np.int64]:
    arr = data.astype(dtype)
    memory_mb = arr.nbytes / 1_000_000
    print(f"{dtype.__name__:6} - {memory_mb:.2f} MB ({arr.dtype.itemsize} bytes/item)")

print()
for dtype in [np.float16, np.float32, np.float64]:
    arr = data.astype(dtype)
    memory_mb = arr.nbytes / 1_000_000
    print(f"{dtype.__name__:8} - {memory_mb:.2f} MB ({arr.dtype.itemsize} bytes/item)")

# Real-world example: Image data
print("\n--- Real-World Example: Image Data ---")
# HD image: 1920 × 1080 × 3 (RGB channels)
height, width, channels = 1920, 1080, 3
num_pixels = height * width * channels

print(f"HD Image ({height}×{width}×{channels} RGB):")
print(f"  uint8 (0-255):    {num_pixels * 1 / 1_000_000:.2f} MB")
print(f"  uint16 (0-65535): {num_pixels * 2 / 1_000_000:.2f} MB")
print(f"  float32:          {num_pixels * 4 / 1_000_000:.2f} MB")
print(f"  float64:          {num_pixels * 8 / 1_000_000:.2f} MB")

print("\n✅ Use uint8 for images - 8x smaller than float64!")

# ============================================================
# PART 4: Type Conversion (Casting)
# ============================================================

print("\n" + "="*60)
print("PART 4: TYPE CONVERSION (CASTING)")
print("="*60)

# Float to integer (truncates decimal)
float_arr = np.array([1.7, 2.3, 3.9, 4.5])
print(f"Original (float): {float_arr}")

int_arr = float_arr.astype(np.int32)
print(f"After astype(int32): {int_arr}")
print("⚠️  Decimals are TRUNCATED, not rounded!")

# Rounding before conversion
rounded_arr = np.round(float_arr).astype(np.int32)
print(f"After round then int: {rounded_arr}")

# Integer to float
int_vals = np.array([10, 20, 30])
float_vals = int_vals.astype(np.float32)
print(f"\nInteger to float: {int_vals} → {float_vals}")

# Boolean conversions
print("\n--- Boolean Conversions ---")
numbers = np.array([0, 1, 2, -1, 0.0])
as_bool = numbers.astype(bool)
print(f"Numbers: {numbers}")
print(f"As bool: {as_bool}")
print("Rule: 0 → False, everything else → True")

bool_arr = np.array([True, False, True])
as_int = bool_arr.astype(np.int32)
print(f"\nBooleans: {bool_arr}")
print(f"As int:   {as_int}")
print("Rule: True → 1, False → 0")

# String conversions
print("\n--- String Conversions ---")
str_numbers = np.array(['1', '2', '3', '4'])
print(f"String array: {str_numbers}, dtype: {str_numbers.dtype}")

numeric = str_numbers.astype(np.int32)
print(f"Converted to int: {numeric}, dtype: {numeric.dtype}")

# Numbers to strings
nums = np.array([10, 20, 30])
strs = nums.astype(str)
print(f"\nNumbers: {nums}, dtype: {nums.dtype}")
print(f"As strings: {strs}, dtype: {strs.dtype}")

# ============================================================
# PART 5: Integer Overflow and Underflow
# ============================================================

print("\n" + "="*60)
print("PART 5: INTEGER OVERFLOW AND UNDERFLOW")
print("="*60)

print("⚠️  NumPy does NOT warn about overflow by default!")

# int8 range: -128 to 127
arr = np.array([100, 120], dtype=np.int8)
print(f"\nOriginal (int8): {arr}")

# This overflows!
result = arr + 50
print(f"After +50: {result}")
print("Expected: [150, 170]")
print("Actual: Wrapped around due to overflow!")

print("\n--- Understanding Overflow ---")
print(f"int8 max value: {np.iinfo(np.int8).max}")
print(f"100 + 50 = 150, but max is 127")
print(f"Wraps around: 150 - 256 = -106")

# Safe alternative: use larger type
arr_safe = np.array([100, 120], dtype=np.int32)
result_safe = arr_safe + 50
print(f"\nWith int32: {arr_safe} + 50 = {result_safe}")

# Checking type limits
print("\n--- Type Limits ---")
for dtype in [np.int8, np.int16, np.int32, np.int64]:
    info = np.iinfo(dtype)
    print(f"{dtype.__name__:6} - min: {info.min:20,}, max: {info.max:20,}")

print()
for dtype in [np.float16, np.float32, np.float64]:
    info = np.finfo(dtype)
    print(f"{dtype.__name__:8} - min: {info.min:.2e}, max: {info.max:.2e}")

# ============================================================
# PART 6: Real-World Example - Optimizing Sensor Data
# ============================================================

print("\n" + "="*60)
print("PART 6: REAL-WORLD - OPTIMIZING SENSOR DATA")
print("="*60)

# Simulating 1 year of temperature readings (every minute)
# 365 days × 24 hours × 60 minutes = 525,600 readings
num_readings = 365 * 24 * 60

print(f"Scenario: {num_readings:,} temperature readings")
print("Temperature range: -50°C to 50°C")
print()

# Bad approach: float64 (default)
np.random.seed(42)
temps_float64 = np.random.uniform(-50, 50, num_readings)
memory_float64 = temps_float64.nbytes / 1_000_000

print("--- Approach 1: float64 (default) ---")
print(f"Memory: {memory_float64:.2f} MB")
print(f"Sample: {temps_float64[:5]}")

# Better: float32 (sufficient precision)
temps_float32 = temps_float64.astype(np.float32)
memory_float32 = temps_float32.nbytes / 1_000_000

print("\n--- Approach 2: float32 (sufficient precision) ---")
print(f"Memory: {memory_float32:.2f} MB")
print(f"Savings: {memory_float64 - memory_float32:.2f} MB ({(1 - memory_float32/memory_float64)*100:.0f}%)")
print(f"Sample: {temps_float32[:5]}")

# Best: Scale to int16 (if precision allows)
# Store as: (temp + 50) * 100 → -50.00°C to 50.00°C maps to 0 to 10000
# Precision: 0.01°C (1/100th of a degree)
temps_scaled = ((temps_float64 + 50) * 100).astype(np.int16)
memory_int16 = temps_scaled.nbytes / 1_000_000

print("\n--- Approach 3: Scaled int16 (±50°C, 0.01° precision) ---")
print(f"Memory: {memory_int16:.2f} MB")
print(f"Savings: {memory_float64 - memory_int16:.2f} MB ({(1 - memory_int16/memory_float64)*100:.0f}%)")
print(f"Sample (scaled): {temps_scaled[:5]}")

# Decode back to original
temps_decoded = (temps_scaled / 100.0) - 50
print(f"Sample (decoded): {temps_decoded[:5]}")
print(f"Error: {np.abs(temps_float64[:5] - temps_decoded[:5]).max():.6f}°C")

print(f"\n✅ Using int16 saves {memory_float64 - memory_int16:.2f} MB ({(1 - memory_int16/memory_float64)*100:.0f}%)")

# ============================================================
# PART 7: Structured Arrays (Record Arrays)
# ============================================================

print("\n" + "="*60)
print("PART 7: STRUCTURED ARRAYS (RECORD ARRAYS)")
print("="*60)

print("Structured arrays store heterogeneous data (like database records)")

# Define structure
dt = np.dtype([
    ('name', 'U20'),      # Unicode string, max 20 chars
    ('age', 'i4'),        # 32-bit integer
    ('salary', 'f8'),     # 64-bit float
    ('is_manager', '?')   # Boolean
])

# Create structured array
employees = np.array([
    ('Alice Johnson', 28, 75000.0, False),
    ('Bob Smith', 35, 92000.0, True),
    ('Charlie Davis', 42, 105000.0, True),
    ('Diana Lee', 31, 68000.0, False),
    ('Eve Wilson', 29, 71000.0, False)
], dtype=dt)

print("\nEmployee Records:")
print(employees)

# Access by field name
print("\n--- Accessing Fields ---")
print(f"Names: {employees['name']}")
print(f"Ages: {employees['age']}")
print(f"Salaries: {employees['salary']}")

# Statistics on specific fields
print("\n--- Field Statistics ---")
print(f"Average age: {employees['age'].mean():.1f}")
print(f"Average salary: ${employees['salary'].mean():,.2f}")
print(f"Number of managers: {employees['is_manager'].sum()}")

# Filtering
print("\n--- Filtering Records ---")
high_earners = employees[employees['salary'] > 80000]
print(f"Employees earning > $80k:")
for emp in high_earners:
    print(f"  {emp['name']}: ${emp['salary']:,.0f}")

# Sorting by field
print("\n--- Sorting by Salary ---")
sorted_by_salary = np.sort(employees, order='salary')
print("Top 3 earners:")
for emp in sorted_by_salary[-3:][::-1]:
    print(f"  {emp['name']}: ${emp['salary']:,.0f}")

# Modifying fields
print("\n--- Giving 10% Raise to Managers ---")
manager_mask = employees['is_manager']
employees['salary'][manager_mask] *= 1.10

print("Updated salaries:")
for emp in employees:
    role = "Manager" if emp['is_manager'] else "Employee"
    print(f"  {emp['name']:20} ({role}): ${emp['salary']:,.2f}")

# ============================================================
# PART 8: String Types
# ============================================================

print("\n" + "="*60)
print("PART 8: STRING TYPES")
print("="*60)

# Unicode strings (U)
unicode_arr = np.array(['Hello', 'World', 'NumPy'], dtype='U10')
print(f"Unicode array: {unicode_arr}")
print(f"dtype: {unicode_arr.dtype}")
print(f"Memory per item: {unicode_arr.dtype.itemsize} bytes")

# Byte strings (S) - more compact
byte_arr = np.array([b'Hello', b'World', b'NumPy'], dtype='S10')
print(f"\nByte array: {byte_arr}")
print(f"dtype: {byte_arr.dtype}")
print(f"Memory per item: {byte_arr.dtype.itemsize} bytes")

# String operations
print("\n--- String Operations ---")
names = np.array(['Alice', 'Bob', 'Charlie', 'David'])
print(f"Names: {names}")

# Length of each string
lengths = np.char.str_len(names)
print(f"Lengths: {lengths}")

# Uppercase
upper = np.char.upper(names)
print(f"Uppercase: {upper}")

# Concatenation
last_names = np.array(['Smith', 'Jones', 'Brown', 'Wilson'])
full_names = np.char.add(np.char.add(names, ' '), last_names)
print(f"Full names: {full_names}")

# Comparison
starts_with_a = np.char.startswith(names, 'A')
print(f"Starts with 'A': {starts_with_a}")
print(f"Names starting with 'A': {names[starts_with_a]}")

# ============================================================
# PART 9: Datetime Types
# ============================================================

print("\n" + "="*60)
print("PART 9: DATETIME TYPES")
print("="*60)

# Create datetime array
dates = np.array(['2024-01-01', '2024-01-15', '2024-02-01', '2024-03-01'], 
                 dtype='datetime64')
print(f"Dates: {dates}")
print(f"dtype: {dates.dtype}")

# Date arithmetic
print("\n--- Date Arithmetic ---")
days_later = dates + np.timedelta64(10, 'D')  # Add 10 days
print(f"10 days later: {days_later}")

# Date range
print("\n--- Date Range ---")
date_range = np.arange('2024-01-01', '2024-01-10', dtype='datetime64[D]')
print(f"Jan 1-9, 2024: {date_range}")

# Time deltas
print("\n--- Time Deltas ---")
delta = dates[1] - dates[0]
print(f"Days between {dates[0]} and {dates[1]}: {delta}")

# Extract components (requires conversion)
dates_dt = dates.astype('datetime64[D]')
print(f"\nDates as 'Day' precision: {dates_dt}")

# Business day calculations
print("\n--- Business Days ---")
start = np.datetime64('2024-01-01')  # Monday
business_days = np.busday_count(start, start + np.timedelta64(30, 'D'))
print(f"Business days in first 30 days of 2024: {business_days}")

# ============================================================
# PART 10: Type Coercion in Operations
# ============================================================

print("\n" + "="*60)
print("PART 10: TYPE COERCION IN OPERATIONS")
print("="*60)

print("NumPy automatically promotes types in mixed operations:")

# Integer + Float = Float
int_arr = np.array([1, 2, 3], dtype=np.int32)
float_arr = np.array([1.5, 2.5, 3.5], dtype=np.float32)

result = int_arr + float_arr
print(f"\nint32 + float32 = {result.dtype}")
print(f"  {int_arr} + {float_arr} = {result}")

# Different integer sizes
int8_arr = np.array([10, 20], dtype=np.int8)
int32_arr = np.array([100, 200], dtype=np.int32)

result = int8_arr + int32_arr
print(f"\nint8 + int32 = {result.dtype}")
print(f"  {int8_arr} + {int32_arr} = {result}")

# Type promotion rules
print("\n--- Type Promotion Hierarchy ---")
print("bool < int8 < int16 < int32 < int64 < float32 < float64")
print("Operations promote to the 'higher' type")

# Complex example
a = np.array([1], dtype=np.int8)
b = np.array([2], dtype=np.int16)
c = np.array([3.0], dtype=np.float32)

result = a + b + c
print(f"\nint8 + int16 + float32 = {result.dtype}")
print(f"Result: {result}")

# ============================================================
# PART 11: Memory-Efficient Data Loading
# ============================================================

print("\n" + "="*60)
print("PART 11: MEMORY-EFFICIENT DATA LOADING")
print("="*60)

# Create sample CSV data
csv_content = """id,temperature,humidity,pressure
1,22.5,45.2,1013.25
2,23.1,47.8,1012.80
3,21.9,46.5,1013.10
4,22.8,44.9,1013.50
5,23.5,48.1,1012.95"""

# Save to file
with open('sensor_data.csv', 'w') as f:
    f.write(csv_content)

# Load with default types
print("--- Loading with Default Types ---")
data_default = np.genfromtxt('sensor_data.csv', delimiter=',', skip_header=1)
print(f"dtype: {data_default.dtype}")
print(f"Memory: {data_default.nbytes} bytes")
print(data_default)

# Load with optimized types
print("\n--- Loading with Optimized Types ---")
data_optimized = np.genfromtxt('sensor_data.csv', delimiter=',', skip_header=1,
                               dtype=[('id', 'i2'), ('temp', 'f4'), 
                                      ('humidity', 'f4'), ('pressure', 'f4')])
print(f"dtypes: {data_optimized.dtype}")
print(f"Memory: {data_optimized.nbytes} bytes")
print(f"Savings: {data_default.nbytes - data_optimized.nbytes} bytes")
print(data_optimized)

# Clean up
import os
os.remove('sensor_data.csv')

# ============================================================
# PART 12: Best Practices Summary
# ============================================================

print("\n" + "="*60)
print("PART 12: DTYPE BEST PRACTICES")
print("="*60)

print("""
✅ DO:
1. Use smallest type that fits your data range
2. Use int8/uint8 for images (0-255)
3. Use float32 instead of float64 when precision allows
4. Use structured arrays for heterogeneous data
5. Check type limits before operations (np.iinfo, np.finfo)
6. Specify dtype when creating arrays for clarity
7. Use astype() for explicit conversions

❌ DON'T:
1. Use float64 everywhere (wastes memory)
2. Ignore overflow warnings
3. Use 'object' dtype for numeric data (very slow!)
4. Assume type conversions round (they truncate!)
5. Mix types without understanding coercion rules

💡 MEMORY OPTIMIZATION:
- Images: uint8 (8× smaller than float64)
- Counts/IDs: int16 or int32 (not int64 unless needed)
- Percentages: float32 (not float64)
- Flags: bool (smallest)
- Scaled integers: Store fixed-point as int16/int32

⚠️  WATCH OUT:
- Integer overflow wraps silently!
- float16 has limited range/precision
- String arrays have fixed max length
- Type promotion can surprise you
- astype() creates a COPY (memory cost)
""")

print("\n" + "="*60)
print("DTYPE CHEAT SHEET")
print("="*60)

print("""
COMMON USE CASES:

Image pixels:        uint8
Image processing:    float32
Indices/counts:      int32 or int16
Flags/masks:         bool
IDs (small):         int32
IDs (large):         int64
Money (cents):       int64 (avoid float for currency!)
Scientific:          float64
ML features:         float32
Percentages:         float32
Coordinates:         float32 or float64
Timestamps:          datetime64
Durations:           timedelta64
Names/labels:        U20 (Unicode, 20 chars)
Categories:          int8 or int16 (as codes)

MEMORY SIZES:
bool:     1 byte
int8:     1 byte
int16:    2 bytes
int32:    4 bytes
int64:    8 bytes
float16:  2 bytes
float32:  4 bytes
float64:  8 bytes
""")
