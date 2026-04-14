"""
LESSON 21: DATA TYPES AND CONVERSION IN PANDAS
================================================================================

What You Will Learn:
- Deep dive into all Pandas dtypes
- Memory impact of different dtype choices
- Nullable integer and boolean types
- String dtype vs object dtype
- Working with datetime and timedelta
- Automatic type inference and its pitfalls
- Optimizing a real dataset for memory efficiency
- Best practices for dtype management in production

Real World Usage:
- Reducing memory footprint of large datasets
- Preventing silent type errors in pipelines
- Correctly representing dates for time series analysis
- Optimizing DataFrames before saving to disk or passing to ML models
- Handling nullable integers in database-sourced data

Dataset Used:
NYC Yellow Taxi Trip Data (public, no login required)
URL: https://raw.githubusercontent.com/plotly/datasets/master/uber-rides-data1.csv
Fallback: Titanic dataset
https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv

================================================================================
"""

import numpy as np
import pandas as pd
import sys

print("=" * 70)
print("LESSON 21: DATA TYPES AND CONVERSION IN PANDAS")
print("=" * 70)


# ==============================================================================
# SECTION 1: LOAD DATASET
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 1: LOAD DATASET")
print("=" * 70)

primary_url = "https://raw.githubusercontent.com/plotly/datasets/master/uber-rides-data1.csv"
fallback_url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

def load_dataset():
    """Load primary dataset with fallback."""
    try:
        print("Loading Uber rides dataset from:")
        print(primary_url)
        df = pd.read_csv(primary_url)
        print("Primary dataset loaded successfully.")
        return df, "uber"
    except Exception as e:
        print(f"Primary load failed: {e}")
        print("\nFalling back to Titanic dataset from:")
        print(fallback_url)
        df = pd.read_csv(fallback_url)
        print("Fallback dataset loaded successfully.")
        return df, "titanic"

df_raw, dataset_name = load_dataset()

print(f"\nDataset: {dataset_name}")
print(f"Shape: {df_raw.shape}")
print(f"Columns: {list(df_raw.columns)}")
print("\nFirst 5 rows:")
print(df_raw.head())
print("\nData types as loaded:")
print(df_raw.dtypes)


# ==============================================================================
# SECTION 2: PANDAS DTYPE OVERVIEW
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 2: PANDAS DTYPE OVERVIEW")
print("=" * 70)

dtype_overview = """
PANDAS DTYPES AND THEIR PURPOSE
================================

NUMERIC TYPES:
  int8, int16, int32, int64       Signed integers (different memory sizes)
  uint8, uint16, uint32, uint64   Unsigned integers (no negative, larger max)
  float16, float32, float64       Floating point numbers
  Int8, Int16, Int32, Int64       Nullable signed integers (capital I)
  UInt8, UInt16, UInt32, UInt64   Nullable unsigned integers (capital U)

TEXT TYPES:
  object                          Python string objects (flexible but slow)
  string                          Pandas StringDtype (consistent, NA-aware)

BOOLEAN:
  bool                            Python bool (cannot hold NA)
  boolean                         Pandas BooleanDtype (nullable)

CATEGORICAL:
  category                        Finite set of values (memory efficient)

DATE AND TIME:
  datetime64[ns]                  Timestamps with nanosecond precision
  timedelta64[ns]                 Time differences
  period                          Fixed-frequency time periods
  DatetimeTZDtype                 Timezone-aware timestamps

MEMORY SIZES (bytes per element):
  int8 / uint8      1 byte    Range: -128 to 127 / 0 to 255
  int16 / uint16    2 bytes   Range: -32768 to 32767 / 0 to 65535
  int32 / uint32    4 bytes   Range: -2.1B to 2.1B / 0 to 4.3B
  int64 / uint64    8 bytes   Range: very large
  float16           2 bytes   ~3 decimal digits precision
  float32           4 bytes   ~7 decimal digits precision
  float64           8 bytes   ~15 decimal digits precision
"""
print(dtype_overview)


# ==============================================================================
# SECTION 3: MEMORY IMPACT OF DTYPES
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 3: MEMORY IMPACT OF DTYPES")
print("=" * 70)

# ------------------------------------------------------------------------------
# 3.1 Memory Usage of Current Dataset
# ------------------------------------------------------------------------------
print("\n--- 3.1 Memory Usage of Current Dataset ---")

def memory_report(data, label="DataFrame"):
    """Print memory usage per column and total."""
    mem = data.memory_usage(deep=True)
    total_kb = mem.sum() / 1024
    total_mb = total_kb / 1024

    print(f"\nMemory Report: {label}")
    print(f"{'Column':<25} {'Dtype':<20} {'Memory (KB)':>12}")
    print("-" * 60)

    for col in data.columns:
        col_mem_kb = data[col].memory_usage(deep=True) / 1024
        print(f"{col:<25} {str(data[col].dtype):<20} {col_mem_kb:>12.2f}")

    print("-" * 60)
    print(f"{'TOTAL':<25} {'':<20} {total_kb:>12.2f} KB")
    print(f"{'':25} {'':<20} {total_mb:>12.4f} MB")

memory_report(df_raw, f"Original {dataset_name} data")

# ------------------------------------------------------------------------------
# 3.2 Side by Side Comparison of Dtype Memory Usage
# ------------------------------------------------------------------------------
print("\n--- 3.2 Side by Side: Same Data, Different Dtypes ---")

# Create demonstration array with 1 million values
n = 1_000_000
values = np.random.randint(0, 100, n)

dtype_tests = [
    ("int8",    pd.array(values, dtype="int8")),
    ("int16",   pd.array(values, dtype="int16")),
    ("int32",   pd.array(values, dtype="int32")),
    ("int64",   pd.array(values, dtype="int64")),
    ("float32", pd.array(values.astype(float), dtype="float32")),
    ("float64", pd.array(values.astype(float), dtype="float64")),
    ("category",pd.Categorical(values)),
    ("object",  pd.array(values.astype(str), dtype="object")),
]

print(f"\nMemory usage for {n:,} elements with the same integer values (0-100):")
print(f"{'Dtype':<12} {'Memory (MB)':>14} {'Notes'}")
print("-" * 60)

int64_mem = None
for dtype_name, arr in dtype_tests:
    s = pd.Series(arr)
    mem_mb = s.memory_usage(deep=True) / 1024 / 1024
    if dtype_name == "int64":
        int64_mem = mem_mb

    saving = ""
    if int64_mem and dtype_name != "int64":
        pct = (1 - mem_mb / int64_mem) * 100
        saving = f"{pct:.0f}% smaller than int64" if pct > 0 else "larger than int64"

    print(f"{dtype_name:<12} {mem_mb:>12.2f} MB   {saving}")

print("\nKey insight: Choosing int8 instead of int64 can save 87.5% memory")
print("for columns with values in range -128 to 127")


# ==============================================================================
# SECTION 4: INTEGER DTYPES IN DETAIL
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 4: INTEGER DTYPES IN DETAIL")
print("=" * 70)

# ------------------------------------------------------------------------------
# 4.1 Standard Integer Types
# ------------------------------------------------------------------------------
print("\n--- 4.1 Standard Integer Types ---")

int_types = ["int8", "int16", "int32", "int64"]

print(f"{'Dtype':<10} {'Min':>25} {'Max':>25} {'Bytes':>8}")
print("-" * 72)
for dtype in int_types:
    info = np.iinfo(dtype)
    print(f"{dtype:<10} {info.min:>25,} {info.max:>25,} {np.dtype(dtype).itemsize:>8}")

# Overflow demonstration
print("\nOverflow risk demonstration:")
s_int8 = pd.Series([120, 125, 127], dtype="int8")
print(f"int8 Series: {s_int8.tolist()}")
try:
    result = s_int8 + 10
    print(f"After +10: {result.tolist()}  (silent overflow!)")
except Exception as e:
    print(f"Error: {e}")

# Safe: use int16 when values might exceed int8
s_int16 = s_int8.astype("int16") + 10
print(f"After upcasting to int16 then +10: {s_int16.tolist()}")

# ------------------------------------------------------------------------------
# 4.2 Nullable Integer Types (Int64 with capital I)
# ------------------------------------------------------------------------------
print("\n--- 4.2 Nullable Integer Types ---")

explanation = """
Standard int64 cannot hold NaN values.
When a column has missing integers (common in database data),
Pandas converts it to float64 to accommodate NaN.
Nullable integer types (Int64 capital I) solve this.
"""
print(explanation)

# The classic problem: integers that have missing values
print("Problem: integer column with NaN becomes float64")
s_float = pd.Series([1, 2, None, 4, 5])
print(f"  pd.Series([1, 2, None, 4, 5]) dtype: {s_float.dtype}")
print(f"  Values: {s_float.tolist()}")
print("  The integers became 1.0, 2.0 etc (float) due to NaN")

# Solution: nullable integer
print("\nSolution: use nullable integer dtype Int64 (capital I)")
s_int64_nullable = pd.Series([1, 2, None, 4, 5], dtype="Int64")
print(f"  pd.Series([1, 2, None, 4, 5], dtype='Int64') dtype: {s_int64_nullable.dtype}")
print(f"  Values: {s_int64_nullable.tolist()}")
print("  Integers remain integers, NaN is pd.NA")

# Memory comparison
mem_float = s_float.memory_usage(deep=True)
mem_nullable = s_int64_nullable.memory_usage(deep=True)
print(f"\nMemory usage comparison (5 elements):")
print(f"  float64 with NaN:      {mem_float} bytes")
print(f"  Int64 nullable:        {mem_nullable} bytes")

# Operations still work
print("\nOperations on nullable integer Series:")
print(f"  Sum (skips NA): {s_int64_nullable.sum()}")
print(f"  Mean (skips NA): {s_int64_nullable.mean()}")
print(f"  Filled NA: {s_int64_nullable.fillna(0).tolist()}")

# ------------------------------------------------------------------------------
# 4.3 When to Use Which Integer Type
# ------------------------------------------------------------------------------
print("\n--- 4.3 When to Use Which Integer Type ---")

guidance = """
int8    Values -128 to 127         Small flags, scores out of 100
int16   Values -32768 to 32767     Year numbers, small counts
int32   Values up to ~2 billion    Most typical integer data
int64   Very large integers         Default, IDs, timestamps
Int64   Integer but has missing    Database foreign keys, optional fields
uint8   0 to 255                   Pixel values, byte data
uint16  0 to 65535                 Port numbers, image dimensions
"""
print(guidance)


# ==============================================================================
# SECTION 5: FLOAT DTYPES IN DETAIL
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 5: FLOAT DTYPES IN DETAIL")
print("=" * 70)

# ------------------------------------------------------------------------------
# 5.1 Float Precision Comparison
# ------------------------------------------------------------------------------
print("\n--- 5.1 Float Precision Comparison ---")

float_types = ["float16", "float32", "float64"]

print(f"{'Dtype':<12} {'Bytes':>8} {'Approx Precision':>20}")
print("-" * 44)
for dtype in float_types:
    info = np.finfo(dtype)
    print(f"{dtype:<12} {np.dtype(dtype).itemsize:>8} {info.precision:>20} decimal digits")

# Precision loss demonstration
print("\nPrecision loss in float16 vs float64:")
value = 3.141592653589793

f16 = np.float16(value)
f32 = np.float32(value)
f64 = np.float64(value)

print(f"  Original:   {value}")
print(f"  float16:    {f16}  (significant precision loss)")
print(f"  float32:    {f32}")
print(f"  float64:    {f64}")

# ------------------------------------------------------------------------------
# 5.2 When float32 is Acceptable
# ------------------------------------------------------------------------------
print("\n--- 5.2 When float32 is Acceptable ---")

print("float32 is acceptable when:")
print("  - Data has limited real-world precision (e.g., sensor readings to 2 dp)")
print("  - Working with ML models that use float32 internally")
print("  - Memory is constrained and marginal precision loss is acceptable")
print("\nfloat64 is required when:")
print("  - Financial calculations (money amounts)")
print("  - Scientific calculations requiring high precision")
print("  - Aggregations that accumulate many small differences")

# Example: float32 sufficient for temperatures
temps_f64 = pd.Series([22.5, 18.3, 25.1, 30.7], dtype="float64")
temps_f32 = pd.Series([22.5, 18.3, 25.1, 30.7], dtype="float32")

print(f"\nTemperature readings:")
print(f"  float64: {temps_f64.tolist()}")
print(f"  float32: {temps_f32.tolist()}")
print(f"  Difference acceptable for sensor data: {all(abs(temps_f64 - temps_f32) < 0.001)}")


# ==============================================================================
# SECTION 6: OBJECT VS STRING DTYPE
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 6: OBJECT VS STRING DTYPE")
print("=" * 70)

# ------------------------------------------------------------------------------
# 6.1 The Object Dtype Problem
# ------------------------------------------------------------------------------
print("\n--- 6.1 The Object Dtype Problem ---")

explanation = """
'object' dtype is Pandas default for text columns.
Under the hood it stores Python str objects (or any Python object).
Problems with object dtype:
  - Slower string operations
  - Higher memory usage
  - Ambiguous: can store mixed types silently
  - No consistent handling of NA values

'string' dtype (StringDtype) was added in Pandas 1.0 to fix this:
  - Consistent NA handling using pd.NA
  - Slightly better performance for string operations
  - Clear intent: this column holds only strings
"""
print(explanation)

# Create example
names_object = pd.Series(["Alice", "Bob", None, "Charlie"], dtype="object")
names_string = pd.Series(["Alice", "Bob", None, "Charlie"], dtype="string")

print("object dtype Series:")
print(f"  Values: {names_object.tolist()}")
print(f"  Dtype: {names_object.dtype}")
print(f"  NA type: {type(names_object[2])}")

print("\nstring dtype Series:")
print(f"  Values: {names_string.tolist()}")
print(f"  Dtype: {names_string.dtype}")
print(f"  NA type: {type(names_string[2])}")

print("\nNA comparison:")
print(f"  object NA is np.nan:  {names_object[2] is np.nan}")
print(f"  string NA is pd.NA:   {names_string[2] is pd.NA}")

# ------------------------------------------------------------------------------
# 6.2 Memory and Performance Comparison
# ------------------------------------------------------------------------------
print("\n--- 6.2 Memory Comparison: object vs string vs category ---")

n = 100_000
words = ["apple", "banana", "cherry", "date", "elderberry"]
data = np.random.choice(words, n)

s_object   = pd.Series(data, dtype="object")
s_string   = pd.Series(data, dtype="string")
s_category = pd.Series(data, dtype="category")

print(f"Series with {n:,} string values (5 unique words):")
print(f"  object dtype:   {s_object.memory_usage(deep=True)/1024:.1f} KB")
print(f"  string dtype:   {s_string.memory_usage(deep=True)/1024:.1f} KB")
print(f"  category dtype: {s_category.memory_usage(deep=True)/1024:.1f} KB")

print("\nKey rule:")
print("  Use category when column has low cardinality (few unique values)")
print("  Use string for high cardinality free-text columns")
print("  Avoid object unless you need to store mixed types")


# ==============================================================================
# SECTION 7: BOOLEAN DTYPE
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 7: BOOLEAN DTYPE")
print("=" * 70)

# ------------------------------------------------------------------------------
# 7.1 bool vs boolean (Nullable)
# ------------------------------------------------------------------------------
print("\n--- 7.1 bool vs boolean (Nullable) ---")

# Standard bool cannot hold NA
s_bool = pd.Series([True, False, True])
print(f"Standard bool Series dtype: {s_bool.dtype}")
print(f"Values: {s_bool.tolist()}")

# What happens when NA is introduced
s_bool_with_na = pd.Series([True, False, None])
print(f"\nBool Series with None - dtype becomes: {s_bool_with_na.dtype}")
print(f"Values: {s_bool_with_na.tolist()}")
print("Becomes object dtype - problematic!")

# Nullable boolean
s_boolean = pd.Series([True, False, None], dtype="boolean")
print(f"\nNullable boolean dtype: {s_boolean.dtype}")
print(f"Values: {s_boolean.tolist()}")
print(f"NA value: {s_boolean[2]}, type: {type(s_boolean[2])}")

# Operations still work correctly
print(f"\nOperations on nullable boolean:")
print(f"  Any True: {s_boolean.any()}")
print(f"  All True: {s_boolean.all()}")
print(f"  Sum (counts True): {s_boolean.sum()}")

# ------------------------------------------------------------------------------
# 7.2 Boolean from Conditions
# ------------------------------------------------------------------------------
print("\n--- 7.2 Creating Boolean Columns from Conditions ---")

df = df_raw.copy()

if dataset_name == "titanic":
    # Create boolean feature columns
    df["is_adult"] = (df["Age"] >= 18).astype("boolean")
    df["is_survived"] = (df["Survived"] == 1).astype("boolean")
    df["is_first_class"] = (df["Pclass"] == 1).astype("boolean")

    print("Boolean columns created from conditions:")
    print(df[["Age", "is_adult", "Survived", "is_survived", "Pclass", "is_first_class"]].head(10))

    print("\nMemory usage of boolean vs int columns:")
    print(f"  Survived (int64):    {df['Survived'].memory_usage(deep=True)} bytes")
    print(f"  is_survived (bool):  {df['is_survived'].memory_usage(deep=True)} bytes")
else:
    print("Uber dataset - demonstrating boolean on available columns")
    first_col = df.columns[0]
    df["example_bool"] = (df[first_col].notna()).astype("boolean")
    print(df[["example_bool"]].head())


# ==============================================================================
# SECTION 8: CATEGORY DTYPE IN DEPTH
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 8: CATEGORY DTYPE IN DEPTH")
print("=" * 70)

# ------------------------------------------------------------------------------
# 8.1 How Category Works Internally
# ------------------------------------------------------------------------------
print("\n--- 8.1 How Category Works Internally ---")

explanation = """
Category dtype stores:
  - An array of integer CODES (one per row)
  - A small array of CATEGORIES (the unique values)

Instead of storing "male" 577 times, it stores:
  - categories = ["female", "male"]
  - codes      = [1, 0, 1, 1, 0, 0, ...] (integers 0/1)

This is why it saves memory for low-cardinality columns.
"""
print(explanation)

if dataset_name == "titanic":
    sex_object   = df_raw["Sex"].astype("object")
    sex_category = df_raw["Sex"].astype("category")

    print(f"Sex column ({sex_object.nunique()} unique values):")
    print(f"  object dtype memory:   {sex_object.memory_usage(deep=True):,} bytes")
    print(f"  category dtype memory: {sex_category.memory_usage(deep=True):,} bytes")
    print(f"\nCategory codes (first 10): {sex_category.cat.codes[:10].tolist()}")
    print(f"Category values: {sex_category.cat.categories.tolist()}")

# ------------------------------------------------------------------------------
# 8.2 Ordered Categories
# ------------------------------------------------------------------------------
print("\n--- 8.2 Ordered Categories ---")

# Ordered categories allow comparison operators
size_series = pd.Series(
    ["small", "large", "medium", "small", "large", "medium"],
    dtype="category"
)
print("Unordered category:")
print(f"  Categories: {size_series.cat.categories.tolist()}")

# Add ordering
size_ordered = pd.Categorical(
    size_series,
    categories=["small", "medium", "large"],
    ordered=True
)
size_series_ordered = pd.Series(size_ordered)
print("\nOrdered category (small < medium < large):")
print(f"  Categories: {size_series_ordered.cat.categories.tolist()}")
print(f"  Is ordered: {size_series_ordered.cat.ordered}")

# Comparison now works
print(f"\nComparison 'small' < 'large': {(size_series_ordered == 'small').any()}")

# Sorting respects order
print(f"Sorted: {sorted(size_series_ordered.unique())}")

# ------------------------------------------------------------------------------
# 8.3 When to Use Category
# ------------------------------------------------------------------------------
print("\n--- 8.3 When to Use Category ---")

guidance = """
USE category when:
  - Column has low cardinality (less than 50% unique values)
  - Column values are repeated frequently (Sex, Pclass, country codes)
  - You want to enable ordered comparisons (small < medium < large)
  - Memory is a concern

DO NOT USE category when:
  - Column has high cardinality (names, emails, IDs)
  - Values change frequently (adding new categories is costly)
  - You are doing heavy string operations
"""
print(guidance)

if dataset_name == "titanic":
    print("Cardinality check for Titanic columns:")
    for col in df_raw.select_dtypes(include=["object"]).columns:
        unique_count = df_raw[col].nunique()
        total = len(df_raw)
        ratio = unique_count / total
        recommendation = "USE category" if ratio < 0.5 else "KEEP as string"
        print(f"  {col:<12}: {unique_count:>4} unique / {total} total "
              f"({ratio*100:.1f}%)  -> {recommendation}")


# ==============================================================================
# SECTION 9: DATETIME DTYPE IN DEPTH
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 9: DATETIME DTYPE IN DEPTH")
print("=" * 70)

# ------------------------------------------------------------------------------
# 9.1 Converting Strings to Datetime
# ------------------------------------------------------------------------------
print("\n--- 9.1 Converting Strings to Datetime ---")

# Common formats seen in real data
date_strings = pd.Series([
    "2024-01-15",
    "2024-02-20",
    "2024-03-10",
    "2024-11-05"
])

# Method 1: pd.to_datetime (most flexible)
dates_auto = pd.to_datetime(date_strings)
print("Auto-parsed dates:")
print(f"  Input dtype: {date_strings.dtype}")
print(f"  Output dtype: {dates_auto.dtype}")
print(f"  Values: {dates_auto.tolist()}")

# Specific format for performance (faster on large datasets)
dates_format = pd.to_datetime(date_strings, format="%Y-%m-%d")
print(f"\nExplicit format parsing: {dates_format.dtype}")

# Various input formats
messy_dates = pd.Series(["15/01/2024", "20/02/2024", "10/03/2024"])
dates_messy = pd.to_datetime(messy_dates, format="%d/%m/%Y")
print(f"\nEuropean format dates: {dates_messy.tolist()}")

# Handle errors gracefully
mixed_dates = pd.Series(["2024-01-15", "not_a_date", "2024-03-10"])
dates_coerced = pd.to_datetime(mixed_dates, errors="coerce")
print(f"\nCoerced bad dates to NaT: {dates_coerced.tolist()}")

# ------------------------------------------------------------------------------
# 9.2 Extracting Date Components
# ------------------------------------------------------------------------------
print("\n--- 9.2 Extracting Date Components ---")

# Create a demo DataFrame with dates
date_df = pd.DataFrame({
    "timestamp": pd.date_range("2024-01-01", periods=10, freq="D"),
    "value": np.random.randint(10, 100, 10)
})

# Extract components using .dt accessor
date_df["year"]        = date_df["timestamp"].dt.year
date_df["month"]       = date_df["timestamp"].dt.month
date_df["day"]         = date_df["timestamp"].dt.day
date_df["day_of_week"] = date_df["timestamp"].dt.day_name()
date_df["is_weekend"]  = date_df["timestamp"].dt.dayofweek >= 5
date_df["quarter"]     = date_df["timestamp"].dt.quarter
date_df["week"]        = date_df["timestamp"].dt.isocalendar().week

print("Date components extracted:")
print(date_df)

# ------------------------------------------------------------------------------
# 9.3 Timedelta - Duration Between Dates
# ------------------------------------------------------------------------------
print("\n--- 9.3 Timedelta - Duration Between Dates ---")

# Common use case: compute age, tenure, or duration
start_dates = pd.Series(pd.to_datetime([
    "2020-01-01",
    "2019-06-15",
    "2023-03-22"
]))
end_date = pd.Timestamp("2024-06-01")

# Compute duration
durations = end_date - start_dates
print("Durations from start to reference date:")
print(f"  Dtype: {durations.dtype}")
print(f"  Values: {durations.tolist()}")

# Extract useful units from timedelta
print("\nExtracted duration components:")
print(f"  Days: {durations.dt.days.tolist()}")
print(f"  Months (approx): {(durations.dt.days / 30.44).round(1).tolist()}")
print(f"  Years (approx): {(durations.dt.days / 365.25).round(2).tolist()}")

# ------------------------------------------------------------------------------
# 9.4 Timezone Handling
# ------------------------------------------------------------------------------
print("\n--- 9.4 Timezone Handling ---")

# Create timezone-naive datetime
naive_dt = pd.Series(pd.date_range("2024-01-01", periods=3, freq="h"))
print(f"Timezone-naive: {naive_dt.tolist()}")
print(f"Dtype: {naive_dt.dtype}")

# Localize to a timezone
aware_dt = naive_dt.dt.tz_localize("UTC")
print(f"\nAfter UTC localize: {aware_dt.tolist()}")
print(f"Dtype: {aware_dt.dtype}")

# Convert to another timezone
ny_dt = aware_dt.dt.tz_convert("America/New_York")
print(f"\nConverted to New York: {ny_dt.tolist()}")

print("\nReal world rule:")
print("  Store all timestamps in UTC internally")
print("  Convert to local timezone only for display")


# ==============================================================================
# SECTION 10: TYPE INFERENCE PITFALLS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 10: TYPE INFERENCE PITFALLS")
print("=" * 70)

# ------------------------------------------------------------------------------
# 10.1 Mixed Type Columns
# ------------------------------------------------------------------------------
print("\n--- 10.1 Mixed Type Columns ---")

# Pandas will use object dtype when types are mixed
mixed = pd.Series([1, 2, "three", 4, 5])
print(f"Mixed types series dtype: {mixed.dtype}")
print(f"Values: {mixed.tolist()}")
print("All elements stored as Python objects - operations will fail!")

# Attempt numeric operation
try:
    result = mixed * 2
    print(f"Mixed * 2: {result.tolist()}")
except TypeError as e:
    print(f"TypeError on numeric operation: {e}")

# How to detect and fix
print("\nDetecting mixed types:")
type_check = mixed.apply(type)
print(type_check.tolist())

# Coerce to numeric
mixed_numeric = pd.to_numeric(mixed, errors="coerce")
print(f"\nAfter pd.to_numeric(errors='coerce'): {mixed_numeric.tolist()}")
print(f"Dtype: {mixed_numeric.dtype}")

# ------------------------------------------------------------------------------
# 10.2 Integers Becoming Floats Due to NaN
# ------------------------------------------------------------------------------
print("\n--- 10.2 Integers Silently Becoming Floats ---")

# This is a very common source of bugs in database-sourced data
data_with_gap = pd.Series([1, 2, None, 4, 5])
print(f"Integer data with None:")
print(f"  Input: [1, 2, None, 4, 5]")
print(f"  Dtype: {data_with_gap.dtype}  <- became float!")
print(f"  Values: {data_with_gap.tolist()}")

# Fix with nullable integer
data_fixed = pd.Series([1, 2, None, 4, 5], dtype="Int64")
print(f"\nWith nullable Int64:")
print(f"  Dtype: {data_fixed.dtype}")
print(f"  Values: {data_fixed.tolist()}")

# Real-world implication
print("\nWhy this matters:")
print("  Downstream joins on float keys may fail silently")
print("  Equality checks: 1.0 == 1 but hash(1.0) == hash(1) in Python")
print("  Better to use Int64 and be explicit about missing integer data")

# ------------------------------------------------------------------------------
# 10.3 Numeric Columns Stored as Strings
# ------------------------------------------------------------------------------
print("\n--- 10.3 Numeric Columns Stored as Strings ---")

# Common when reading from CSV files with formatting
numeric_as_string = pd.Series(["1,200", "3,450", "890", "12,000"])
print(f"Numeric strings with commas: {numeric_as_string.tolist()}")

# Attempt direct conversion - fails
try:
    pd.to_numeric(numeric_as_string)
except ValueError as e:
    print(f"Direct conversion fails: {e}")

# Correct approach: remove formatting first
cleaned = numeric_as_string.str.replace(",", "", regex=False)
converted = pd.to_numeric(cleaned)
print(f"After removing commas and converting: {converted.tolist()}")
print(f"Dtype: {converted.dtype}")

# Currency strings
currency_strings = pd.Series(["$1,200.50", "$3,450.00", "$890.99"])
cleaned_currency = (
    currency_strings
    .str.replace("$", "", regex=False)
    .str.replace(",", "", regex=False)
)
converted_currency = pd.to_numeric(cleaned_currency)
print(f"\nCurrency strings: {currency_strings.tolist()}")
print(f"After cleaning: {converted_currency.tolist()}")


# ==============================================================================
# SECTION 11: OPTIMIZING A FULL DATASET
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 11: OPTIMIZING A FULL DATASET")
print("=" * 70)

def optimize_dtypes(data, verbose=True):
    """
    Automatically optimize DataFrame dtypes to reduce memory usage.

    Strategy:
      - Integer columns: downcast to smallest fitting int type
      - Float columns: downcast to float32 if precision allows
      - Object columns with low cardinality: convert to category
      - Object columns that look like dates: convert to datetime

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame
    verbose : bool
        Print optimization steps

    Returns
    -------
    pd.DataFrame
        Optimized DataFrame
    """
    df_opt = data.copy()
    original_mem = df_opt.memory_usage(deep=True).sum()

    if verbose:
        print("Starting dtype optimization...")

    for col in df_opt.columns:
        col_type = df_opt[col].dtype

        # Integer columns: downcast
        if pd.api.types.is_integer_dtype(col_type):
            col_min = df_opt[col].min()
            col_max = df_opt[col].max()

            if col_min >= 0:
                # Unsigned
                if col_max <= 255:
                    df_opt[col] = df_opt[col].astype("uint8")
                elif col_max <= 65535:
                    df_opt[col] = df_opt[col].astype("uint16")
                elif col_max <= 4294967295:
                    df_opt[col] = df_opt[col].astype("uint32")
            else:
                # Signed
                if col_min >= -128 and col_max <= 127:
                    df_opt[col] = df_opt[col].astype("int8")
                elif col_min >= -32768 and col_max <= 32767:
                    df_opt[col] = df_opt[col].astype("int16")
                elif col_min >= -2147483648 and col_max <= 2147483647:
                    df_opt[col] = df_opt[col].astype("int32")

            if verbose:
                print(f"  {col}: {col_type} -> {df_opt[col].dtype} (integer downcast)")

        # Float columns: downcast to float32
        elif pd.api.types.is_float_dtype(col_type):
            df_opt[col] = df_opt[col].astype("float32")
            if verbose:
                print(f"  {col}: {col_type} -> float32 (float downcast)")

        # Object columns: check cardinality
        elif col_type == object:
            num_unique = df_opt[col].nunique()
            num_total = len(df_opt[col])
            ratio = num_unique / num_total

            if ratio < 0.5:
                df_opt[col] = df_opt[col].astype("category")
                if verbose:
                    print(f"  {col}: object -> category "
                          f"({num_unique} unique, {ratio*100:.1f}% cardinality)")
            else:
                if verbose:
                    print(f"  {col}: object kept "
                          f"({num_unique} unique, {ratio*100:.1f}% cardinality)")

    optimized_mem = df_opt.memory_usage(deep=True).sum()
    saving = (1 - optimized_mem / original_mem) * 100

    if verbose:
        print(f"\nMemory before: {original_mem / 1024:.1f} KB")
        print(f"Memory after:  {optimized_mem / 1024:.1f} KB")
        print(f"Memory saved:  {saving:.1f}%")

    return df_opt


# Apply optimization to loaded dataset
print("Applying dtype optimization to loaded dataset...")
df_optimized = optimize_dtypes(df_raw)

print("\nDtype comparison before and after:")
print(f"{'Column':<20} {'Before':<15} {'After':<15}")
print("-" * 52)
for col in df_raw.columns:
    before = str(df_raw[col].dtype)
    after  = str(df_optimized[col].dtype)
    changed = " <-- changed" if before != after else ""
    print(f"{col:<20} {before:<15} {after:<15}{changed}")


# ==============================================================================
# SECTION 12: TYPE CONVERSION REFERENCE
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 12: TYPE CONVERSION REFERENCE")
print("=" * 70)

print("\n--- Conversion Methods Summary ---")

conversion_guide = """
FROM              TO              METHOD
--------------    -----------     ----------------------------------------
object/string     int/float       pd.to_numeric(s, errors='coerce')
object/string     datetime        pd.to_datetime(s, errors='coerce')
object/string     category        s.astype('category')
object/string     boolean         s.map({'yes': True, 'no': False})
int               float           s.astype('float32')
int               bool            s.astype('bool')
int               Int64 nullable  s.astype('Int64')
float             int (if whole)  s.round().astype('Int64')
float             bool            s.astype('bool')
any               object          s.astype('object')
any               string          s.astype('string')
datetime          period          s.dt.to_period('M')
datetime string   timestamp       pd.to_datetime(s, format='%Y-%m-%d')

SAFE CONVERSION PATTERN:
  Always use errors='coerce' when converting from strings
  Then fill or drop resulting NaN values
  Then validate the result with assertions
"""
print(conversion_guide)


# ==============================================================================
# SECTION 13: VALIDATION AFTER TYPE CONVERSION
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 13: VALIDATION AFTER TYPE CONVERSION")
print("=" * 70)

def validate_dtypes(data, expected_dtypes):
    """
    Validate that DataFrame columns have expected dtypes.

    Parameters
    ----------
    data : pd.DataFrame
    expected_dtypes : dict
        {column_name: expected_dtype_string}

    Returns
    -------
    bool
        True if all checks pass
    """
    print("Validating dtypes...")
    all_passed = True

    for col, expected in expected_dtypes.items():
        if col not in data.columns:
            print(f"  [FAIL] Column '{col}' not found in DataFrame")
            all_passed = False
            continue

        actual = str(data[col].dtype)
        if actual == expected:
            print(f"  [PASS] {col}: {actual}")
        else:
            print(f"  [FAIL] {col}: expected {expected}, got {actual}")
            all_passed = False

    return all_passed


# Define expected dtypes for a cleaned Titanic dataset
if dataset_name == "titanic":
    expected = {
        "PassengerId": "int64",
        "Survived": "int64",
        "Pclass": "int64",
        "Sex": "object",
        "Age": "float64",
        "Fare": "float64"
    }
    validate_dtypes(df_raw, expected)


# ==============================================================================
# SECTION 14: COMMON PITFALLS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 14: COMMON PITFALLS")
print("=" * 70)

pitfalls = """
Pitfall 1: Assuming integer columns stay integer after merge or groupby
   - Operations like groupby mean convert int to float
   - Merges can introduce NaN which promotes int to float
   - Always check dtypes after any multi-step operation

Pitfall 2: Using astype() without handling NaN first
   - s.astype('int64') fails if NaN values are present
   - Fix: use s.dropna().astype('int64') or nullable Int64

Pitfall 3: Storing IDs as floats
   - If PassengerId = 1.0 due to NaN promotion, joins may fail
   - Fix: use nullable Int64 to keep integer IDs intact

Pitfall 4: Not specifying format in pd.to_datetime()
   - Auto-inference is slow on large datasets
   - Ambiguous dates (01/02/2024 is Jan 2 or Feb 1?)
   - Always specify format= for production code

Pitfall 5: Converting high-cardinality columns to category
   - category is efficient only for low-cardinality columns
   - Adding new values requires updating categories list
   - For names/emails, string dtype is better

Pitfall 6: Comparing after type change
   - 1 == 1.0 is True, but '1' == 1 is False in Python
   - After string to numeric conversion, old string comparisons break
   - Always update all filter conditions after dtype changes

Pitfall 7: Using object dtype for intended numeric columns
   - CSV reading may give you '123' instead of 123
   - Always check dtypes immediately after loading
   - Use df.dtypes and df.head() together to catch this
"""
print(pitfalls)


# ==============================================================================
# SECTION 15: PRACTICE EXERCISES
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 15: PRACTICE EXERCISES")
print("=" * 70)

if dataset_name == "titanic":
    print("Exercise 1: Convert Survived to nullable boolean")
    df_ex = df_raw.copy()
    df_ex["Survived"] = df_ex["Survived"].astype("boolean")
    print(f"  Dtype: {df_ex['Survived'].dtype}")
    print(f"  Survival rate: {df_ex['Survived'].mean():.3f}")

    print("\nExercise 2: Downcast Age to smallest fitting nullable int")
    df_ex["Age_filled"] = df_ex["Age"].fillna(-1).astype("int16")
    print(f"  Dtype: {df_ex['Age_filled'].dtype}")
    print(f"  Memory: {df_ex['Age_filled'].memory_usage(deep=True)} bytes")
    print(f"  vs float64: {df_ex['Age'].memory_usage(deep=True)} bytes")

    print("\nExercise 3: Find columns suitable for category dtype")
    for col in df_raw.select_dtypes(include=["object"]).columns:
        ratio = df_raw[col].nunique() / len(df_raw)
        suitable = "YES" if ratio < 0.5 else "NO"
        print(f"  {col:<15} cardinality {ratio:.2%}  suitable: {suitable}")

    print("\nExercise 4: Compare memory before and after full optimization")
    print(f"  Before: {df_raw.memory_usage(deep=True).sum() / 1024:.1f} KB")
    df_opt_ex = optimize_dtypes(df_raw, verbose=False)
    print(f"  After:  {df_opt_ex.memory_usage(deep=True).sum() / 1024:.1f} KB")


# ==============================================================================
# SECTION 16: KEY TAKEAWAYS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 16: KEY TAKEAWAYS")
print("=" * 70)

takeaways = """
1. Always inspect dtypes after loading with df.dtypes and df.head()
2. Use nullable types (Int64, boolean) for integer/bool columns with NaN
3. Prefer category dtype for low-cardinality string columns
4. Use float32 instead of float64 when precision loss is acceptable
5. Use string dtype instead of object for text columns
6. Always specify format= in pd.to_datetime() for production code
7. Store timestamps in UTC, convert to local timezone only for display
8. Use pd.to_numeric(errors='coerce') for safe string to number conversion
9. Validate dtypes explicitly after any multi-step transformation
10. Memory optimization through dtype selection can save 50-80% of memory
"""
print(takeaways)


# ==============================================================================
# NEXT LESSON PREVIEW
# ==============================================================================
print("\n" + "=" * 70)
print("NEXT LESSON PREVIEW")
print("=" * 70)

preview = """
Lesson 22: Renaming Columns and Working with Index

You will learn:
- Renaming columns with rename() and set_axis()
- Bulk renaming patterns (lowercase, strip spaces, replace characters)
- Setting meaningful index columns
- Resetting and reindexing DataFrames
- Multi-level (hierarchical) index basics
- Index alignment behavior in operations
- Best practices for column and index naming in production
"""
print(preview)

print("\n" + "=" * 70)
print("END OF LESSON 21")
print("=" * 70)
print("\nYou now understand all Pandas dtypes, their memory implications,")
print("and how to choose and convert types correctly for production pipelines.")