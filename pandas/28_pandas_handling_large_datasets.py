"""
LESSON 28: HANDLING LARGE DATASETS EFFICIENTLY
================================================================================

What You Will Learn:
- Memory profiling of DataFrames
- Dtype optimization for memory reduction
- Chunked reading with pd.read_csv(chunksize=)
- Selecting only needed columns on load
- Efficient filtering strategies before computation
- Using categorical dtype for string columns
- Sparse data representation
- Memory-efficient aggregation patterns
- When to consider alternatives (NumPy, DuckDB)
- Building memory-efficient pipelines

Real World Usage:
- Processing multi-gigabyte CSV files on limited RAM
- Loading only relevant columns from wide datasets
- Aggregating log files too large to fit in memory
- Building ETL pipelines for large scale data
- Reducing cloud compute costs through memory optimization
- Preprocessing large datasets for machine learning

Dataset Used:
NYC Yellow Taxi Trip Records (public, no login required)
URL: https://raw.githubusercontent.com/plotly/datasets/master/uber-rides-data1.csv
Fallback: Titanic dataset with synthetic expansion
https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv

================================================================================
"""

import numpy as np
import pandas as pd
import time
import io
import re
import sys
import os
import gc

print("=" * 70)
print("LESSON 28: HANDLING LARGE DATASETS EFFICIENTLY")
print("=" * 70)


# ==============================================================================
# SECTION 1: SETUP AND CREATE LARGE SYNTHETIC DATASET
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 1: SETUP AND CREATE LARGE SYNTHETIC DATASET")
print("=" * 70)

np.random.seed(42)

def to_snake(name):
    """Convert column name to snake_case."""
    name = str(name).strip()
    name = re.sub(r"[\s\-\.]+", "_", name)
    name = re.sub(r"[^\w]", "", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_").lower()


# Create a realistic large dataset in memory (simulates a large CSV file)
# We use 1 million rows to demonstrate real memory issues
N_ROWS = 1_000_000

print(f"Generating synthetic transaction dataset with {N_ROWS:,} rows...")

# Realistic e-commerce transaction data
regions      = ["NORTH", "SOUTH", "EAST", "WEST", "CENTRAL"]
categories   = ["electronics", "clothing", "home", "sports", "food", "books"]
payment_types = ["credit_card", "debit_card", "cash", "digital_wallet"]
statuses     = ["completed", "pending", "cancelled", "refunded"]

raw_data = {
    "transaction_id": np.arange(1, N_ROWS + 1, dtype=np.int64),
    "customer_id":    np.random.randint(1, 50_001, N_ROWS).astype(np.int32),
    "product_id":     np.random.randint(1, 10_001, N_ROWS).astype(np.int32),
    "region":         np.random.choice(regions, N_ROWS),
    "category":       np.random.choice(categories, N_ROWS),
    "payment_type":   np.random.choice(payment_types, N_ROWS),
    "status":         np.random.choice(statuses, N_ROWS),
    "quantity":       np.random.randint(1, 11, N_ROWS).astype(np.int8),
    "unit_price":     np.round(np.random.uniform(5.0, 500.0, N_ROWS), 2).astype(np.float32),
    "discount_pct":   np.round(np.random.uniform(0.0, 0.5, N_ROWS), 4).astype(np.float32),
    "rating":         np.random.choice([1, 2, 3, 4, 5], N_ROWS).astype(np.int8),
    "year":           np.random.choice([2021, 2022, 2023], N_ROWS).astype(np.int16),
    "month":          np.random.randint(1, 13, N_ROWS).astype(np.int8),
    "day":            np.random.randint(1, 29, N_ROWS).astype(np.int8),
}

# Add computed column
raw_data["total_price"] = np.round(
    raw_data["quantity"] * raw_data["unit_price"] * (1 - raw_data["discount_pct"]),
    2
).astype(np.float32)

# Inject some missing values
missing_idx = np.random.choice(N_ROWS, size=N_ROWS // 20, replace=False)
rating_array = raw_data["rating"].astype(float)
rating_array[missing_idx[:N_ROWS // 40]] = np.nan
raw_data["rating"] = rating_array

print(f"Dataset generated successfully.")
print(f"Columns: {list(raw_data.keys())}")
print(f"Total columns: {len(raw_data)}")

# Serialize to CSV in memory (simulates reading from disk)
# We will use this buffer throughout the lesson
df_initial = pd.DataFrame(raw_data)

# Write to an in-memory buffer
buffer = io.StringIO()
df_initial.to_csv(buffer, index=False)
csv_content = buffer.getvalue()
csv_size_mb = sys.getsizeof(csv_content) / 1024 / 1024

print(f"\nSimulated CSV file size: {csv_size_mb:.1f} MB")
print(f"Initial DataFrame memory: {df_initial.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")


# ==============================================================================
# SECTION 2: MEMORY PROFILING
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 2: MEMORY PROFILING")
print("=" * 70)

# ------------------------------------------------------------------------------
# 2.1 Per-Column Memory Usage
# ------------------------------------------------------------------------------
print("\n--- 2.1 Per-Column Memory Usage ---")

def memory_profile(data, label="DataFrame"):
    """
    Print detailed memory usage per column and total.

    Parameters
    ----------
    data : pd.DataFrame
    label : str

    Returns
    -------
    pd.DataFrame with memory report
    """
    mem_usage = data.memory_usage(deep=True)
    total_mb = mem_usage.sum() / 1024 / 1024
    mem_usage = mem_usage.drop("Index", errors="ignore")

    report = pd.DataFrame({
        "dtype":     data.dtypes,
        "null_count": data.isnull().sum(),
        "nunique":   data.nunique(),
        "mem_kb":    (mem_usage / 1024).round(2)
    })

    report["mem_pct"] = (report["mem_kb"] / report["mem_kb"].sum() * 100).round(1)

    print(f"\nMemory Profile: {label}")
    print(f"{'Column':<18} {'Dtype':<14} {'Nulls':>7} {'Unique':>8} {'KB':>10} {'%':>6}")
    print("-" * 68)
    for col, row in report.iterrows():
        print(
            f"{col:<18} {str(row['dtype']):<14} "
            f"{int(row['null_count']):>7} {int(row['nunique']):>8} "
            f"{row['mem_kb']:>10.1f} {row['mem_pct']:>6.1f}%"
        )
    print("-" * 68)
    print(f"{'TOTAL':<18} {'':<14} {'':<7} {'':<8} {total_mb*1024:>10.1f} {'100%':>6}")
    print(f"Total memory: {total_mb:.2f} MB")
    return report

report = memory_profile(df_initial, "Initial DataFrame (1M rows)")

# ------------------------------------------------------------------------------
# 2.2 Understanding Object Dtype Cost
# ------------------------------------------------------------------------------
print("\n--- 2.2 Why Object Dtype is Expensive ---")

# Object dtype stores pointers to Python string objects
# Each Python string object has 50+ bytes of overhead

string_col = df_initial["region"]
category_col = string_col.astype("category")

obj_mem = string_col.memory_usage(deep=True) / 1024
cat_mem = category_col.memory_usage(deep=True) / 1024

print(f"'region' column ({string_col.nunique()} unique values):")
print(f"  object dtype memory:   {obj_mem:.1f} KB")
print(f"  category dtype memory: {cat_mem:.1f} KB")
print(f"  Memory saved:          {(1 - cat_mem/obj_mem)*100:.1f}%")

print(f"\nWhy: object stores {N_ROWS:,} Python string objects")
print(f"     category stores {N_ROWS:,} int codes + {string_col.nunique()} string labels")


# ==============================================================================
# SECTION 3: DTYPE OPTIMIZATION
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 3: DTYPE OPTIMIZATION")
print("=" * 70)

# ------------------------------------------------------------------------------
# 3.1 Automatic Dtype Optimizer
# ------------------------------------------------------------------------------
print("\n--- 3.1 Automatic Dtype Optimizer ---")

def optimize_dtypes(data, verbose=True):
    """
    Optimize DataFrame dtypes to minimize memory usage.

    Strategy:
      Integer columns  -> smallest fitting int type
      Float columns    -> float32 (where precision allows)
      Object columns   -> category (if cardinality < 50%)

    Parameters
    ----------
    data : pd.DataFrame
    verbose : bool

    Returns
    -------
    pd.DataFrame with optimized dtypes
    dict with optimization report
    """
    df_opt = data.copy()
    original_mem = df_opt.memory_usage(deep=True).sum()
    report = {}

    for col in df_opt.columns:
        dtype_before = df_opt[col].dtype
        col_data = df_opt[col]

        # Integer columns: find smallest type
        if pd.api.types.is_integer_dtype(dtype_before):
            col_clean = col_data.dropna()
            if len(col_clean) == 0:
                continue
            col_min = int(col_clean.min())
            col_max = int(col_clean.max())

            if col_min >= 0:
                if col_max <= 255:
                    new_dtype = np.uint8
                elif col_max <= 65535:
                    new_dtype = np.uint16
                elif col_max <= 4_294_967_295:
                    new_dtype = np.uint32
                else:
                    new_dtype = np.uint64
            else:
                if col_min >= -128 and col_max <= 127:
                    new_dtype = np.int8
                elif col_min >= -32_768 and col_max <= 32_767:
                    new_dtype = np.int16
                elif col_min >= -2_147_483_648 and col_max <= 2_147_483_647:
                    new_dtype = np.int32
                else:
                    new_dtype = np.int64

            df_opt[col] = col_data.astype(new_dtype)

        # Float columns: try float32
        elif pd.api.types.is_float_dtype(dtype_before):
            if dtype_before == np.float64:
                df_opt[col] = col_data.astype(np.float32)

        # Object columns: try category
        elif dtype_before == object:
            n_unique = col_data.nunique()
            n_total = len(col_data)
            if n_unique / n_total < 0.5:
                df_opt[col] = col_data.astype("category")

        dtype_after = df_opt[col].dtype
        mem_before = data[col].memory_usage(deep=True) / 1024
        mem_after = df_opt[col].memory_usage(deep=True) / 1024
        saved = mem_before - mem_after

        report[col] = {
            "before":   str(dtype_before),
            "after":    str(dtype_after),
            "kb_saved": round(saved, 1)
        }

        if verbose and str(dtype_before) != str(dtype_after):
            print(f"  {col:<18}: {str(dtype_before):<12} -> {str(dtype_after):<14} "
                  f"(saved {saved:.1f} KB)")

    optimized_mem = df_opt.memory_usage(deep=True).sum()
    saving_mb = (original_mem - optimized_mem) / 1024 / 1024
    saving_pct = (1 - optimized_mem / original_mem) * 100

    if verbose:
        print(f"\nMemory before: {original_mem/1024/1024:.2f} MB")
        print(f"Memory after:  {optimized_mem/1024/1024:.2f} MB")
        print(f"Saved:         {saving_mb:.2f} MB ({saving_pct:.1f}%)")

    return df_opt, report


print("Optimizing dtypes on 1M row DataFrame:")
df_optimized, opt_report = optimize_dtypes(df_initial)

# Verify data integrity
assert len(df_optimized) == len(df_initial), "Row count changed"
assert list(df_optimized.columns) == list(df_initial.columns), "Columns changed"
print("\nData integrity verified: same rows and columns")

# Compare memory profiles
_ = memory_profile(df_optimized, "After Dtype Optimization")

# ------------------------------------------------------------------------------
# 3.2 Specifying dtypes at Load Time
# ------------------------------------------------------------------------------
print("\n--- 3.2 Specifying dtypes When Loading CSV ---")

explanation = """
The most efficient approach is to specify dtypes WHEN loading.
This prevents Pandas from inferring (and often choosing) large dtypes.
Avoids the two-step: load-then-convert memory spike.
"""
print(explanation)

# Define optimal dtypes upfront
optimal_dtypes = {
    "transaction_id": np.int32,
    "customer_id":    np.int32,
    "product_id":     np.int32,
    "quantity":       np.int8,
    "unit_price":     np.float32,
    "discount_pct":   np.float32,
    "total_price":    np.float32,
    "year":           np.int16,
    "month":          np.int8,
    "day":            np.int8,
}

# Categorical columns at load time
categorical_cols = ["region", "category", "payment_type", "status"]

# Simulate loading with dtype specification
buffer_copy = io.StringIO(csv_content)
df_typed = pd.read_csv(
    buffer_copy,
    dtype=optimal_dtypes
)
# Convert categorical columns after load
for col in categorical_cols:
    if col in df_typed.columns:
        df_typed[col] = df_typed[col].astype("category")

print("Loaded with specified dtypes:")
print(f"  Memory usage: {df_typed.memory_usage(deep=True).sum()/1024/1024:.2f} MB")
print(f"  Column dtypes:")
for col in df_typed.columns:
    print(f"    {col:<18}: {df_typed[col].dtype}")


# ==============================================================================
# SECTION 4: LOADING ONLY NEEDED COLUMNS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 4: LOADING ONLY NEEDED COLUMNS")
print("=" * 70)

explanation = """
One of the simplest and most impactful optimizations.
If your analysis uses 5 of 50 columns, load only those 5.
This can reduce memory usage by 90% before any other optimization.

usecols parameter in pd.read_csv() handles this efficiently.
The skipped columns are never read from disk into memory.
"""
print(explanation)

# ------------------------------------------------------------------------------
# 4.1 usecols for Column Selection at Load Time
# ------------------------------------------------------------------------------
print("\n--- 4.1 usecols Parameter ---")

all_columns = list(df_initial.columns)
needed_columns = ["transaction_id", "customer_id", "region", "category", "total_price"]

print(f"Total columns available: {len(all_columns)}")
print(f"Columns needed: {needed_columns}")

# Load with all columns
buffer_all = io.StringIO(csv_content)
start = time.perf_counter()
df_all_cols = pd.read_csv(buffer_all)
t_all = time.perf_counter() - start
mem_all = df_all_cols.memory_usage(deep=True).sum() / 1024 / 1024

# Load with only needed columns
buffer_subset = io.StringIO(csv_content)
start = time.perf_counter()
df_subset_cols = pd.read_csv(buffer_subset, usecols=needed_columns)
t_subset = time.perf_counter() - start
mem_subset = df_subset_cols.memory_usage(deep=True).sum() / 1024 / 1024

print(f"\n{'':20} {'All Cols':>12} {'5 Cols':>12}")
print("-" * 46)
print(f"{'Columns loaded':<20} {len(df_all_cols.columns):>12} {len(df_subset_cols.columns):>12}")
print(f"{'Memory (MB)':<20} {mem_all:>12.2f} {mem_subset:>12.2f}")
print(f"{'Load time (s)':<20} {t_all:>12.3f} {t_subset:>12.3f}")
print(f"{'Memory saved':<20} {'':>12} {(1-mem_subset/mem_all)*100:>11.1f}%")

# Validate same row count
assert len(df_all_cols) == len(df_subset_cols), "Row count mismatch"
print(f"\nRow count verified: {len(df_subset_cols):,} rows in both")

# ------------------------------------------------------------------------------
# 4.2 Filtering Rows at Load Time with nrows and skiprows
# ------------------------------------------------------------------------------
print("\n--- 4.2 Load Only a Subset of Rows ---")

# Load first N rows for exploration
buffer_nrows = io.StringIO(csv_content)
df_sample = pd.read_csv(buffer_nrows, nrows=10_000)
print(f"Quick sample load with nrows=10000:")
print(f"  Shape: {df_sample.shape}")
print(f"  Memory: {df_sample.memory_usage(deep=True).sum()/1024:.1f} KB")

# Load with row skip (useful for resuming partial loads)
buffer_skip = io.StringIO(csv_content)
# Get header row count
header = pd.read_csv(io.StringIO(csv_content), nrows=0)
header_row_count = 1

df_skip = pd.read_csv(
    buffer_skip,
    skiprows=range(1, 100_001),  # Skip first 100K data rows (keep header)
    nrows=50_000,
    names=list(header.columns)
)
print(f"\nLoad rows 100001 to 150000 using skiprows:")
print(f"  Shape: {df_skip.shape}")


# ==============================================================================
# SECTION 5: CHUNKED READING AND PROCESSING
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 5: CHUNKED READING AND PROCESSING")
print("=" * 70)

explanation = """
Chunked reading processes large files in pieces.
Instead of loading 10GB into RAM, process 100MB at a time.

pd.read_csv(chunksize=N) returns a TextFileReader iterator.
Each iteration yields a DataFrame with N rows.

This is the foundation of out-of-core (larger than RAM) processing.

Key patterns:
  1. Collect and concat results from each chunk
  2. Aggregate progressively across chunks
  3. Filter and write each chunk to output
"""
print(explanation)

# ------------------------------------------------------------------------------
# 5.1 Basic Chunked Reading
# ------------------------------------------------------------------------------
print("\n--- 5.1 Basic Chunked Reading ---")

CHUNK_SIZE = 100_000

buffer_chunks = io.StringIO(csv_content)
reader = pd.read_csv(buffer_chunks, chunksize=CHUNK_SIZE)

print(f"Processing in chunks of {CHUNK_SIZE:,} rows...")
print(f"Expected chunks: {N_ROWS // CHUNK_SIZE} + remainder")

chunk_summaries = []
n_chunks = 0

for chunk in reader:
    n_chunks += 1
    # Process each chunk
    chunk_summary = {
        "chunk":     n_chunks,
        "rows":      len(chunk),
        "total_revenue": chunk["total_price"].sum(),
        "avg_price":     chunk["unit_price"].mean(),
        "n_customers":   chunk["customer_id"].nunique()
    }
    chunk_summaries.append(chunk_summary)
    if n_chunks % 2 == 0:
        print(f"  Chunk {n_chunks}: {len(chunk):,} rows processed")

print(f"\nTotal chunks processed: {n_chunks}")

# Combine chunk summaries
summary_df = pd.DataFrame(chunk_summaries)
print("\nPer-chunk summary:")
print(summary_df.round(2))

# Aggregate across chunks
total_revenue = summary_df["total_revenue"].sum()
overall_avg = summary_df.apply(
    lambda r: r["avg_price"] * r["rows"], axis=1
).sum() / summary_df["rows"].sum()

print(f"\nAggregated results:")
print(f"  Total revenue: ${total_revenue:,.2f}")
print(f"  Overall avg price: ${overall_avg:.2f}")

# Validate against single-load result
expected_revenue = df_initial["total_price"].sum()
match = abs(total_revenue - expected_revenue) < 1.0  # Allow floating point diff
print(f"\nRevenue matches single-load calculation: {match}")

# ------------------------------------------------------------------------------
# 5.2 Chunked Aggregation (Group-By Across Chunks)
# ------------------------------------------------------------------------------
print("\n--- 5.2 Aggregation Across Chunks ---")

explanation = """
Challenge: groupby() needs ALL data in memory.
Solution: Aggregate per chunk, then merge results.

Pattern:
  1. Compute partial aggregation in each chunk
  2. Combine partial results after all chunks
  3. Final aggregation on the combined partial results

This works for sum, count, min, max but not directly for mean.
Mean requires: total_sum / total_count from all chunks.
"""
print(explanation)

buffer_agg = io.StringIO(csv_content)
reader_agg = pd.read_csv(buffer_agg, chunksize=CHUNK_SIZE)

# Accumulate partial results
partial_results = []

for chunk in reader_agg:
    # Compute partial aggregation per chunk
    partial = chunk.groupby("category").agg(
        total_revenue   = ("total_price", "sum"),
        total_quantity  = ("quantity",    "sum"),
        transaction_count = ("transaction_id", "count"),
    ).reset_index()
    partial_results.append(partial)

# Combine all partial results
all_partials = pd.concat(partial_results, ignore_index=True)

# Final aggregation (combine partial sums and counts)
final_by_category = all_partials.groupby("category").agg(
    total_revenue     = ("total_revenue",      "sum"),
    total_quantity    = ("total_quantity",      "sum"),
    transaction_count = ("transaction_count",  "sum"),
).reset_index()

# Compute derived metrics after combining
final_by_category["avg_transaction_value"] = (
    final_by_category["total_revenue"] / final_by_category["transaction_count"]
).round(2)

final_by_category["revenue_share"] = (
    final_by_category["total_revenue"] /
    final_by_category["total_revenue"].sum() * 100
).round(2)

print("Category revenue (computed via chunks):")
print(final_by_category.sort_values("total_revenue", ascending=False).to_string(index=False))

# Validate against in-memory calculation
expected = df_initial.groupby("category")["total_price"].sum()
for _, row in final_by_category.iterrows():
    expected_val = expected[row["category"]]
    actual_val   = row["total_revenue"]
    assert abs(actual_val - expected_val) < 1.0, (
        f"Category {row['category']}: expected {expected_val:.2f}, got {actual_val:.2f}"
    )
print("\nValidation passed: chunk aggregation matches in-memory calculation")

# ------------------------------------------------------------------------------
# 5.3 Chunked Filtering and Writing
# ------------------------------------------------------------------------------
print("\n--- 5.3 Filter and Write Chunks ---")

explanation = """
Common ETL pattern: read large file, filter rows, write to output.
Never need to hold entire file in memory at once.
"""
print(explanation)

# Simulate filter: keep only completed electronics transactions
output_buffer = io.StringIO()
rows_written = 0
chunks_read = 0

buffer_filter = io.StringIO(csv_content)
reader_filter = pd.read_csv(
    buffer_filter,
    chunksize=CHUNK_SIZE,
    usecols=["transaction_id", "category", "status", "total_price", "region"]
)

for i, chunk in enumerate(reader_filter):
    chunks_read += 1

    # Apply filter
    filtered = chunk[
        (chunk["category"] == "electronics") &
        (chunk["status"] == "completed")
    ]

    # Write header only on first chunk
    if i == 0:
        filtered.to_csv(output_buffer, index=False, mode="w")
    else:
        filtered.to_csv(output_buffer, index=False, mode="a", header=False)

    rows_written += len(filtered)

print(f"Chunks processed: {chunks_read}")
print(f"Rows written after filter: {rows_written:,}")
print(f"Filter rate: {rows_written / N_ROWS * 100:.1f}% of input rows matched")

# Validate
expected_filtered = df_initial[
    (df_initial["category"] == "electronics") &
    (df_initial["status"] == "completed")
]
assert abs(rows_written - len(expected_filtered)) < 5, "Filter count mismatch"
print(f"Validation: expected {len(expected_filtered):,}, got {rows_written:,}")


# ==============================================================================
# SECTION 6: MEMORY-EFFICIENT OPERATIONS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 6: MEMORY-EFFICIENT OPERATIONS")
print("=" * 70)

# ------------------------------------------------------------------------------
# 6.1 Avoid Intermediate Copies
# ------------------------------------------------------------------------------
print("\n--- 6.1 Avoid Unnecessary Copies ---")

df_work = df_optimized.copy()

print("Tracking memory for different operation patterns:")

# Bad: Creates multiple intermediate DataFrames
def bad_pipeline(data):
    """Creates multiple full DataFrame copies."""
    step1 = data[data["status"] == "completed"]         # Copy 1
    step2 = step1[step1["category"] == "electronics"]   # Copy 2
    step3 = step2[["customer_id", "total_price"]]       # Copy 3
    step4 = step3.dropna()                              # Copy 4
    return step4

# Good: Build filter mask, apply once
def good_pipeline(data):
    """Single filter pass, minimal copies."""
    mask = (
        (data["status"] == "completed") &
        (data["category"] == "electronics")
    )
    return data.loc[mask, ["customer_id", "total_price"]].dropna()

start = time.perf_counter()
result_bad = bad_pipeline(df_work)
t_bad = time.perf_counter() - start

start = time.perf_counter()
result_good = good_pipeline(df_work)
t_good = time.perf_counter() - start

print(f"Chained filter (bad):   {t_bad*1000:.2f} ms")
print(f"Single mask (good):     {t_good*1000:.2f} ms")
print(f"Speedup:                {t_bad/t_good:.1f}x")
print(f"\nBoth return same rows: {len(result_bad) == len(result_good)}")

# ------------------------------------------------------------------------------
# 6.2 In-place Operations
# ------------------------------------------------------------------------------
print("\n--- 6.2 In-place vs New Object ---")

df_inplace_test = df_optimized[["total_price"]].copy()

print(f"Memory before: {df_inplace_test.memory_usage(deep=True).sum()/1024:.1f} KB")

# Create new object (uses extra memory temporarily)
start = time.perf_counter()
df_new_obj = df_inplace_test["total_price"] * 1.1
t_new = time.perf_counter() - start

# In-place (no new allocation for data)
start = time.perf_counter()
df_inplace_test["total_price"] = df_inplace_test["total_price"] * 1.1
t_inplace = time.perf_counter() - start

print(f"New object (*= with assignment): {t_new*1000:.2f} ms")
print(f"In-place assignment:             {t_inplace*1000:.2f} ms")

# ------------------------------------------------------------------------------
# 6.3 Use query() for Readable Filtered Views
# ------------------------------------------------------------------------------
print("\n--- 6.3 query() for Filtered Computation ---")

df_q = df_optimized.copy()

# Method 1: Boolean mask
start = time.perf_counter()
r1 = df_q[(df_q["year"] == 2023) & (df_q["rating"] > 3)]["total_price"].sum()
t1 = time.perf_counter() - start

# Method 2: query()
start = time.perf_counter()
r2 = df_q.query("year == 2023 and rating > 3")["total_price"].sum()
t2 = time.perf_counter() - start

print(f"Boolean mask: {t1*1000:.2f} ms -> ${r1:,.2f}")
print(f"query():      {t2*1000:.2f} ms -> ${r2:,.2f}")
print(f"Match: {abs(r1 - r2) < 0.01}")
print("\nquery() is more readable; performance is similar to boolean mask")

# ------------------------------------------------------------------------------
# 6.4 Efficient GroupBy Patterns
# ------------------------------------------------------------------------------
print("\n--- 6.4 Efficient GroupBy Patterns ---")

df_gb = df_optimized.copy()

# Bad: Multiple separate groupby calls on same group
start = time.perf_counter()
rev_by_region    = df_gb.groupby("region")["total_price"].sum()
count_by_region  = df_gb.groupby("region")["transaction_id"].count()
avg_by_region    = df_gb.groupby("region")["total_price"].mean()
t_multiple = time.perf_counter() - start

# Good: Single groupby call with agg()
start = time.perf_counter()
combined = df_gb.groupby("region").agg(
    total_revenue     = ("total_price",    "sum"),
    transaction_count = ("transaction_id", "count"),
    avg_transaction   = ("total_price",    "mean")
)
t_single = time.perf_counter() - start

print(f"Multiple groupby calls: {t_multiple*1000:.2f} ms")
print(f"Single agg() call:      {t_single*1000:.2f} ms")
print(f"Speedup: {t_multiple/t_single:.1f}x")
print("\nResult from single agg():")
print(combined.round(2))

# ------------------------------------------------------------------------------
# 6.5 Sparse Arrays for High-NaN Columns
# ------------------------------------------------------------------------------
print("\n--- 6.5 Sparse Arrays for High-NaN Columns ---")

explanation = """
If a column is mostly NaN or a single fill value,
use Sparse dtype to avoid storing all those zeros/NaNs.

This is useful for:
  - One-hot encoded columns (mostly 0s)
  - Columns where most rows have no value
  - Interaction features in ML
"""
print(explanation)

# Create a mostly-null column (simulate optional feature)
n = 100_000
mostly_null = pd.Series(
    np.where(np.random.rand(n) < 0.05, np.random.rand(n), np.nan)
)
mostly_zero = pd.Series(
    np.where(np.random.rand(n) < 0.05, np.random.randint(1, 10, n).astype(float), 0.0)
)

print(f"Dense float64 NaN column:  {mostly_null.memory_usage(deep=True)/1024:.1f} KB")

sparse_null = mostly_null.astype(pd.SparseDtype("float64", fill_value=np.nan))
print(f"Sparse float64 NaN column: {sparse_null.memory_usage(deep=True)/1024:.1f} KB")
print(f"Null density: {mostly_null.isna().mean():.1%}")

dense_zero_mem = mostly_zero.memory_usage(deep=True) / 1024
sparse_zero = mostly_zero.astype(pd.SparseDtype("float64", fill_value=0.0))
sparse_zero_mem = sparse_zero.memory_usage(deep=True) / 1024

print(f"\nDense float64 mostly-zero: {dense_zero_mem:.1f} KB")
print(f"Sparse float64 mostly-zero:{sparse_zero_mem:.1f} KB")
print(f"Zero density: {(mostly_zero == 0).mean():.1%}")
print(f"Memory saved: {(1 - sparse_zero_mem/dense_zero_mem)*100:.1f}%")


# ==============================================================================
# SECTION 7: GARBAGE COLLECTION AND MEMORY RELEASE
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 7: GARBAGE COLLECTION AND MEMORY RELEASE")
print("=" * 70)

explanation = """
Python garbage collection handles most memory automatically.
But with large DataFrames, explicit cleanup helps avoid
running out of RAM during multi-step pipelines.

Best practices:
  1. del large_df  to release reference
  2. gc.collect()  to force garbage collection
  3. Use context managers for temporary data
  4. Process in chunks instead of keeping everything in memory
"""
print(explanation)

# ------------------------------------------------------------------------------
# 7.1 Explicit Deletion
# ------------------------------------------------------------------------------
print("\n--- 7.1 Explicit Deletion ---")

def get_memory_mb():
    """Rough memory estimate of large DataFrames in environment."""
    return df_initial.memory_usage(deep=True).sum() / 1024 / 1024

# Create large temporary DataFrame
temp_large = pd.DataFrame(
    np.random.rand(500_000, 10),
    columns=[f"col_{i}" for i in range(10)]
)
temp_mem = temp_large.memory_usage(deep=True).sum() / 1024 / 1024
print(f"Temporary DataFrame memory: {temp_mem:.2f} MB")

# Release memory explicitly
del temp_large
collected = gc.collect()
print(f"After del + gc.collect(): {collected} objects collected")
print("Memory released back to Python pool")

# ------------------------------------------------------------------------------
# 7.2 Pipeline with Explicit Cleanup
# ------------------------------------------------------------------------------
print("\n--- 7.2 Multi-Step Pipeline with Cleanup ---")

def memory_efficient_pipeline(csv_buffer, chunk_size=100_000):
    """
    Demonstrate explicit memory management in a pipeline.

    Steps:
      1. Load and filter chunk by chunk
      2. Aggregate
      3. Release intermediate data
      4. Return final results only
    """
    print("  Running memory-efficient pipeline...")

    partial_aggs = []

    reader = pd.read_csv(
        io.StringIO(csv_buffer),
        chunksize=chunk_size,
        usecols=["region", "category", "total_price", "status"]
    )

    for i, chunk in enumerate(reader):
        # Step 1: Filter
        filtered = chunk[chunk["status"] == "completed"].copy()

        # Step 2: Aggregate
        agg = filtered.groupby(["region", "category"]).agg(
            revenue = ("total_price", "sum"),
            count   = ("total_price", "count")
        ).reset_index()
        partial_aggs.append(agg)

        # Step 3: Explicit cleanup of chunk data
        del chunk, filtered, agg
        gc.collect()

    # Combine and final aggregation
    combined = pd.concat(partial_aggs, ignore_index=True)
    del partial_aggs
    gc.collect()

    final = combined.groupby(["region", "category"]).agg(
        total_revenue = ("revenue", "sum"),
        total_orders  = ("count",   "sum")
    ).reset_index()

    del combined
    gc.collect()

    print(f"  Pipeline complete: {len(final)} result rows")
    return final


result = memory_efficient_pipeline(csv_content)
print("\nFinal aggregated result (first 10 rows):")
print(result.head(10).round(2))


# ==============================================================================
# SECTION 8: LOADING STRATEGIES COMPARISON
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 8: LOADING STRATEGIES COMPARISON")
print("=" * 70)

strategies = {}

# Strategy 1: Default load (all columns, default dtypes)
buffer_s1 = io.StringIO(csv_content)
start = time.perf_counter()
df_s1 = pd.read_csv(buffer_s1)
strategies["1_default"] = {
    "desc": "Default load",
    "time_ms": (time.perf_counter() - start) * 1000,
    "mem_mb": df_s1.memory_usage(deep=True).sum() / 1024 / 1024,
    "rows": len(df_s1)
}
del df_s1; gc.collect()

# Strategy 2: Optimized dtypes at load
buffer_s2 = io.StringIO(csv_content)
start = time.perf_counter()
df_s2 = pd.read_csv(buffer_s2, dtype=optimal_dtypes)
for col in categorical_cols:
    if col in df_s2.columns:
        df_s2[col] = df_s2[col].astype("category")
strategies["2_optimized_dtypes"] = {
    "desc": "Optimized dtypes",
    "time_ms": (time.perf_counter() - start) * 1000,
    "mem_mb": df_s2.memory_usage(deep=True).sum() / 1024 / 1024,
    "rows": len(df_s2)
}
del df_s2; gc.collect()

# Strategy 3: Column selection + optimized dtypes
needed = ["transaction_id", "region", "category", "total_price", "status", "year"]
buffer_s3 = io.StringIO(csv_content)
start = time.perf_counter()
df_s3 = pd.read_csv(
    buffer_s3,
    usecols=needed,
    dtype={
        "transaction_id": np.int32,
        "total_price":    np.float32,
        "year":           np.int16
    }
)
df_s3["region"]   = df_s3["region"].astype("category")
df_s3["category"] = df_s3["category"].astype("category")
df_s3["status"]   = df_s3["status"].astype("category")
strategies["3_cols_and_dtypes"] = {
    "desc": "Cols + dtypes",
    "time_ms": (time.perf_counter() - start) * 1000,
    "mem_mb": df_s3.memory_usage(deep=True).sum() / 1024 / 1024,
    "rows": len(df_s3)
}
del df_s3; gc.collect()

# Strategy 4: Sample only
buffer_s4 = io.StringIO(csv_content)
start = time.perf_counter()
df_s4 = pd.read_csv(buffer_s4, nrows=50_000)
strategies["4_sample"] = {
    "desc": "Sample 50K rows",
    "time_ms": (time.perf_counter() - start) * 1000,
    "mem_mb": df_s4.memory_usage(deep=True).sum() / 1024 / 1024,
    "rows": len(df_s4)
}
del df_s4; gc.collect()

# Print comparison
print(f"\n{'Strategy':<22} {'Time (ms)':>12} {'Memory (MB)':>14} {'Rows':>10}")
print("-" * 62)
baseline_mem = strategies["1_default"]["mem_mb"]
for key, s in strategies.items():
    savings = (1 - s["mem_mb"] / baseline_mem) * 100
    savings_str = f"({savings:.0f}% less)" if key != "1_default" else "(baseline)"
    print(
        f"{s['desc']:<22} {s['time_ms']:>12.1f} "
        f"{s['mem_mb']:>12.2f} MB {s['rows']:>10,}  {savings_str}"
    )


# ==============================================================================
# SECTION 9: WHEN TO CONSIDER ALTERNATIVES
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 9: WHEN TO CONSIDER ALTERNATIVES TO PANDAS")
print("=" * 70)

alternatives = """
Pandas is excellent but has limits. Consider alternatives when:

SCENARIO                           | ALTERNATIVE        | REASON
-----------------------------------|--------------------|--------------------------
File larger than available RAM     | Dask               | Parallel out-of-core
Very fast SQL-like queries         | DuckDB             | Columnar, in-process SQL
Huge arrays (pure numeric)         | NumPy directly     | No DataFrame overhead
Very fast DataFrame operations     | Polars             | Rust-based, lazy eval
Distributed computation            | PySpark            | Multi-node processing
Streaming data                     | Pandas chunked     | Built-in solution

STAYING WITH PANDAS:
When data fits in RAM with dtype optimization, Pandas is the right tool.
The optimizations in this lesson often make Pandas viable for datasets
10-20x larger than the naive default load.

QUICK DECISION RULE:
  < 100 MB:   Load normally, optimize if needed
  100 MB - 2 GB: Use dtype optimization and column selection
  2 GB - 10 GB:  Use chunked processing with Pandas
  > 10 GB:    Consider Dask, Polars, or DuckDB

Note: DuckDB can be used alongside Pandas.
You can query Pandas DataFrames directly with DuckDB SQL.
"""
print(alternatives)

# Demonstrate DuckDB-style pattern using Pandas (without installing DuckDB)
print("\n--- Simulating DuckDB-style Aggregate Query ---")

# In DuckDB this would be:
# SELECT region, category, SUM(total_price) as revenue
# FROM transactions
# WHERE status = 'completed' AND year = 2023
# GROUP BY region, category
# ORDER BY revenue DESC

# With Pandas
result_sql_style = (
    df_optimized
    .query("status == 'completed' and year == 2023")
    .groupby(["region", "category"], observed=True)
    .agg(revenue=("total_price", "sum"))
    .reset_index()
    .sort_values("revenue", ascending=False)
    .head(10)
)

print("SQL-equivalent query result (Pandas):")
print(result_sql_style.round(2).to_string(index=False))


# ==============================================================================
# SECTION 10: REAL WORLD PIPELINE - LARGE FILE PROCESSING
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 10: REAL WORLD PIPELINE - LARGE FILE PROCESSING")
print("=" * 70)

def process_large_transaction_file(
    csv_data,
    chunk_size=100_000,
    target_categories=None,
    target_status="completed"
):
    """
    Production-grade pipeline for processing large transaction file.

    Parameters
    ----------
    csv_data : str
        CSV content as string (simulates file path in production)
    chunk_size : int
        Rows per chunk
    target_categories : list or None
        Filter to these categories; None means all
    target_status : str
        Filter to this transaction status

    Returns
    -------
    dict with aggregated results
    """
    print(f"\nPipeline configuration:")
    print(f"  Chunk size:       {chunk_size:,}")
    print(f"  Target status:    {target_status}")
    print(f"  Target categories:{target_categories or 'ALL'}")

    # Only load columns we need
    cols_needed = [
        "transaction_id", "customer_id", "region",
        "category", "status", "total_price", "quantity", "year", "month"
    ]

    dtype_spec = {
        "transaction_id": np.int32,
        "customer_id":    np.int32,
        "total_price":    np.float32,
        "quantity":       np.int8,
        "year":           np.int16,
        "month":          np.int8
    }

    reader = pd.read_csv(
        io.StringIO(csv_data),
        usecols=cols_needed,
        dtype=dtype_spec,
        chunksize=chunk_size
    )

    # Accumulators
    partial_region_agg  = []
    partial_category_agg = []
    total_rows_read     = 0
    total_rows_matched  = 0

    for chunk_num, chunk in enumerate(reader, 1):

        # Convert low-cardinality to category
        for col in ["region", "category", "status"]:
            if col in chunk.columns:
                chunk[col] = chunk[col].astype("category")

        total_rows_read += len(chunk)

        # Apply filters
        mask = chunk["status"] == target_status
        if target_categories is not None:
            mask &= chunk["category"].isin(target_categories)

        filtered = chunk[mask].copy()
        total_rows_matched += len(filtered)

        if len(filtered) == 0:
            del chunk, filtered
            continue

        # Partial aggregation by region
        region_agg = filtered.groupby("region", observed=True).agg(
            revenue       = ("total_price", "sum"),
            orders        = ("transaction_id", "count"),
            total_qty     = ("quantity", "sum"),
            unique_cust   = ("customer_id", "nunique")
        ).reset_index()
        partial_region_agg.append(region_agg)

        # Partial aggregation by category
        cat_agg = filtered.groupby("category", observed=True).agg(
            revenue = ("total_price", "sum"),
            orders  = ("transaction_id", "count")
        ).reset_index()
        partial_category_agg.append(cat_agg)

        # Cleanup
        del chunk, filtered, region_agg, cat_agg
        gc.collect()

    # Combine partials
    print(f"\n  Total rows read:    {total_rows_read:,}")
    print(f"  Rows matched:       {total_rows_matched:,}")
    print(f"  Filter rate:        {total_rows_matched/total_rows_read*100:.1f}%")

    if not partial_region_agg:
        return {}

    # Final region summary
    region_combined = pd.concat(partial_region_agg, ignore_index=True)
    final_region = region_combined.groupby("region").agg(
        total_revenue   = ("revenue",     "sum"),
        total_orders    = ("orders",      "sum"),
        total_quantity  = ("total_qty",   "sum"),
        unique_customers= ("unique_cust", "sum")
    ).reset_index()
    final_region["avg_order_value"] = (
        final_region["total_revenue"] / final_region["total_orders"]
    ).round(2)

    del region_combined, partial_region_agg
    gc.collect()

    # Final category summary
    cat_combined = pd.concat(partial_category_agg, ignore_index=True)
    final_category = cat_combined.groupby("category").agg(
        total_revenue = ("revenue", "sum"),
        total_orders  = ("orders",  "sum")
    ).reset_index()
    final_category["revenue_share_pct"] = (
        final_category["total_revenue"] /
        final_category["total_revenue"].sum() * 100
    ).round(2)

    del cat_combined, partial_category_agg
    gc.collect()

    return {
        "rows_read":    total_rows_read,
        "rows_matched": total_rows_matched,
        "by_region":    final_region,
        "by_category":  final_category
    }


# Run the pipeline
pipeline_result = process_large_transaction_file(
    csv_content,
    chunk_size=200_000,
    target_categories=["electronics", "clothing", "sports"],
    target_status="completed"
)

print("\n=== PIPELINE RESULTS ===")
print("\nRevenue by Region:")
print(
    pipeline_result["by_region"]
    .sort_values("total_revenue", ascending=False)
    .round(2)
    .to_string(index=False)
)

print("\nRevenue by Category:")
print(
    pipeline_result["by_category"]
    .sort_values("total_revenue", ascending=False)
    .round(2)
    .to_string(index=False)
)


# ==============================================================================
# SECTION 11: VALIDATION AFTER OPTIMIZATION
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 11: VALIDATION AFTER OPTIMIZATION")
print("=" * 70)

def validate_optimized_df(original, optimized, tolerance=0.01):
    """
    Validate that dtype optimization preserved data integrity.

    Parameters
    ----------
    original : pd.DataFrame
    optimized : pd.DataFrame
    tolerance : float
        Allowable relative difference for numeric values
    """
    print("Running validation checks...")
    passed = 0
    total  = 0

    # Check 1: Same shape
    total += 1
    assert original.shape == optimized.shape, (
        f"Shape mismatch: {original.shape} vs {optimized.shape}"
    )
    print(f"  [PASS] Shape preserved: {original.shape}")
    passed += 1

    # Check 2: Same columns
    total += 1
    assert list(original.columns) == list(optimized.columns), "Column mismatch"
    print(f"  [PASS] Columns match: {len(original.columns)} columns")
    passed += 1

    # Check 3: Numeric column totals within tolerance
    numeric_cols = original.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        total += 1
        orig_sum = float(original[col].sum())
        opt_sum  = float(optimized[col].sum())
        if orig_sum != 0:
            rel_diff = abs(orig_sum - opt_sum) / abs(orig_sum)
            assert rel_diff <= tolerance, (
                f"Column {col}: relative diff {rel_diff:.6f} > tolerance {tolerance}"
            )
        print(f"  [PASS] {col}: sum within tolerance ({tolerance*100:.0f}%)")
        passed += 1

    # Check 4: No new nulls introduced
    total += 1
    orig_nulls = original.isnull().sum().sum()
    opt_nulls  = optimized.isnull().sum().sum()
    assert orig_nulls == opt_nulls, (
        f"Null count changed: {orig_nulls} -> {opt_nulls}"
    )
    print(f"  [PASS] Null count preserved: {orig_nulls} nulls")
    passed += 1

    print(f"\nValidation complete: {passed}/{total} checks passed")
    return True


validate_optimized_df(df_initial, df_optimized)


# ==============================================================================
# SECTION 12: COMMON PITFALLS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 12: COMMON PITFALLS")
print("=" * 70)

pitfalls = """
Pitfall 1: Converting to category for high-cardinality columns
   - Names, emails, transaction IDs have near-unique values
   - category dtype INCREASES memory for high cardinality
   - Rule: Only use category when unique values < 50% of row count

Pitfall 2: Integer overflow after downcasting
   - Downcasting int64 to int8 silently wraps if values exceed range
   - Always check min and max before downcasting
   - Use the optimize_dtypes function shown in this lesson

Pitfall 3: Losing float precision with float32
   - float32 has ~7 significant digits vs float64 with ~15
   - Never use float32 for financial calculations requiring exact cents
   - Safe for: sensor readings, scores, prices in summary stats

Pitfall 4: Forgetting that chunked groupby needs post-combination
   - Chunked groupby gives partial results per chunk
   - You must combine and re-aggregate after all chunks
   - SUM of SUMs is correct; MEAN of MEANs is NOT

Pitfall 5: Not releasing intermediate DataFrames
   - Each filter step creates a new DataFrame if not using masks
   - del df_intermediate followed by gc.collect() in long pipelines
   - Use method chaining to avoid named intermediates

Pitfall 6: Using usecols list that includes non-existent column
   - Raises ValueError: Usecols do not match columns
   - Always validate column names against file header first

Pitfall 7: Reading entire file to get shape or column names
   - pd.read_csv(file, nrows=0) gives header only (no data)
   - pd.read_csv(file, nrows=5) gives a quick sample
   - Never load entire GB file just to check shape

Pitfall 8: Applying transformations after chunked load incompletely
   - If you transform inside chunk loop, apply same transform when reading
   - Inconsistent transforms on different chunks cause silent errors
"""
print(pitfalls)


# ==============================================================================
# SECTION 13: SUMMARY TABLE
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 13: SUMMARY TABLE")
print("=" * 70)

summary = """
Technique                          | Syntax / Pattern
-----------------------------------|------------------------------------------------
Profile memory per column          | df.memory_usage(deep=True)
Optimize all dtypes automatically  | optimize_dtypes(df) [custom function]
Specify dtypes at load             | pd.read_csv(f, dtype={"col": np.int32})
Load only needed columns           | pd.read_csv(f, usecols=["a","b","c"])
Load sample of rows                | pd.read_csv(f, nrows=10000)
Skip rows at start                 | pd.read_csv(f, skiprows=range(1,100001))
Chunked reading                    | pd.read_csv(f, chunksize=100000)
Convert to category                | df["col"].astype("category")
Convert to smaller int             | df["col"].astype(np.int32)
Convert to float32                 | df["col"].astype(np.float32)
Sparse array for mostly-null col   | s.astype(pd.SparseDtype("float64", np.nan))
Combine filters into one mask      | mask = (cond1) & (cond2); df[mask]
Force garbage collection           | import gc; del df; gc.collect()
Efficient multi-metric groupby     | df.groupby("col").agg(a=("c","sum"),b=("c","mean"))
Get only header from file          | pd.read_csv(file, nrows=0)
"""
print(summary)


# ==============================================================================
# SECTION 14: PRACTICE EXERCISES
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 14: PRACTICE EXERCISES")
print("=" * 70)

print("Exercise 1: Profile and compare memory before/after optimization")
sample_1m = df_initial.sample(100_000, random_state=42)
before_mem = sample_1m.memory_usage(deep=True).sum() / 1024 / 1024
optimized_sample, _ = optimize_dtypes(sample_1m, verbose=False)
after_mem = optimized_sample.memory_usage(deep=True).sum() / 1024 / 1024
print(f"  Before: {before_mem:.2f} MB")
print(f"  After:  {after_mem:.2f} MB")
print(f"  Saved:  {(1 - after_mem/before_mem)*100:.1f}%")

print("\nExercise 2: Chunked computation of average order value per region")
buffer_ex2 = io.StringIO(csv_content)
region_sums   = {}
region_counts = {}

for chunk in pd.read_csv(
    buffer_ex2,
    chunksize=200_000,
    usecols=["region", "total_price"]
):
    for region, grp in chunk.groupby("region"):
        region_sums[region]   = region_sums.get(region, 0.0) + float(grp["total_price"].sum())
        region_counts[region] = region_counts.get(region, 0) + len(grp)

avg_by_region = {
    r: round(region_sums[r] / region_counts[r], 2)
    for r in region_sums
}
print("  Average order value by region (chunked):")
for r, avg in sorted(avg_by_region.items()):
    print(f"    {r}: ${avg:,.2f}")

print("\nExercise 3: Find top 3 categories by volume in 2023")
buffer_ex3 = io.StringIO(csv_content)
cat_2023 = []
for chunk in pd.read_csv(
    buffer_ex3,
    chunksize=200_000,
    usecols=["category", "total_price", "year"],
    dtype={"year": np.int16, "total_price": np.float32}
):
    filtered = chunk[chunk["year"] == 2023]
    partial  = filtered.groupby("category")["total_price"].sum().reset_index()
    cat_2023.append(partial)

top3_2023 = (
    pd.concat(cat_2023)
    .groupby("category")["total_price"]
    .sum()
    .nlargest(3)
)
print("  Top 3 categories in 2023 by revenue:")
for cat, rev in top3_2023.items():
    print(f"    {cat}: ${rev:,.2f}")


# ==============================================================================
# SECTION 15: KEY TAKEAWAYS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 15: KEY TAKEAWAYS")
print("=" * 70)

takeaways = """
1.  Always profile memory immediately after loading: df.memory_usage(deep=True)

2.  Specify dtypes at load time with dtype= parameter in pd.read_csv()
    This avoids memory spikes from load-then-convert pattern

3.  Use usecols= to load only the columns your analysis needs
    This is often the single biggest optimization available

4.  Convert low-cardinality string columns to category dtype
    Typically saves 60-95% memory for string columns

5.  Downcast integers to smallest fitting type: int8, int16, int32
    Always validate min/max before downcasting to prevent overflow

6.  Use float32 instead of float64 where precision allows
    Saves 50% memory; avoid for exact financial calculations

7.  Process large files in chunks using chunksize= parameter
    Allows working with files larger than available RAM

8.  For chunked groupby: aggregate per chunk, concat, re-aggregate
    Never compute mean of means - always track sum and count separately

9.  Delete intermediate DataFrames and call gc.collect() in long pipelines
    Prevents memory accumulation across processing steps

10. When Pandas optimizations are not enough, consider
    DuckDB for SQL queries, Polars for speed, or Dask for scale
"""
print(takeaways)


# ==============================================================================
# NEXT LESSON PREVIEW
# ==============================================================================
print("\n" + "=" * 70)
print("NEXT LESSON PREVIEW")
print("=" * 70)

preview = """
Lesson 29: Data Validation and Pipeline Building

You will learn:
- Writing assertion-based validation functions
- Validating schema, ranges, uniqueness, and referential integrity
- Building reusable validation decorators
- Structuring a full data pipeline with validation at each step
- Logging validation results
- Handling validation failures gracefully
- Testing pipeline correctness with known inputs
"""
print(preview)

print("\n" + "=" * 70)
print("END OF LESSON 28")
print("=" * 70)
print("\nYou can now handle datasets far larger than naive Pandas allows.")
print("Dtype optimization, column selection, and chunked processing")
print("are the three pillars of memory-efficient data engineering.")