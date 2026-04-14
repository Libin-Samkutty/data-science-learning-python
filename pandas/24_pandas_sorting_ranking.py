"""
LESSON 24: SORTING AND RANKING
================================================================================

What You Will Learn:
- Sorting by single and multiple columns
- Ascending and descending sort
- Handling NaN values during sort
- Ranking rows with different ranking methods
- Sorting and ranking within groups
- Finding top N and bottom N records
- Stable sort and its importance in production
- Real world leaderboard and report generation examples

Real World Usage:
- Generating ranked leaderboards for sales teams
- Finding top and bottom performing products
- Sorting customer records for priority queues
- Ranking students or employees by performance
- Ordering time series data correctly before analysis
- Producing sorted reports for business dashboards

Dataset Used:
World University Rankings dataset (public, no login required)
URL: https://raw.githubusercontent.com/dsrscientist/dataset1/master/world_university_rankings.csv
Fallback: Titanic dataset
https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv

================================================================================
"""

import numpy as np
import pandas as pd
import re

print("=" * 70)
print("LESSON 24: SORTING AND RANKING")
print("=" * 70)


# ==============================================================================
# SECTION 1: LOAD DATASET
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 1: LOAD DATASET")
print("=" * 70)

primary_url  = "https://raw.githubusercontent.com/dsrscientist/dataset1/master/world_university_rankings.csv"
fallback_url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

def to_snake(name):
    name = str(name).strip()
    name = re.sub(r"[\s\-\.]+", "_", name)
    name = re.sub(r"[^\w]", "", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_").lower()

def load_dataset():
    try:
        print("Loading World University Rankings dataset from:")
        print(primary_url)
        df = pd.read_csv(primary_url, encoding="latin-1")
        print("Primary dataset loaded successfully.")
        return df, "universities"
    except Exception as e:
        print(f"Primary load failed: {e}")
        print("\nFalling back to Titanic dataset from:")
        print(fallback_url)
        df = pd.read_csv(fallback_url)
        print("Fallback dataset loaded successfully.")
        return df, "titanic"

df_raw, dataset_name = load_dataset()

df_raw.columns = [to_snake(col) for col in df_raw.columns]

print(f"\nDataset: {dataset_name}")
print(f"Shape: {df_raw.shape}")
print(f"Columns: {list(df_raw.columns)}")
print("\nFirst 5 rows:")
print(df_raw.head())
print("\nData types:")
print(df_raw.dtypes)


# ==============================================================================
# SECTION 2: SORTING BASICS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 2: SORTING BASICS")
print("=" * 70)

df = df_raw.copy()

# ------------------------------------------------------------------------------
# 2.1 sort_values - Single Column
# ------------------------------------------------------------------------------
print("\n--- 2.1 sort_values - Single Column ---")

explanation = """
sort_values() sorts the DataFrame by one or more columns.
It returns a new DataFrame by default (does not modify in place).
Always reassign: df = df.sort_values(...)
"""
print(explanation)

if dataset_name == "universities":
    sort_col = df.select_dtypes(include=[np.number]).columns[0]
else:
    sort_col = "fare"

# Ascending sort (default)
df_asc = df.sort_values(sort_col)
print(f"Sorted by {sort_col} ascending (default):")
print(df_asc[[sort_col]].head(5))
print(f"  Min value at top: {df_asc[sort_col].iloc[0]}")

# Descending sort
df_desc = df.sort_values(sort_col, ascending=False)
print(f"\nSorted by {sort_col} descending:")
print(df_desc[[sort_col]].head(5))
print(f"  Max value at top: {df_desc[sort_col].iloc[0]}")

# Verify sort is correct
assert df_asc[sort_col].is_monotonic_increasing or \
       df_asc[sort_col].isnull().any(), "Ascending sort verification failed"
print(f"\nAscending sort verified: is_monotonic_increasing = "
      f"{df_asc[sort_col].dropna().is_monotonic_increasing}")

# ------------------------------------------------------------------------------
# 2.2 sort_values - Multiple Columns
# ------------------------------------------------------------------------------
print("\n--- 2.2 sort_values - Multiple Columns ---")

explanation = """
Sorting by multiple columns works like SQL ORDER BY col1, col2.
The first column has highest priority.
If two rows have the same value in col1, they are ordered by col2.
"""
print(explanation)

if dataset_name == "titanic":
    # Sort by class ascending, then fare descending within each class
    df_multi = df.sort_values(
        by=["pclass", "fare"],
        ascending=[True, False]
    )
    print("Sorted by pclass ASC then fare DESC within each class:")
    print(df_multi[["pclass", "fare", "name"]].head(10))

    # Verify multi-sort
    for pclass_val in sorted(df["pclass"].dropna().unique()):
        group = df_multi[df_multi["pclass"] == pclass_val]["fare"]
        assert group.dropna().is_monotonic_decreasing, \
            f"Fare not descending within class {pclass_val}"
    print("\nMulti-column sort verified for all classes")

else:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) >= 2:
        col1, col2 = numeric_cols[0], numeric_cols[1]
        df_multi = df.sort_values(
            by=[col1, col2],
            ascending=[False, True]
        )
        print(f"Sorted by {col1} DESC then {col2} ASC:")
        print(df_multi[[col1, col2]].head(10))

# ------------------------------------------------------------------------------
# 2.3 sort_index - Sorting by Row Labels
# ------------------------------------------------------------------------------
print("\n--- 2.3 sort_index - Sorting by Row Labels ---")

# After filtering or shuffling, index may be out of order
df_shuffled = df.sample(frac=1, random_state=42)
print(f"Shuffled index (first 5): {df_shuffled.index[:5].tolist()}")

# Sort back to original order
df_sorted_idx = df_shuffled.sort_index()
print(f"After sort_index (first 5): {df_sorted_idx.index[:5].tolist()}")

assert df_sorted_idx.index.is_monotonic_increasing, "Index not sorted"
print("Index sort verified: is_monotonic_increasing = True")

# Sort index descending
df_idx_desc = df_shuffled.sort_index(ascending=False)
print(f"\nDescending index (first 3): {df_idx_desc.index[:3].tolist()}")

# ------------------------------------------------------------------------------
# 2.4 Sorting String Columns
# ------------------------------------------------------------------------------
print("\n--- 2.4 Sorting String Columns ---")

if dataset_name == "titanic":
    # Sort alphabetically by name
    df_name_sort = df.sort_values("name")
    print("Sorted alphabetically by name:")
    print(df_name_sort["name"].head(5).tolist())

    # Sort by name descending (reverse alphabetical)
    df_name_desc = df.sort_values("name", ascending=False)
    print("\nReverse alphabetical:")
    print(df_name_desc["name"].head(5).tolist())

else:
    str_cols = df.select_dtypes(include=["object"]).columns
    if len(str_cols) > 0:
        col = str_cols[0]
        df_str_sort = df.sort_values(col)
        print(f"Sorted by string column '{col}':")
        print(df_str_sort[col].head(5).tolist())


# ==============================================================================
# SECTION 3: HANDLING NaN IN SORT
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 3: HANDLING NaN IN SORT")
print("=" * 70)

# ------------------------------------------------------------------------------
# 3.1 Default NaN Behavior
# ------------------------------------------------------------------------------
print("\n--- 3.1 Default NaN Behavior ---")

explanation = """
By default, NaN values are placed at the END of the sorted result
regardless of whether sorting is ascending or descending.

Control NaN placement with na_position parameter:
  na_position='last'   (default) NaN goes to the end
  na_position='first'  NaN goes to the beginning
"""
print(explanation)

if dataset_name == "titanic":
    nan_col = "age"
else:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    nan_col = numeric_cols[df[numeric_cols].isnull().any()].tolist()
    nan_col = nan_col[0] if nan_col else numeric_cols[0]

nan_count = df[nan_col].isnull().sum()
print(f"Column '{nan_col}' has {nan_count} NaN values")

# NaN at end (default)
df_nan_last = df.sort_values(nan_col, ascending=True, na_position="last")
print(f"\nAscending sort, NaN at end (default):")
print(f"  First value: {df_nan_last[nan_col].iloc[0]}")
print(f"  Last value:  {df_nan_last[nan_col].iloc[-1]}")
print(f"  Last 3 values: {df_nan_last[nan_col].tail(3).tolist()}")

# NaN at beginning
df_nan_first = df.sort_values(nan_col, ascending=True, na_position="first")
print(f"\nAscending sort, NaN at beginning:")
print(f"  First 3 values: {df_nan_first[nan_col].head(3).tolist()}")
print(f"  Last value: {df_nan_first[nan_col].iloc[-1]}")

# ------------------------------------------------------------------------------
# 3.2 Sorting After Filling NaN
# ------------------------------------------------------------------------------
print("\n--- 3.2 Sorting After Filling NaN ---")

# Fill NaN before sort for full control
df_filled_sort = df.copy()
fill_val = df[nan_col].median()
df_filled_sort[nan_col] = df_filled_sort[nan_col].fillna(fill_val)
df_filled_sort = df_filled_sort.sort_values(nan_col)

print(f"Sorted after filling NaN with median ({fill_val:.1f}):")
print(f"  NaN count after fill: {df_filled_sort[nan_col].isnull().sum()}")
print(f"  Min: {df_filled_sort[nan_col].min()}")
print(f"  Max: {df_filled_sort[nan_col].max()}")


# ==============================================================================
# SECTION 4: STABLE SORT
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 4: STABLE SORT")
print("=" * 70)

explanation = """
A STABLE SORT preserves the original order of rows that have
equal values in the sort key.

Pandas sort_values() is stable by default (kind='quicksort' in NumPy
but Pandas uses mergesort which is stable).

Why stability matters in production:
  - If you sort by date and rows have the same date,
    the original row order is preserved within that date
  - Reproducible results: same input always gives same output
  - Required when secondary order is defined by insertion order

kind parameter options:
  'quicksort'  (not stable, faster for large data)
  'mergesort'  (stable, default in Pandas sort)
  'heapsort'   (not stable)
  'stable'     (alias for stable sort)
"""
print(explanation)

if dataset_name == "titanic":
    # Demonstrate stable sort
    # Sort by pclass: passengers with same class keep their original row order
    df_stable = df.sort_values("pclass", kind="mergesort")
    df_unstable = df.sort_values("pclass", kind="quicksort")

    print("Stable sort (mergesort) - rows with same pclass keep original order")
    print("First 5 rows in class 1 (stable):")
    print(df_stable[df_stable["pclass"] == 1][["pclass", "name"]].head(5))

    # Verify: within same class, PassengerId order should be preserved
    # (since original data is ordered by PassengerId)
    class1_stable = df_stable[df_stable["pclass"] == 1]["passengerid"]
    print(f"\nPassengerIds in class 1 (first 5, stable): {class1_stable.head(5).tolist()}")
    print("Original insertion order is preserved within each class group")

else:
    # Sort by first string column stably
    str_cols = df.select_dtypes(include=["object"]).columns
    if len(str_cols) > 0:
        sort_col = str_cols[0]
        df_stable = df.sort_values(sort_col, kind="mergesort")
        print(f"Stable sort by '{sort_col}':")
        print(df_stable[sort_col].head(5).tolist())


# ==============================================================================
# SECTION 5: RANKING
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 5: RANKING")
print("=" * 70)

# ------------------------------------------------------------------------------
# 5.1 Basic Ranking with rank()
# ------------------------------------------------------------------------------
print("\n--- 5.1 Basic Ranking with rank() ---")

explanation = """
rank() assigns a rank to each value in a Series.
By default rank starts from 1 (smallest value gets rank 1).
Ties are handled according to the method parameter.

Rank does NOT sort the DataFrame.
It adds a rank number to each row based on a column's values.
"""
print(explanation)

if dataset_name == "titanic":
    rank_col = "fare"
else:
    rank_col = df.select_dtypes(include=[np.number]).columns[0]

# Default rank (ascending: smallest = rank 1)
df["rank_asc"] = df[rank_col].rank()
print(f"Rank of {rank_col} (ascending, smallest = rank 1):")
if dataset_name == "titanic":
    print(df[["name", rank_col, "rank_asc"]].sort_values("rank_asc").head(5))
else:
    print(df[[rank_col, "rank_asc"]].sort_values("rank_asc").head(5))

# Descending rank (largest = rank 1)
df["rank_desc"] = df[rank_col].rank(ascending=False)
print(f"\nRank of {rank_col} (descending, largest = rank 1):")
if dataset_name == "titanic":
    print(df[["name", rank_col, "rank_desc"]].sort_values("rank_desc").head(5))
else:
    print(df[[rank_col, "rank_desc"]].sort_values("rank_desc").head(5))

# ------------------------------------------------------------------------------
# 5.2 Tie-Breaking Methods
# ------------------------------------------------------------------------------
print("\n--- 5.2 Tie-Breaking Methods ---")

explanation = """
When multiple rows have the same value, ties must be broken.
rank() method parameter controls this:

  'average'  (default)  Tied rows get the average of the ranks they span
  'min'                  Tied rows all get the LOWEST rank in the tie group
  'max'                  Tied rows all get the HIGHEST rank in the tie group
  'first'                Tied rows are ranked in order of appearance
  'dense'                Like 'min' but no gaps in rank numbers
"""
print(explanation)

# Create a simple example with ties
tie_example = pd.Series([10, 20, 20, 30, 30, 30, 40])

methods = ["average", "min", "max", "first", "dense"]
print("Values: [10, 20, 20, 30, 30, 30, 40]")
print(f"\n{'Value':<8}", end="")
for method in methods:
    print(f"{method:>10}", end="")
print()
print("-" * (8 + 10 * len(methods)))

for i, val in enumerate(tie_example):
    print(f"{val:<8}", end="")
    for method in methods:
        rank_val = tie_example.rank(method=method).iloc[i]
        print(f"{rank_val:>10.1f}", end="")
    print()

print("""
Explanation:
  average: The three 30s occupy ranks 4, 5, 6 so each gets (4+5+6)/3 = 5.0
  min:     The three 30s all get rank 4 (lowest in their group)
  max:     The three 30s all get rank 6 (highest in their group)
  first:   The three 30s get ranks 4, 5, 6 in row order
  dense:   The three 30s all get rank 4, next value gets rank 5 (no gap)
""")

# Real world choice guide
print("When to use each method:")
print("  average: statistical analysis, matches Excel RANK.AVG")
print("  min:     competition ranking (tied for 3rd place)")
print("  dense:   leaderboards where you want 1st, 2nd, 3rd with no gaps")
print("  first:   unique rank for every row regardless of ties")
print("  max:     percentile calculations")

# ------------------------------------------------------------------------------
# 5.3 Percentile Rank
# ------------------------------------------------------------------------------
print("\n--- 5.3 Percentile Rank ---")

explanation = """
pct=True in rank() returns fractional rank between 0 and 1.
This is equivalent to a percentile position.
A value with pct_rank=0.9 is higher than 90% of all values.
"""
print(explanation)

df["pct_rank"] = df[rank_col].rank(pct=True)

print(f"Percentile rank of {rank_col}:")
if dataset_name == "titanic":
    sample = df[["name", rank_col, "pct_rank"]].sort_values(rank_col, ascending=False).head(5)
else:
    sample = df[[rank_col, "pct_rank"]].sort_values(rank_col, ascending=False).head(5)
print(sample.round(4))

# Interpret
if dataset_name == "titanic":
    top_pct = df[df["pct_rank"] >= 0.95]
    print(f"\nPassengers in top 5% by fare: {len(top_pct)}")
    print(f"Min fare in top 5%: {top_pct[rank_col].min():.2f}")

# ------------------------------------------------------------------------------
# 5.4 Handling NaN in Rank
# ------------------------------------------------------------------------------
print("\n--- 5.4 Handling NaN in Rank ---")

print("NaN handling options in rank():")
print("  na_option='keep'   (default) NaN stays NaN in rank")
print("  na_option='top'    NaN gets lowest rank (treated as smallest)")
print("  na_option='bottom' NaN gets highest rank (treated as largest)")

nan_demo = pd.Series([10, np.nan, 30, np.nan, 50])
print(f"\nSeries: {nan_demo.tolist()}")
print(f"rank(na_option='keep'):   {nan_demo.rank(na_option='keep').tolist()}")
print(f"rank(na_option='top'):    {nan_demo.rank(na_option='top').tolist()}")
print(f"rank(na_option='bottom'): {nan_demo.rank(na_option='bottom').tolist()}")


# ==============================================================================
# SECTION 6: SORTING AND RANKING WITHIN GROUPS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 6: SORTING AND RANKING WITHIN GROUPS")
print("=" * 70)

# ------------------------------------------------------------------------------
# 6.1 Rank Within Groups Using transform
# ------------------------------------------------------------------------------
print("\n--- 6.1 Rank Within Groups Using transform ---")

explanation = """
To rank within each group independently, use groupby + transform.
This is essential for leaderboards broken down by category.
Example: Rank each passenger within their own class,
not against all passengers.
"""
print(explanation)

if dataset_name == "titanic":
    df["fare_rank_within_class"] = df.groupby("pclass")["fare"].rank(
        ascending=False,
        method="dense"
    )

    print("Fare rank within each passenger class (rank 1 = highest fare):")
    result = df[["pclass", "name", "fare", "fare_rank_within_class"]]
    # Show top 3 per class
    for pclass_val in sorted(df["pclass"].dropna().unique()):
        group_top = result[result["pclass"] == pclass_val].sort_values(
            "fare_rank_within_class"
        ).head(3)
        print(f"\n  Class {int(pclass_val)} top 3 by fare:")
        print(group_top.to_string(index=False))

else:
    numeric_col = df.select_dtypes(include=[np.number]).columns[0]
    str_col     = df.select_dtypes(include=["object"]).columns[0]

    df["rank_within_group"] = df.groupby(str_col)[numeric_col].rank(
        ascending=False,
        method="dense"
    )
    print(f"Rank of {numeric_col} within each {str_col} group:")
    print(df[[str_col, numeric_col, "rank_within_group"]].head(10))

# ------------------------------------------------------------------------------
# 6.2 Sort Within Groups Using groupby + apply
# ------------------------------------------------------------------------------
print("\n--- 6.2 Sort Within Groups ---")

if dataset_name == "titanic":
    # Sort passengers within each class by fare descending
    df_sorted_within = df.sort_values(
        ["pclass", "fare"], ascending=[True, False]
    ).reset_index(drop=True)

    print("Passengers sorted by fare within each class:")
    print(df_sorted_within[["pclass", "name", "fare"]].head(9))

else:
    numeric_col = df.select_dtypes(include=[np.number]).columns[0]
    str_col     = df.select_dtypes(include=["object"]).columns[0]

    df_sorted_within = df.groupby(
        str_col, group_keys=False
    ).apply(
        lambda g: g.sort_values(numeric_col, ascending=False)
    )
    print(f"Sorted by {numeric_col} within each {str_col}:")
    print(df_sorted_within[[str_col, numeric_col]].head(9))

# ------------------------------------------------------------------------------
# 6.3 Cumulative Rank and Running Position
# ------------------------------------------------------------------------------
print("\n--- 6.3 Cumulative Metrics Using cumsum and cumcount ---")

if dataset_name == "titanic":
    # Sort by fare and compute cumulative sum
    df_cum = df.sort_values("fare", ascending=False).copy()
    df_cum["cumulative_fare"] = df_cum["fare"].cumsum()
    df_cum["running_position"] = range(1, len(df_cum) + 1)
    df_cum["pct_total_fare"] = (
        df_cum["cumulative_fare"] / df_cum["fare"].sum() * 100
    ).round(2)

    print("Cumulative fare analysis (top 10 by fare):")
    print(df_cum[["name", "fare", "cumulative_fare",
                   "running_position", "pct_total_fare"]].head(10))

    # How many top passengers account for 50% of total fare revenue
    top_50pct = df_cum[df_cum["pct_total_fare"] <= 50]
    print(f"\nTop {len(top_50pct)} passengers account for 50% of total fare collected")

else:
    numeric_col = df.select_dtypes(include=[np.number]).columns[0]
    df_cum = df.sort_values(numeric_col, ascending=False).copy()
    df_cum["cumulative"] = df_cum[numeric_col].cumsum()
    df_cum["running_position"] = range(1, len(df_cum) + 1)
    print(f"Cumulative {numeric_col} (top 5):")
    print(df_cum[[numeric_col, "cumulative", "running_position"]].head(5))


# ==============================================================================
# SECTION 7: TOP N AND BOTTOM N
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 7: TOP N AND BOTTOM N")
print("=" * 70)

# ------------------------------------------------------------------------------
# 7.1 nlargest and nsmallest
# ------------------------------------------------------------------------------
print("\n--- 7.1 nlargest and nsmallest ---")

explanation = """
nlargest(n, col)  returns the n rows with largest values in col
nsmallest(n, col) returns the n rows with smallest values in col

These are faster than sort_values().head(n) for large DataFrames
because they do not sort the entire DataFrame.
"""
print(explanation)

if dataset_name == "titanic":
    value_col = "fare"
else:
    value_col = df.select_dtypes(include=[np.number]).columns[0]

# Top 5
top5 = df.nlargest(5, value_col)
print(f"Top 5 by {value_col}:")
if dataset_name == "titanic":
    print(top5[["name", "pclass", value_col]].to_string(index=False))
else:
    print(top5[[value_col]].to_string(index=False))

# Bottom 5
bottom5 = df.nsmallest(5, value_col)
print(f"\nBottom 5 by {value_col}:")
if dataset_name == "titanic":
    print(bottom5[["name", "pclass", value_col]].to_string(index=False))
else:
    print(bottom5[[value_col]].to_string(index=False))

# nlargest on Series
print(f"\nnlargest on Series - top 3 {value_col} values:")
print(df[value_col].nlargest(3).to_string())

# ------------------------------------------------------------------------------
# 7.2 Top N Per Group
# ------------------------------------------------------------------------------
print("\n--- 7.2 Top N Per Group ---")

if dataset_name == "titanic":
    # Top 2 fares per class
    top2_per_class = df.groupby(
        "pclass", group_keys=False
    )[["pclass", "name", "fare"]].apply(
        lambda g: g.nlargest(2, "fare")
    )
    print("Top 2 fares per passenger class:")
    print(top2_per_class.to_string(index=False))

else:
    numeric_col = df.select_dtypes(include=[np.number]).columns[0]
    str_col     = df.select_dtypes(include=["object"]).columns[0]

    top2_per_group = df.groupby(
        str_col, group_keys=False
    )[[str_col, numeric_col]].apply(
        lambda g: g.nlargest(2, numeric_col)
    )
    print(f"Top 2 by {numeric_col} per {str_col} group:")
    print(top2_per_group.head(10).to_string(index=False))

# ------------------------------------------------------------------------------
# 7.3 Quantile-Based Segmentation
# ------------------------------------------------------------------------------
print("\n--- 7.3 Quantile-Based Segmentation ---")

if dataset_name == "titanic":
    # Segment passengers by fare into quartiles
    df["fare_quartile"] = pd.qcut(
        df["fare"],
        q=4,
        labels=["Q1_low", "Q2_mid_low", "Q3_mid_high", "Q4_high"]
    )

    quartile_summary = df.groupby("fare_quartile", observed=True).agg(
        count         = ("fare",     "count"),
        min_fare      = ("fare",     "min"),
        max_fare      = ("fare",     "max"),
        survival_rate = ("survived", "mean"),
    ).round(3)

    print("Passengers segmented by fare quartile:")
    print(quartile_summary)

else:
    numeric_col = df.select_dtypes(include=[np.number]).columns[0]
    df["quartile"] = pd.qcut(
        df[numeric_col].dropna(),
        q=4,
        labels=["Q1", "Q2", "Q3", "Q4"]
    )
    print(f"Quartile distribution of {numeric_col}:")
    print(df["quartile"].value_counts().sort_index())


# ==============================================================================
# SECTION 8: REAL WORLD EXAMPLE - LEADERBOARD PIPELINE
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 8: REAL WORLD EXAMPLE - LEADERBOARD PIPELINE")
print("=" * 70)

if dataset_name == "titanic":
    print("Scenario: Generate a passenger fare leaderboard with group rankings")

    # Step 1: Prepare clean data
    leaderboard_df = df[["passengerid", "name", "pclass", "sex",
                          "age", "fare", "survived"]].copy()
    leaderboard_df = leaderboard_df.dropna(subset=["fare"])

    # Step 2: Overall rank by fare
    leaderboard_df["overall_rank"] = leaderboard_df["fare"].rank(
        ascending=False, method="dense"
    ).astype(int)

    # Step 3: Rank within class
    leaderboard_df["class_rank"] = leaderboard_df.groupby("pclass")["fare"].rank(
        ascending=False, method="dense"
    ).astype(int)

    # Step 4: Percentile
    leaderboard_df["fare_percentile"] = (
        leaderboard_df["fare"].rank(pct=True) * 100
    ).round(1)

    # Step 5: Fare tier label
    leaderboard_df["fare_tier"] = pd.qcut(
        leaderboard_df["fare"],
        q=[0, 0.25, 0.5, 0.75, 1.0],
        labels=["Budget", "Economy", "Premium", "Luxury"]
    )

    # Step 6: Sort by overall rank
    leaderboard_df = leaderboard_df.sort_values("overall_rank")

    # Step 7: Validate
    assert leaderboard_df["overall_rank"].min() == 1, "Rank does not start at 1"
    assert leaderboard_df["class_rank"].min() == 1, "Class rank error"
    assert leaderboard_df["fare_percentile"].between(0, 100).all(), \
        "Percentile out of range"
    print("Validation passed: all rank assertions satisfied")

    # Step 8: Display leaderboard
    print("\nTop 15 Passenger Fare Leaderboard:")
    print("-" * 100)
    display_cols = ["overall_rank", "class_rank", "name", "pclass",
                    "fare", "fare_percentile", "fare_tier", "survived"]
    print(leaderboard_df[display_cols].head(15).to_string(index=False))

    # Step 9: Leaderboard summary by tier
    print("\nTier Summary:")
    tier_summary = leaderboard_df.groupby("fare_tier", observed=True).agg(
        passenger_count = ("passengerid",    "count"),
        avg_fare        = ("fare",           "mean"),
        min_fare        = ("fare",           "min"),
        max_fare        = ("fare",           "max"),
        survival_rate   = ("survived",       "mean"),
    ).round(3)
    print(tier_summary)

else:
    print("Scenario: Generate university ranking leaderboard")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    str_cols     = df.select_dtypes(include=["object"]).columns.tolist()

    if len(numeric_cols) > 0:
        score_col = numeric_cols[0]
        leaderboard_df = df.copy().dropna(subset=[score_col])
        leaderboard_df["overall_rank"] = leaderboard_df[score_col].rank(
            ascending=False, method="dense"
        ).astype(int)
        leaderboard_df["percentile"] = (
            leaderboard_df[score_col].rank(pct=True) * 100
        ).round(1)
        leaderboard_df = leaderboard_df.sort_values("overall_rank")

        print(f"Top 10 by {score_col}:")
        print(leaderboard_df[[score_col, "overall_rank", "percentile"]].head(10))


# ==============================================================================
# SECTION 9: SORTING CATEGORICAL COLUMNS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 9: SORTING CATEGORICAL COLUMNS")
print("=" * 70)

# ------------------------------------------------------------------------------
# 9.1 Default Categorical Sort
# ------------------------------------------------------------------------------
print("\n--- 9.1 Default Categorical Sort ---")

explanation = """
When a column has category dtype with ordered=True,
sort_values() respects the category order, not alphabetical order.
This is important for ordinal data like sizes, grades, or priorities.
"""
print(explanation)

# Create example with logical order
priority_data = pd.DataFrame({
    "task":     ["Task A", "Task B", "Task C", "Task D", "Task E"],
    "priority": ["High", "Low", "Critical", "Medium", "Low"],
    "hours":    [3, 1, 8, 2, 1]
})

# Without category: sorts alphabetically (Critical, High, Low, Medium)
df_alpha = priority_data.sort_values("priority")
print("Default alphabetical sort of priority:")
print(df_alpha[["task", "priority"]].to_string(index=False))

# With ordered category: sorts in logical order
priority_order = pd.CategoricalDtype(
    categories=["Low", "Medium", "High", "Critical"],
    ordered=True
)
priority_data["priority"] = priority_data["priority"].astype(priority_order)

df_logical = priority_data.sort_values("priority")
print("\nWith ordered category (Low < Medium < High < Critical):")
print(df_logical[["task", "priority"]].to_string(index=False))

# Also enables comparisons
print("\nTasks with priority above Medium:")
above_medium = priority_data[priority_data["priority"] > "Medium"]
print(above_medium[["task", "priority"]].to_string(index=False))


# ==============================================================================
# SECTION 10: PERFORMANCE NOTES
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 10: PERFORMANCE NOTES")
print("=" * 70)

performance_notes = """
SORT PERFORMANCE:
  sort_values()           O(n log n) - same as any comparison sort
  nlargest(k, col)        O(n log k) - faster than full sort when k << n
  nsmallest(k, col)       O(n log k) - same advantage

RANK PERFORMANCE:
  rank() with method='average'  Fast, C-optimized
  rank() with method='first'    Requires stable sort internally
  groupby + transform(rank)     Slower, applies per group in Python

BEST PRACTICES:
  1. Use nlargest/nsmallest when you only need top or bottom N
     It is faster than sort_values().head(n) for large DataFrames

  2. Use sort_values with kind='mergesort' when order of tied
     rows matters (stable sort)

  3. Pre-filter before sorting to reduce sort input size
     df[df["col"] > threshold].sort_values("other_col")

  4. Sorting on multiple columns is not significantly slower
     than sorting on one column

  5. sort_index() is faster than sort_values() on the index column
     because it operates directly on the index structure

  6. rank() with pct=True avoids a separate division step
     and is more readable than computing manually
"""
print(performance_notes)


# ==============================================================================
# SECTION 11: COMMON PITFALLS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 11: COMMON PITFALLS")
print("=" * 70)

pitfalls = """
Pitfall 1: Forgetting sort_values() does not modify in place
   WRONG: df.sort_values("col")          <- df is unchanged
   RIGHT: df = df.sort_values("col")

Pitfall 2: NaN placement surprises
   NaN goes to END by default in both ascending and descending sort
   If you need NaN first: na_position='first'
   If you need NaN excluded: sort after dropna()

Pitfall 3: rank() starts at 1.0 (float) not 0
   Default rank output is float64 even for integer-like values
   Cast with .astype(int) only after verifying no NaN in rank output

Pitfall 4: Confusing sort and rank
   sort_values() reorders rows in the DataFrame
   rank() adds a new column with rank numbers, does not reorder

Pitfall 5: Alphabetical sort of numeric strings
   "10" sorts before "9" alphabetically but after numerically
   Always convert to numeric dtype before sorting numeric data

Pitfall 6: Group rank with wrong method
   method='average' produces float ranks (4.0 not 4)
   Use method='dense' for integer-like leaderboard ranks
   Then cast to int after checking no NaN

Pitfall 7: nlargest with tied values includes extra rows
   nlargest(3, "col") may return more than 3 rows if keep='all'
   Default keep='first' returns exactly n rows
"""
print(pitfalls)


# ==============================================================================
# SECTION 12: SUMMARY TABLE
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 12: SUMMARY TABLE")
print("=" * 70)

summary = """
Operation                              | Syntax
---------------------------------------|----------------------------------------------
Sort ascending by one column           | df.sort_values("col")
Sort descending                        | df.sort_values("col", ascending=False)
Sort by multiple columns               | df.sort_values(["c1","c2"], ascending=[T,F])
Sort by index                          | df.sort_index()
NaN at top in sort                     | df.sort_values("col", na_position="first")
Stable sort                            | df.sort_values("col", kind="mergesort")
Rank ascending (smallest = 1)          | df["col"].rank()
Rank descending (largest = 1)          | df["col"].rank(ascending=False)
Rank with dense method                 | df["col"].rank(method="dense")
Percentile rank                        | df["col"].rank(pct=True)
Rank NaN as bottom                     | df["col"].rank(na_option="bottom")
Rank within groups                     | df.groupby("g")["col"].rank(method="dense")
Top N rows                             | df.nlargest(n, "col")
Bottom N rows                          | df.nsmallest(n, "col")
Top N per group                        | df.groupby("g").apply(lambda g: g.nlargest(n,"col"))
Cumulative sum after sort              | df.sort_values("col")["val"].cumsum()
Quantile segmentation                  | pd.qcut(df["col"], q=4, labels=[...])
Sort with ordered category             | col.astype(CategoricalDtype(ordered=True))
"""
print(summary)


# ==============================================================================
# SECTION 13: PRACTICE EXERCISES
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 13: PRACTICE EXERCISES")
print("=" * 70)

if dataset_name == "titanic":
    print("Exercise 1: Sort by survival then by fare descending")
    ex1 = df.sort_values(
        by=["survived", "fare"],
        ascending=[False, False]
    )
    print("  Top 5 (survived=1, highest fare):")
    print(ex1[["name", "survived", "fare"]].head(5).to_string(index=False))

    print("\nExercise 2: Dense rank by age, NaN ranked last")
    df_ex2 = df.copy()
    df_ex2["age_rank"] = df_ex2["age"].rank(
        method="dense",
        ascending=False,
        na_option="bottom"
    ).astype("Int64")
    print("  Oldest passengers:")
    print(df_ex2[["name", "age", "age_rank"]].nlargest(5, "age").to_string(index=False))

    print("\nExercise 3: Top 2 survivors per class by fare")
    ex3 = df[df["survived"] == 1].groupby(
        "pclass", group_keys=False
    )[["pclass", "name", "fare"]].apply(
        lambda g: g.nlargest(2, "fare")
    )
    print(ex3.to_string(index=False))

    print("\nExercise 4: Decile segmentation of fare")
    df_ex4 = df.dropna(subset=["fare"]).copy()
    df_ex4["fare_decile"] = pd.qcut(
        df_ex4["fare"],
        q=10,
        labels=[f"D{i}" for i in range(1, 11)]
    )
    decile_survival = df_ex4.groupby(
        "fare_decile", observed=True
    )["survived"].mean().round(3)
    print("  Survival rate by fare decile:")
    print(decile_survival)

else:
    numeric_col = df.select_dtypes(include=[np.number]).columns[0]

    print(f"Exercise 1: Sort by {numeric_col} descending, show top 5")
    ex1 = df.nlargest(5, numeric_col)
    print(ex1[[numeric_col]].to_string())

    print(f"\nExercise 2: Dense rank of {numeric_col}")
    df[f"{numeric_col}_rank"] = df[numeric_col].rank(
        method="dense", ascending=False
    ).astype("Int64")
    print(df[[numeric_col, f"{numeric_col}_rank"]].head(5))

    print(f"\nExercise 3: Quartile segmentation of {numeric_col}")
    df_ex3 = df.dropna(subset=[numeric_col]).copy()
    df_ex3["quartile"] = pd.qcut(
        df_ex3[numeric_col],
        q=4,
        labels=["Q1_bottom", "Q2", "Q3", "Q4_top"]
    )
    print(df_ex3["quartile"].value_counts().sort_index())


# ==============================================================================
# SECTION 14: KEY TAKEAWAYS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 14: KEY TAKEAWAYS")
print("=" * 70)

takeaways = """
1.  sort_values() does not modify in place - always reassign
2.  Use ascending=[True, False] list for multi-column sort directions
3.  NaN goes to end by default in both ascending and descending sort
4.  Use kind='mergesort' for stable sort when tied row order matters
5.  rank() adds rank numbers without reordering the DataFrame
6.  Use method='dense' for leaderboard-style ranking with no gaps
7.  Use pct=True for percentile ranks between 0 and 1
8.  Use groupby + transform for ranking within groups independently
9.  nlargest and nsmallest are faster than sort_values().head(n)
10. Use ordered CategoricalDtype for sorting ordinal string data correctly
11. Always validate rank output: check min, max, and NaN counts
12. pd.qcut creates equal-frequency bins for quantile segmentation
"""
print(takeaways)


# ==============================================================================
# NEXT LESSON PREVIEW
# ==============================================================================
print("\n" + "=" * 70)
print("NEXT LESSON PREVIEW")
print("=" * 70)

preview = """
Lesson 25: Merging, Joining, and Concatenation

You will learn:
- pd.merge() for database-style joins (inner, left, right, outer)
- pd.concat() for stacking DataFrames vertically and horizontally
- join() as a shorthand for index-based merges
- Handling duplicate columns after merge
- Validating merge results with indicator and validate parameters
- Detecting and debugging merge problems
- Real world multi-table data assembly pipeline
"""
print(preview)

print("\n" + "=" * 70)
print("END OF LESSON 24")
print("=" * 70)
print("\nYou can now sort DataFrames by any combination of columns,")
print("rank values with full control over ties and NaN handling,")
print("and build production-grade leaderboards with group rankings.")