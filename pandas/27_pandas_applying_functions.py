"""
LESSON 27: APPLYING FUNCTIONS (apply, map, transform, vectorize)
================================================================================

What You Will Learn:
- When to use apply vs map vs vectorized operations
- Row-wise and column-wise function application with apply()
- Series.map() for element-wise transformation and lookup
- DataFrame.map() for element-wise transformation
- transform() for group-aware operations that preserve shape
- Performance hierarchy and when NOT to use apply
- Lambda functions for quick inline transformations
- Building reusable transformation pipelines
- Common production patterns

Real World Usage:
- Deriving new columns from complex business logic
- Encoding categorical variables with lookup tables
- Applying domain-specific calculations row by row
- Cleaning and normalizing text fields
- Feature engineering for machine learning
- Scoring and ranking records

Dataset Used:
Student performance dataset (public, no login required)
URL: https://raw.githubusercontent.com/dsrscientist/dataset1/master/student_scores.csv
Fallback: Titanic dataset
https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv

================================================================================
"""

import numpy as np
import pandas as pd
import time
import re

print("=" * 70)
print("LESSON 27: APPLYING FUNCTIONS")
print("apply, map, transform, and vectorized operations")
print("=" * 70)


# ==============================================================================
# SECTION 1: LOAD DATASET
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 1: LOAD DATASET")
print("=" * 70)

primary_url  = "https://raw.githubusercontent.com/dsrscientist/dataset1/master/student_scores.csv"
fallback_url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"


def to_snake(name):
    """Convert column name to snake_case."""
    name = str(name).strip()
    name = re.sub(r"[\s\-\.]+", "_", name)
    name = re.sub(r"[^\w]", "", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_").lower()


def load_dataset():
    try:
        print("Loading student scores dataset from:")
        print(primary_url)
        df = pd.read_csv(primary_url)
        print("Primary dataset loaded successfully.")
        return df, "students"
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
print("\nMissing values:")
print(df_raw.isnull().sum())


# ==============================================================================
# SECTION 2: THE FUNCTION APPLICATION HIERARCHY
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 2: THE FUNCTION APPLICATION HIERARCHY")
print("=" * 70)

hierarchy = """
FASTEST to SLOWEST for applying transformations in Pandas:

1. VECTORIZED NUMPY/PANDAS OPERATIONS
   df["col"] * 2, np.log(df["col"]), df["col"].str.lower()
   Operates on entire column at once in compiled C code.
   Always prefer this when possible.

2. SERIES.map() or DATAFRAME.map()
   Element-wise. Best for lookups and simple transformations.
   Faster than apply() for element-level work.

3. SERIES.apply() or DATAFRAME.apply()
   Applies a Python function to each element, row, or column.
   Flexible but slower because Python function overhead per call.

4. PYTHON LOOPS (for row in df.iterrows())
   Never use loops for row-by-row computation on large DataFrames.
   Orders of magnitude slower than all options above.

RULE: Always ask yourself if a vectorized approach exists first.
Only fall back to apply() when vectorization is not possible.
"""
print(hierarchy)


# ==============================================================================
# SECTION 3: VECTORIZED OPERATIONS (BASELINE COMPARISON)
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 3: VECTORIZED OPERATIONS (BASELINE)")
print("=" * 70)

# Create a numeric dataset for benchmarking
np.random.seed(42)
n = 100_000
bench_df = pd.DataFrame({
    "score_a": np.random.uniform(0, 100, n),
    "score_b": np.random.uniform(0, 100, n),
    "score_c": np.random.uniform(0, 100, n),
    "category": np.random.choice(["A", "B", "C", "D"], n),
    "weight":   np.random.uniform(0.5, 1.5, n)
})

print(f"Benchmark dataset: {n:,} rows")
print(bench_df.head(3))

# ------------------------------------------------------------------------------
# 3.1 Vectorized vs apply() performance comparison
# ------------------------------------------------------------------------------
print("\n--- 3.1 Vectorized vs apply() Performance ---")

# Task: Compute weighted average of three scores

# Method 1: Vectorized (FAST)
start = time.perf_counter()
result_vec = (
    bench_df["score_a"] * 0.3 +
    bench_df["score_b"] * 0.4 +
    bench_df["score_c"] * 0.3
)
time_vec = time.perf_counter() - start

# Method 2: apply() row-wise (SLOW)
start = time.perf_counter()
result_apply = bench_df.apply(
    lambda row: row["score_a"] * 0.3 + row["score_b"] * 0.4 + row["score_c"] * 0.3,
    axis=1
)
time_apply = time.perf_counter() - start

# Validate both produce same result
assert np.allclose(result_vec, result_apply), "Results differ!"

print(f"Vectorized:       {time_vec*1000:.2f} ms")
print(f"apply() row-wise: {time_apply*1000:.2f} ms")
print(f"Speedup:          {time_apply / time_vec:.0f}x faster with vectorization")

print("\nConclusion: Always use vectorized math over apply() for numeric columns.")


# ==============================================================================
# SECTION 4: Series.apply()
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 4: Series.apply()")
print("=" * 70)

explanation = """
Series.apply(func) applies func to each element of the Series.
Returns a new Series with the results.

Use when:
- The logic is complex and cannot be expressed vectorially
- You need to call an external function per element
- Logic involves conditional branching on multiple criteria

Avoid when:
- Simple arithmetic (use +, -, *, /)
- Built-in string methods (use .str accessor)
- NumPy ufuncs can do the same job
"""
print(explanation)

df = df_raw.copy()

# ------------------------------------------------------------------------------
# 4.1 Simple Function Application
# ------------------------------------------------------------------------------
print("\n--- 4.1 Simple Function Application ---")

if dataset_name == "titanic":
    # Apply to extract title from name
    def extract_title(name):
        """Extract title from Titanic name string."""
        if pd.isna(name):
            return "Unknown"
        match = re.search(r",\s*([A-Za-z]+)\.", str(name))
        return match.group(1) if match else "Unknown"

    df["title"] = df["name"].apply(extract_title)

    print("Extracted title from name:")
    print(df[["name", "title"]].head(8))
    print("\nTitle value counts:")
    print(df["title"].value_counts())

else:
    # Numeric example: classify a score
    numeric_col = df.select_dtypes(include=[np.number]).columns[0]

    def classify_score(val):
        """Classify numeric value into bands."""
        if pd.isna(val):
            return "Unknown"
        if val >= 90:
            return "Excellent"
        if val >= 75:
            return "Good"
        if val >= 50:
            return "Average"
        return "Below Average"

    df["score_band"] = df[numeric_col].apply(classify_score)
    print(f"Score band from {numeric_col}:")
    print(df[[numeric_col, "score_band"]].head(10))
    print("\nBand distribution:")
    print(df["score_band"].value_counts())

# ------------------------------------------------------------------------------
# 4.2 apply() with Arguments (args and kwargs)
# ------------------------------------------------------------------------------
print("\n--- 4.2 apply() with Arguments ---")

def scale_value(val, min_val, max_val, clip=True):
    """
    Scale value to [0, 1] range using min-max normalization.

    Parameters
    ----------
    val : float
    min_val : float
        Minimum value for scaling
    max_val : float
        Maximum value for scaling
    clip : bool
        Whether to clip result to [0, 1]
    """
    if pd.isna(val):
        return np.nan
    if max_val == min_val:
        return 0.0
    scaled = (val - min_val) / (max_val - min_val)
    if clip:
        return float(np.clip(scaled, 0.0, 1.0))
    return float(scaled)


numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if len(numeric_cols) > 0:
    col = numeric_cols[0]
    col_min = df[col].min()
    col_max = df[col].max()

    # Pass extra args to apply using args= and kwargs=
    df[f"{col}_scaled"] = df[col].apply(
        scale_value,
        args=(col_min, col_max),
        clip=True
    )

    print(f"Scaled {col} to [0, 1]:")
    print(df[[col, f"{col}_scaled"]].describe().round(4))

# ------------------------------------------------------------------------------
# 4.3 Lambda Functions with apply()
# ------------------------------------------------------------------------------
print("\n--- 4.3 Lambda Functions ---")

explanation = """
Lambda is an inline anonymous function.
Use for simple, short logic that does not need a name.
For anything more than one line, write a named function instead.
"""
print(explanation)

if dataset_name == "titanic":
    # Count words in name using lambda
    df["name_word_count"] = df["name"].apply(
        lambda x: len(str(x).split()) if pd.notna(x) else 0
    )
    print("Word count in passenger name:")
    print(df[["name", "name_word_count"]].head(5))

else:
    if len(numeric_cols) > 0:
        col = numeric_cols[0]
        df["absolute_deviation"] = df[col].apply(
            lambda x: abs(x - df[col].mean()) if pd.notna(x) else np.nan
        )
        print(f"Absolute deviation from mean of {col}:")
        print(df[[col, "absolute_deviation"]].head(5).round(4))


# ==============================================================================
# SECTION 5: DATAFRAME.apply() (ROW-WISE AND COLUMN-WISE)
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 5: DATAFRAME.apply()")
print("=" * 70)

explanation = """
DataFrame.apply(func, axis) applies func along an axis.

  axis=0  (default): Function receives each COLUMN as a Series
  axis=1           : Function receives each ROW as a Series

Column-wise (axis=0) is fast, similar to Series.apply on each column.
Row-wise (axis=1) is slow: Python overhead for every single row.

Use row-wise apply ONLY when the logic genuinely needs multiple columns.
If you only need one column, use Series.apply instead.
"""
print(explanation)

# ------------------------------------------------------------------------------
# 5.1 Column-wise apply (axis=0)
# ------------------------------------------------------------------------------
print("\n--- 5.1 Column-wise apply (axis=0) ---")

# Apply a function to each column
numeric_df = bench_df[["score_a", "score_b", "score_c"]].head(1000)

def column_stats(series):
    """Compute custom statistics for a column."""
    return pd.Series({
        "mean":   series.mean(),
        "std":    series.std(),
        "cv":     series.std() / series.mean() if series.mean() != 0 else np.nan,
        "iqr":    series.quantile(0.75) - series.quantile(0.25),
        "range":  series.max() - series.min()
    })

col_stats = numeric_df.apply(column_stats, axis=0)
print("Custom column statistics (axis=0):")
print(col_stats.round(4))

# ------------------------------------------------------------------------------
# 5.2 Row-wise apply (axis=1)
# ------------------------------------------------------------------------------
print("\n--- 5.2 Row-wise apply (axis=1) ---")

explanation = """
Row-wise apply is legitimate when logic genuinely involves multiple columns.
Example: Assign a tier based on a combination of columns.
"""
print(explanation)

def assign_tier(row):
    """
    Assign passenger tier based on multiple criteria.
    Needs pclass AND fare, so row-wise apply is justified.
    """
    if pd.isna(row.get("pclass")) or pd.isna(row.get("fare")):
        return "Unknown"
    if row["pclass"] == 1 and row["fare"] > 100:
        return "Platinum"
    if row["pclass"] == 1:
        return "Gold"
    if row["pclass"] == 2:
        return "Silver"
    return "Standard"


if dataset_name == "titanic":
    df["tier"] = df.apply(assign_tier, axis=1)
    print("Passenger tier (based on class + fare):")
    print(df[["pclass", "fare", "tier"]].head(10))
    print("\nTier distribution:")
    print(df["tier"].value_counts())

else:
    # Multi-column score aggregation
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) >= 2:
        col_a = numeric_cols[0]
        col_b = numeric_cols[1]

        def classify_combined(row):
            """Classify based on two scores."""
            a = row.get(col_a, np.nan)
            b = row.get(col_b, np.nan)
            if pd.isna(a) or pd.isna(b):
                return "Unknown"
            avg = (a + b) / 2
            if avg >= 75:
                return "High"
            if avg >= 50:
                return "Mid"
            return "Low"

        df["combined_tier"] = df.apply(classify_combined, axis=1)
        print(f"Combined tier from {col_a} and {col_b}:")
        print(df[[col_a, col_b, "combined_tier"]].head(8))

# ------------------------------------------------------------------------------
# 5.3 apply() Returning a Series (Multiple Output Columns)
# ------------------------------------------------------------------------------
print("\n--- 5.3 apply() Returning Multiple Columns ---")

if dataset_name == "titanic":
    def parse_name_parts(name):
        """
        Parse Titanic name string into components.
        Format: 'Lastname, Title. Firstname Middlename'
        Returns Series so apply() expands it to multiple columns.
        """
        if pd.isna(name):
            return pd.Series({"last_name": None, "title": None, "first_name": None})
        name = str(name)
        try:
            last, rest = name.split(",", 1)
            title_match = re.search(r"\s*([A-Za-z]+)\.\s*", rest)
            title = title_match.group(1).strip() if title_match else ""
            first = re.sub(r"\s*[A-Za-z]+\.\s*", "", rest).strip()
        except Exception:
            last, title, first = name, "", ""
        return pd.Series({
            "last_name":  last.strip(),
            "title":      title,
            "first_name": first
        })

    # apply() expanding a returned Series into multiple new columns
    name_parts = df["name"].apply(parse_name_parts)
    df = pd.concat([df, name_parts], axis=1)

    print("Name parsed into components:")
    print(df[["name", "last_name", "title", "first_name"]].head(8))

else:
    # Return multiple metrics per row
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) >= 3:
        def row_metrics(row):
            """Return multiple per-row metrics."""
            vals = row[numeric_cols[:3]].dropna()
            return pd.Series({
                "row_mean":  vals.mean(),
                "row_std":   vals.std(),
                "row_max":   vals.max(),
                "row_range": vals.max() - vals.min()
            })

        row_stats = df.apply(row_metrics, axis=1)
        df = pd.concat([df, row_stats], axis=1)
        print("Per-row metrics returned from apply:")
        print(df[["row_mean", "row_std", "row_max", "row_range"]].head(8).round(3))


# ==============================================================================
# SECTION 6: Series.map()
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 6: Series.map()")
print("=" * 70)

explanation = """
Series.map() applies a function or lookup to each element.

Three usage forms:
  1. Function:    s.map(func)              Apply a callable to each element
  2. Dictionary:  s.map({"A": 1, "B": 2}) Lookup replacement (NA for missing)
  3. Series:      s.map(other_series)      Align and map by index

Key difference from apply():
  - map() is element-level ONLY (no axis parameter)
  - map() passes NaN through by default (does not call func on NaN)
  - map() is typically faster than apply() for element operations

Most common use: replace/encode categorical values using a dictionary.
"""
print(explanation)

# ------------------------------------------------------------------------------
# 6.1 map() with a Dictionary (Most Common Pattern)
# ------------------------------------------------------------------------------
print("\n--- 6.1 map() with a Dictionary ---")

if dataset_name == "titanic":
    # Encode embarkation port
    embarked_map = {
        "S": "Southampton",
        "C": "Cherbourg",
        "Q": "Queenstown"
    }
    df["embarkation_city"] = df["embarked"].map(embarked_map)
    print("Embarkation port expanded via map():")
    print(df[["embarked", "embarkation_city"]].drop_duplicates().dropna())

    # Encode sex as binary
    sex_map = {"male": 0, "female": 1}
    df["sex_encoded"] = df["sex"].map(sex_map)
    print("\nSex encoded as binary:")
    print(df[["sex", "sex_encoded"]].drop_duplicates())

    # Notice: unmapped values become NaN
    test_series = pd.Series(["male", "female", "unknown", None])
    test_result = test_series.map(sex_map)
    print("\nmap() with unmapped value 'unknown':")
    print(pd.DataFrame({"input": test_series, "output": test_result}))
    print("Note: 'unknown' and None both become NaN in output")

else:
    str_cols = df.select_dtypes(include=["object"]).columns
    if len(str_cols) > 0:
        col = str_cols[0]
        unique_vals = df[col].dropna().unique()
        # Build encoding map from unique values
        encoding_map = {val: idx for idx, val in enumerate(sorted(unique_vals))}
        df[f"{col}_encoded"] = df[col].map(encoding_map)
        print(f"Encoded {col} using map():")
        print(df[[col, f"{col}_encoded"]].drop_duplicates().head(10))

# ------------------------------------------------------------------------------
# 6.2 map() with a Function
# ------------------------------------------------------------------------------
print("\n--- 6.2 map() with a Function ---")

if dataset_name == "titanic":
    def fare_tier(fare):
        """Classify fare into tier."""
        if pd.isna(fare):
            return "Unknown"
        if fare > 100:
            return "High"
        if fare > 30:
            return "Medium"
        return "Low"

    df["fare_tier"] = df["fare"].map(fare_tier)

    print("Fare tier via map(func):")
    print(df[["fare", "fare_tier"]].head(10))
    print("\nTier distribution:")
    print(df["fare_tier"].value_counts())

else:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) > 0:
        col = numeric_cols[0]
        df[f"{col}_grade"] = df[col].map(
            lambda x: "Pass" if pd.notna(x) and x >= 50 else "Fail"
        )
        print(f"Graded {col}:")
        print(df[[col, f"{col}_grade"]].head(8))

# ------------------------------------------------------------------------------
# 6.3 map() with a Series (Index-Aligned Lookup)
# ------------------------------------------------------------------------------
print("\n--- 6.3 map() with a Series (Lookup Table) ---")

if dataset_name == "titanic":
    # Build a survival rate per class Series
    survival_rate_by_class = df.groupby("pclass")["survived"].mean()
    print("Survival rate by class (lookup Series):")
    print(survival_rate_by_class.round(3))

    # Map this lookup onto the pclass column
    df["class_survival_rate"] = df["pclass"].map(survival_rate_by_class)
    print("\nSurvival rate added per row from lookup:")
    print(df[["pclass", "survived", "class_survival_rate"]].head(10).round(3))

else:
    str_cols = df.select_dtypes(include=["object"]).columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(str_cols) > 0 and len(numeric_cols) > 0:
        group_col = str_cols[0]
        value_col = numeric_cols[0]
        group_means = df.groupby(group_col)[value_col].mean()
        df["group_mean"] = df[group_col].map(group_means)
        print(f"Group mean of {value_col} by {group_col} mapped onto rows:")
        print(df[[group_col, value_col, "group_mean"]].head(8).round(3))


# ==============================================================================
# SECTION 7: DATAFRAME.map() (ELEMENT-WISE ON ENTIRE DATAFRAME)
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 7: DATAFRAME.map()")
print("=" * 70)

explanation = """
DataFrame.map(func) applies func to every single element in the DataFrame.
Equivalent to applymap() in older Pandas versions (deprecated in Pandas 2.1).

Use when:
- You want to transform ALL values regardless of column
- Formatting output (rounding, adding units, masking)
- Cleaning values consistently across all columns

Note: This is different from DataFrame.apply() which operates on
full columns or rows.
"""
print(explanation)

# ------------------------------------------------------------------------------
# 7.1 Rounding All Numeric Values
# ------------------------------------------------------------------------------
print("\n--- 7.1 Rounding All Values ---")

sample_numeric = df.select_dtypes(include=[np.number]).head(5)
print("Original values (first 5 rows, numeric only):")
print(sample_numeric)

# Round all values to 2 decimal places
rounded = sample_numeric.map(lambda x: round(float(x), 2) if pd.notna(x) else x)
print("\nAfter DataFrame.map(round to 2dp):")
print(rounded)

# ------------------------------------------------------------------------------
# 7.2 Masking Sensitive Values
# ------------------------------------------------------------------------------
print("\n--- 7.2 Masking Sensitive Values ---")

# Real world: mask values above a threshold for privacy
if dataset_name == "titanic":
    fare_only = df[["fare"]].head(10).copy()
    print("Original fares:")
    print(fare_only.T)

    masked = fare_only.map(
        lambda x: "***" if pd.notna(x) and x > 50 else x
    )
    print("\nFares above 50 masked:")
    print(masked.T)

# ------------------------------------------------------------------------------
# 7.3 Type Check Across All Cells
# ------------------------------------------------------------------------------
print("\n--- 7.3 Check Type of Every Cell ---")

mixed_df = pd.DataFrame({
    "col_a": [1, "two", 3.0, None],
    "col_b": ["x", 2, "y", 4]
})

print("Mixed DataFrame:")
print(mixed_df)

type_check = mixed_df.map(lambda x: type(x).__name__)
print("\nType of each cell:")
print(type_check)


# ==============================================================================
# SECTION 8: TRANSFORM FOR GROUP-AWARE OPERATIONS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 8: TRANSFORM FOR GROUP-AWARE OPERATIONS")
print("=" * 70)

explanation = """
transform() applies a function within each group and returns a result
with the SAME shape as the original DataFrame (preserves index).

Key differences:
  agg()       returns ONE row per group (reduces shape)
  transform() returns ONE value per ORIGINAL row (same shape)

Use transform() for:
  - Adding group statistics as a new column
  - Group-based normalization (z-score within group)
  - Filling missing values with group statistics
  - Creating deviation from group mean

This is one of the most powerful and underused features in Pandas.
"""
print(explanation)

# ------------------------------------------------------------------------------
# 8.1 Group Statistics Back to Rows
# ------------------------------------------------------------------------------
print("\n--- 8.1 Group Statistics Back to Every Row ---")

if dataset_name == "titanic":
    # Add class-level survival rate to each row
    df["class_avg_fare"] = df.groupby("pclass")["fare"].transform("mean")
    df["fare_vs_class_avg"] = df["fare"] - df["class_avg_fare"]
    df["fare_above_class_avg"] = df["fare_vs_class_avg"] > 0

    print("Class average fare added to each row:")
    print(df[["pclass", "fare", "class_avg_fare", "fare_vs_class_avg"]].head(12).round(2))

    # Verify shape preserved
    assert len(df) == len(df_raw), "transform changed row count"
    print(f"\nShape verified: {len(df)} rows preserved")

else:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    str_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if len(str_cols) > 0 and len(numeric_cols) > 0:
        group_col = str_cols[0]
        value_col = numeric_cols[0]
        df["group_mean_value"] = df.groupby(group_col)[value_col].transform("mean")
        df["deviation_from_group"] = df[value_col] - df["group_mean_value"]
        print(f"Group mean of {value_col} by {group_col} on every row:")
        print(df[[group_col, value_col, "group_mean_value", "deviation_from_group"]].head(10).round(3))

# ------------------------------------------------------------------------------
# 8.2 Z-Score Normalization Within Groups
# ------------------------------------------------------------------------------
print("\n--- 8.2 Z-Score Within Group (Group Normalization) ---")

def zscore_normalize(series):
    """Compute z-score for a series."""
    mean = series.mean()
    std  = series.std()
    if std == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - mean) / std


if dataset_name == "titanic":
    df["fare_zscore_within_class"] = df.groupby("pclass")["fare"].transform(
        zscore_normalize
    )

    print("Z-score of fare within each passenger class:")
    verification = df.groupby("pclass")["fare_zscore_within_class"].agg(
        ["mean", "std", "min", "max"]
    ).round(4)
    print(verification)
    print("\nMean ~0 and std ~1 within each group: normalization confirmed")

else:
    str_cols = df.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(str_cols) > 0 and len(numeric_cols) > 0:
        group_col = str_cols[0]
        value_col = numeric_cols[0]
        df[f"{value_col}_zscore"] = df.groupby(group_col)[value_col].transform(
            zscore_normalize
        )
        print(f"Z-score of {value_col} within {group_col}:")
        print(df.groupby(group_col)[f"{value_col}_zscore"].agg(["mean", "std"]).round(4))

# ------------------------------------------------------------------------------
# 8.3 Fill Missing Values with Group Statistics
# ------------------------------------------------------------------------------
print("\n--- 8.3 Fill Missing Values with Group Statistics ---")

if dataset_name == "titanic":
    print(f"Missing Age before: {df['age'].isnull().sum()}")

    # Fill missing age with median of same class + sex group
    df["age_filled"] = df.groupby(["pclass", "sex"])["age"].transform(
        lambda x: x.fillna(x.median())
    )
    # Second pass: fill any remaining with overall median
    df["age_filled"] = df["age_filled"].fillna(df["age_filled"].median())

    print(f"Missing Age after group fill: {df['age_filled'].isnull().sum()}")

    print("\nGroup medians used:")
    print(df.groupby(["pclass", "sex"])["age"].median().round(1))

else:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    str_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if len(str_cols) > 0 and len(numeric_cols) > 0:
        group_col = str_cols[0]
        value_col = numeric_cols[0]
        missing_before = df[value_col].isnull().sum()
        df[f"{value_col}_filled"] = df.groupby(group_col)[value_col].transform(
            lambda x: x.fillna(x.median())
        )
        missing_after = df[f"{value_col}_filled"].isnull().sum()
        print(f"Missing {value_col} before: {missing_before}")
        print(f"Missing {value_col} after:  {missing_after}")


# ==============================================================================
# SECTION 9: COMPARISON OF ALL METHODS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 9: COMPARISON OF ALL METHODS")
print("=" * 70)

n_bench = 50_000
compare_df = pd.DataFrame({
    "score": np.random.uniform(0, 100, n_bench),
    "group": np.random.choice(["A", "B", "C"], n_bench)
})

# Task: Multiply score by 2
print(f"Task: Multiply score by 2 ({n_bench:,} rows)")
print("-" * 50)

# Method 1: Vectorized (BEST)
start = time.perf_counter()
r1 = compare_df["score"] * 2
t1 = time.perf_counter() - start
print(f"1. Vectorized (col * 2):         {t1*1000:>7.2f} ms")

# Method 2: numpy ufunc
start = time.perf_counter()
r2 = np.multiply(compare_df["score"].values, 2)
t2 = time.perf_counter() - start
print(f"2. np.multiply (ufunc):          {t2*1000:>7.2f} ms")

# Method 3: Series.map with function
start = time.perf_counter()
r3 = compare_df["score"].map(lambda x: x * 2)
t3 = time.perf_counter() - start
print(f"3. Series.map (lambda):          {t3*1000:>7.2f} ms")

# Method 4: Series.apply with function
start = time.perf_counter()
r4 = compare_df["score"].apply(lambda x: x * 2)
t4 = time.perf_counter() - start
print(f"4. Series.apply (lambda):        {t4*1000:>7.2f} ms")

# Method 5: DataFrame.apply row-wise (WORST)
start = time.perf_counter()
r5 = compare_df.apply(lambda row: row["score"] * 2, axis=1)
t5 = time.perf_counter() - start
print(f"5. DataFrame.apply row-wise:     {t5*1000:>7.2f} ms")

# Validate all produce same result
assert np.allclose(r1, r3), "Methods 1 and 3 differ"
assert np.allclose(r1, r4), "Methods 1 and 4 differ"
assert np.allclose(r1, r5), "Methods 1 and 5 differ"

print("\nAll methods produce identical results.")
print("\nSpeedup of vectorized over DataFrame.apply row-wise:")
print(f"  {t5 / t1:.0f}x faster")


# ==============================================================================
# SECTION 10: PRACTICAL PATTERNS FOR PRODUCTION CODE
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 10: PRACTICAL PATTERNS FOR PRODUCTION CODE")
print("=" * 70)

# ------------------------------------------------------------------------------
# 10.1 Prefer Vectorized String Operations Over apply
# ------------------------------------------------------------------------------
print("\n--- 10.1 String Operations: .str vs apply ---")

if dataset_name == "titanic":
    # WRONG: using apply for string operations
    start = time.perf_counter()
    result_apply = df["sex"].apply(lambda x: str(x).upper() if pd.notna(x) else x)
    t_apply = time.perf_counter() - start

    # RIGHT: use .str accessor
    start = time.perf_counter()
    result_str = df["sex"].str.upper()
    t_str = time.perf_counter() - start

    assert result_apply.dropna().tolist() == result_str.dropna().tolist(), "Mismatch"

    print(f"apply() for str.upper():   {t_apply*1000:.3f} ms")
    print(f".str.upper() (vectorized): {t_str*1000:.3f} ms")
    print(f"Speedup: {t_apply/t_str:.1f}x faster with .str accessor")

# ------------------------------------------------------------------------------
# 10.2 Prefer np.where Over apply for Conditionals
# ------------------------------------------------------------------------------
print("\n--- 10.2 Conditionals: np.where vs apply ---")

if dataset_name == "titanic":
    # WRONG: apply for simple condition
    start = time.perf_counter()
    df["adult_apply"] = df["age"].apply(
        lambda x: "Adult" if pd.notna(x) and x >= 18 else "Minor"
    )
    t_apply = time.perf_counter() - start

    # RIGHT: np.where (vectorized)
    start = time.perf_counter()
    df["adult_vec"] = np.where(df["age"] >= 18, "Adult", "Minor")
    t_vec = time.perf_counter() - start

    print(f"apply() for condition:     {t_apply*1000:.3f} ms")
    print(f"np.where() (vectorized):   {t_vec*1000:.3f} ms")
    print(f"Speedup: {t_apply/t_vec:.1f}x faster with np.where")

# ------------------------------------------------------------------------------
# 10.3 Prefer pd.cut / pd.qcut Over apply for Binning
# ------------------------------------------------------------------------------
print("\n--- 10.3 Binning: pd.cut vs apply ---")

if dataset_name == "titanic":
    fare_data = df["fare"].dropna()

    # WRONG: apply for binning
    start = time.perf_counter()
    result_apply = df["fare"].apply(
        lambda x: "Low" if pd.notna(x) and x < 30 else
                  ("Medium" if x < 100 else "High") if pd.notna(x) else "Unknown"
    )
    t_apply = time.perf_counter() - start

    # RIGHT: pd.cut
    start = time.perf_counter()
    result_cut = pd.cut(
        df["fare"],
        bins=[-1, 30, 100, float("inf")],
        labels=["Low", "Medium", "High"]
    )
    t_cut = time.perf_counter() - start

    print(f"apply() for binning:   {t_apply*1000:.3f} ms")
    print(f"pd.cut() (vectorized): {t_cut*1000:.3f} ms")
    print(f"Speedup: {t_apply/t_cut:.1f}x faster with pd.cut")
    print(f"\npd.cut result:")
    print(result_cut.value_counts())

# ------------------------------------------------------------------------------
# 10.4 Prefer np.select for Multiple Conditions
# ------------------------------------------------------------------------------
print("\n--- 10.4 Multiple Conditions: np.select vs apply ---")

if dataset_name == "titanic":
    # WRONG: apply with multiple conditions
    def assign_label_apply(row):
        if pd.isna(row["pclass"]) or pd.isna(row["fare"]):
            return "Unknown"
        if row["pclass"] == 1 and row["fare"] > 100:
            return "VIP"
        if row["pclass"] == 1:
            return "First"
        if row["pclass"] == 2:
            return "Business"
        return "Economy"

    start = time.perf_counter()
    result_apply = df.apply(assign_label_apply, axis=1)
    t_apply = time.perf_counter() - start

    # RIGHT: np.select (vectorized)
    start = time.perf_counter()
    conditions = [
        (df["pclass"] == 1) & (df["fare"] > 100),
        df["pclass"] == 1,
        df["pclass"] == 2
    ]
    choices = ["VIP", "First", "Business"]
    result_select = np.select(conditions, choices, default="Economy")
    t_select = time.perf_counter() - start

    print(f"apply() row-wise:      {t_apply*1000:.3f} ms")
    print(f"np.select() (vectorized): {t_select*1000:.3f} ms")
    print(f"Speedup: {t_apply/t_select:.1f}x faster with np.select")

    # Show result
    df["label_vec"] = result_select
    print(f"\nLabel distribution:")
    print(df["label_vec"].value_counts())

# ------------------------------------------------------------------------------
# 10.5 Reusable Transformation Pipeline
# ------------------------------------------------------------------------------
print("\n--- 10.5 Reusable Transformation Pipeline ---")

def build_features(data):
    """
    Build a complete feature set from raw Titanic data.
    Demonstrates the correct tool for each transformation type.

    Parameters
    ----------
    data : pd.DataFrame

    Returns
    -------
    pd.DataFrame with engineered features
    """
    df_feat = data.copy()

    if dataset_name != "titanic":
        print("  (Feature pipeline designed for Titanic dataset)")
        return df_feat

    print("Building features...")

    # Vectorized: numeric transformations
    df_feat["fare_log"] = np.log1p(df_feat["fare"].fillna(0))
    df_feat["fare_per_person"] = df_feat["fare"] / (
        df_feat["sibsp"].fillna(0) + df_feat["parch"].fillna(0) + 1
    )
    print("  Vectorized numeric features: done")

    # np.where: binary condition
    df_feat["is_child"] = np.where(
        df_feat["age"].fillna(99) < 15, 1, 0
    )
    df_feat["is_alone"] = np.where(
        (df_feat["sibsp"].fillna(0) + df_feat["parch"].fillna(0)) == 0, 1, 0
    )
    print("  np.where binary features: done")

    # np.select: multi-condition feature
    family_size = df_feat["sibsp"].fillna(0) + df_feat["parch"].fillna(0) + 1
    df_feat["family_size"] = family_size
    family_conditions = [
        family_size == 1,
        (family_size >= 2) & (family_size <= 4),
        family_size >= 5
    ]
    df_feat["family_type"] = np.select(
        family_conditions,
        ["Alone", "Small", "Large"],
        default="Unknown"
    )
    print("  np.select multi-condition features: done")

    # map(): lookup encoding
    sex_map = {"male": 0, "female": 1}
    df_feat["sex_encoded"] = df_feat["sex"].map(sex_map)

    port_map = {"S": 0, "C": 1, "Q": 2}
    df_feat["port_encoded"] = df_feat["embarked"].map(port_map)
    print("  map() encoding features: done")

    # pd.cut: binning
    df_feat["age_bin"] = pd.cut(
        df_feat["age"].fillna(df_feat["age"].median()),
        bins=[0, 12, 18, 35, 60, 120],
        labels=["child", "teen", "young_adult", "adult", "senior"]
    )
    df_feat["fare_bin"] = pd.qcut(
        df_feat["fare"].fillna(df_feat["fare"].median()),
        q=4,
        labels=["Q1", "Q2", "Q3", "Q4"]
    )
    print("  pd.cut/qcut binning features: done")

    # transform(): group statistics
    df_feat["class_avg_fare"] = df_feat.groupby("pclass")["fare"].transform("mean")
    df_feat["fare_ratio_to_class"] = (
        df_feat["fare"] / df_feat["class_avg_fare"]
    ).fillna(1.0)
    print("  transform() group features: done")

    # apply(): only for genuinely complex multi-column logic
    def deck_from_cabin(cabin_val):
        """Extract deck letter from cabin code."""
        if pd.isna(cabin_val) or cabin_val == "Unknown":
            return "Unknown"
        match = re.match(r"([A-Za-z])", str(cabin_val))
        return match.group(1).upper() if match else "Unknown"

    df_feat["deck"] = df_feat["cabin"].apply(deck_from_cabin)
    print("  apply() complex parsing: done")

    return df_feat


features_df = build_features(df)

if dataset_name == "titanic":
    new_cols = [
        "fare_log", "fare_per_person", "is_child", "is_alone",
        "family_type", "sex_encoded", "port_encoded",
        "age_bin", "fare_bin", "deck"
    ]
    available = [c for c in new_cols if c in features_df.columns]
    print("\nEngineered features sample:")
    print(features_df[available].head(10))


# ==============================================================================
# SECTION 11: DECISION GUIDE
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 11: DECISION GUIDE - WHICH TOOL TO USE")
print("=" * 70)

guide = """
TASK                                      | CORRECT TOOL
------------------------------------------|-----------------------------------
Multiply column by scalar                 | df["col"] * scalar
Add two columns                           | df["a"] + df["b"]
Apply log / sqrt / trig                   | np.log(df["col"])
Uppercase string column                   | df["col"].str.upper()
Check if string contains pattern          | df["col"].str.contains("pat")
Simple condition (A or B)                 | np.where(condition, "A", "B")
Multiple conditions with different values | np.select(conditions, choices)
Bin numeric into ranges                   | pd.cut(df["col"], bins)
Bin into equal-frequency bins             | pd.qcut(df["col"], q)
Replace values using lookup               | df["col"].map(dictionary)
Add group mean to each row                | groupby().transform("mean")
Fill NaN with group statistic             | groupby().transform(lambda x: x.fillna(x.median()))
Z-score within group                      | groupby().transform(zscore_func)
Compute per-row statistics (multi-col)    | df.apply(func, axis=1)
Parse complex string per element          | df["col"].apply(parse_func)
Summary stats per column                  | df.apply(stats_func, axis=0)
Format all values in DataFrame            | df.map(format_func)
"""
print(guide)


# ==============================================================================
# SECTION 12: COMMON PITFALLS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 12: COMMON PITFALLS")
print("=" * 70)

pitfalls = """
Pitfall 1: Using apply() where vectorized arithmetic works
   WRONG: df["col"].apply(lambda x: x * 2)
   RIGHT: df["col"] * 2

Pitfall 2: Using apply() where .str accessor works
   WRONG: df["col"].apply(lambda x: x.lower())
   RIGHT: df["col"].str.lower()

Pitfall 3: Using apply(axis=1) where np.where works
   WRONG: df.apply(lambda r: "A" if r["x"] > 5 else "B", axis=1)
   RIGHT: np.where(df["x"] > 5, "A", "B")

Pitfall 4: Expecting map() to handle NaN like apply()
   - map() returns NaN for any value not in dictionary
   - Use fillna() after map() to handle unmapped values

Pitfall 5: Using apply() to compute group statistics
   WRONG: df.apply(lambda r: group_mean[r["group"]], axis=1)
   RIGHT: df.groupby("group")["value"].transform("mean")

Pitfall 6: Modifying original DataFrame inside apply function
   - Functions passed to apply() should be pure (no side effects)
   - Always return a new value, never modify df directly inside apply

Pitfall 7: Using applymap() in Pandas 2.x
   - applymap() is deprecated in Pandas 2.1+
   - Use DataFrame.map() instead

Pitfall 8: Returning inconsistent types from apply()
   - If func returns Series sometimes and scalar other times,
     results will be unpredictable
   - Always ensure all code paths return the same type

Pitfall 9: Using Python loops instead of any of the above
   WRONG: for i, row in df.iterrows(): df.at[i, "col"] = row["x"] * 2
   RIGHT: df["col"] = df["x"] * 2
"""
print(pitfalls)


# ==============================================================================
# SECTION 13: SUMMARY TABLE
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 13: SUMMARY TABLE")
print("=" * 70)

summary = """
Method                       | Level       | Axis    | Returns same shape?
-----------------------------|-------------|---------|----------------------
df["col"] * 2                | Vectorized  | N/A     | Yes (same column)
np.where(cond, a, b)         | Vectorized  | N/A     | Yes (same column)
np.select(conds, choices)    | Vectorized  | N/A     | Yes (same column)
pd.cut() / pd.qcut()         | Vectorized  | N/A     | Yes (same column)
Series.map(dict)             | Element     | N/A     | Yes (same column)
Series.map(func)             | Element     | N/A     | Yes (same column)
Series.apply(func)           | Element     | N/A     | Yes (same column)
DataFrame.map(func)          | Element     | Both    | Yes (same DataFrame)
DataFrame.apply(func, axis=0)| Column      | Columns | No (one row per col)
DataFrame.apply(func, axis=1)| Row         | Rows    | Depends on func output
groupby.transform(func)      | Group       | Rows    | Yes (same shape)
groupby.agg(func)            | Group       | Groups  | No (one row per group)
"""
print(summary)


# ==============================================================================
# SECTION 14: PRACTICE EXERCISES
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 14: PRACTICE EXERCISES")
print("=" * 70)

if dataset_name == "titanic":
    print("Exercise 1: Encode survival as 'Survived'/'Not Survived' using map")
    survived_map = {0: "Not Survived", 1: "Survived"}
    df["survived_label"] = df["survived"].map(survived_map)
    print(df[["survived", "survived_label"]].value_counts())

    print("\nExercise 2: Create 'age_group' using pd.cut")
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 12, 18, 60, 120],
        labels=["Child", "Teen", "Adult", "Senior"]
    )
    print(df[["age", "age_group"]].dropna().head(10))
    print("\nDistribution:")
    print(df["age_group"].value_counts())

    print("\nExercise 3: Add median fare of each class using transform")
    df["median_fare_by_class"] = df.groupby("pclass")["fare"].transform("median")
    print(df[["pclass", "fare", "median_fare_by_class"]].head(10).round(2))

    print("\nExercise 4: Use np.select for priority scoring")
    conditions = [
        (df["pclass"] == 1) & (df["sex"] == "female"),
        (df["pclass"] == 1) & (df["sex"] == "male"),
        (df["pclass"] == 2) & (df["sex"] == "female"),
        (df["pclass"] == 2) & (df["sex"] == "male"),
    ]
    choices = [1, 2, 3, 4]
    df["priority_score"] = np.select(conditions, choices, default=5)
    print(df[["pclass", "sex", "priority_score"]].head(10))
    print("\nPriority distribution:")
    print(df["priority_score"].value_counts().sort_index())

else:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    str_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if len(numeric_cols) >= 1:
        col = numeric_cols[0]
        print(f"Exercise 1: Apply log transform to {col}")
        df[f"{col}_log"] = np.log1p(df[col].clip(lower=0))
        print(df[[col, f"{col}_log"]].head(5).round(4))

        print(f"\nExercise 2: Bin {col} into quartiles with pd.qcut")
        df[f"{col}_quartile"] = pd.qcut(
            df[col].dropna(),
            q=4,
            labels=["Q1", "Q2", "Q3", "Q4"]
        )
        print(df[[col, f"{col}_quartile"]].dropna().head(8))


# ==============================================================================
# SECTION 15: KEY TAKEAWAYS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 15: KEY TAKEAWAYS")
print("=" * 70)

takeaways = """
1.  Vectorized operations (arithmetic, np.where, np.select, pd.cut)
    are ALWAYS faster than apply() for the same task

2.  Use .str accessor for all string operations instead of apply(lambda)

3.  Use Series.map(dict) for lookup/replacement of categorical values

4.  Use Series.apply(func) when the logic is genuinely complex and
    cannot be expressed with vectorized tools

5.  Use DataFrame.apply(func, axis=1) only when the function NEEDS
    values from multiple columns simultaneously

6.  Use DataFrame.apply(func, axis=0) for column-level statistics

7.  Use DataFrame.map(func) to apply a function to every cell
    (replaces deprecated applymap() in Pandas 2.x)

8.  Use groupby().transform() to add group statistics back to
    every original row while preserving shape

9.  Never use Python loops (iterrows, itertuples) for
    row-by-row computation on large DataFrames

10. Always validate that apply() results are correct on a sample
    before running on the full dataset
"""
print(takeaways)


# ==============================================================================
# NEXT LESSON PREVIEW
# ==============================================================================
print("\n" + "=" * 70)
print("NEXT LESSON PREVIEW")
print("=" * 70)

preview = """
Lesson 28: Handling Large Datasets Efficiently

You will learn:
- Memory profiling of DataFrames
- Dtype optimization for memory reduction
- Chunked reading with pd.read_csv(chunksize=)
- Selecting only needed columns on load
- Using categorical dtype for string columns
- When to use alternatives (NumPy, Polars, DuckDB)
- Efficient filtering before computation
- Building memory-efficient pipelines
"""
print(preview)

print("\n" + "=" * 70)
print("END OF LESSON 27")
print("=" * 70)
print("\nYou now know exactly which tool to use for every transformation task.")
print("Vectorized operations first, apply() only when genuinely necessary.")
