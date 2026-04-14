"""
LESSON 31: PIPELINE BUILDING WITH METHOD CHAINING (PRODUCTION PATTERNS)
================================================================================

What You Will Learn:
- Method chaining with assign(), pipe(), query(), and rename()
- Building readable transformation pipelines
- Creating reusable, composable pipeline functions
- Avoiding unnecessary intermediate DataFrames
- Adding checkpoint logs inside method chains
- Parameterizing pipelines with configuration objects
- Testing pipelines for determinism and correctness
- Real-world end-to-end pipeline examples

Real World Usage:
- Building repeatable data preparation workflows
- Creating configurable ETL steps shared across multiple projects
- Ensuring consistent feature engineering across training and inference
- Producing auditable, step-by-step data transformations
- Onboarding new engineers with readable, self-documenting code

Dataset Used:
Superstore Sales Dataset (public, no login required)
URL: https://raw.githubusercontent.com/dsrscientist/dataset1/master/superstore.csv
Fallback: Titanic dataset
https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv

================================================================================
"""

import numpy as np
import pandas as pd
import re
import time
import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any

print("=" * 70)
print("LESSON 31: PIPELINE BUILDING WITH METHOD CHAINING")
print("=" * 70)


# ==============================================================================
# SECTION 1: LOAD DATASET
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 1: LOAD DATASET")
print("=" * 70)

primary_url  = "https://raw.githubusercontent.com/dsrscientist/dataset1/master/superstore.csv"
fallback_url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"


def to_snake(name):
    name = str(name).strip()
    name = re.sub(r"[\s\-\.]+", "_", name)
    name = re.sub(r"[^\w]", "", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_").lower()


def load_dataset():
    try:
        print("Loading Superstore dataset from:")
        print(primary_url)
        df = pd.read_csv(primary_url, encoding="latin-1")
        df.columns = [to_snake(c) for c in df.columns]
        print("Loaded successfully.")
        return df, "superstore"
    except Exception as e:
        print(f"Primary load failed: {e}")
        print("Falling back to Titanic dataset from:")
        print(fallback_url)
        df = pd.read_csv(fallback_url)
        df.columns = [to_snake(c) for c in df.columns]
        print("Fallback loaded successfully.")
        return df, "titanic"


df_raw, dataset_name = load_dataset()

print(f"\nDataset: {dataset_name}")
print(f"Shape: {df_raw.shape}")
print(f"Columns: {list(df_raw.columns)}")
print("\nFirst 5 rows:")
print(df_raw.head())
print("\nDtypes:")
print(df_raw.dtypes)
print("\nMissing values:")
print(df_raw.isnull().sum())


# ==============================================================================
# SECTION 2: WHY METHOD CHAINING MATTERS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 2: WHY METHOD CHAINING MATTERS")
print("=" * 70)

explanation = """
Traditional imperative style creates many intermediate variables:

  df1 = df.dropna(subset=["col_a"])
  df2 = df1.rename(columns={"col_a": "a"})
  df3 = df2[df2["a"] > 0]
  df4 = df3.assign(b=df3["a"] * 2)
  result = df4.sort_values("b")

Problems:
  - Many named intermediates that clutter the namespace
  - Hard to see the transformation sequence at a glance
  - Temptation to inspect and mutate intermediates, causing state bugs

Method chaining applies transformations in a clean sequence:

  result = (
      df
      .dropna(subset=["col_a"])
      .rename(columns={"col_a": "a"})
      .query("a > 0")
      .assign(b=lambda x: x["a"] * 2)
      .sort_values("b")
  )

Benefits:
  - Reads as a list of transformations top-to-bottom
  - No intermediate state to worry about
  - Easy to add, remove, or reorder steps
  - pipe() makes ANY function chainable
"""
print(explanation)


# ==============================================================================
# SECTION 3: THE CORE CHAINING TOOLS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 3: THE CORE CHAINING TOOLS")
print("=" * 70)

# ------------------------------------------------------------------------------
# 3.1 assign() - Add or Overwrite Columns
# ------------------------------------------------------------------------------
print("\n--- 3.1 assign() ---")

explanation = """
assign(col=value_or_callable) adds or overwrites columns.

Key rule: use lambda x: ... to reference columns computed earlier
          in the SAME assign() call (Pandas evaluates left to right).
"""
print(explanation)

df = df_raw.copy()

if dataset_name == "superstore":
    sample = (
        df[["sales", "profit", "quantity", "discount"]]
        .head(8)
        .assign(
            revenue_net   = lambda x: x["sales"] * (1 - x["discount"]),
            profit_margin = lambda x: np.where(
                x["sales"] > 0,
                x["profit"] / x["sales"],
                np.nan
            ),
            is_high_value = lambda x: x["sales"] > x["sales"].median()
        )
    )
else:
    sample = (
        df[["age", "fare", "survived", "pclass"]]
        .head(8)
        .assign(
            fare_log      = lambda x: np.log1p(x["fare"].fillna(0)),
            is_adult      = lambda x: (x["age"].fillna(0) >= 18).astype(int),
            is_high_fare  = lambda x: x["fare"] > x["fare"].median()
        )
    )

print("assign() result:")
print(sample.round(4))

# ------------------------------------------------------------------------------
# 3.2 pipe() - Make Any Function Chainable
# ------------------------------------------------------------------------------
print("\n--- 3.2 pipe() ---")

explanation = """
pipe(func, *args, **kwargs) passes the DataFrame as the first argument
to func, making ANY function chainable.

This lets you include custom functions inside a chain without
breaking the fluent style.
"""
print(explanation)


def log_shape(df_in, label=""):
    """Log shape without transforming. Returns df unchanged."""
    print(f"  [pipe log] {label}: shape={df_in.shape}")
    return df_in


def drop_constant_cols(df_in):
    """Drop columns where every non-null value is identical."""
    non_const = [c for c in df_in.columns if df_in[c].nunique() > 1]
    dropped   = [c for c in df_in.columns if c not in non_const]
    if dropped:
        print(f"  [pipe] dropped constant columns: {dropped}")
    return df_in[non_const]


def clip_numeric(df_in, lower_quantile=0.01, upper_quantile=0.99):
    """Clip numeric columns to specified quantile range."""
    result = df_in.copy()
    numeric_cols = result.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        lo = result[col].quantile(lower_quantile)
        hi = result[col].quantile(upper_quantile)
        result[col] = result[col].clip(lower=lo, upper=hi)
    return result


def standardize_strings(df_in):
    """Strip whitespace from all string columns."""
    result = df_in.copy()
    for col in result.select_dtypes(include=["str", "category"]).columns:
        result[col] = result[col].astype(str).str.strip()
    return result


# Chain using pipe()
piped = (
    df_raw
    .pipe(log_shape, label="raw")
    .pipe(standardize_strings)
    .pipe(log_shape, label="after string clean")
    .pipe(drop_constant_cols)
    .pipe(log_shape, label="after constant drop")
    .pipe(clip_numeric, lower_quantile=0.01, upper_quantile=0.99)
    .pipe(log_shape, label="after clip")
)

print(f"\nFinal piped shape: {piped.shape}")

# ------------------------------------------------------------------------------
# 3.3 query() Inside a Chain
# ------------------------------------------------------------------------------
print("\n--- 3.3 query() Inside a Chain ---")

explanation = """
query(expr) filters rows using a string expression.
Supports referencing Python variables with @ prefix.
Produces cleaner chains than boolean masks.
"""
print(explanation)

if dataset_name == "superstore":
    sales_threshold = df_raw["sales"].quantile(0.75)

    filtered = (
        df_raw
        .pipe(standardize_strings)
        .query("sales > @sales_threshold")
        .query("discount < 0.5")
    )

    print(f"Rows after query filters: {len(filtered):,}")
    print(f"  sales > {sales_threshold:.2f} and discount < 0.5")

else:
    fare_threshold = df_raw["fare"].quantile(0.75)

    filtered = (
        df_raw
        .pipe(standardize_strings)
        .query("fare > @fare_threshold")
        .query("survived == 1")
    )

    print(f"Rows after query filters: {len(filtered):,}")
    print(f"  fare > {fare_threshold:.2f} and survived == 1")

# ------------------------------------------------------------------------------
# 3.4 rename() and rename_axis() in Chains
# ------------------------------------------------------------------------------
print("\n--- 3.4 rename() in Chains ---")

if dataset_name == "superstore":
    renamed = (
        df_raw[["order_id", "sales", "profit", "region"]]
        .rename(columns={
            "order_id": "id",
            "sales":    "revenue",
            "profit":   "net_profit",
            "region":   "zone"
        })
        .head(5)
    )
else:
    renamed = (
        df_raw[["passengerid", "survived", "pclass", "fare"]]
        .rename(columns={
            "passengerid": "id",
            "survived":    "outcome",
            "pclass":      "ticket_class",
            "fare":        "ticket_price"
        })
        .head(5)
    )

print("Renamed columns:")
print(renamed)


# ==============================================================================
# SECTION 4: BUILDING REUSABLE PIPELINE FUNCTIONS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 4: BUILDING REUSABLE PIPELINE FUNCTIONS")
print("=" * 70)

explanation = """
A well-designed pipeline function:
  - Takes a DataFrame as first argument
  - Returns a transformed DataFrame
  - Has explicit parameters for all configurable behaviour
  - Validates required columns at the start
  - Is usable with .pipe() inside a chain
  - Has a clear docstring

This makes functions composable, testable, and reusable across projects.
"""
print(explanation)


def require_columns(df_in, cols, step_name=""):
    """Raise KeyError if any expected column is absent."""
    missing = [c for c in cols if c not in df_in.columns]
    if missing:
        raise KeyError(f"[{step_name}] Missing columns: {missing}")


def fill_missing_with_group_stat(
    df_in,
    target_col,
    group_cols,
    stat="median",
    global_fallback=True
):
    """
    Fill NaN in target_col using per-group statistics.

    Parameters
    ----------
    df_in : pd.DataFrame
    target_col : str
        Column to fill
    group_cols : list of str
        Grouping columns
    stat : str
        Aggregation to use ('median', 'mean', 'mode')
    global_fallback : bool
        Fill remaining NaN with global stat after group fill

    Returns
    -------
    pd.DataFrame
    """
    require_columns(df_in, [target_col] + group_cols, "fill_missing_with_group_stat")

    result = df_in.copy()

    if stat == "mode":
        result[target_col] = result.groupby(group_cols)[target_col].transform(
            lambda x: x.fillna(x.mode().iloc[0] if len(x.mode()) > 0 else np.nan)
        )
    elif stat == "mean":
        result[target_col] = result.groupby(group_cols)[target_col].transform(
            lambda x: x.fillna(x.mean())
        )
    else:
        result[target_col] = result.groupby(group_cols)[target_col].transform(
            lambda x: x.fillna(x.median())
        )

    if global_fallback and result[target_col].isnull().any():
        global_stat = (
            result[target_col].median()
            if stat in ("median", "mean")
            else result[target_col].mode().iloc[0]
        )
        result[target_col] = result[target_col].fillna(global_stat)

    return result


def cap_outliers(df_in, col, method="iqr", factor=1.5):
    """
    Cap outliers in a numeric column.

    Parameters
    ----------
    df_in : pd.DataFrame
    col : str
    method : str
        'iqr' for IQR-based, 'zscore' for 3-sigma
    factor : float
        IQR multiplier or sigma multiplier

    Returns
    -------
    pd.DataFrame
    """
    require_columns(df_in, [col], "cap_outliers")

    result = df_in.copy()
    series = result[col].dropna()

    if method == "iqr":
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lo  = q1 - factor * iqr
        hi  = q3 + factor * iqr
    else:
        mean = series.mean()
        std  = series.std()
        lo   = mean - factor * std
        hi   = mean + factor * std

    result[col] = result[col].clip(lower=lo, upper=hi)
    return result


def encode_ordinal(df_in, col, order):
    """
    Encode an ordinal column as integers based on specified order.

    Parameters
    ----------
    df_in : pd.DataFrame
    col : str
    order : list
        Ordered list of values from lowest to highest

    Returns
    -------
    pd.DataFrame with new column col + '_encoded'
    """
    require_columns(df_in, [col], "encode_ordinal")

    mapping = {v: i for i, v in enumerate(order)}
    result  = df_in.copy()
    result[f"{col}_encoded"] = result[col].map(mapping)
    return result


def add_group_features(
    df_in,
    group_cols,
    value_col,
    aggregations=None
):
    """
    Add group-level statistics as new columns.

    Parameters
    ----------
    df_in : pd.DataFrame
    group_cols : list of str
    value_col : str
    aggregations : dict or None
        {new_col_name: agg_func} e.g. {"group_mean": "mean"}
        Defaults to mean, std, and count.

    Returns
    -------
    pd.DataFrame
    """
    require_columns(df_in, group_cols + [value_col], "add_group_features")

    if aggregations is None:
        aggregations = {
            f"{value_col}_group_mean":  "mean",
            f"{value_col}_group_std":   "std",
            f"{value_col}_group_count": "count"
        }

    result = df_in.copy()

    for new_col, agg_func in aggregations.items():
        result[new_col] = result.groupby(group_cols)[value_col].transform(agg_func)

    return result


# Test the reusable functions

if dataset_name == "superstore":
    test_df = df_raw[["order_id", "sales", "profit", "region", "category"]].head(20).copy()
    test_df.loc[3, "sales"] = np.nan

    result_fill = fill_missing_with_group_stat(test_df, "sales", ["region"], stat="median")
    print("fill_missing_with_group_stat result:")
    print(result_fill[["order_id", "sales"]].head(6))

    result_cap = cap_outliers(df_raw.head(50), "sales", method="iqr", factor=1.5)
    print(f"\ncap_outliers: sales max before={df_raw['sales'].head(50).max():.2f}  after={result_cap['sales'].max():.2f}")

    result_grp = add_group_features(
        df_raw.head(50), ["region"], "sales",
        {"region_avg_sales": "mean", "region_std_sales": "std"}
    )
    print("\nadd_group_features result (first 5 rows):")
    print(result_grp[["region", "sales", "region_avg_sales", "region_std_sales"]].head(5).round(2))

else:
    test_df = df_raw[["passengerid", "fare", "pclass", "sex"]].head(20).copy()
    test_df.loc[3, "fare"] = np.nan

    result_fill = fill_missing_with_group_stat(test_df, "fare", ["pclass"], stat="median")
    print("fill_missing_with_group_stat result:")
    print(result_fill[["passengerid", "fare"]].head(6))

    result_cap = cap_outliers(df_raw.head(50), "fare", method="iqr", factor=1.5)
    print(f"\ncap_outliers: fare max before={df_raw['fare'].head(50).max():.2f}  after={result_cap['fare'].max():.2f}")

    result_grp = add_group_features(
        df_raw.head(50), ["pclass"], "fare",
        {"class_avg_fare": "mean", "class_std_fare": "std"}
    )
    print("\nadd_group_features result (first 5 rows):")
    print(result_grp[["pclass", "fare", "class_avg_fare", "class_std_fare"]].head(5).round(2))


# ==============================================================================
# SECTION 5: PIPELINE CONFIGURATION OBJECTS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 5: PIPELINE CONFIGURATION OBJECTS")
print("=" * 70)

explanation = """
Hard-coded values inside pipeline functions make them:
  - Difficult to tune
  - Hard to reproduce different experiment configurations
  - Impossible to unit test in isolation

Use configuration dataclasses to:
  - Collect all parameters in one place
  - Allow different configs for training vs inference
  - Document defaults explicitly
  - Enable serialisation (asdict) for logging
"""
print(explanation)


@dataclass
class PipelineConfig:
    """
    Central configuration for the data cleaning pipeline.

    Attributes
    ----------
    drop_duplicate_rows : bool
    missing_fill_stat : str
        'mean', 'median', or 'mode'
    outlier_method : str
        'iqr' or 'zscore'
    outlier_factor : float
    min_non_null_fraction : float
        Drop columns where more than this fraction is null
    categorical_cardinality_threshold : float
        Convert to category if unique/total < threshold
    clip_quantile_low : float
    clip_quantile_high : float
    random_seed : int
    """
    drop_duplicate_rows:              bool  = True
    missing_fill_stat:                str   = "median"
    outlier_method:                   str   = "iqr"
    outlier_factor:                   float = 1.5
    min_non_null_fraction:            float = 0.50
    categorical_cardinality_threshold:float = 0.50
    clip_quantile_low:                float = 0.01
    clip_quantile_high:               float = 0.99
    random_seed:                      int   = 42

    def summary(self):
        print("\nPipeline Configuration:")
        for k, v in asdict(self).items():
            print(f"  {k:<40}: {v}")


# Create configs for different use cases
config_default = PipelineConfig()

config_strict = PipelineConfig(
    missing_fill_stat  = "mean",
    outlier_method     = "zscore",
    outlier_factor     = 3.0,
    clip_quantile_low  = 0.001,
    clip_quantile_high = 0.999
)

config_permissive = PipelineConfig(
    outlier_factor        = 3.0,
    min_non_null_fraction = 0.20
)

print("Default config:")
config_default.summary()

print("\nStrict config:")
config_strict.summary()


# ==============================================================================
# SECTION 6: A CONFIGURABLE PRODUCTION PIPELINE
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 6: A CONFIGURABLE PRODUCTION PIPELINE")
print("=" * 70)


def drop_high_null_columns(df_in, min_non_null_fraction=0.50):
    """Drop columns where non-null fraction is below threshold."""
    non_null_frac = 1 - df_in.isnull().mean()
    keep_cols = non_null_frac[non_null_frac >= min_non_null_fraction].index.tolist()
    dropped   = [c for c in df_in.columns if c not in keep_cols]
    if dropped:
        print(f"  [pipe] dropped high-null columns: {dropped}")
    return df_in[keep_cols]


def convert_low_cardinality_to_category(df_in, threshold=0.50):
    """Convert object columns with low cardinality to category."""
    result = df_in.copy()
    for col in result.select_dtypes(include=["str"]).columns:
        ratio = result[col].nunique() / len(result)
        if ratio < threshold:
            result[col] = result[col].astype("category")
    return result


def drop_duplicates_step(df_in):
    before = len(df_in)
    result = df_in.drop_duplicates()
    after  = len(result)
    if before != after:
        print(f"  [pipe] dropped {before - after} duplicate rows")
    return result


def build_superstore_features(df_in):
    """Superstore-specific feature engineering."""
    require_columns(df_in, ["sales", "profit", "quantity", "discount"], "build_superstore_features")
    return (
        df_in
        .assign(
            revenue_net    = lambda x: x["sales"] * (1 - x["discount"]),
            profit_margin  = lambda x: np.where(x["sales"] > 0, x["profit"] / x["sales"], np.nan),
            discount_pct   = lambda x: x["discount"] * 100,
            is_discounted  = lambda x: x["discount"] > 0,
            qty_bin        = lambda x: pd.cut(
                x["quantity"],
                bins=[0, 2, 5, 10, 100],
                labels=["1-2", "3-5", "6-10", "11+"]
            )
        )
    )


def build_titanic_features(df_in):
    """Titanic-specific feature engineering."""
    require_columns(df_in, ["survived", "pclass", "fare"], "build_titanic_features")
    result = df_in.copy()

    if "sibsp" in result.columns and "parch" in result.columns:
        result["family_size"] = result["sibsp"] + result["parch"] + 1
        result["is_alone"]    = (result["family_size"] == 1).astype(int)

    result = result.assign(
        fare_log     = lambda x: np.log1p(x["fare"].fillna(0)),
        fare_bin     = lambda x: pd.qcut(
            x["fare"].fillna(x["fare"].median()),
            q=4,
            labels=["Q1", "Q2", "Q3", "Q4"],
            duplicates="drop"
        ),
        is_adult     = lambda x: np.where(
            x["age"].notna(),
            (x["age"] >= 18).astype(int),
            np.nan
        )
    )

    if "sex" in result.columns:
        result["sex_encoded"] = result["sex"].map({"male": 0, "female": 1})

    return result


def run_production_pipeline(df_in, config):
    """
    Execute a configurable data preparation pipeline using method chaining.

    Parameters
    ----------
    df_in : pd.DataFrame
    config : PipelineConfig

    Returns
    -------
    pd.DataFrame
    """
    print(f"\n{'='*60}")
    print("PRODUCTION PIPELINE")
    print(f"Dataset: {dataset_name}  |  Seed: {config.random_seed}")
    print("=" * 60)

    np.random.seed(config.random_seed)

    pipeline_start = time.perf_counter()

    # Determine which numeric columns to fill
    num_cols_with_nulls = [
        c for c in df_in.select_dtypes(include=[np.number]).columns
        if df_in[c].isnull().any()
    ]

    # Build the chain
    result = (
        df_in

        # Step 1: Structural cleanup
        .pipe(log_shape, "raw input")
        .pipe(drop_duplicates_step if config.drop_duplicate_rows else (lambda x: x))
        .pipe(standardize_strings)
        .pipe(log_shape, "after structural cleanup")

        # Step 2: Column filtering
        .pipe(drop_high_null_columns, config.min_non_null_fraction)
        .pipe(log_shape, "after column filtering")

        # Step 3: Type optimization
        .pipe(convert_low_cardinality_to_category, config.categorical_cardinality_threshold)
        .pipe(log_shape, "after type optimization")
    )

    # Step 4: Fill missing values per column (cannot do inside chain easily)
    print(f"  [pipe log] filling missing values in: {num_cols_with_nulls}")
    for col in num_cols_with_nulls:
        if col in result.columns:
            fill_val = (
                result[col].mean()
                if config.missing_fill_stat == "mean"
                else result[col].median()
            )
            result[col] = result[col].fillna(fill_val)

    result = (
        result
        .pipe(log_shape, "after fill missing")

        # Step 5: Outlier handling
        .pipe(clip_numeric,
              lower_quantile=config.clip_quantile_low,
              upper_quantile=config.clip_quantile_high)
        .pipe(log_shape, "after outlier clipping")
    )

    # Step 6: Feature engineering (dataset-specific)
    if dataset_name == "superstore":
        result = result.pipe(build_superstore_features)
    else:
        result = result.pipe(build_titanic_features)

    result = result.pipe(log_shape, "after feature engineering")

    elapsed = (time.perf_counter() - pipeline_start) * 1000
    print(f"\nPipeline complete in {elapsed:.1f} ms")
    print(f"Input:  {df_in.shape}")
    print(f"Output: {result.shape}")

    return result


df_processed = run_production_pipeline(df_raw, config_default)

print("\nNew columns created:")
new_cols = [c for c in df_processed.columns if c not in df_raw.columns]
print(new_cols)

print("\nProcessed data sample:")
if dataset_name == "superstore":
    show_cols = ["sales", "profit", "revenue_net", "profit_margin", "is_discounted", "qty_bin"]
else:
    show_cols = ["survived", "fare", "fare_log", "fare_bin", "family_size", "sex_encoded"]
show_cols = [c for c in show_cols if c in df_processed.columns]
print(df_processed[show_cols].head(8))


# ==============================================================================
# SECTION 7: TESTING PIPELINES FOR DETERMINISM AND CORRECTNESS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 7: TESTING PIPELINES FOR DETERMINISM AND CORRECTNESS")
print("=" * 70)

explanation = """
A production pipeline MUST be:

1. DETERMINISTIC
   Same input + same config = same output every time.
   Random operations must use a fixed seed.

2. SHAPE-PRESERVING or PREDICTABLY SHAPE-CHANGING
   Filters reduce rows. Feature engineering adds columns.
   Any unexpected shape change is a bug.

3. NULL-FREE IN KEY COLUMNS after cleaning steps
   Key columns must not have NaN after the fill step.

4. COLUMN-STABLE
   Columns that existed in input must still exist in output
   unless explicitly removed.

5. VALUE-RANGE-RESPECTING
   Clipping and capping must hold after all operations.
"""
print(explanation)


def run_pipeline_tests(df_input, config):
    """Run a suite of correctness tests on the pipeline."""
    results = {}
    failed  = 0

    def record(name, passed, msg=""):
        results[name] = passed
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}" + (f": {msg}" if msg else ""))
        return passed

    print("\nRunning pipeline tests...")

    # Run pipeline twice with same config
    out1 = run_production_pipeline(df_input.head(100), config)
    out2 = run_production_pipeline(df_input.head(100), config)

    # Test 1: Determinism
    try:
        numeric1 = out1.select_dtypes(include=[np.number])
        numeric2 = out2.select_dtypes(include=[np.number])
        det = np.allclose(numeric1.values, numeric2.values, equal_nan=True)
        if not record("determinism_numeric", det, "outputs differ between runs"):
            failed += 1
    except Exception as e:
        record("determinism_numeric", False, str(e))
        failed += 1

    # Test 2: Output has rows
    has_rows = len(out1) > 0
    if not record("output_has_rows", has_rows, f"got {len(out1)} rows"):
        failed += 1

    # Test 3: Input columns preserved (allowing additions)
    input_cols  = set(df_input.columns)
    output_cols = set(out1.columns)
    preserved   = input_cols.issubset(output_cols)
    missing_cols = input_cols - output_cols
    if not record("input_columns_preserved", preserved, f"missing: {missing_cols}"):
        failed += 1

    # Test 4: No NaN in numeric columns after filling
    numeric_cols   = out1.select_dtypes(include=[np.number]).columns
    numeric_na     = out1[numeric_cols].isnull().sum().sum()
    if not record("no_null_after_fill", numeric_na == 0, f"{numeric_na} nulls remain"):
        failed += 1

    # Test 5: No duplicate rows
    dup_count = out1.duplicated().sum()
    if config.drop_duplicate_rows:
        if not record("no_duplicates_after_drop", dup_count == 0, f"{dup_count} dupes remain"):
            failed += 1

    # Test 6: No new columns lost (same column set both runs)
    same_cols = sorted(out1.columns.tolist()) == sorted(out2.columns.tolist())
    if not record("consistent_columns_across_runs", same_cols):
        failed += 1

    # Test 7: Feature columns exist (dataset-specific)
    if dataset_name == "superstore":
        expected_features = ["revenue_net", "profit_margin", "is_discounted"]
    else:
        expected_features = ["fare_log", "fare_bin"]

    for feat in expected_features:
        if not record(f"feature_exists_{feat}", feat in out1.columns):
            failed += 1

    # Test 8: Empty DataFrame edge case
    df_empty  = df_input.head(0)
    try:
        out_empty = run_production_pipeline(df_empty, config)
        if not record("handles_empty_input", len(out_empty) == 0):
            failed += 1
    except Exception as e:
        if not record("handles_empty_input", False, str(e)):
            failed += 1

    # Test 9: Single row edge case
    df_single = df_input.head(1)
    try:
        out_single = run_production_pipeline(df_single, config)
        if not record("handles_single_row", len(out_single) == 1):
            failed += 1
    except Exception as e:
        if not record("handles_single_row", False, str(e)):
            failed += 1

    total  = len(results)
    passed = total - failed
    print(f"\nTest summary: {passed}/{total} passed")
    return results


test_results = run_pipeline_tests(df_raw, config_default)


# ==============================================================================
# SECTION 8: COMPARING PIPELINE CONFIGS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 8: COMPARING PIPELINE CONFIGS")
print("=" * 70)

print("Running pipeline with three different configurations...")

configs = {
    "default":    config_default,
    "strict":     config_strict,
    "permissive": config_permissive,
}

comparison_rows = []
for config_name, cfg in configs.items():
    start = time.perf_counter()
    out   = run_production_pipeline(df_raw, cfg)
    elapsed_ms = (time.perf_counter() - start) * 1000

    numeric_cols = out.select_dtypes(include=[np.number]).columns
    comparison_rows.append({
        "config":           config_name,
        "rows":             len(out),
        "cols":             len(out.columns),
        "nulls_remaining":  int(out[numeric_cols].isnull().sum().sum()),
        "memory_mb":        round(out.memory_usage(deep=True).sum() / 1024 / 1024, 2),
        "elapsed_ms":       round(elapsed_ms, 1),
        "new_cols":         len([c for c in out.columns if c not in df_raw.columns])
    })

comparison_df = pd.DataFrame(comparison_rows)
print("\nConfig comparison:")
print(comparison_df.to_string(index=False))


# ==============================================================================
# SECTION 9: REAL-WORLD END-TO-END EXAMPLE
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 9: REAL-WORLD END-TO-END EXAMPLE")
print("=" * 70)

print("Scenario: Prepare a dataset for downstream analysis or ML training")

# Pick numeric target and group columns based on dataset
if dataset_name == "superstore":
    group_col   = "category"
    value_col   = "revenue_net"
    target_col  = "profit_margin"
else:
    group_col   = "pclass"
    value_col   = "fare_log"
    target_col  = "survived"

# Run the full pipeline
df_ready = run_production_pipeline(df_raw, config_default)

# Additional domain-specific steps chained on after the base pipeline

if dataset_name == "superstore" and "revenue_net" in df_ready.columns:
    df_ready = (
        df_ready
        .pipe(add_group_features,
              group_cols=[group_col],
              value_col=value_col,
              aggregations={
                  "category_avg_revenue": "mean",
                  "category_std_revenue": "std",
                  "category_order_count": "count"
              })
        .assign(
            revenue_vs_category_avg = lambda x: (
                x[value_col] - x["category_avg_revenue"]
            )
        )
    )

elif dataset_name == "titanic" and "fare_log" in df_ready.columns:
    df_ready = (
        df_ready
        .pipe(add_group_features,
              group_cols=["pclass"],
              value_col="fare_log",
              aggregations={
                  "class_avg_fare_log":  "mean",
                  "class_std_fare_log":  "std",
                  "class_passenger_count": "count"
              })
        .assign(
            fare_vs_class_avg = lambda x: (
                x["fare_log"] - x["class_avg_fare_log"]
            )
        )
    )

print("\nFinal dataset ready for analysis/ML:")
print(f"  Shape: {df_ready.shape}")
print(f"  Columns: {list(df_ready.columns)}")
print("\nSample rows:")
final_sample_cols = [c for c in df_ready.columns if c in df_ready.columns][:10]
print(df_ready[final_sample_cols].head(5).round(4))

# Quick summary statistics
print("\nNumeric column statistics:")
print(df_ready.select_dtypes(include=[np.number]).describe().round(2))


# ==============================================================================
# SECTION 10: COMMON PITFALLS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 10: COMMON PITFALLS")
print("=" * 70)

pitfalls = """
Pitfall 1: Referencing earlier columns inside the SAME assign() call
   WRONG: df.assign(a=df["x"]*2, b=df["a"]+1)
          df["a"] does not yet exist when b is computed
   RIGHT: df.assign(a=lambda x: x["x"]*2, b=lambda x: x["a"]+1)
          lambda receives the updated DataFrame at each step

Pitfall 2: Using pipe() with functions that modify in place
   - pipe() expects a function that returns a new/modified DataFrame
   - Functions that return None (or sort/fillna inplace) break the chain
   - Always return the DataFrame explicitly from pipe() functions

Pitfall 3: Hard-coding column names inside pipeline functions
   - Hard-coded names make functions brittle and non-reusable
   - Pass column names as explicit parameters with defaults

Pitfall 4: Losing track of shape changes across a long chain
   - Insert pipe(log_shape, 'label') checkpoints liberally during dev
   - Remove or reduce them for production after validation

Pitfall 5: Config mutations breaking reproducibility
   - Never mutate config objects inside pipeline steps
   - Pass config values by value (float, int, str) not by reference

Pitfall 6: Missing test for empty and single-row DataFrames
   - Many transformations silently fail on edge cases
   - Always test pipeline on head(0) and head(1)

Pitfall 7: Chaining after groupby without reset_index
   - groupby + agg returns MultiIndex which breaks most chaining
   - Always .reset_index() before continuing the chain

Pitfall 8: apply() inside a chain on large DataFrames
   - apply(row-wise) is very slow
   - Use assign + vectorized expressions or np.where inside chain
"""
print(pitfalls)


# ==============================================================================
# SECTION 11: SUMMARY TABLE
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 11: SUMMARY TABLE")
print("=" * 70)

summary = """
Operation                               | Syntax
----------------------------------------|------------------------------------------
Add / overwrite columns                 | df.assign(col=lambda x: x["a"]*2)
Call any function in chain              | df.pipe(my_func, arg1, kwarg=val)
Log shape without transforming          | df.pipe(log_shape, "label")
Filter rows (readable)                  | df.query("col > 5 and other == 'x'")
Reference Python variable in query      | df.query("col > @threshold")
Rename inside chain                     | df.rename(columns={"old": "new"})
Sort inside chain                       | df.sort_values("col", ascending=False)
Drop duplicates inside chain            | df.drop_duplicates()
Select columns inside chain             | df[["col1", "col2"]]
Drop columns inside chain               | df.drop(columns=["col"])
Convert dtype inside chain              | df.assign(col=lambda x: x["col"].astype("category"))
Group features inside chain             | df.pipe(add_group_features, ...)
Clip outliers inside chain              | df.pipe(cap_outliers, "col", method="iqr")
Fill missing inside chain               | df.pipe(fill_missing_with_group_stat, ...)
Multiple configs                        | PipelineConfig dataclass + run_production_pipeline
Test determinism                        | Run twice, np.allclose both outputs
Test edge cases                         | head(0), head(1), all-null column
"""
print(summary)


# ==============================================================================
# SECTION 12: PRACTICE EXERCISES
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 12: PRACTICE EXERCISES")
print("=" * 70)

print("Exercise 1: Add a log_nulls helper and use it inside a chain")


def log_nulls(df_in, label=""):
    """Print null counts per column without transforming."""
    n_nulls = df_in.isnull().sum()
    n_nulls = n_nulls[n_nulls > 0]
    if len(n_nulls) == 0:
        print(f"  [null log] {label}: no nulls")
    else:
        print(f"  [null log] {label}:")
        for col, n in n_nulls.items():
            print(f"    {col}: {n}")
    return df_in


exercise1 = (
    df_raw
    .pipe(log_nulls, "before fill")
    .pipe(lambda x: x.fillna({c: x[c].median() for c in x.select_dtypes(include=[np.number]).columns}))
    .pipe(log_nulls, "after fill")
)
print(f"  Shape: {exercise1.shape}")

print("\nExercise 2: Build a chain that filters, adds a feature, and summarizes")
if dataset_name == "superstore":
    q75 = df_raw["sales"].quantile(0.75)
    ex2 = (
        df_raw
        .query("sales > @q75")
        .assign(profit_flag=lambda x: np.where(x["profit"] > 0, "profit", "loss"))
        .groupby("profit_flag")["sales"]
        .agg(["sum", "mean", "count"])
        .reset_index()
    )
else:
    ex2 = (
        df_raw
        .query("survived == 1")
        .assign(fare_band=lambda x: pd.cut(
            x["fare"].fillna(x["fare"].median()),
            bins=4,
            labels=["Low", "Mid", "High", "VeryHigh"]
        ))
        .groupby("fare_band", observed=True)["survived"]
        .agg(["sum", "count"])
        .reset_index()
    )
print(ex2)

print("\nExercise 3: Compare row counts under two configs")
out_default    = run_production_pipeline(df_raw.head(200), config_default)
out_permissive = run_production_pipeline(df_raw.head(200), config_permissive)
print(f"  Default config rows:    {len(out_default)}")
print(f"  Permissive config rows: {len(out_permissive)}")

print("\nExercise 4: Write a test that verifies no new NaN appears after clip_numeric")
df_no_nulls = df_raw.select_dtypes(include=[np.number]).dropna()
clipped     = clip_numeric(df_no_nulls)
nulls_after = clipped.isnull().sum().sum()
print(f"  Nulls after clipping (should be 0): {nulls_after}")
assert nulls_after == 0, "clip_numeric introduced NaN"
print("  Assertion passed.")


# ==============================================================================
# SECTION 13: KEY TAKEAWAYS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 13: KEY TAKEAWAYS")
print("=" * 70)

takeaways = """
1.  Method chaining produces readable, top-to-bottom data pipelines
    with no intermediate variable clutter

2.  Use assign(col=lambda x: ...) to reference prior columns in the
    same assign() call safely

3.  pipe(func, *args) makes ANY function chainable without breaking
    the fluent style

4.  Insert pipe(log_shape, "label") and pipe(log_nulls, "label")
    freely during development; prune for production

5.  query() inside chains is cleaner than boolean masks for filtering

6.  Encapsulate all tunable parameters in a PipelineConfig dataclass
    to enable reproducible experiments and easy tuning

7.  Separate pipeline functions into single-responsibility steps
    that accept a DataFrame, transform it, and return it

8.  Always test pipelines with the same input twice to verify determinism

9.  Always test on edge cases: empty DataFrame and single-row DataFrame

10. Always require_columns() at the start of any function that depends
    on specific column names to fail early with a clear error

11. Use config objects to version different pipeline settings rather
    than hard-coding values inside functions

12. A good pipeline is one that new engineers can read top-to-bottom
    and understand without needing to trace execution mentally
"""
print(takeaways)


# ==============================================================================
# NEXT LESSON PREVIEW
# ==============================================================================
print("\n" + "=" * 70)
print("NEXT LESSON PREVIEW")
print("=" * 70)

preview = """
Lesson 32: Exploratory Data Analysis (EDA)

You will learn:
- Systematic EDA workflow from first load to insight
- Univariate analysis: distributions, outliers, missing patterns
- Bivariate analysis: correlations, group comparisons
- Multivariate analysis: pivot summaries, feature interactions
- EDA-specific Pandas patterns and helpers
- Building a reusable EDA report function
- Preparing findings for stakeholder communication
"""
print(preview)

print("\n" + "=" * 70)
print("END OF LESSON 31")
print("=" * 70)
print("\nYou can now build production-grade method-chained pipelines,")
print("parameterized with config objects, tested for determinism and")
print("correctness, and composed from single-responsibility functions.")