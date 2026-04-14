"""
LESSON 30: DEBUGGING DATA PIPELINES
================================================================================

What You Will Learn:
- A systematic debugging workflow for data pipelines
- Inspecting intermediate outputs at every step (shape, schema, dtypes)
- Tracing NaN and incorrect value propagation
- Detecting silent row loss or row explosion
- Detecting silent type coercion (object, float promotion)
- Debugging joins/merges (indicator, validate, row count checks)
- Debugging groupby outputs (index shape, MultiIndex surprises)
- Writing diagnostic helper functions (checkpoints)
- Reproducing pipeline bugs with minimal examples
- Building a self-auditing pipeline with checkpoints

Real World Usage:
- Debugging ETL pipelines producing wrong dashboard numbers
- Finding where duplicates entered a dataset
- Tracing why a join suddenly produced fewer rows
- Catching upstream schema changes (renamed columns)
- Investigating sudden spikes/drops after cleaning steps

Dataset Used:
Superstore dataset (public, no login required)
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
from dataclasses import dataclass, asdict

print("=" * 70)
print("LESSON 30: DEBUGGING DATA PIPELINES")
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
        print("Primary dataset loaded successfully.")
        return df, "superstore"
    except Exception as e:
        print(f"Primary load failed: {e}")
        print("\nFalling back to Titanic dataset from:")
        print(fallback_url)
        df = pd.read_csv(fallback_url)
        df.columns = [to_snake(c) for c in df.columns]
        print("Fallback dataset loaded successfully.")
        return df, "titanic"


df_raw, dataset_name = load_dataset()

print(f"\nDataset: {dataset_name}")
print(f"Shape: {df_raw.shape}")
print(f"Columns: {list(df_raw.columns)}")
print("\nHead:")
print(df_raw.head())
print("\nDtypes:")
print(df_raw.dtypes)
print("\nMissing values:")
print(df_raw.isnull().sum())


# ==============================================================================
# SECTION 2: THE DEBUGGING WORKFLOW
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 2: THE DEBUGGING WORKFLOW")
print("=" * 70)

workflow = """
A practical pipeline debugging workflow:

1. REPRODUCE
   - Fix seed, pin input data version, isolate the failing run
   - Reduce to the smallest subset that still shows the problem

2. CHECK INVARIANTS AT EVERY STEP
   - shape: rows/cols
   - schema: required columns exist
   - dtypes: numeric vs object
   - key constraints: uniqueness and nulls
   - numeric ranges: non-negative, expected bounds
   - nulls: missing value counts
   - duplicates: duplicated rows or keys

3. BISECT THE PIPELINE
   - Add checkpoints after each step
   - Find first step where output diverges from expectation

4. IDENTIFY THE FAILURE MODE
   - Silent row loss (over-filtering, inner join)
   - Row explosion (many-to-many merge)
   - NaN propagation (missing fill, type conversion)
   - Type coercion (int -> float, numeric -> object)
   - Misaligned index (assignment causes NaN)
   - MultiIndex surprises (groupby)

5. FIX WITH GUARDRAILS
   - Add explicit checks and assertions
   - Use merge validate=, indicator=True
   - Log key metrics and small samples
"""
print(workflow)


# ==============================================================================
# SECTION 3: DIAGNOSTIC HELPERS (CHECKPOINTS)
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 3: DIAGNOSTIC HELPERS (CHECKPOINTS)")
print("=" * 70)

@dataclass
class Checkpoint:
    name: str
    rows: int
    cols: int
    nulls_total: int
    duplicated_rows: int
    memory_mb: float
    dtypes_summary: dict
    timestamp: str

def dtype_summary(df):
    """Return dtype counts as a dictionary."""
    s = df.dtypes.astype(str).value_counts()
    return s.to_dict()

def memory_mb(df):
    return float(df.memory_usage(deep=True).sum() / 1024 / 1024)

def make_checkpoint(df, name):
    """Collect a snapshot of DataFrame state."""
    cp = Checkpoint(
        name=name,
        rows=int(df.shape[0]),
        cols=int(df.shape[1]),
        nulls_total=int(df.isnull().sum().sum()),
        duplicated_rows=int(df.duplicated().sum()),
        memory_mb=round(memory_mb(df), 3),
        dtypes_summary=dtype_summary(df),
        timestamp=datetime.datetime.now().isoformat(timespec="seconds"),
    )
    return cp

def print_checkpoint(cp):
    """Pretty print a checkpoint."""
    print(f"\n[CHECKPOINT] {cp.name}")
    print(f"  Timestamp:      {cp.timestamp}")
    print(f"  Shape:          ({cp.rows:,}, {cp.cols})")
    print(f"  Nulls total:    {cp.nulls_total:,}")
    print(f"  Duplicated rows:{cp.duplicated_rows:,}")
    print(f"  Memory:         {cp.memory_mb:.3f} MB")
    print(f"  Dtypes:         {cp.dtypes_summary}")

def require_columns(df, cols, where=""):
    """Raise with a clear error if required columns are missing."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{where}: missing required columns: {missing}")

def require_non_null(df, cols, where=""):
    """Ensure key columns are not null."""
    for c in cols:
        if c in df.columns:
            n = int(df[c].isnull().sum())
            if n > 0:
                raise ValueError(f"{where}: column '{c}' has {n} nulls")

def require_unique(df, cols, where=""):
    """Ensure columns (single or composite) are unique."""
    if isinstance(cols, str):
        cols = [cols]
    require_columns(df, cols, where)
    dup = int(df[cols].duplicated().sum())
    if dup > 0:
        raise ValueError(f"{where}: duplicate keys found for {cols} (duplicates={dup})")

def print_small_sample(df, cols, n=5, where=""):
    """Print a small sample of selected columns."""
    print(f"\n[SAMPLE] {where} columns={cols} n={n}")
    cols_existing = [c for c in cols if c in df.columns]
    if not cols_existing:
        print("  No requested columns exist in DataFrame.")
        return
    print(df[cols_existing].head(n))


cp0 = make_checkpoint(df_raw, "loaded_raw")
print_checkpoint(cp0)


# ==============================================================================
# SECTION 4: A DELIBERATELY BUGGY PIPELINE (FOR DEBUGGING PRACTICE)
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 4: A DELIBERATELY BUGGY PIPELINE")
print("=" * 70)

explanation = """
We will build a small pipeline that intentionally contains common bugs:
- Over-filtering causing row loss
- Merge key mismatch and many-to-many explosion risk
- Bad type conversion causing NaN
- Silent NaN propagation in computed fields

Then we will debug it using checkpoints and targeted inspections.
"""
print(explanation)

df = df_raw.copy()

# Choose column names depending on dataset
if dataset_name == "superstore":
    # Ensure key columns exist
    required = ["order_id", "sales", "profit", "region", "category"]
    require_columns(df, required, where="initial")
else:
    required = ["passengerid", "survived", "pclass", "sex", "fare"]
    require_columns(df, required, where="initial")

# Create a lookup table for a merge
if dataset_name == "superstore":
    # Simulate a region lookup with a deliberate issue: duplicated key (WEST appears twice)
    region_lookup = pd.DataFrame({
        "region": ["Central", "East", "South", "West", "West"],
        "manager": ["M1", "M2", "M3", "M4", "M4_DUP"]
    })
else:
    # Simulate class lookup with a deliberate issue: duplicated key for pclass
    region_lookup = pd.DataFrame({
        "pclass": [1, 2, 3, 3],
        "class_label": ["First", "Second", "Third", "Third_DUP"]
    })

print("\nLookup table (intentionally includes duplicate key):")
print(region_lookup)


def step_1_filter(df_in):
    """Bug: over-filtering can drop too many rows."""
    out = df_in.copy()
    if dataset_name == "superstore":
        # Bug: strict filter accidentally removes most data
        out = out[(out["sales"] > 5000) & (out["profit"] > 1000)]
    else:
        out = out[(out["fare"] > 200) & (out["survived"] == 1)]
    return out


def step_2_merge_lookup(df_in):
    """
    Bug: merge can cause row explosion due to many-to-many keys.
    Also missing validation means this change can go unnoticed.
    """
    if dataset_name == "superstore":
        out = pd.merge(df_in, region_lookup, on="region", how="left")
    else:
        out = pd.merge(df_in, region_lookup, on="pclass", how="left")
    return out


def step_3_bad_type_conversion(df_in):
    """Bug: force numeric conversion on a column that is already numeric or has commas."""
    out = df_in.copy()
    if dataset_name == "superstore":
        # Convert sales to numeric by string manipulation (unnecessary and risky)
        # This intentionally introduces NaN if unexpected formatting exists.
        out["sales_str"] = out["sales"].astype(str)
        out["sales_str"] = out["sales_str"].str.replace(",", "", regex=False)
        out["sales_num"] = pd.to_numeric(out["sales_str"], errors="coerce")
    else:
        out["fare_str"] = out["fare"].astype(str)
        out["fare_num"] = pd.to_numeric(out["fare_str"], errors="coerce")
    return out


def step_4_compute_metrics(df_in):
    """Bug: compute ratio without guarding against NaN or zero."""
    out = df_in.copy()
    if dataset_name == "superstore":
        out["profit_margin"] = out["profit"] / out["sales_num"]
    else:
        out["fare_ratio"] = out["fare_num"] / out["fare_num"].mean()
    return out


def run_buggy_pipeline(df_in):
    checkpoints = []

    x = df_in
    checkpoints.append(make_checkpoint(x, "start"))

    x = step_1_filter(x)
    checkpoints.append(make_checkpoint(x, "after_step_1_filter"))

    x = step_2_merge_lookup(x)
    checkpoints.append(make_checkpoint(x, "after_step_2_merge_lookup"))

    x = step_3_bad_type_conversion(x)
    checkpoints.append(make_checkpoint(x, "after_step_3_bad_type_conversion"))

    x = step_4_compute_metrics(x)
    checkpoints.append(make_checkpoint(x, "after_step_4_compute_metrics"))

    return x, checkpoints


buggy_out, cps = run_buggy_pipeline(df)

print("\nCheckpoints from buggy pipeline:")
for cp in cps:
    print_checkpoint(cp)

print("\nBuggy pipeline output head:")
print(buggy_out.head())


# ==============================================================================
# SECTION 5: DEBUGGING THE BUGGY PIPELINE STEP BY STEP
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 5: DEBUGGING STEP BY STEP")
print("=" * 70)

# ------------------------------------------------------------------------------
# 5.1 Detect Silent Row Loss
# ------------------------------------------------------------------------------
print("\n--- 5.1 Detect Silent Row Loss ---")

rows_start = cps[0].rows
rows_after_filter = cps[1].rows
print(f"Rows start:        {rows_start:,}")
print(f"Rows after filter: {rows_after_filter:,}")
drop_pct = (1 - rows_after_filter / rows_start) * 100 if rows_start else 0
print(f"Rows dropped:      {rows_start - rows_after_filter:,} ({drop_pct:.1f}%)")

print("\nDebugging action:")
print("  Inspect filter thresholds and distribution before filtering")

if dataset_name == "superstore":
    print("\nSales and profit quantiles:")
    print(df_raw[["sales", "profit"]].quantile([0.5, 0.75, 0.9, 0.95, 0.99]).round(2))

    # Show how many rows would remain under different thresholds
    for sales_thr, profit_thr in [(1000, 100), (2000, 200), (5000, 1000)]:
        kept = df_raw[(df_raw["sales"] > sales_thr) & (df_raw["profit"] > profit_thr)]
        print(f"  Threshold sales>{sales_thr}, profit>{profit_thr}: kept {len(kept):,} rows")
else:
    print("\nFare quantiles:")
    print(df_raw["fare"].quantile([0.5, 0.75, 0.9, 0.95, 0.99]).round(2))
    kept = df_raw[(df_raw["fare"] > 200) & (df_raw["survived"] == 1)]
    print(f"Rows with fare>200 and survived==1: {len(kept):,}")

print("\nConclusion: Over-filtering is a common root cause of empty or tiny outputs.")


# ------------------------------------------------------------------------------
# 5.2 Detect Row Explosion from Merge
# ------------------------------------------------------------------------------
print("\n--- 5.2 Detect Row Explosion from Merge ---")

rows_before_merge = cps[1].rows
rows_after_merge  = cps[2].rows
print(f"Rows before merge: {rows_before_merge:,}")
print(f"Rows after merge:  {rows_after_merge:,}")

if rows_before_merge > 0:
    ratio = rows_after_merge / rows_before_merge
    print(f"Row multiplier:    {ratio:.2f}x")
else:
    print("No rows before merge; cannot evaluate merge explosion.")

print("\nDebugging action:")
print("  Check key uniqueness on the right table and use merge validate=")

if dataset_name == "superstore":
    key = "region"
else:
    key = "pclass"

dup_right = int(region_lookup[key].duplicated().sum())
print(f"Right lookup duplicates on key '{key}': {dup_right}")

print("\nFix pattern:")
print("  pd.merge(left, right, on=key, how='left', validate='many_to_one')")
print("  If validate fails, your lookup table is not unique on key")

try:
    if dataset_name == "superstore":
        _ = pd.merge(
            step_1_filter(df),
            region_lookup,
            on="region",
            how="left",
            validate="many_to_one"
        )
    else:
        _ = pd.merge(
            step_1_filter(df),
            region_lookup,
            on="pclass",
            how="left",
            validate="many_to_one"
        )
    print("Validation passed unexpectedly (no duplicates).")
except Exception as e:
    print(f"Validation correctly failed: {e}")

print("\nFix:")
print("  Deduplicate lookup table or correct the join key.")
fixed_lookup = region_lookup.drop_duplicates(subset=[key], keep="first")
print("Fixed lookup table:")
print(fixed_lookup)

# Re-run merge with validate
if dataset_name == "superstore":
    merged_fixed = pd.merge(
        step_1_filter(df),
        fixed_lookup,
        on="region",
        how="left",
        validate="many_to_one",
        indicator=True
    )
else:
    merged_fixed = pd.merge(
        step_1_filter(df),
        fixed_lookup,
        on="pclass",
        how="left",
        validate="many_to_one",
        indicator=True
    )

print("\nMerge with indicator shows match status:")
print(merged_fixed["_merge"].value_counts())


# ------------------------------------------------------------------------------
# 5.3 Detect NaN Introduced by Bad Type Conversion
# ------------------------------------------------------------------------------
print("\n--- 5.3 Detect NaN Introduced by Type Conversion ---")

if dataset_name == "superstore":
    conv_col = "sales_num"
    base_col = "sales"
else:
    conv_col = "fare_num"
    base_col = "fare"

tmp = step_3_bad_type_conversion(merged_fixed)
n_nan = int(tmp[conv_col].isnull().sum())
print(f"NaN count in {conv_col}: {n_nan}")

print("\nDebugging action:")
print("  Compare original column dtype and sample values")
print(f"Original {base_col} dtype: {df_raw[base_col].dtype}")
print_small_sample(df_raw, [base_col], n=5, where="raw sample")

print("\nIf base column is already numeric, do not convert through string.")
print("If base column is string numeric, use pd.to_numeric once.")

# Better conversion approach (safe):
tmp2 = merged_fixed.copy()
tmp2[conv_col] = pd.to_numeric(tmp2[base_col], errors="coerce")
print(f"\nSafe conversion NaN count in {conv_col}: {int(tmp2[conv_col].isnull().sum())}")


# ------------------------------------------------------------------------------
# 5.4 Detect NaN/Inf in Computed Metrics
# ------------------------------------------------------------------------------
print("\n--- 5.4 Detect NaN/Inf in Derived Metrics ---")

computed = step_4_compute_metrics(tmp2)

if dataset_name == "superstore":
    metric_col = "profit_margin"
    denom_col = "sales_num"
else:
    metric_col = "fare_ratio"
    denom_col = "fare_num"

n_nan_metric = int(computed[metric_col].isnull().sum())
n_inf_metric = int(np.isinf(computed[metric_col]).sum()) if pd.api.types.is_numeric_dtype(computed[metric_col]) else 0

print(f"{metric_col} NaNs: {n_nan_metric}")
print(f"{metric_col} Infs: {n_inf_metric}")

print("\nDebugging action:")
print("  Check denominators for zeros or NaN before division")
if denom_col in computed.columns:
    denom_nulls = int(computed[denom_col].isnull().sum())
    denom_zeros = int((computed[denom_col] == 0).sum()) if pd.api.types.is_numeric_dtype(computed[denom_col]) else 0
    print(f"Denominator {denom_col} nulls: {denom_nulls}")
    print(f"Denominator {denom_col} zeros: {denom_zeros}")

print("\nFix pattern: safe division with np.where and fillna")
fixed = tmp2.copy()

if dataset_name == "superstore":
    denom = fixed["sales_num"]
    fixed["profit_margin"] = np.where(
        denom > 0,
        fixed["profit"] / denom,
        np.nan
    )
    fixed["profit_margin"] = fixed["profit_margin"].fillna(0.0)
else:
    denom_mean = fixed["fare_num"].mean()
    fixed["fare_ratio"] = np.where(
        fixed["fare_num"].notna() & (denom_mean != 0),
        fixed["fare_num"] / denom_mean,
        np.nan
    )
    fixed["fare_ratio"] = fixed["fare_ratio"].fillna(0.0)

print(f"After fix, {metric_col} NaNs: {int(fixed[metric_col].isnull().sum())}")


# ==============================================================================
# SECTION 6: BUILD A SELF-AUDITING PIPELINE (CHECKPOINTS + INVARIANTS)
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 6: SELF-AUDITING PIPELINE")
print("=" * 70)

explanation = """
A self-auditing pipeline:
- Runs each step
- Captures a checkpoint after each step
- Checks invariants (schema, row count expectations, key constraints)
- Stops early or prints clear diagnostics

This is a production pattern: pipeline + observability.
"""
print(explanation)

def run_pipeline_with_checkpoints(df_in, steps):
    """
    Run a list of steps with checkpoints and basic invariant checks.

    Parameters
    ----------
    df_in : pd.DataFrame
    steps : list of tuples
        [(name, func, invariant_func_or_none), ...]

    Returns
    -------
    (final_df, checkpoints)
    """
    checkpoints = []
    x = df_in.copy()
    checkpoints.append(make_checkpoint(x, "start"))
    print_checkpoint(checkpoints[-1])

    for name, func, invariant in steps:
        t0 = time.perf_counter()
        x = func(x)
        elapsed = (time.perf_counter() - t0) * 1000

        cp = make_checkpoint(x, name)
        checkpoints.append(cp)

        print_checkpoint(cp)
        print(f"  Step time: {elapsed:.1f} ms")

        if invariant is not None:
            invariant(x, checkpoints)

    return x, checkpoints

def invariant_basic(df_step, checkpoints):
    """Basic invariants that should usually hold."""
    # Must have at least 1 row unless intentionally filtering to empty
    if df_step.shape[0] == 0:
        raise ValueError("Invariant failed: DataFrame became empty")

    # No duplicate columns
    if df_step.columns.duplicated().any():
        dups = df_step.columns[df_step.columns.duplicated()].tolist()
        raise ValueError(f"Invariant failed: duplicate columns: {dups}")

def invariant_no_row_explosion(df_step, checkpoints, max_multiplier=5.0):
    """Detect sudden row explosion relative to previous checkpoint."""
    prev = checkpoints[-2].rows
    curr = checkpoints[-1].rows
    if prev > 0 and curr / prev > max_multiplier:
        raise ValueError(f"Row explosion detected: {prev} -> {curr} ({curr/prev:.2f}x)")

def step_filter_reasonable(df_in):
    """A safer filter step that is less likely to wipe out the dataset."""
    out = df_in.copy()
    if dataset_name == "superstore":
        # Use quantile-based threshold instead of hard-coded huge values
        sales_thr = out["sales"].quantile(0.90)
        out = out[out["sales"] >= sales_thr]
    else:
        # Keep passengers with known fare and valid survived values
        out = out[out["fare"].notna() & out["survived"].isin([0, 1])]
    return out

def step_merge_lookup_safe(df_in):
    """Safe merge with validate and indicator."""
    if dataset_name == "superstore":
        lk = region_lookup.drop_duplicates(subset=["region"])
        out = pd.merge(df_in, lk, on="region", how="left", validate="many_to_one", indicator=True)
    else:
        lk = region_lookup.drop_duplicates(subset=["pclass"])
        out = pd.merge(df_in, lk, on="pclass", how="left", validate="many_to_one", indicator=True)
    return out

def step_safe_numeric(df_in):
    """Safe numeric conversion (if needed) without string round-trip."""
    out = df_in.copy()
    if dataset_name == "superstore":
        out["sales_num"] = pd.to_numeric(out["sales"], errors="coerce")
    else:
        out["fare_num"] = pd.to_numeric(out["fare"], errors="coerce")
    return out

def step_safe_metrics(df_in):
    """Compute derived metrics with guards."""
    out = df_in.copy()
    if dataset_name == "superstore":
        denom = out["sales_num"]
        out["profit_margin"] = np.where(denom > 0, out["profit"] / denom, np.nan)
        out["profit_margin"] = out["profit_margin"].fillna(0.0)
    else:
        denom_mean = out["fare_num"].mean()
        out["fare_ratio"] = np.where(out["fare_num"].notna() & (denom_mean != 0), out["fare_num"] / denom_mean, np.nan)
        out["fare_ratio"] = out["fare_ratio"].fillna(0.0)
    return out

steps = [
    ("filter_reasonable", step_filter_reasonable, invariant_basic),
    ("merge_lookup_safe", step_merge_lookup_safe, lambda df_s, cps_s: invariant_no_row_explosion(df_s, cps_s, 2.0)),
    ("safe_numeric",      step_safe_numeric,      invariant_basic),
    ("safe_metrics",      step_safe_metrics,      invariant_basic),
]

try:
    final_df, final_cps = run_pipeline_with_checkpoints(df_raw, steps)
    print("\nFinal output sample:")
    cols = ["sales", "profit", "sales_num", "profit_margin"] if dataset_name == "superstore" else ["fare", "fare_num", "fare_ratio"]
    print_small_sample(final_df, cols + (["_merge"] if "_merge" in final_df.columns else []), n=8, where="final")
except Exception as e:
    print("\nPipeline failed with clear error:")
    print(str(e))


# ==============================================================================
# SECTION 7: MINIMAL REPRODUCTION EXAMPLES (MREs)
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 7: MINIMAL REPRODUCTION EXAMPLES (MREs)")
print("=" * 70)

explanation = """
When you find a bug, build a minimal reproduction example:
- Smallest possible DataFrame that still produces the issue
- Easy to reason about and test
- Can be included in unit tests to prevent regressions
"""
print(explanation)

# ------------------------------------------------------------------------------
# 7.1 MRE: Many-to-many merge row explosion
# ------------------------------------------------------------------------------
print("\n--- 7.1 MRE: Many-to-many merge ---")

left = pd.DataFrame({"key": ["A", "A", "B"], "x": [1, 2, 3]})
right = pd.DataFrame({"key": ["A", "A", "B", "B"], "y": [10, 20, 30, 40]})

m = pd.merge(left, right, on="key", how="inner")
print("Left:")
print(left)
print("\nRight:")
print(right)
print("\nMerged (row explosion):")
print(m)
print(f"\nRows left: {len(left)}, rows right: {len(right)}, rows merged: {len(m)}")

print("\nFix pattern: enforce uniqueness on lookup side or validate relationship.")
try:
    pd.merge(left, right, on="key", validate="many_to_one")
except Exception as e:
    print(f"validate='many_to_one' correctly fails: {e}")

# ------------------------------------------------------------------------------
# 7.2 MRE: Index misalignment on assignment
# ------------------------------------------------------------------------------
print("\n--- 7.2 MRE: Index misalignment ---")

df_mre = pd.DataFrame({"val": [10, 20, 30, 40]}, index=[10, 11, 12, 13])
s = pd.Series([1, 2], index=[10, 12])

df_mre["bad_assign"] = s
print("Assignment aligns by index labels:")
print(df_mre)
print("Rows 11 and 13 become NaN because s has no entries for them.")

df_mre["good_assign"] = s.reindex(df_mre.index).fillna(0).astype(int)
print("\nFix: align explicitly with reindex + fill:")
print(df_mre)


# ==============================================================================
# SECTION 8: COMMON PITFALLS CHECKLIST
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 8: COMMON PITFALLS CHECKLIST")
print("=" * 70)

pitfalls = """
1. Over-filtering:
   - Row count drops to near zero
   - Fix: use quantiles, inspect distributions, add row count checks

2. Wrong join type:
   - Inner join drops unmatched rows unexpectedly
   - Fix: indicator=True, use left join when preserving base table

3. Many-to-many join:
   - Row count explodes unexpectedly
   - Fix: validate=, ensure lookup keys are unique

4. Type coercion:
   - int becomes float after NaN appears, object appears unexpectedly
   - Fix: explicit dtypes, to_numeric(errors='coerce'), nullable Int64

5. NaN propagation:
   - Derived columns become NaN or inf
   - Fix: fillna, guards for division by zero, validate denominators

6. Misaligned index:
   - Assignments create NaN due to index alignment
   - Fix: use .values for positional assign or align with reindex()

7. MultiIndex surprises:
   - groupby results have MultiIndex that breaks downstream code
   - Fix: reset_index(), flatten columns after agg

8. Silent duplicate columns after merge:
   - value_x/value_y appear, later code uses wrong one
   - Fix: suffixes=, rename before merge

9. Hidden duplicates:
   - duplicates in keys cause joins and groupby to behave oddly
   - Fix: require_unique checks, duplicate reports

10. Debugging without checkpoints:
    - Hard to know where problem was introduced
    - Fix: checkpoint after each pipeline step, log shape/nulls/dtypes
"""
print(pitfalls)


# ==============================================================================
# SECTION 9: PRACTICE EXERCISES
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 9: PRACTICE EXERCISES")
print("=" * 70)

print("Exercise 1: Add a checkpoint after filtering and print top values")
if dataset_name == "superstore":
    filtered = df_raw[df_raw["sales"] > df_raw["sales"].quantile(0.95)].copy()
    cp = make_checkpoint(filtered, "exercise_filtered_top_5pct_sales")
    print_checkpoint(cp)
    print_small_sample(filtered, ["order_id", "sales", "profit", "region", "category"], n=5, where="exercise1")
else:
    filtered = df_raw[df_raw["fare"] > df_raw["fare"].quantile(0.95)].copy()
    cp = make_checkpoint(filtered, "exercise_filtered_top_5pct_fare")
    print_checkpoint(cp)
    print_small_sample(filtered, ["passengerid", "fare", "pclass", "sex", "survived"], n=5, where="exercise1")

print("\nExercise 2: Identify which step first introduces new NaN values")
# Compare consecutive checkpoints from the buggy pipeline
for i in range(1, len(cps)):
    prev = cps[i-1]
    curr = cps[i]
    if curr.nulls_total > prev.nulls_total:
        print(f"  NaNs increased at: {curr.name} ({prev.nulls_total} -> {curr.nulls_total})")
        break
else:
    print("  No NaN increase detected in the buggy pipeline checkpoints")

print("\nExercise 3: Verify key uniqueness in the lookup table")
key_col = "region" if dataset_name == "superstore" else "pclass"
dup = int(region_lookup[key_col].duplicated().sum())
print(f"  Duplicate keys in lookup '{key_col}': {dup}")
print("  Fix: lookup.drop_duplicates(subset=[key_col], keep='first')")

print("\nExercise 4: Build a minimal DataFrame that reproduces a row explosion")
left2 = pd.DataFrame({"id": [1, 1, 1], "v": [10, 20, 30]})
right2 = pd.DataFrame({"id": [1, 1], "w": [100, 200]})
m2 = pd.merge(left2, right2, on="id")
print(f"  Left rows: {len(left2)}, Right rows: {len(right2)}, Merged rows: {len(m2)}")
print(m2)


# ==============================================================================
# SECTION 10: KEY TAKEAWAYS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 10: KEY TAKEAWAYS")
print("=" * 70)

takeaways = """
1. Debug pipelines by adding checkpoints after every step:
   shape, null counts, duplicates, dtypes, memory.

2. The most common pipeline failures are:
   - row loss (filters, inner joins)
   - row explosion (many-to-many joins)
   - NaN propagation (missing fill, division by zero)
   - dtype changes (int -> float, object creep)
   - index misalignment (assignment produces NaN)

3. Always validate merges:
   - indicator=True to see match status
   - validate='many_to_one' or 'one_to_one' to prevent row explosion
   - pre-check uniqueness of lookup keys

4. Use quantile-based filters instead of hard-coded thresholds when possible.

5. Build minimal reproduction examples for bugs and add them as tests.

6. Prefer a self-auditing pipeline in production:
   - step logging
   - invariant checks
   - clear failures
"""
print(takeaways)


# ==============================================================================
# NEXT LESSON PREVIEW
# ==============================================================================
print("\n" + "=" * 70)
print("NEXT LESSON PREVIEW")
print("=" * 70)

preview = """
Lesson 31: Pipeline Building with Method Chaining (Production Patterns)

You will learn:
- Method chaining with assign(), pipe(), and query()
- Building readable transformation pipelines
- Creating reusable pipeline functions
- Avoiding intermediate DataFrames
- Adding checkpoint logs inside method chains
- Using configuration objects to parameterize pipelines
- Testing pipelines and ensuring deterministic outputs
"""
print(preview)

print("\n" + "=" * 70)
print("END OF LESSON 30")
print("=" * 70)
print("\nYou can now debug data pipelines systematically using checkpoints,")
print("invariant checks, and minimal reproduction examples.")