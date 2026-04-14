"""
LESSON 23: AGGREGATIONS AND GROUPBY
================================================================================

What You Will Learn:
- The split-apply-combine pattern
- groupby mechanics and how it works internally
- Aggregating with single and multiple functions
- Grouping by multiple columns
- transform vs agg vs filter
- Named aggregations with agg()
- Pivot tables as an alternative to groupby
- Real world sales and customer segmentation examples

Real World Usage:
- Calculating revenue per region or product category
- Customer segmentation by spending behavior
- Computing per-group statistics for dashboards
- Generating summary reports from raw transaction data
- Feature engineering: adding group-level stats to each row

Dataset Used:
Superstore Sales Dataset (public, no login required)
URL: https://raw.githubusercontent.com/dsrscientist/dataset1/master/superstore.csv
Fallback: Titanic dataset
https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv

================================================================================
"""

import numpy as np
import pandas as pd

print("=" * 70)
print("LESSON 23: AGGREGATIONS AND GROUPBY")
print("=" * 70)


# ==============================================================================
# SECTION 1: LOAD DATASET
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 1: LOAD DATASET")
print("=" * 70)

primary_url  = "https://raw.githubusercontent.com/dsrscientist/dataset1/master/superstore.csv"
fallback_url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

def load_dataset():
    try:
        print("Loading Superstore Sales dataset from:")
        print(primary_url)
        df = pd.read_csv(primary_url, encoding="latin-1")
        print("Primary dataset loaded successfully.")
        return df, "superstore"
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
print("\nData types:")
print(df_raw.dtypes)

# Standardize column names to snake_case
import re

def to_snake(name):
    name = str(name).strip()
    name = re.sub(r"[\s\-\.]+", "_", name)
    name = re.sub(r"[^\w]", "", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_").lower()

df_raw.columns = [to_snake(col) for col in df_raw.columns]
print("\nStandardized column names:")
print(list(df_raw.columns))


# ==============================================================================
# SECTION 2: THE SPLIT-APPLY-COMBINE PATTERN
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 2: THE SPLIT-APPLY-COMBINE PATTERN")
print("=" * 70)

explanation = """
groupby() follows the Split-Apply-Combine pattern:

1. SPLIT
   Divide the DataFrame into groups based on one or more columns.
   Each unique value (or combination) becomes a group.

2. APPLY
   Apply a function to each group independently.
   This can be aggregation, transformation, or filtering.

3. COMBINE
   Collect all group results back into a single DataFrame or Series.

Example:
   Sales data has 1000 rows and a 'Region' column.
   Split:   Group rows by Region (North, South, East, West)
   Apply:   Calculate total Sales for each group
   Combine: Return a Series with one value per Region

This pattern replaces many loops in data analysis.
"""
print(explanation)

# Manual demonstration of split-apply-combine without groupby
if dataset_name == "titanic":
    df = df_raw.copy()

    # MANUAL WAY (slow, verbose)
    print("Manual split-apply-combine (what groupby replaces):")
    for pclass in sorted(df["pclass"].unique()):
        group = df[df["pclass"] == pclass]
        rate  = group["survived"].mean()
        print(f"  Class {pclass}: survival rate = {rate:.3f}")

    # GROUPBY WAY (fast, clean)
    print("\nEquivalent using groupby:")
    result = df.groupby("pclass")["survived"].mean()
    print(result)
    print("\ngroupby is faster and requires far less code")


# ==============================================================================
# SECTION 3: GROUPBY BASICS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 3: GROUPBY BASICS")
print("=" * 70)

df = df_raw.copy()

# ------------------------------------------------------------------------------
# 3.1 Single Column Groupby
# ------------------------------------------------------------------------------
print("\n--- 3.1 Single Column Groupby ---")

if dataset_name == "superstore":
    group_col    = "category"
    numeric_col  = "sales"
else:
    group_col    = "pclass"
    numeric_col  = "fare"

# The groupby object itself is lazy - no computation yet
grouped = df.groupby(group_col)
print(f"Type of groupby object: {type(grouped)}")
print(f"Number of groups: {grouped.ngroups}")
print(f"Group keys: {list(grouped.groups.keys())}")

# Apply an aggregation to trigger computation
result = grouped[numeric_col].sum()
print(f"\nTotal {numeric_col} per {group_col}:")
print(result)
print(f"\nResult type: {type(result)}")
print(f"Result dtype: {result.dtype}")

# ------------------------------------------------------------------------------
# 3.2 Common Aggregation Functions
# ------------------------------------------------------------------------------
print("\n--- 3.2 Common Aggregation Functions ---")

agg_functions = {
    "sum":    grouped[numeric_col].sum(),
    "mean":   grouped[numeric_col].mean(),
    "median": grouped[numeric_col].median(),
    "min":    grouped[numeric_col].min(),
    "max":    grouped[numeric_col].max(),
    "count":  grouped[numeric_col].count(),
    "std":    grouped[numeric_col].std(),
    "var":    grouped[numeric_col].var(),
}

print(f"Aggregations on {numeric_col} grouped by {group_col}:")
print(f"\n{'Function':<12}", end="")
for key in grouped.groups.keys():
    print(f"{str(key)[:12]:>14}", end="")
print()
print("-" * (12 + 14 * grouped.ngroups))

for func_name, result in agg_functions.items():
    print(f"{func_name:<12}", end="")
    for val in result.values:
        print(f"{val:>14.2f}", end="")
    print()

# ------------------------------------------------------------------------------
# 3.3 Inspecting Groups
# ------------------------------------------------------------------------------
print("\n--- 3.3 Inspecting Groups ---")

# Iterate over groups (useful for debugging)
print("First 2 rows of each group:")
for name, group_df in grouped:
    print(f"\nGroup: {name}")
    print(group_df[[group_col, numeric_col]].head(2))

# Get a single group
if dataset_name == "superstore":
    single_group_key = list(grouped.groups.keys())[0]
else:
    single_group_key = 1

single_group = grouped.get_group(single_group_key)
print(f"\nget_group('{single_group_key}'):")
print(f"  Shape: {single_group.shape}")
print(single_group[[group_col, numeric_col]].head(3))

# Group sizes
print("\nGroup sizes:")
print(grouped.size())


# ==============================================================================
# SECTION 4: MULTIPLE COLUMN GROUPBY
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 4: MULTIPLE COLUMN GROUPBY")
print("=" * 70)

# ------------------------------------------------------------------------------
# 4.1 Group by Two Columns
# ------------------------------------------------------------------------------
print("\n--- 4.1 Group by Two Columns ---")

if dataset_name == "superstore":
    group_cols = ["category", "region"]
    value_col  = "sales"
else:
    group_cols = ["pclass", "sex"]
    value_col  = "survived"

result_multi = df.groupby(group_cols)[value_col].mean()
print(f"Mean {value_col} grouped by {group_cols}:")
print(result_multi)
print(f"\nResult index type: {type(result_multi.index)}")
print("Result has a MultiIndex")

# Flatten the MultiIndex by resetting
result_flat = result_multi.reset_index()
print("\nAfter reset_index() - flat DataFrame:")
print(result_flat)

# ------------------------------------------------------------------------------
# 4.2 Multiple Groups Multiple Values
# ------------------------------------------------------------------------------
print("\n--- 4.2 Multiple Groups, Multiple Values ---")

if dataset_name == "superstore":
    numeric_cols = ["sales", "profit", "quantity"]
    result_mv = df.groupby("category")[numeric_cols].mean()
else:
    numeric_cols = ["fare", "age"]
    result_mv = df.groupby("pclass")[numeric_cols].mean()

print(f"Mean of multiple columns grouped by single column:")
print(result_mv.round(2))

# ------------------------------------------------------------------------------
# 4.3 as_index Parameter
# ------------------------------------------------------------------------------
print("\n--- 4.3 as_index=False (Returns Flat DataFrame Directly) ---")

# as_index=False avoids reset_index() step
result_flat_direct = df.groupby(
    group_cols,
    as_index=False
)[value_col].mean()

print("groupby with as_index=False:")
print(result_flat_direct)
print(f"Type: {type(result_flat_direct)}")
print("Same as groupby().agg().reset_index() in one step")


# ==============================================================================
# SECTION 5: MULTIPLE AGGREGATIONS WITH agg()
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 5: MULTIPLE AGGREGATIONS WITH agg()")
print("=" * 70)

# ------------------------------------------------------------------------------
# 5.1 agg() with a List of Functions
# ------------------------------------------------------------------------------
print("\n--- 5.1 agg() with List of Functions ---")

if dataset_name == "superstore":
    agg_result = df.groupby("category")["sales"].agg(
        ["sum", "mean", "median", "std", "count", "min", "max"]
    )
else:
    agg_result = df.groupby("pclass")["fare"].agg(
        ["sum", "mean", "median", "std", "count", "min", "max"]
    )

print("Multiple aggregations with agg():")
print(agg_result.round(2))

# ------------------------------------------------------------------------------
# 5.2 Named Aggregations (Most Production-Friendly Pattern)
# ------------------------------------------------------------------------------
print("\n--- 5.2 Named Aggregations ---")

explanation = """
Named aggregations use a dictionary to specify:
  output_column_name = pd.NamedAgg(column="source_col", aggfunc="function")

Or the simpler shorthand:
  output_column_name = ("source_col", "function")

This avoids MultiIndex columns and produces clean, readable output.
"""
print(explanation)

if dataset_name == "superstore":
    named_agg = df.groupby("category").agg(
        total_sales    = ("sales",    "sum"),
        avg_sales      = ("sales",    "mean"),
        total_profit   = ("profit",   "sum"),
        avg_profit     = ("profit",   "mean"),
        total_orders   = ("order_id", "count"),
        avg_quantity   = ("quantity", "mean"),
        max_discount   = ("discount", "max"),
    ).round(2)
else:
    named_agg = df.groupby("pclass").agg(
        passenger_count = ("survived",  "count"),
        survival_rate   = ("survived",  "mean"),
        avg_fare        = ("fare",      "mean"),
        max_fare        = ("fare",      "max"),
        avg_age         = ("age",       "mean"),
        min_age         = ("age",       "min"),
    ).round(2)

named_agg = named_agg.reset_index()
print("Named aggregation result:")
print(named_agg)
print(f"\nColumn names are clean: {list(named_agg.columns)}")
print("No MultiIndex, no flattening required")

# ------------------------------------------------------------------------------
# 5.3 Custom Aggregation Functions
# ------------------------------------------------------------------------------
print("\n--- 5.3 Custom Aggregation Functions ---")

# Define custom aggregation functions
def coefficient_of_variation(series):
    """CV = std / mean - measures relative variability."""
    mean = series.mean()
    return series.std() / mean if mean != 0 else np.nan

def top_10_pct_mean(series):
    """Mean of top 10% values."""
    threshold = series.quantile(0.9)
    return series[series >= threshold].mean()

def iqr(series):
    """Interquartile range."""
    return series.quantile(0.75) - series.quantile(0.25)

if dataset_name == "superstore":
    custom_agg = df.groupby("category")["sales"].agg(
        cv            = coefficient_of_variation,
        top_10pct_avg = top_10_pct_mean,
        iqr_range     = iqr,
    ).round(2)
else:
    custom_agg = df.groupby("pclass")["fare"].agg(
        cv            = coefficient_of_variation,
        top_10pct_avg = top_10_pct_mean,
        iqr_range     = iqr,
    ).round(2)

print("Custom aggregation functions:")
print(custom_agg)

# ------------------------------------------------------------------------------
# 5.4 Different Aggregations for Different Columns
# ------------------------------------------------------------------------------
print("\n--- 5.4 Different Aggregations per Column ---")

if dataset_name == "superstore":
    mixed_agg = df.groupby("category").agg({
        "sales":    ["sum", "mean"],
        "profit":   ["sum", "mean"],
        "quantity": "sum",
        "discount": "max",
    })
    # Flatten MultiIndex columns
    mixed_agg.columns = ["_".join(c).strip() for c in mixed_agg.columns]
    mixed_agg = mixed_agg.reset_index()
else:
    mixed_agg = df.groupby("pclass").agg({
        "fare":     ["mean", "max"],
        "age":      ["mean", "min"],
        "survived": "sum",
    })
    mixed_agg.columns = ["_".join(c).strip() for c in mixed_agg.columns]
    mixed_agg = mixed_agg.reset_index()

print("Different aggregations per column:")
print(mixed_agg.round(2))


# ==============================================================================
# SECTION 6: TRANSFORM - GROUP STATS BACK TO ORIGINAL ROWS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 6: TRANSFORM - GROUP STATS BACK TO ORIGINAL ROWS")
print("=" * 70)

# ------------------------------------------------------------------------------
# 6.1 What transform Does
# ------------------------------------------------------------------------------
print("\n--- 6.1 What transform Does ---")

explanation = """
agg()       returns one row per group (reduces data)
transform() returns one value per ORIGINAL row (same length as input)

transform is used for:
  - Adding group-level statistics as new columns
  - Normalizing within groups (z-score by group)
  - Filling missing values with group statistics
  - Feature engineering for machine learning
"""
print(explanation)

if dataset_name == "superstore":
    demo_df = df[["category", "sales", "profit"]].copy()

    # Add group mean as a new column
    demo_df["category_avg_sales"] = df.groupby("category")["sales"].transform("mean")
    demo_df["sales_vs_avg"]       = demo_df["sales"] - demo_df["category_avg_sales"]
    demo_df["pct_of_avg"]         = (demo_df["sales"] / demo_df["category_avg_sales"] * 100).round(1)

    print("Original rows with group statistics added via transform:")
    print(demo_df.head(10).round(2))

    # Verify transform preserved shape
    assert len(demo_df) == len(df), "transform changed shape - something is wrong"
    print(f"\nShape preserved: {demo_df.shape} == {df.shape}")

else:
    demo_df = df[["pclass", "sex", "fare", "age"]].copy()

    # Add group mean as a new column
    demo_df["class_avg_fare"] = df.groupby("pclass")["fare"].transform("mean")
    demo_df["fare_vs_avg"]    = (demo_df["fare"] - demo_df["class_avg_fare"]).round(2)

    print("Original rows with group mean fare added via transform:")
    print(demo_df.head(10).round(2))
    assert len(demo_df) == len(df), "transform changed shape"
    print(f"\nShape preserved: {demo_df.shape}")

# ------------------------------------------------------------------------------
# 6.2 Within-Group Normalization (Z-Score)
# ------------------------------------------------------------------------------
print("\n--- 6.2 Within-Group Normalization (Z-Score) ---")

def zscore(series):
    """Standardize a series to zero mean, unit variance."""
    mean = series.mean()
    std  = series.std()
    return (series - mean) / std if std > 0 else series * 0

if dataset_name == "superstore":
    df["sales_zscore_within_category"] = df.groupby("category")["sales"].transform(zscore)
    print("Z-score of sales within each category:")
    print(df.groupby("category")["sales_zscore_within_category"].agg(
        ["mean", "std", "min", "max"]
    ).round(4))
    print("\nMean ~0 and std ~1 within each group confirms correct normalization")

else:
    df["fare_zscore_within_class"] = df.groupby("pclass")["fare"].transform(zscore)
    print("Z-score of fare within each passenger class:")
    print(df.groupby("pclass")["fare_zscore_within_class"].agg(
        ["mean", "std", "min", "max"]
    ).round(4))

# ------------------------------------------------------------------------------
# 6.3 Filling Missing Values with Group Statistics
# ------------------------------------------------------------------------------
print("\n--- 6.3 Filling Missing Values with Group Statistics ---")

if dataset_name == "titanic":
    print(f"Missing Age values: {df['age'].isnull().sum()}")

    df["age_filled"] = df.groupby(["pclass", "sex"])["age"].transform(
        lambda x: x.fillna(x.median())
    )
    # Fill any remaining with overall median
    df["age_filled"] = df["age_filled"].fillna(df["age_filled"].median())

    print(f"Missing Age after group fill: {df['age_filled'].isnull().sum()}")
    print("\nGroup medians used for filling:")
    print(df.groupby(["pclass", "sex"])["age"].median().round(1))


# ==============================================================================
# SECTION 7: FILTER - KEEP OR DROP ENTIRE GROUPS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 7: FILTER - KEEP OR DROP ENTIRE GROUPS")
print("=" * 70)

# ------------------------------------------------------------------------------
# 7.1 What filter Does
# ------------------------------------------------------------------------------
print("\n--- 7.1 What filter Does ---")

explanation = """
filter() applies a function to each group.
If the function returns True, the group is kept.
If it returns False, ALL rows in that group are removed.

Use cases:
  - Remove groups that are too small (not enough data)
  - Remove groups where the mean does not meet a threshold
  - Quality control: discard groups with too many missing values
"""
print(explanation)

if dataset_name == "superstore":
    # Keep only categories with total sales above threshold
    threshold = df.groupby("category")["sales"].sum().mean()
    print(f"Mean total sales across categories: {threshold:.2f}")

    df_filtered_groups = df.groupby("category").filter(
        lambda g: g["sales"].sum() > threshold
    )

    print(f"\nRows before filter: {len(df)}")
    print(f"Rows after filter:  {len(df_filtered_groups)}")
    print(f"Categories kept:    {df_filtered_groups['category'].unique()}")

else:
    # Keep only passenger classes with more than 200 passengers
    size_threshold = 200
    df_filtered_groups = df.groupby("pclass").filter(
        lambda g: len(g) > size_threshold
    )
    group_sizes = df.groupby("pclass").size()
    print(f"Group sizes:\n{group_sizes}")
    print(f"\nKeeping only groups with more than {size_threshold} passengers")
    print(f"Rows before filter: {len(df)}")
    print(f"Rows after filter:  {len(df_filtered_groups)}")
    print(f"Classes kept: {sorted(df_filtered_groups['pclass'].unique())}")

# ------------------------------------------------------------------------------
# 7.2 filter vs Boolean Mask
# ------------------------------------------------------------------------------
print("\n--- 7.2 filter vs Boolean Mask ---")

print("filter removes ALL rows in groups that fail the condition.")
print("Boolean mask removes individual rows regardless of group.")
print("\nUse filter when group membership determines inclusion/exclusion.")
print("Use boolean mask when individual row values determine inclusion.")


# ==============================================================================
# SECTION 8: GROUPBY WITH CUSTOM FUNCTIONS USING apply()
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 8: GROUPBY WITH apply()")
print("=" * 70)

# ------------------------------------------------------------------------------
# 8.1 apply() for Complex Per-Group Logic
# ------------------------------------------------------------------------------
print("\n--- 8.1 apply() for Complex Per-Group Logic ---")

explanation = """
apply() passes each group as a DataFrame to a function.
The function can return:
  - A scalar (result is a Series, one value per group)
  - A Series (result is a DataFrame)
  - A DataFrame (result is a larger DataFrame)

Use apply() when agg() and transform() are not flexible enough.
Note: apply() is slower than agg() and transform().
Prefer agg/transform when possible.
"""
print(explanation)

if dataset_name == "superstore":
    def group_summary(group_df):
        """Return a summary Series for each group."""
        return pd.Series({
            "total_sales":      group_df["sales"].sum(),
            "avg_profit_margin": (group_df["profit"] / group_df["sales"]).mean(),
            "high_value_orders": (group_df["sales"] > 1000).sum(),
            "order_count":       len(group_df),
        })

    summary = df.groupby("category").apply(group_summary, include_groups=False)
    print("Per-group summary using apply():")
    print(summary.round(4))

else:
    def group_summary(group_df):
        return pd.Series({
            "count":            len(group_df),
            "survival_rate":    group_df["survived"].mean(),
            "avg_fare":         group_df["fare"].mean(),
            "pct_female":       (group_df["sex"] == "female").mean(),
            "missing_age_pct":  group_df["age"].isnull().mean(),
        })

    summary = df.groupby("pclass").apply(group_summary, include_groups=False)
    print("Per-group summary using apply():")
    print(summary.round(4))

# ------------------------------------------------------------------------------
# 8.2 apply() for Top N per Group
# ------------------------------------------------------------------------------
print("\n--- 8.2 Top N Rows Per Group ---")

if dataset_name == "superstore":
    def top_n(group_df, n=3, col="sales"):
        return group_df.nlargest(n, col)

    top3_per_category = df.groupby(
        "category", group_keys=False
    ).apply(top_n, n=3, col="sales")

    print("Top 3 sales orders per category:")
    print(top3_per_category[["category", "sales", "profit"]].head(12))

else:
    top3_per_class = (
        df.sort_values("fare", ascending=False)
        .groupby("pclass", group_keys=False)
        .head(3)
        .sort_values(["pclass", "fare"], ascending=[True, False])
    )

    print("Top 3 fares per passenger class:")
    print(top3_per_class[["pclass", "name", "fare", "survived"]].head(9))


# ==============================================================================
# SECTION 9: PIVOT TABLES
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 9: PIVOT TABLES")
print("=" * 70)

# ------------------------------------------------------------------------------
# 9.1 What is a Pivot Table
# ------------------------------------------------------------------------------
print("\n--- 9.1 What is a Pivot Table ---")

explanation = """
A pivot table summarizes data by crossing two categorical variables.
  - rows    = one categorical column
  - columns = another categorical column
  - values  = numeric column to aggregate

It is equivalent to groupby with two columns, then unstacking.
Useful for building cross-tabulation reports.
"""
print(explanation)

if dataset_name == "superstore":
    pivot = pd.pivot_table(
        df,
        values="sales",
        index="category",
        columns="region",
        aggfunc="sum",
        fill_value=0
    )
    print("Pivot table: Total Sales by Category and Region")
    print(pivot.round(2))

    # Add margins (row and column totals)
    pivot_with_totals = pd.pivot_table(
        df,
        values="sales",
        index="category",
        columns="region",
        aggfunc="sum",
        fill_value=0,
        margins=True,
        margins_name="Total"
    )
    print("\nWith row and column totals (margins=True):")
    print(pivot_with_totals.round(2))

else:
    pivot = pd.pivot_table(
        df,
        values="survived",
        index="pclass",
        columns="sex",
        aggfunc="mean",
        fill_value=0
    )
    print("Pivot table: Survival Rate by Pclass and Sex")
    print(pivot.round(3))

    pivot_with_totals = pd.pivot_table(
        df,
        values="survived",
        index="pclass",
        columns="sex",
        aggfunc="mean",
        fill_value=0,
        margins=True,
        margins_name="Overall"
    )
    print("\nWith overall rates (margins=True):")
    print(pivot_with_totals.round(3))

# ------------------------------------------------------------------------------
# 9.2 Multiple Value Aggregations in Pivot Table
# ------------------------------------------------------------------------------
print("\n--- 9.2 Multiple Aggregations in Pivot Table ---")

if dataset_name == "superstore":
    pivot_multi = pd.pivot_table(
        df,
        values=["sales", "profit"],
        index="category",
        columns="region",
        aggfunc="mean",
        fill_value=0
    )
    print("Multiple values in pivot table:")
    print(pivot_multi.round(2))

else:
    pivot_multi = pd.pivot_table(
        df,
        values=["survived", "fare"],
        index="pclass",
        columns="sex",
        aggfunc="mean",
        fill_value=0
    )
    print("Multiple values in pivot table:")
    print(pivot_multi.round(2))

# ------------------------------------------------------------------------------
# 9.3 crosstab - Frequency Tables
# ------------------------------------------------------------------------------
print("\n--- 9.3 crosstab - Frequency Tables ---")

if dataset_name == "titanic":
    # Count of passengers by class and gender
    ct = pd.crosstab(df["pclass"], df["sex"])
    print("Count of passengers by Pclass and Sex:")
    print(ct)

    # With normalize to get proportions
    ct_pct = pd.crosstab(df["pclass"], df["sex"], normalize="index")
    print("\nRow-wise proportions:")
    print(ct_pct.round(3))

else:
    ct = pd.crosstab(df["category"], df["region"])
    print("Count of orders by Category and Region:")
    print(ct)

    ct_pct = pd.crosstab(df["category"], df["region"], normalize="index")
    print("\nRow-wise proportions:")
    print(ct_pct.round(3))


# ==============================================================================
# SECTION 10: REAL WORLD EXAMPLE - SALES ANALYSIS PIPELINE
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 10: REAL WORLD EXAMPLE - SALES ANALYSIS PIPELINE")
print("=" * 70)

if dataset_name == "superstore":
    print("Building a complete sales analysis report...")

    # Step 1: Category performance summary
    print("\nSTEP 1: Category Performance Summary")
    cat_summary = df.groupby("category").agg(
        total_sales      = ("sales",    "sum"),
        total_profit     = ("profit",   "sum"),
        avg_order_value  = ("sales",    "mean"),
        order_count      = ("order_id", "count"),
        avg_discount     = ("discount", "mean"),
    ).round(2)

    cat_summary["profit_margin_pct"] = (
        cat_summary["total_profit"] / cat_summary["total_sales"] * 100
    ).round(2)
    cat_summary["revenue_share_pct"] = (
        cat_summary["total_sales"] / cat_summary["total_sales"].sum() * 100
    ).round(2)
    cat_summary = cat_summary.reset_index()
    print(cat_summary)

    # Step 2: Regional performance
    print("\nSTEP 2: Regional Performance")
    region_summary = df.groupby("region").agg(
        total_sales    = ("sales",  "sum"),
        total_profit   = ("profit", "sum"),
        order_count    = ("sales",  "count"),
    ).round(2)
    region_summary["profit_margin_pct"] = (
        region_summary["total_profit"] / region_summary["total_sales"] * 100
    ).round(2)
    print(region_summary.sort_values("total_sales", ascending=False))

    # Step 3: Add group statistics back to original rows
    print("\nSTEP 3: Enriching original data with group stats")
    df["category_avg_sales"] = df.groupby("category")["sales"].transform("mean")
    df["is_above_category_avg"] = df["sales"] > df["category_avg_sales"]
    pct_above = df["is_above_category_avg"].mean() * 100
    print(f"  Orders above their category average: {pct_above:.1f}%")

    # Step 4: Find underperforming subcategories
    if "sub_category" in df.columns:
        print("\nSTEP 4: Underperforming Subcategories (negative profit)")
        subcat_profit = df.groupby("sub_category").agg(
            total_profit = ("profit", "sum"),
            order_count  = ("profit", "count"),
        ).reset_index()
        underperforming = subcat_profit[subcat_profit["total_profit"] < 0]
        print(underperforming.sort_values("total_profit"))

    # Step 5: Validation
    print("\nSTEP 5: Validation")
    total_from_groups = cat_summary["total_sales"].sum()
    total_from_df     = df["sales"].sum()
    match = abs(total_from_groups - total_from_df) < 0.01
    print(f"  Sum from groups: {total_from_groups:.2f}")
    print(f"  Sum from df:     {total_from_df:.2f}")
    print(f"  Totals match:    {match}")

else:
    print("Building Titanic survival analysis report...")

    print("\nSTEP 1: Survival by Class and Gender")
    survival_report = df.groupby(["pclass", "sex"]).agg(
        passenger_count = ("survived", "count"),
        survived_count  = ("survived", "sum"),
        survival_rate   = ("survived", "mean"),
        avg_fare        = ("fare",     "mean"),
        avg_age         = ("age",      "mean"),
    ).round(3).reset_index()
    print(survival_report)

    print("\nSTEP 2: Flag passengers above median fare in their class")
    df["class_median_fare"]   = df.groupby("pclass")["fare"].transform("median")
    df["above_class_median"]  = df["fare"] > df["class_median_fare"]

    enriched_summary = df.groupby(["pclass", "above_class_median"]).agg(
        count          = ("survived", "count"),
        survival_rate  = ("survived", "mean"),
    ).round(3)
    print("\nSurvival rate by class and whether fare is above class median:")
    print(enriched_summary)


# ==============================================================================
# SECTION 11: PERFORMANCE NOTES
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 11: PERFORMANCE NOTES")
print("=" * 70)

performance_notes = """
FASTEST TO SLOWEST for groupby operations:

1. Built-in agg functions: sum, mean, min, max, count, std
   - Implemented in Cython, very fast
   - df.groupby("col")["val"].sum()

2. Named agg with built-in functions:
   - df.groupby("col").agg(total=("val", "sum"))

3. transform with built-in functions:
   - df.groupby("col")["val"].transform("mean")

4. Custom functions in agg():
   - df.groupby("col")["val"].agg(my_func)
   - Slower because Python function is called per group

5. apply() with custom function:
   - Slowest but most flexible
   - Each group is passed as a full DataFrame
   - Avoid for large datasets when agg or transform can do the job

TIPS:
  - Use agg() with named aggregations for clean readable output
  - Use transform() when you need group stats on every row
  - Use apply() only when agg/transform cannot express the logic
  - Use as_index=False to skip reset_index() step
  - Pre-filter data before groupby to reduce group sizes
"""
print(performance_notes)


# ==============================================================================
# SECTION 12: COMMON PITFALLS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 12: COMMON PITFALLS")
print("=" * 70)

pitfalls = """
Pitfall 1: Forgetting that groupby result has MultiIndex
   - Grouping by two columns creates MultiIndex in result
   - Fix: use reset_index() or as_index=False

Pitfall 2: Using apply() where agg() would work
   - apply() is 10-100x slower than built-in agg functions
   - Fix: check if your logic can be expressed with agg()

Pitfall 3: NaN values in group keys
   - By default groupby drops rows where the group key is NaN
   - Fix: df.groupby("col", dropna=False) to include NaN as a group

Pitfall 4: MultiIndex columns after dict agg
   - df.groupby("col").agg({"a": ["mean", "std"]}) produces MultiIndex columns
   - Fix: flatten with "_".join(c) or use named aggregations

Pitfall 5: transform shape mismatch
   - transform function must return same-length Series as input group
   - Returning a scalar from transform is fine (broadcast to all rows)

Pitfall 6: Modifying the original DataFrame through a group
   - Never modify group_df inside apply() to change the original
   - Fix: work on copies and return the result

Pitfall 7: Expecting filter to work like a boolean mask
   - filter keeps or drops ENTIRE groups, not individual rows
   - Fix: use boolean mask for row-level filtering
"""
print(pitfalls)


# ==============================================================================
# SECTION 13: SUMMARY TABLE
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 13: SUMMARY TABLE")
print("=" * 70)

summary = """
Operation                          | Syntax
-----------------------------------|------------------------------------------
Group and sum                      | df.groupby("col")["val"].sum()
Group by multiple columns          | df.groupby(["c1", "c2"])["val"].mean()
Multiple aggregations              | df.groupby("col")["val"].agg(["sum","mean"])
Named aggregations                 | df.groupby("col").agg(name=("col","func"))
Different agg per column           | df.groupby("col").agg({"a":"sum","b":"mean"})
Custom aggregation function        | df.groupby("col")["val"].agg(my_func)
Add group stat to each row         | df.groupby("col")["val"].transform("mean")
Within-group normalization         | df.groupby("col")["val"].transform(zscore)
Keep/drop entire groups            | df.groupby("col").filter(lambda g: len(g)>10)
Complex per-group logic            | df.groupby("col").apply(my_func)
Top N per group                    | df.groupby("col",group_keys=False).apply(top_n)
Flat result without reset_index    | df.groupby("col", as_index=False)["val"].sum()
Include NaN group keys             | df.groupby("col", dropna=False)
Pivot table                        | pd.pivot_table(df, values, index, columns, aggfunc)
Frequency cross-tabulation         | pd.crosstab(df["c1"], df["c2"])
Inspect group sizes                | df.groupby("col").size()
Get one group                      | df.groupby("col").get_group("value")
"""
print(summary)


# ==============================================================================
# SECTION 14: PRACTICE EXERCISES
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 14: PRACTICE EXERCISES")
print("=" * 70)

if dataset_name == "titanic":
    print("Exercise 1: Survival rate by embarkation port")
    ex1 = df.groupby("embarked").agg(
        count         = ("survived", "count"),
        survival_rate = ("survived", "mean"),
        avg_fare      = ("fare",     "mean"),
    ).round(3)
    print(ex1)

    print("\nExercise 2: Add family size and group survival stats")
    df["family_size"] = df["sibsp"] + df["parch"] + 1
    df["family_category"] = pd.cut(
        df["family_size"],
        bins=[0, 1, 4, 20],
        labels=["alone", "small_family", "large_family"]
    )
    ex2 = df.groupby("family_category").agg(
        count         = ("survived", "count"),
        survival_rate = ("survived", "mean"),
    ).round(3)
    print(ex2)

    print("\nExercise 3: Keep only embarkation ports with survival rate above 0.35")
    ex3 = df.groupby("embarked").filter(
        lambda g: g["survived"].mean() > 0.35
    )
    print(f"  Ports kept: {ex3['embarked'].unique()}")
    print(f"  Rows retained: {len(ex3)}")

    print("\nExercise 4: Pivot table of survival by class and embarkation port")
    ex4 = pd.pivot_table(
        df,
        values="survived",
        index="pclass",
        columns="embarked",
        aggfunc="mean",
        fill_value=0,
        margins=True
    )
    print(ex4.round(3))

else:
    print("Exercise 1: Profit margin by sub-category")
    if "sub_category" in df.columns:
        ex1 = df.groupby("sub_category").agg(
            total_sales  = ("sales",  "sum"),
            total_profit = ("profit", "sum"),
        )
        ex1["margin_pct"] = (ex1["total_profit"] / ex1["total_sales"] * 100).round(2)
        print(ex1.sort_values("margin_pct"))

    print("\nExercise 2: Pivot table of average sales by category and region")
    ex2 = pd.pivot_table(
        df,
        values="sales",
        index="category",
        columns="region",
        aggfunc="mean",
        fill_value=0,
        margins=True
    )
    print(ex2.round(2))


# ==============================================================================
# SECTION 15: KEY TAKEAWAYS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 15: KEY TAKEAWAYS")
print("=" * 70)

takeaways = """
1.  groupby follows split-apply-combine: split into groups, apply function,
    combine results
2.  agg() reduces each group to one row - use for summary statistics
3.  transform() keeps the original shape - use for adding group stats per row
4.  filter() keeps or drops entire groups - use for group-level quality control
5.  apply() is the most flexible but slowest - use only when agg/transform
    cannot express the logic
6.  Named aggregations produce clean column names with no MultiIndex
7.  Use as_index=False to get a flat DataFrame without calling reset_index()
8.  Pivot tables are a readable alternative to two-column groupby
9.  Use dropna=False in groupby to include NaN group keys
10. Always validate group totals against the full DataFrame after aggregation
"""
print(takeaways)


# ==============================================================================
# NEXT LESSON PREVIEW
# ==============================================================================
print("\n" + "=" * 70)
print("NEXT LESSON PREVIEW")
print("=" * 70)

preview = """
Lesson 24: Sorting and Ranking

You will learn:
- Sorting by single and multiple columns
- Ascending and descending sort
- Handling NaN values in sort
- Ranking rows with different ranking methods
- Sorting and ranking within groups
- Finding top N and bottom N records
- Stable sort and its importance in production
"""
print(preview)

print("\n" + "=" * 70)
print("END OF LESSON 23")
print("=" * 70)
print("\nYou can now aggregate data with groupby, transform group stats back")
print("to original rows, filter entire groups, and build pivot tables.")