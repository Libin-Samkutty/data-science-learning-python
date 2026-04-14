"""
LESSON 22: RENAMING COLUMNS AND WORKING WITH INDEX
================================================================================

What You Will Learn:
- Renaming columns with rename() and set_axis()
- Bulk renaming patterns (lowercase, strip spaces, replace characters)
- Setting and resetting the index
- Reindexing DataFrames
- Multi-level (hierarchical) index basics
- Index alignment behavior in operations
- Best practices for column and index naming in production

Real World Usage:
- Standardizing column names from different data sources
- Setting a meaningful primary key as the index for fast lookup
- Aligning data from multiple sources before merging
- Preparing column names for SQL export or ML frameworks
- Building consistent naming conventions across pipelines

Dataset Used:
World Bank Country Data (public, no login required)
URL: https://raw.githubusercontent.com/cs109/2014_data/master/countries.csv
Fallback: Titanic dataset
https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv

================================================================================
"""

import numpy as np
import pandas as pd

print("=" * 70)
print("LESSON 22: RENAMING COLUMNS AND WORKING WITH INDEX")
print("=" * 70)


# ==============================================================================
# SECTION 1: LOAD DATASET
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 1: LOAD DATASET")
print("=" * 70)

primary_url  = "https://raw.githubusercontent.com/cs109/2014_data/master/countries.csv"
fallback_url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

def load_dataset():
    try:
        print("Loading World Bank country dataset from:")
        print(primary_url)
        df = pd.read_csv(primary_url)
        print("Primary dataset loaded successfully.")
        return df, "countries"
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


# ==============================================================================
# SECTION 2: WHY COLUMN NAMING MATTERS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 2: WHY COLUMN NAMING MATTERS")
print("=" * 70)

explanation = """
In production pipelines, column names matter for several reasons:

1. CONSISTENCY
   Different data sources use different naming conventions.
   Standardizing early prevents bugs downstream.

2. READABILITY
   'monthly_revenue_usd' is clearer than 'MthlyRevUSD' or 'col_3'

3. COMPATIBILITY
   - SQL: column names cannot have spaces
   - Python: dot notation fails with spaces or special characters
   - ML frameworks: expect specific column name formats

4. DEBUGGING
   Clear names make it easier to trace errors through a pipeline

COMMON CONVENTIONS:
   snake_case       monthly_revenue    (most common in Python/SQL)
   camelCase        monthlyRevenue     (JavaScript style)
   PascalCase       MonthlyRevenue     (rare in data work)
   ALL_CAPS         MONTHLY_REVENUE    (SQL constants)
"""
print(explanation)

# Create a messy column name example for demonstration
messy_df = pd.DataFrame({
    "First Name ": ["Alice", "Bob", "Charlie"],
    "Last Name":   ["Smith", "Jones", "Brown"],
    " Age(Years)": [25, 30, 35],
    "Monthly Revenue (USD)": [4500, 6200, 5800],
    "IsActive?": [True, True, False],
    "Dept.Code": ["ENG", "MKT", "FIN"]
})

print("\nExample: Messy column names from a raw data export:")
print("Columns:", list(messy_df.columns))
print(messy_df)


# ==============================================================================
# SECTION 3: RENAMING SPECIFIC COLUMNS WITH rename()
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 3: RENAMING SPECIFIC COLUMNS WITH rename()")
print("=" * 70)

# ------------------------------------------------------------------------------
# 3.1 Rename Using a Dictionary
# ------------------------------------------------------------------------------
print("\n--- 3.1 Rename Using a Dictionary ---")

df = df_raw.copy()

print("Columns before renaming:")
print(list(df.columns))

# rename() with a dictionary: {old_name: new_name}
# Only the columns listed are renamed, others remain unchanged
if dataset_name == "titanic":
    df_renamed = df.rename(columns={
        "PassengerId": "passenger_id",
        "Survived":    "survived",
        "Pclass":      "passenger_class",
        "Name":        "full_name",
        "Sex":         "gender",
        "Age":         "age",
        "SibSp":       "siblings_spouses",
        "Parch":       "parents_children",
        "Ticket":      "ticket_number",
        "Fare":        "fare_usd",
        "Cabin":       "cabin_code",
        "Embarked":    "embarkation_port"
    })
else:
    df_renamed = df.rename(columns={
        col: col.lower().replace(" ", "_")
        for col in df.columns
    })

print("\nColumns after renaming:")
print(list(df_renamed.columns))
print("\nFirst 3 rows:")
print(df_renamed.head(3))

# Key behavior: rename() returns a new DataFrame
# Original is unchanged unless inplace=True (avoid inplace in pipelines)
print("\nOriginal columns unchanged:")
print(list(df.columns)[:4], "...")

# ------------------------------------------------------------------------------
# 3.2 Rename Using a Function
# ------------------------------------------------------------------------------
print("\n--- 3.2 Rename Using a Function ---")

# Pass a function to apply to every column name
df_lower = df.rename(columns=str.lower)
print("All columns lowercased:")
print(list(df_lower.columns))

# Custom function
def clean_col_name(name):
    """Standardize column name to snake_case."""
    return (
        name.strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
            .replace(".", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("?", "")
    )

df_messy_cleaned = messy_df.rename(columns=clean_col_name)
print("\nMessy columns after clean_col_name function:")
print("Before:", list(messy_df.columns))
print("After: ", list(df_messy_cleaned.columns))
print(df_messy_cleaned)

# ------------------------------------------------------------------------------
# 3.3 Rename Index Labels
# ------------------------------------------------------------------------------
print("\n--- 3.3 Rename Index Labels ---")

# rename() also works on the index
small_df = df.head(5).copy()
small_df = small_df.rename(index={
    0: "row_a",
    1: "row_b",
    2: "row_c",
    3: "row_d",
    4: "row_e"
})
print("DataFrame with renamed index labels:")
print(small_df[df.columns[:3]])


# ==============================================================================
# SECTION 4: BULK RENAMING PATTERNS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 4: BULK RENAMING PATTERNS")
print("=" * 70)

# ------------------------------------------------------------------------------
# 4.1 Direct Assignment to df.columns
# ------------------------------------------------------------------------------
print("\n--- 4.1 Direct Assignment to df.columns ---")

# You can replace ALL column names at once by assigning a list
# This is fast but fragile: order must match exactly
df_bulk = df.copy()
print(f"Number of columns: {len(df_bulk.columns)}")

# Safest bulk rename: use list comprehension on existing names
df_bulk.columns = [col.lower() for col in df_bulk.columns]
print("After bulk lowercase:")
print(list(df_bulk.columns))

# ------------------------------------------------------------------------------
# 4.2 Using set_axis()
# ------------------------------------------------------------------------------
print("\n--- 4.2 Using set_axis() ---")

# set_axis is cleaner than direct column assignment in pipelines
# because it can be chained

df_axis = df.copy()
new_columns = [col.lower().replace(" ", "_") for col in df_axis.columns]

df_axis = df_axis.set_axis(new_columns, axis="columns")
print("After set_axis with new column list:")
print(list(df_axis.columns))

# Also works on index axis
df_axis_indexed = df.head(3).set_axis(["alpha", "beta", "gamma"], axis="index")
print("\nAfter set_axis on index:")
print(df_axis_indexed[df.columns[:3]])

# ------------------------------------------------------------------------------
# 4.3 Production-Grade Column Standardization Function
# ------------------------------------------------------------------------------
print("\n--- 4.3 Production-Grade Column Standardization ---")

import re

def standardize_columns(data):
    """
    Standardize all column names to snake_case.

    Rules applied:
      - Strip leading and trailing whitespace
      - Replace spaces and hyphens with underscores
      - Remove special characters except underscores
      - Collapse multiple underscores into one
      - Convert to lowercase
      - Ensure name does not start with a digit

    Parameters
    ----------
    data : pd.DataFrame

    Returns
    -------
    pd.DataFrame with standardized column names
    """
    def to_snake(name):
        name = str(name).strip()
        # Replace spaces, hyphens, dots with underscore
        name = re.sub(r"[\s\-\.]+", "_", name)
        # Remove characters that are not alphanumeric or underscore
        name = re.sub(r"[^\w]", "", name)
        # Collapse multiple underscores
        name = re.sub(r"_+", "_", name)
        # Strip trailing/leading underscores
        name = name.strip("_")
        # Lowercase
        name = name.lower()
        # Prefix with col_ if starts with digit
        if name and name[0].isdigit():
            name = "col_" + name
        return name

    new_cols = [to_snake(col) for col in data.columns]

    # Detect and handle duplicate column names after standardization
    seen = {}
    final_cols = []
    for col in new_cols:
        if col in seen:
            seen[col] += 1
            final_cols.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            final_cols.append(col)

    result = data.copy()
    result.columns = final_cols
    return result


# Test on messy columns
print("Testing standardize_columns on messy DataFrame:")
print("Original columns:", list(messy_df.columns))
df_standardized = standardize_columns(messy_df)
print("Standardized:    ", list(df_standardized.columns))
print(df_standardized)

# Test duplicate handling
df_dup_cols = pd.DataFrame([[1, 2, 3]], columns=["Name", "name", "NAME"])
df_dup_fixed = standardize_columns(df_dup_cols)
print("\nDuplicate column name handling:")
print("Before:", list(df_dup_cols.columns))
print("After: ", list(df_dup_fixed.columns))


# ==============================================================================
# SECTION 5: SETTING THE INDEX
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 5: SETTING THE INDEX")
print("=" * 70)

# ------------------------------------------------------------------------------
# 5.1 What is the Index?
# ------------------------------------------------------------------------------
print("\n--- 5.1 What is the Index? ---")

explanation = """
The index is the row label of a DataFrame.

Default index: 0, 1, 2, 3, ... (RangeIndex)

Setting a meaningful index:
  - Makes .loc lookups faster (hash-based)
  - Makes the data self-describing
  - Required for time series analysis (DatetimeIndex)
  - Required for correct alignment when combining DataFrames

When NOT to set a custom index:
  - When you need simple integer-based row access
  - When the candidate key has duplicates or missing values
  - When you plan to use .iloc extensively
"""
print(explanation)

df = df_raw.copy()

print("Default RangeIndex:")
print(f"  Type: {type(df.index)}")
print(f"  Values (first 5): {df.index[:5].tolist()}")

# ------------------------------------------------------------------------------
# 5.2 set_index()
# ------------------------------------------------------------------------------
print("\n--- 5.2 set_index() ---")

if dataset_name == "titanic":
    # Set PassengerId as index
    df_idx = df.set_index("PassengerId")
    print("After set_index('PassengerId'):")
    print(f"  Index type: {type(df_idx.index)}")
    print(f"  Index name: {df_idx.index.name}")
    print(f"  First 5 index values: {df_idx.index[:5].tolist()}")
    print(f"  Columns: {list(df_idx.columns)}")
    print(df_idx.head(3))

    # PassengerId is now the index, no longer a column
    print(f"\nPassengerId still in columns: {'PassengerId' in df_idx.columns}")

    # Fast label-based lookup
    print("\nLookup passenger 5 with .loc[5]:")
    print(df_idx.loc[5][["Name", "Age", "Survived"]])

    # Lookup multiple passengers
    print("\nLookup passengers 10, 20, 30 with .loc[[10, 20, 30]]:")
    print(df_idx.loc[[10, 20, 30]][["Name", "Age", "Survived"]])

else:
    first_col = df.columns[0]
    df_idx = df.set_index(first_col)
    print(f"After set_index('{first_col}'):")
    print(df_idx.head(3))

# ------------------------------------------------------------------------------
# 5.3 keep vs drop the column when setting index
# ------------------------------------------------------------------------------
print("\n--- 5.3 Keeping the Column When Setting Index ---")

if dataset_name == "titanic":
    # Default: drop=True removes the column from the DataFrame
    df_drop_col = df.set_index("PassengerId", drop=True)
    print("set_index with drop=True (default):")
    print(f"  'PassengerId' in columns: {'PassengerId' in df_drop_col.columns}")

    # drop=False keeps the column AND sets it as index
    df_keep_col = df.set_index("PassengerId", drop=False)
    print("\nset_index with drop=False:")
    print(f"  'PassengerId' in columns: {'PassengerId' in df_keep_col.columns}")
    print(f"  Index name: {df_keep_col.index.name}")

# ------------------------------------------------------------------------------
# 5.4 Setting a String or Date Column as Index
# ------------------------------------------------------------------------------
print("\n--- 5.4 Setting a String or Date Column as Index ---")

if dataset_name == "titanic":
    # Setting Name as index (not ideal due to duplicates but demonstrates concept)
    df_name_idx = df.copy()
    df_name_idx = df_name_idx.drop_duplicates(subset=["Name"])
    df_name_idx = df_name_idx.set_index("Name")
    print("After set_index('Name'):")
    print(df_name_idx.head(3)[["Age", "Sex", "Pclass", "Survived"]])

    # Look up by name
    name_lookup = "Braund, Mr. Owen Harris"
    print(f"\nLookup by name '{name_lookup}':")
    print(df_name_idx.loc[name_lookup][["Age", "Sex", "Survived"]])

# Demo: DateTime index (important for time series)
print("\nDateTime index demonstration:")
date_range = pd.date_range("2024-01-01", periods=10, freq="D")
ts_df = pd.DataFrame({
    "sales":       np.random.randint(100, 500, 10),
    "customers":   np.random.randint(10, 100, 10),
    "region":      np.random.choice(["North", "South", "East"], 10)
}, index=date_range)

print(ts_df)
print(f"\nIndex type: {type(ts_df.index)}")

# Date-based lookup
print("\nLookup by date range:")
print(ts_df.loc["2024-01-03":"2024-01-06"])


# ==============================================================================
# SECTION 6: RESETTING THE INDEX
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 6: RESETTING THE INDEX")
print("=" * 70)

# ------------------------------------------------------------------------------
# 6.1 reset_index() Basics
# ------------------------------------------------------------------------------
print("\n--- 6.1 reset_index() Basics ---")

if dataset_name == "titanic":
    df_idx = df.set_index("PassengerId")
    print("Before reset_index:")
    print(f"  Index: {df_idx.index[:3].tolist()}")
    print(f"  Columns: {list(df_idx.columns)[:4]}")

    # Reset brings index back as a column
    df_reset = df_idx.reset_index()
    print("\nAfter reset_index():")
    print(f"  Index: {df_reset.index[:3].tolist()}")
    print(f"  Columns: {list(df_reset.columns)[:4]}")
    print(f"  PassengerId back as column: {'PassengerId' in df_reset.columns}")

    # drop=True discards the index instead of making it a column
    df_reset_drop = df_idx.reset_index(drop=True)
    print("\nAfter reset_index(drop=True):")
    print(f"  Index: {df_reset_drop.index[:3].tolist()}")
    print(f"  PassengerId kept as column: {'PassengerId' in df_reset_drop.columns}")

# ------------------------------------------------------------------------------
# 6.2 When to Use reset_index
# ------------------------------------------------------------------------------
print("\n--- 6.2 When to Use reset_index ---")

when_to_use = """
Reset the index when:
  - After filtering, index has gaps (0, 2, 5, 8, ...) and you need 0, 1, 2, ...
  - After groupby, the group keys become a MultiIndex that you want as columns
  - Before saving to CSV (avoid writing confusing non-default index)
  - Before passing to ML frameworks that expect a clean integer index
"""
print(when_to_use)

if dataset_name == "titanic":
    # Filtering creates gaps in index
    survivors = df[df["Survived"] == 1].copy()
    print(f"Survivors index (first 10): {survivors.index[:10].tolist()}")

    survivors_reset = survivors.reset_index(drop=True)
    print(f"After reset_index(drop=True): {survivors_reset.index[:10].tolist()}")


# ==============================================================================
# SECTION 7: REINDEXING
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 7: REINDEXING")
print("=" * 70)

# ------------------------------------------------------------------------------
# 7.1 reindex() Explained
# ------------------------------------------------------------------------------
print("\n--- 7.1 reindex() Explained ---")

explanation = """
reindex() changes the labels of rows or columns to match a new list.
Rows or columns not in the new list are dropped.
New labels not in the original data get NaN.

Use cases:
  - Align two DataFrames to the same index before combining
  - Ensure all expected columns are present (add NaN for missing)
  - Reorder columns or rows to a specific order
"""
print(explanation)

# Simple example: reindexing rows
sample = pd.DataFrame({
    "value": [10, 20, 30, 40, 50]
}, index=[0, 1, 2, 3, 4])

print("Original DataFrame:")
print(sample)

# Reindex to add new row labels (creates NaN for missing)
reindexed = sample.reindex([0, 1, 2, 5, 6])
print("\nAfter reindex([0, 1, 2, 5, 6]):")
print(reindexed)
print("Rows 5 and 6 did not exist - filled with NaN")

# Reindex with fill value
reindexed_filled = sample.reindex([0, 1, 2, 5, 6], fill_value=0)
print("\nAfter reindex with fill_value=0:")
print(reindexed_filled)

# ------------------------------------------------------------------------------
# 7.2 Reindexing Columns
# ------------------------------------------------------------------------------
print("\n--- 7.2 Reindexing Columns ---")

df_col_demo = df.head(3).copy()
original_cols = list(df_col_demo.columns)
print("Original columns:", original_cols)

# Reorder and select columns
desired_cols = ["Survived", "Name", "Age", "Sex", "Pclass", "NonExistentCol"]
df_reindexed_cols = df_col_demo.reindex(columns=desired_cols)
print("\nAfter reindex(columns=desired_cols):")
print("Columns:", list(df_reindexed_cols.columns))
print(df_reindexed_cols)
print("NonExistentCol filled with NaN as it was not in original")

# ------------------------------------------------------------------------------
# 7.3 Aligning Two DataFrames
# ------------------------------------------------------------------------------
print("\n--- 7.3 Aligning Two DataFrames ---")

# Scenario: two monthly sales tables with different months present
df_store_a = pd.DataFrame({
    "sales": [100, 200, 300, 400]
}, index=["Jan", "Feb", "Mar", "Apr"])

df_store_b = pd.DataFrame({
    "sales": [150, 180, 350]
}, index=["Jan", "Mar", "May"])

print("Store A sales:")
print(df_store_a)
print("\nStore B sales:")
print(df_store_b)

# Align both to same index (union of all months)
all_months = df_store_a.index.union(df_store_b.index)
df_a_aligned = df_store_a.reindex(all_months, fill_value=0)
df_b_aligned = df_store_b.reindex(all_months, fill_value=0)

print("\nAligned Store A:")
print(df_a_aligned)
print("\nAligned Store B:")
print(df_b_aligned)

# Now comparison is valid
df_combined = pd.DataFrame({
    "store_a": df_a_aligned["sales"],
    "store_b": df_b_aligned["sales"]
})
df_combined["difference"] = df_combined["store_a"] - df_combined["store_b"]
print("\nCombined comparison:")
print(df_combined)


# ==============================================================================
# SECTION 8: MULTI-LEVEL (HIERARCHICAL) INDEX
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 8: MULTI-LEVEL (HIERARCHICAL) INDEX")
print("=" * 70)

# ------------------------------------------------------------------------------
# 8.1 Creating a MultiIndex
# ------------------------------------------------------------------------------
print("\n--- 8.1 Creating a MultiIndex ---")

explanation = """
A MultiIndex (hierarchical index) allows multiple levels of labels.
This naturally represents structured data like:
  - Country > City
  - Year > Month
  - Region > Store > Product
"""
print(explanation)

# Create from arrays
arrays = [
    ["North", "North", "South", "South", "East", "East"],
    ["Q1", "Q2", "Q1", "Q2", "Q1", "Q2"]
]
multi_idx = pd.MultiIndex.from_arrays(arrays, names=["Region", "Quarter"])

sales_data = pd.DataFrame({
    "revenue": [1200, 1500, 900, 1100, 1400, 1600],
    "units":   [120, 150, 90, 110, 140, 160]
}, index=multi_idx)

print("DataFrame with MultiIndex (Region, Quarter):")
print(sales_data)
print(f"\nIndex type: {type(sales_data.index)}")
print(f"Index names: {sales_data.index.names}")
print(f"Index levels: {sales_data.index.levels}")

# ------------------------------------------------------------------------------
# 8.2 Selecting Data from MultiIndex
# ------------------------------------------------------------------------------
print("\n--- 8.2 Selecting Data from MultiIndex ---")

# Select all rows for a region
print("All North rows:")
print(sales_data.loc["North"])

# Select specific region and quarter
print("\nNorth Q1:")
print(sales_data.loc[("North", "Q1")])

# Select across first level
print("\nAll Q1 rows (using xs - cross-section):")
print(sales_data.xs("Q1", level="Quarter"))

# ------------------------------------------------------------------------------
# 8.3 MultiIndex from groupby (Common Pattern)
# ------------------------------------------------------------------------------
print("\n--- 8.3 MultiIndex from groupby (Common Pattern) ---")

if dataset_name == "titanic":
    # groupby creates MultiIndex in result
    grouped = df.groupby(["Pclass", "Sex"])["Survived"].agg(
        ["mean", "count", "sum"]
    )
    print("groupby result has MultiIndex:")
    print(grouped)
    print(f"\nIndex type: {type(grouped.index)}")

    # Reset to flat DataFrame
    grouped_flat = grouped.reset_index()
    print("\nAfter reset_index() - flat DataFrame:")
    print(grouped_flat)

# ------------------------------------------------------------------------------
# 8.4 Flattening MultiIndex Columns
# ------------------------------------------------------------------------------
print("\n--- 8.4 Flattening MultiIndex Columns ---")

if dataset_name == "titanic":
    # agg with multiple functions creates MultiIndex columns
    agg_df = df.groupby("Pclass")[["Age", "Fare"]].agg(["mean", "max"])
    print("MultiIndex columns from groupby agg:")
    print(agg_df)
    print(f"\nColumn type: {type(agg_df.columns)}")
    print(f"Column levels: {agg_df.columns.tolist()}")

    # Flatten MultiIndex columns to single level
    agg_df.columns = ["_".join(col).strip() for col in agg_df.columns.values]
    print("\nAfter flattening MultiIndex columns:")
    print(agg_df)
    print(f"Columns: {list(agg_df.columns)}")


# ==============================================================================
# SECTION 9: INDEX ALIGNMENT IN OPERATIONS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 9: INDEX ALIGNMENT IN OPERATIONS")
print("=" * 70)

# ------------------------------------------------------------------------------
# 9.1 Automatic Index Alignment
# ------------------------------------------------------------------------------
print("\n--- 9.1 Automatic Index Alignment ---")

explanation = """
When combining two Series or DataFrames, Pandas aligns on index labels.
This is a powerful feature but can produce unexpected NaN values
if the indexes do not match perfectly.
"""
print(explanation)

s1 = pd.Series([10, 20, 30], index=["a", "b", "c"])
s2 = pd.Series([100, 200, 300], index=["a", "c", "d"])

print("Series 1:")
print(s1)
print("\nSeries 2:")
print(s2)

result = s1 + s2
print("\ns1 + s2 (automatic alignment):")
print(result)
print("'b' is NaN because it only exists in s1")
print("'d' is NaN because it only exists in s2")

# ------------------------------------------------------------------------------
# 9.2 Disabling Alignment
# ------------------------------------------------------------------------------
print("\n--- 9.2 Controlling Alignment Behavior ---")

# Use fill_value to replace NaN in mismatched positions
result_filled = s1.add(s2, fill_value=0)
print("s1.add(s2, fill_value=0):")
print(result_filled)
print("Missing positions filled with 0 before operation")

# Align explicitly before operating
s1_aligned, s2_aligned = s1.align(s2, fill_value=0)
print("\nExplicit alignment with s1.align(s2, fill_value=0):")
print("s1 aligned:", s1_aligned.tolist())
print("s2 aligned:", s2_aligned.tolist())

# ------------------------------------------------------------------------------
# 9.3 Index Alignment Pitfall in DataFrames
# ------------------------------------------------------------------------------
print("\n--- 9.3 Index Alignment Pitfall ---")

# After filtering, index is not contiguous
df_filtered = df[df["Survived"] == 1].copy() if dataset_name == "titanic" \
              else df.head(10).copy()

print(f"Filtered DataFrame index (first 5): {df_filtered.index[:5].tolist()}")

# Assigning a new Series with default index causes misalignment
new_values = pd.Series(range(len(df_filtered)))  # 0, 1, 2, 3, ...
df_filtered["new_col"] = new_values

nan_count = df_filtered["new_col"].isnull().sum()
print(f"\nAssigned Series with default index to filtered DataFrame:")
print(f"  NaN count in new_col: {nan_count}")
print("  Misalignment: new_values has index 0,1,2,... but df has 1,3,6,...")

# Correct: use .values to ignore index
df_filtered["new_col_correct"] = new_values.values
nan_count_correct = df_filtered["new_col_correct"].isnull().sum()
print(f"\nUsing .values to bypass alignment:")
print(f"  NaN count in new_col_correct: {nan_count_correct}")
print("  .values strips index labels so assignment is purely positional")


# ==============================================================================
# SECTION 10: REAL WORLD EXAMPLE - STANDARDIZING A PIPELINE
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 10: REAL WORLD EXAMPLE - STANDARDIZING A PIPELINE")
print("=" * 70)

print("Scenario: Load raw data, standardize names, set index, validate.")

def prepare_dataframe(data, id_column=None, verbose=True):
    """
    Standardize a raw DataFrame for use in a production pipeline.

    Steps:
      1. Standardize column names to snake_case
      2. Set a meaningful index if id_column is provided
      3. Remove duplicate rows
      4. Validate output

    Parameters
    ----------
    data : pd.DataFrame
        Raw input DataFrame
    id_column : str or None
        Column to set as index after standardization
    verbose : bool
        Print each step

    Returns
    -------
    pd.DataFrame
        Prepared DataFrame
    """
    import re

    def to_snake(name):
        name = str(name).strip()
        name = re.sub(r"[\s\-\.]+", "_", name)
        name = re.sub(r"[^\w]", "", name)
        name = re.sub(r"_+", "_", name)
        name = name.strip("_").lower()
        if name and name[0].isdigit():
            name = "col_" + name
        return name

    result = data.copy()

    # Step 1: standardize column names
    original_cols = list(result.columns)
    result.columns = [to_snake(col) for col in result.columns]
    if verbose:
        print(f"\nStep 1: Standardized {len(original_cols)} column names")
        for old, new in zip(original_cols, result.columns):
            if old != new:
                print(f"   '{old}' -> '{new}'")

    # Step 2: remove duplicates
    before = len(result)
    result = result.drop_duplicates()
    if verbose:
        print(f"\nStep 2: Removed {before - len(result)} duplicate rows")

    # Step 3: set index
    if id_column is not None:
        snake_id = to_snake(id_column)
        if snake_id in result.columns:
            result = result.set_index(snake_id)
            if verbose:
                print(f"\nStep 3: Set '{snake_id}' as index")
        else:
            if verbose:
                print(f"\nStep 3: Column '{snake_id}' not found, skipping index set")

    # Step 4: validate
    assert result.index.is_unique or id_column is None, \
        "Index has duplicate values after set_index"
    if verbose:
        print(f"\nStep 4: Validation passed")
        print(f"\nFinal shape: {result.shape}")
        print(f"Final columns: {list(result.columns)}")

    return result


# Apply to loaded dataset
if dataset_name == "titanic":
    df_prepared = prepare_dataframe(df_raw, id_column="PassengerId")
else:
    df_prepared = prepare_dataframe(df_raw)

print("\nFirst 3 rows of prepared DataFrame:")
print(df_prepared.head(3))


# ==============================================================================
# SECTION 11: COMMON PITFALLS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 11: COMMON PITFALLS")
print("=" * 70)

pitfalls = """
Pitfall 1: Setting a non-unique column as index
   - .loc lookups return multiple rows unexpectedly
   - Joins produce cartesian products
   - Fix: verify uniqueness before set_index
   - Check: df[col].is_unique

Pitfall 2: Forgetting that rename() does not modify in place
   - df.rename(columns={'a': 'b'})  <- does nothing to df
   - Fix: df = df.rename(columns={'a': 'b'})

Pitfall 3: Index misalignment when combining DataFrames
   - Adding a column from another filtered DataFrame
   - Fix: use .values to bypass index alignment

Pitfall 4: MultiIndex columns after groupby agg
   - ('Age', 'mean') column name breaks downstream code
   - Fix: flatten with "_".join(col) pattern immediately

Pitfall 5: Losing index after operations
   - Some operations reset the index silently
   - Always check df.index after complex transforms

Pitfall 6: Duplicate column names after standardization
   - 'First Name' and 'first name' both become 'first_name'
   - Fix: detect and suffix duplicates as shown in standardize_columns

Pitfall 7: Writing index to CSV unintentionally
   - df.to_csv('file.csv') includes the index as an extra column
   - Fix: df.to_csv('file.csv', index=False) for default integer index
"""
print(pitfalls)


# ==============================================================================
# SECTION 12: SUMMARY TABLE
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 12: SUMMARY TABLE")
print("=" * 70)

summary = """
Operation                          | Syntax
-----------------------------------|------------------------------------------
Rename specific columns            | df.rename(columns={"old": "new"})
Rename all columns with function   | df.rename(columns=str.lower)
Replace all column names           | df.columns = [new_list]
Set new column names (chainable)   | df.set_axis(new_list, axis="columns")
Set a column as index              | df.set_index("col")
Set index, keep column             | df.set_index("col", drop=False)
Reset index to default             | df.reset_index()
Reset index and drop it            | df.reset_index(drop=True)
Reindex rows                       | df.reindex([new_labels])
Reindex columns                    | df.reindex(columns=[col_list])
Reindex with fill value            | df.reindex([labels], fill_value=0)
Cross-section from MultiIndex      | df.xs("val", level="LevelName")
Flatten MultiIndex columns         | df.columns = ["_".join(c) for c in df.columns]
Align two Series                   | s1.align(s2, fill_value=0)
Add with alignment control         | s1.add(s2, fill_value=0)
Bypass index alignment             | df["col"] = series.values
Check index uniqueness             | df.index.is_unique
"""
print(summary)


# ==============================================================================
# SECTION 13: PRACTICE EXERCISES
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 13: PRACTICE EXERCISES")
print("=" * 70)

print("Exercise 1: Rename columns to snake_case using rename and a lambda")
df_ex1 = df_raw.copy()
df_ex1 = df_ex1.rename(columns=lambda c: c.lower().replace(" ", "_"))
print("  Columns:", list(df_ex1.columns))

if dataset_name == "titanic":
    print("\nExercise 2: Set Name as index, look up a passenger by name")
    df_ex2 = df_raw.drop_duplicates(subset=["Name"]).set_index("Name")
    target = "Allen, Miss. Elisabeth Walton"
    if target in df_ex2.index:
        print(f"  Lookup '{target}':")
        print(df_ex2.loc[target][["Age", "Pclass", "Survived"]])

    print("\nExercise 3: Align two passenger subsets and combine")
    first_class  = df_raw[df_raw["Pclass"] == 1].set_index("PassengerId")["Fare"]
    second_class = df_raw[df_raw["Pclass"] == 2].set_index("PassengerId")["Fare"]
    fc_aligned, sc_aligned = first_class.align(second_class, fill_value=0)
    print(f"  First class passengers:  {len(first_class)}")
    print(f"  Second class passengers: {len(second_class)}")
    print(f"  After alignment (union): {len(fc_aligned)}")

print("\nExercise 4: Flatten MultiIndex columns from agg")
if dataset_name == "titanic":
    agg_ex = df_raw.groupby("Sex")[["Age", "Fare"]].agg(["mean", "std"])
    print("  Before flattening:", list(agg_ex.columns))
    agg_ex.columns = ["_".join(c) for c in agg_ex.columns]
    agg_ex = agg_ex.reset_index()
    print("  After flattening:", list(agg_ex.columns))
    print(agg_ex)


# ==============================================================================
# SECTION 14: KEY TAKEAWAYS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 14: KEY TAKEAWAYS")
print("=" * 70)

takeaways = """
1.  Standardize column names immediately after loading data
2.  Use snake_case for all column names in Python pipelines
3.  rename() returns a new DataFrame - always reassign
4.  set_index() makes .loc lookups faster and code more readable
5.  Only set unique columns as index
6.  Always reset_index(drop=True) after filtering to avoid index gaps
7.  reindex() is the correct tool for aligning DataFrames before combining
8.  MultiIndex is created by groupby and must often be flattened afterward
9.  Index alignment is automatic but can produce unexpected NaN values
10. Use .values when you want positional assignment without index alignment
11. Always write CSVs with index=False unless you have a meaningful index
"""
print(takeaways)


# ==============================================================================
# NEXT LESSON PREVIEW
# ==============================================================================
print("\n" + "=" * 70)
print("NEXT LESSON PREVIEW")
print("=" * 70)

preview = """
Lesson 23: Aggregations and GroupBy

You will learn:
- groupby mechanics and the split-apply-combine pattern
- Aggregating with single and multiple functions
- Grouping by multiple columns
- Transform vs aggregate vs filter
- Named aggregations with agg()
- Pivot tables as an alternative to groupby
- Real world sales and customer segmentation examples
"""
print(preview)

print("\n" + "=" * 70)
print("END OF LESSON 22")
print("=" * 70)
print("\nYou can now rename columns consistently, set meaningful indexes,")
print("reindex and align DataFrames, and work with hierarchical indexes.")