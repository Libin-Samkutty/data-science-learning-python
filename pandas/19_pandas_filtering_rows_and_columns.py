"""
LESSON 19: FILTERING ROWS AND COLUMNS
================================================================================

What You Will Learn:
- Advanced row filtering patterns
- Filtering columns by data type
- Dropping rows and columns
- Filtering with the query() method
- Dealing with missing values during filtering
- Building reusable filter functions for data pipelines
- Validating filter results

Real World Usage:
- Removing bad or incomplete records before analysis
- Selecting only relevant features for a machine learning model
- Isolating specific customer segments from a CRM dataset
- Cleaning log data by dropping irrelevant columns
- Building configurable filter steps in ETL pipelines

Dataset Used:
World Bank world development indicators (subset)
URL: https://raw.githubusercontent.com/cs109/2014_data/master/countries.csv
Fallback: Titanic dataset if the above is unavailable
https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv

================================================================================
"""

import numpy as np
import pandas as pd
import urllib.request

print("=" * 70)
print("LESSON 19: FILTERING ROWS AND COLUMNS")
print("=" * 70)


# ==============================================================================
# SECTION 1: LOAD AND VALIDATE DATASET
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 1: LOAD AND VALIDATE DATASET")
print("=" * 70)

# We use Titanic for consistency across lessons and because it has
# mixed types, missing values, and many natural filtering scenarios

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

print("Loading dataset from:")
print(url)

df = pd.read_csv(url)

print("\nValidation after loading:")
print("  Shape   :", df.shape)
print("  Columns :", list(df.columns))
print("\nMissing values:")
print(df.isnull().sum())
print("\nFirst 5 rows:")
print(df.head())


# ==============================================================================
# SECTION 2: RECAP - BASIC ROW FILTERING
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 2: RECAP - BASIC ROW FILTERING")
print("=" * 70)

print("Quick recap from Lesson 18 before going deeper.")

# Simple single condition
survived = df[df["Survived"] == 1]
print("\nSurvivors - df[df['Survived'] == 1]:")
print("  Count:", len(survived))

# Combined condition
female_survivors = df[(df["Survived"] == 1) & (df["Sex"] == "female")]
print("\nFemale survivors:")
print("  Count:", len(female_survivors))

# Verify: no accidental inclusion
assert all(female_survivors["Survived"] == 1), "Non-survivors in result"
assert all(female_survivors["Sex"] == "female"), "Males in result"
print("  Assertions passed: all rows are female survivors")


# ==============================================================================
# SECTION 3: ADVANCED ROW FILTERING PATTERNS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 3: ADVANCED ROW FILTERING PATTERNS")
print("=" * 70)

# ------------------------------------------------------------------------------
# 3.1 Filtering with isin
# ------------------------------------------------------------------------------
print("\n--- 3.1 Filtering with isin ---")

# isin is cleaner than multiple OR conditions
# Instead of: (df["Pclass"] == 1) | (df["Pclass"] == 3)
multi_class = df[df["Pclass"].isin([1, 3])]
print("Passengers in class 1 or 3 using isin:")
print("  Count:", len(multi_class))
print("  Class distribution:")
print(multi_class["Pclass"].value_counts())

# Negated isin: passengers NOT in those classes
only_second = df[~df["Pclass"].isin([1, 3])]
print("\nPassengers NOT in class 1 or 3 (negated isin):")
print("  Count:", len(only_second))
print("  Classes present:", only_second["Pclass"].unique())

# ------------------------------------------------------------------------------
# 3.2 Filtering with between
# ------------------------------------------------------------------------------
print("\n--- 3.2 Filtering with between ---")

# between is inclusive on both ends by default
young_adults = df[df["Age"].between(18, 30)]
print("Passengers aged 18 to 30 (inclusive):")
print("  Count:", len(young_adults))
print("  Age range:", young_adults["Age"].min(), "to", young_adults["Age"].max())

# Exclusive on one end using inclusive parameter
young_adults_exclusive = df[df["Age"].between(18, 30, inclusive="left")]
print("\nPassengers aged 18 to 30 (left inclusive, right exclusive):")
print("  Count:", len(young_adults_exclusive))

# Verify the bounds
assert young_adults["Age"].min() >= 18, "Age below 18 found"
assert young_adults["Age"].max() <= 30, "Age above 30 found"
print("\nBound assertions passed")

# ------------------------------------------------------------------------------
# 3.3 Filtering with str accessor
# ------------------------------------------------------------------------------
print("\n--- 3.3 Filtering with str Accessor ---")

# Contains: partial string match
has_title_mr = df[df["Name"].str.contains("Mr\\.", regex=True, na=False)]
print("Passengers with title 'Mr.' in name:")
print("  Count:", len(has_title_mr))
print(has_title_mr[["Name", "Sex", "Age"]].head())

# Startswith and endswith
starts_with_j = df[df["Name"].str.startswith("J", na=False)]
print("\nPassengers whose name starts with J:")
print("  Count:", len(starts_with_j))

# Extract a pattern: title from name (advanced but practical)
# Name format: "Lastname, Title. Firstname"
df["Title"] = df["Name"].str.extract(r",\s([A-Za-z]+)\\.")
print("\nExtracted titles from Name column:")
print(df["Title"].value_counts())

# Filter by extracted title
officers = df[df["Title"].isin(["Col", "Major", "Capt"])]
print("\nMilitary officers on board:")
print("  Count:", len(officers))
print(officers[["Name", "Title", "Age", "Survived"]])

# ------------------------------------------------------------------------------
# 3.4 Filtering Numeric Outliers
# ------------------------------------------------------------------------------
print("\n--- 3.4 Filtering Numeric Outliers ---")

# Method 1: Fixed threshold
fare_cap = 300
no_outliers = df[df["Fare"] <= fare_cap]
print("Removing fare outliers above", fare_cap)
print("  Before:", len(df), "rows")
print("  After :", len(no_outliers), "rows")
print("  Removed:", len(df) - len(no_outliers), "rows")

# Method 2: IQR-based outlier detection (more principled)
q1 = df["Fare"].quantile(0.25)
q3 = df["Fare"].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

no_outliers_iqr = df[df["Fare"].between(lower_bound, upper_bound)]
print("\nIQR-based outlier removal:")
print("  Q1:", round(q1, 2), "Q3:", round(q3, 2), "IQR:", round(iqr, 2))
print("  Lower bound:", round(lower_bound, 2))
print("  Upper bound:", round(upper_bound, 2))
print("  Rows retained:", len(no_outliers_iqr))
print("  Rows removed :", len(df) - len(no_outliers_iqr))

# ------------------------------------------------------------------------------
# 3.5 Filtering Using nunique and value_counts
# ------------------------------------------------------------------------------
print("\n--- 3.5 Filtering Based on Group Size ---")

# Real world use case: remove categories with very few members
# Example: only keep titles that appear more than 5 times
title_counts = df["Title"].value_counts()
print("All title counts:")
print(title_counts)

common_titles = title_counts[title_counts >= 5].index.tolist()
print("\nTitles with 5 or more passengers:", common_titles)

df_common_titles = df[df["Title"].isin(common_titles)]
print("Rows with common titles:", len(df_common_titles))
print("Rows dropped:", len(df) - len(df_common_titles))


# ==============================================================================
# SECTION 4: FILTERING COLUMNS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 4: FILTERING COLUMNS")
print("=" * 70)

# ------------------------------------------------------------------------------
# 4.1 Selecting Columns by Name Pattern
# ------------------------------------------------------------------------------
print("\n--- 4.1 Selecting Columns by Name Pattern ---")

# Columns that start with a specific prefix
print("All columns:", list(df.columns))

# Using list comprehension for flexible column selection
p_columns = [col for col in df.columns if col.startswith("P")]
print("\nColumns starting with P:", p_columns)
print(df[p_columns].head())

# Columns containing a substring
age_related = [col for col in df.columns if "a" in col.lower()]
print("\nColumns containing letter 'a' (case insensitive):", age_related)

# ------------------------------------------------------------------------------
# 4.2 Filtering Columns by Data Type
# ------------------------------------------------------------------------------
print("\n--- 4.2 Filtering Columns by Data Type ---")

# select_dtypes is the correct Pandas way to filter by dtype
print("All column dtypes:")
print(df.dtypes)

# Select only numeric columns
numeric_cols = df.select_dtypes(include=[np.number])
print("\nNumeric columns only:")
print("  Columns:", list(numeric_cols.columns))
print(numeric_cols.head())

# Select only object (string) columns
string_cols = df.select_dtypes(include=["object"])
print("\nString (object) columns only:")
print("  Columns:", list(string_cols.columns))
print(string_cols.head())

# Exclude specific dtypes
non_string = df.select_dtypes(exclude=["object"])
print("\nAll columns except string:")
print("  Columns:", list(non_string.columns))

# Real world use: pass only numeric features to a model
print("\nReal world pattern: select numeric features for ML model")
feature_matrix = df.select_dtypes(include=[np.number]).drop(
    columns=["Survived", "PassengerId"]
)
print("  Feature matrix columns:", list(feature_matrix.columns))
print("  Shape:", feature_matrix.shape)

# ------------------------------------------------------------------------------
# 4.3 Dropping Specific Columns
# ------------------------------------------------------------------------------
print("\n--- 4.3 Dropping Specific Columns ---")

# Drop a single column
# axis=1 means columns, axis=0 means rows
df_no_ticket = df.drop(columns=["Ticket"])
print("After dropping Ticket column:")
print("  Columns:", list(df_no_ticket.columns))

# Drop multiple columns
cols_to_drop = ["Ticket", "Cabin", "PassengerId"]
df_cleaned = df.drop(columns=cols_to_drop)
print("\nAfter dropping Ticket, Cabin, PassengerId:")
print("  Columns:", list(df_cleaned.columns))

# errors='ignore' is useful in pipelines where column may not exist
df_safe_drop = df.drop(columns=["Ticket", "NonExistentColumn"], errors="ignore")
print("\nDrop with errors='ignore' (safe for pipelines):")
print("  Columns:", list(df_safe_drop.columns))

# Validate column was actually removed
assert "Ticket" not in df_cleaned.columns, "Ticket column still present"
assert "Survived" in df_cleaned.columns, "Survived column missing"
print("\nColumn drop assertions passed")


# ==============================================================================
# SECTION 5: DROPPING ROWS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 5: DROPPING ROWS")
print("=" * 70)

# ------------------------------------------------------------------------------
# 5.1 Dropping Rows by Index Label
# ------------------------------------------------------------------------------
print("\n--- 5.1 Dropping Rows by Index Label ---")

# Drop specific rows by index
df_drop_rows = df.drop(index=[0, 1, 2])
print("After dropping rows 0, 1, 2:")
print("  Shape before:", df.shape)
print("  Shape after :", df_drop_rows.shape)
print("  First row index is now:", df_drop_rows.index[0])

# ------------------------------------------------------------------------------
# 5.2 Dropping Rows with Missing Values
# ------------------------------------------------------------------------------
print("\n--- 5.2 Dropping Rows with Missing Values ---")

print("Missing values before drop:")
print(df.isnull().sum())

# Drop rows where ANY column is NaN
df_drop_any = df.dropna()
print("\nAfter dropna() - drop rows with ANY missing value:")
print("  Shape:", df_drop_any.shape)
print("  Rows dropped:", len(df) - len(df_drop_any))

# Drop rows where ALL columns are NaN (rare but safe default)
df_drop_all = df.dropna(how="all")
print("\nAfter dropna(how='all') - drop rows where ALL values are missing:")
print("  Shape:", df_drop_all.shape)

# Drop rows where specific column is NaN
df_drop_age = df.dropna(subset=["Age"])
print("\nAfter dropna(subset=['Age']) - only drop rows with missing Age:")
print("  Shape:", df_drop_age.shape)
print("  Age missing:", df_drop_age["Age"].isnull().sum())

# Drop rows with missing values in any of several columns
df_drop_multi = df.dropna(subset=["Age", "Embarked"])
print("\nAfter dropna(subset=['Age', 'Embarked']):")
print("  Shape:", df_drop_multi.shape)

# threshold: keep rows with at least N non-NaN values
df_thresh = df.dropna(thresh=10)
print("\nAfter dropna(thresh=10) - keep rows with at least 10 non-NaN values:")
print("  Shape:", df_thresh.shape)

# ------------------------------------------------------------------------------
# 5.3 Dropping Duplicate Rows
# ------------------------------------------------------------------------------
print("\n--- 5.3 Dropping Duplicate Rows ---")

# Check for duplicates
print("Duplicate rows in original data:", df.duplicated().sum())

# Create a DataFrame with intentional duplicates for demonstration
df_with_dupes = pd.concat([df.head(5), df.head(3)], ignore_index=True)
print("\nArtificial dataset with duplicates:")
print("  Shape:", df_with_dupes.shape)
print("  Duplicate rows:", df_with_dupes.duplicated().sum())

# Drop duplicates keeping first occurrence
df_no_dupes = df_with_dupes.drop_duplicates()
print("\nAfter drop_duplicates() - keeps first occurrence:")
print("  Shape:", df_no_dupes.shape)

# Drop duplicates keeping last occurrence
df_keep_last = df_with_dupes.drop_duplicates(keep="last")
print("\nAfter drop_duplicates(keep='last'):")
print("  Shape:", df_keep_last.shape)

# Drop duplicates based on specific column subset
df_unique_names = df_with_dupes.drop_duplicates(subset=["Name"])
print("\nDrop duplicates based on Name column only:")
print("  Shape:", df_unique_names.shape)


# ==============================================================================
# SECTION 6: FILTERING WITH query()
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 6: FILTERING WITH query()")
print("=" * 70)

explanation = """
query() allows filtering with a string expression.
Advantages:
- More readable for complex conditions
- Cleaner syntax (no need for df["col"] repeatedly)
- Useful in chained operations and pipelines

Limitations:
- Column names with spaces need backtick quoting
- Cannot use all Python expressions inside query strings
- Slightly slower than direct boolean indexing on very large datasets
"""
print(explanation)

# Basic query
result = df.query("Survived == 1")
print("df.query('Survived == 1'):")
print("  Count:", len(result))

# Equivalent boolean filter for comparison
result_bool = df[df["Survived"] == 1]
print("  Equal to boolean filter:", result.shape == result_bool.shape)

# Query with AND
result2 = df.query("Survived == 1 and Sex == 'female'")
print("\ndf.query(\"Survived == 1 and Sex == 'female'\"):")
print("  Count:", len(result2))

# Query with numeric range
result3 = df.query("Age >= 18 and Age <= 30")
print("\ndf.query('Age >= 18 and Age <= 30'):")
print("  Count:", len(result3))

# Query with Python variable reference using @
fare_limit = 50.0
result4 = df.query("Fare < @fare_limit")
print("\ndf.query('Fare < @fare_limit') where fare_limit =", fare_limit)
print("  Count:", len(result4))

# Query with isin equivalent
result5 = df.query("Pclass in [1, 2]")
print("\ndf.query('Pclass in [1, 2]'):")
print("  Count:", len(result5))

# Multi-line query for readability
result6 = df.query(
    "Pclass == 1 "
    "and Age > 40 "
    "and Survived == 1"
)
print("\nMulti-condition query (first class, over 40, survived):")
print("  Count:", len(result6))
print(result6[["Name", "Age", "Pclass", "Survived"]].head())

# query vs boolean filter: verify same result
result_bool2 = df[
    (df["Pclass"] == 1) &
    (df["Age"] > 40) &
    (df["Survived"] == 1)
]
assert len(result6) == len(result_bool2), "query and boolean filter gave different results"
print("\nVerified: query() and boolean filter produce identical results")


# ==============================================================================
# SECTION 7: FILTERING WHEN MISSING VALUES ARE PRESENT
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 7: FILTERING WHEN MISSING VALUES ARE PRESENT")
print("=" * 70)

explanation = """
Missing values (NaN) cause subtle problems during filtering.
A comparison with NaN always returns False in Pandas.
This means NaN rows are silently excluded from filter results.
You must decide explicitly what to do with them.
"""
print(explanation)

# Demonstrate NaN exclusion
print("Passengers with Age > 30:")
over_30 = df[df["Age"] > 30]
print("  Count:", len(over_30))

print("\nPassengers with Age <= 30:")
not_over_30 = df[df["Age"] <= 30]
print("  Count:", len(not_over_30))

total_accounted = len(over_30) + len(not_over_30)
print("\nTotal accounted for:", total_accounted)
print("Total rows in dataset:", len(df))
missing_age_count = df["Age"].isnull().sum()
print("Missing Age count:", missing_age_count)
print("Difference equals missing:", len(df) - total_accounted == missing_age_count)

print("\nConclusion: NaN rows are silently excluded from both sides of a filter.")
print("You must handle them explicitly before or during filtering.")

# Strategy 1: Drop NaN rows before filtering
df_age_known = df.dropna(subset=["Age"])
result_strategy1 = df_age_known[df_age_known["Age"] > 30]
print("\nStrategy 1 - Drop NaN before filtering:")
print("  Rows with known age:", len(df_age_known))
print("  Over 30:", len(result_strategy1))

# Strategy 2: Fill NaN before filtering
df_age_filled = df.copy()
df_age_filled["Age"] = df_age_filled["Age"].fillna(df_age_filled["Age"].median())
result_strategy2 = df_age_filled[df_age_filled["Age"] > 30]
print("\nStrategy 2 - Fill NaN with median before filtering:")
print("  Median used:", df["Age"].median())
print("  Over 30:", len(result_strategy2))

# Strategy 3: Include NaN rows explicitly in filter result
result_strategy3 = df[df["Age"].isna() | (df["Age"] > 30)]
print("\nStrategy 3 - Keep NaN rows in result explicitly:")
print("  Over 30 or unknown age:", len(result_strategy3))

# Strategy 4: Filter only rows where Age is known and condition met
result_strategy4 = df[df["Age"].notna() & (df["Age"] > 30)]
print("\nStrategy 4 - Known age AND over 30:")
print("  Count:", len(result_strategy4))


# ==============================================================================
# SECTION 8: BUILDING REUSABLE FILTER FUNCTIONS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 8: BUILDING REUSABLE FILTER FUNCTIONS")
print("=" * 70)

explanation = """
In production pipelines, filtering logic should be encapsulated in functions.
This ensures:
- Consistent application across datasets
- Easy testing and debugging
- Readable pipeline code
"""
print(explanation)

# ------------------------------------------------------------------------------
# 8.1 Generic Row Filter Function
# ------------------------------------------------------------------------------
def filter_passengers(
    data,
    survived=None,
    sex=None,
    pclass=None,
    min_age=None,
    max_age=None,
    max_fare=None,
    drop_missing_age=False
):
    """
    Filter passenger DataFrame based on configurable criteria.

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame (expected to have Titanic structure)
    survived : int or None
        Filter by survival status (0 or 1)
    sex : str or None
        Filter by sex ('male' or 'female')
    pclass : list or None
        List of passenger classes to include
    min_age : float or None
        Minimum age (inclusive)
    max_age : float or None
        Maximum age (inclusive)
    max_fare : float or None
        Maximum fare (inclusive)
    drop_missing_age : bool
        Drop rows where Age is NaN before applying age filters

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame (copy)
    """
    result = data.copy()
    initial_count = len(result)

    if drop_missing_age:
        result = result.dropna(subset=["Age"])

    if survived is not None:
        result = result[result["Survived"] == survived]

    if sex is not None:
        result = result[result["Sex"] == sex]

    if pclass is not None:
        result = result[result["Pclass"].isin(pclass)]

    if min_age is not None:
        result = result[result["Age"] >= min_age]

    if max_age is not None:
        result = result[result["Age"] <= max_age]

    if max_fare is not None:
        result = result[result["Fare"] <= max_fare]

    print("  Input rows   :", initial_count)
    print("  Output rows  :", len(result))
    print("  Rows filtered:", initial_count - len(result))

    return result


print("Using filter_passengers function:")

print("\nScenario 1: Female survivors in first class")
seg1 = filter_passengers(df, survived=1, sex="female", pclass=[1])
print(seg1[["Name", "Age", "Pclass", "Survived"]].head())

print("\nScenario 2: Adult males (18-60) who paid under 50")
seg2 = filter_passengers(
    df,
    sex="male",
    min_age=18,
    max_age=60,
    max_fare=50,
    drop_missing_age=True
)
print(seg2[["Name", "Age", "Fare", "Survived"]].head())

print("\nScenario 3: Third class passengers of any gender who survived")
seg3 = filter_passengers(df, survived=1, pclass=[3])
print(seg3[["Name", "Age", "Sex", "Pclass"]].head())

# Validate output of filter function
assert all(seg1["Survived"] == 1), "Non-survivors in Scenario 1"
assert all(seg1["Sex"] == "female"), "Males in Scenario 1"
assert all(seg1["Pclass"] == 1), "Non-first-class in Scenario 1"
print("\nAll validation assertions passed for filter function outputs")

# ------------------------------------------------------------------------------
# 8.2 Column Filter Function
# ------------------------------------------------------------------------------
def select_feature_columns(data, include_dtypes=None, exclude_cols=None):
    """
    Select columns by dtype and exclusion list.

    Parameters
    ----------
    data : pd.DataFrame
    include_dtypes : list or None
        List of dtype strings to include (e.g. ['number', 'object'])
    exclude_cols : list or None
        Column names to explicitly exclude

    Returns
    -------
    pd.DataFrame
    """
    if include_dtypes is not None:
        result = data.select_dtypes(include=include_dtypes)
    else:
        result = data.copy()

    if exclude_cols is not None:
        cols_to_drop = [c for c in exclude_cols if c in result.columns]
        result = result.drop(columns=cols_to_drop)

    print("  Selected columns:", list(result.columns))
    return result


print("\nUsing select_feature_columns function:")
print("\nNumeric features only, excluding ID and target:")
features = select_feature_columns(
    df,
    include_dtypes=["number"],
    exclude_cols=["PassengerId", "Survived"]
)
print(features.head())


# ==============================================================================
# SECTION 9: REAL WORLD EXAMPLE - PASSENGER SEGMENTATION PIPELINE
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 9: REAL WORLD EXAMPLE - PASSENGER SEGMENTATION PIPELINE")
print("=" * 70)

print("Scenario: Prepare three passenger segments for separate analysis.")
print("Segment A: High-risk group (male, third class, adult)")
print("Segment B: Low-risk group (female, first or second class)")
print("Segment C: Children (under 15, any class)")

# Segment A
print("\nSegment A: High-risk males")
segment_a = df[
    (df["Sex"] == "male") &
    (df["Pclass"] == 3) &
    (df["Age"] >= 18)
].dropna(subset=["Age"])

print("  Count:", len(segment_a))
print("  Survival rate: {:.1f}%".format(segment_a["Survived"].mean() * 100))

# Segment B
print("\nSegment B: Lower-risk females")
segment_b = df[
    (df["Sex"] == "female") &
    (df["Pclass"].isin([1, 2]))
]
print("  Count:", len(segment_b))
print("  Survival rate: {:.1f}%".format(segment_b["Survived"].mean() * 100))

# Segment C
print("\nSegment C: Children under 15")
segment_c = df[df["Age"] < 15].dropna(subset=["Age"])
print("  Count:", len(segment_c))
print("  Survival rate: {:.1f}%".format(segment_c["Survived"].mean() * 100))

# Validate segments are mutually exclusive where expected
overlap_ab = pd.merge(
    segment_a[["PassengerId"]],
    segment_b[["PassengerId"]],
    on="PassengerId"
)
print("\nOverlap between Segment A and B:", len(overlap_ab), "passengers")

# Summary report
print("\nSegment Summary:")
print("{:<12} {:>8} {:>16}".format("Segment", "Count", "Survival Rate"))
print("-" * 40)
segments = [
    ("A (high-risk)", segment_a),
    ("B (low-risk)", segment_b),
    ("C (children)", segment_c)
]
for name, seg in segments:
    rate = seg["Survived"].mean() * 100
    print("{:<14} {:>6} {:>14.1f}%".format(name, len(seg), rate))


# ==============================================================================
# SECTION 10: COMMON PITFALLS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 10: COMMON PITFALLS")
print("=" * 70)

print("Pitfall 1: Modifying a filter result without .copy()")
print("  WRONG:")
print("    subset = df[df['Age'] > 30]")
print("    subset['NewCol'] = 1          <- SettingWithCopyWarning")
print("  RIGHT:")
print("    subset = df[df['Age'] > 30].copy()")
print("    subset['NewCol'] = 1          <- safe, modifies only the copy")

print("\nPitfall 2: Forgetting that NaN rows are silently excluded")
print("  Always check: len(filtered) + len(complement) == len(original)")
over_30_count = len(df[df["Age"] > 30])
under_30_count = len(df[df["Age"] <= 30])
nan_age_count = df["Age"].isnull().sum()
print("  Over 30:", over_30_count)
print("  Under or equal 30:", under_30_count)
print("  NaN Age:", nan_age_count)
print("  Sum:", over_30_count + under_30_count + nan_age_count, "== Total:", len(df))

print("\nPitfall 3: Using = instead of == inside a filter")
print("  WRONG: df[df['Survived'] = 1]  <- SyntaxError")
print("  RIGHT: df[df['Survived'] == 1]")

print("\nPitfall 4: drop() without inplace or reassignment has no effect")
df_original = df.copy()
df_original.drop(columns=["Ticket"])         # No effect on df_original
print("  df.drop(columns=['Ticket']) without reassignment:")
print("  'Ticket' still in df:", "Ticket" in df_original.columns)
df_original = df_original.drop(columns=["Ticket"])
print("  After reassignment - 'Ticket' still in df:", "Ticket" in df_original.columns)

print("\nPitfall 5: dropna() drops more rows than expected")
print("  Always use subset= to target specific columns")
print("  df.dropna() may drop valid rows due to NaN in unrelated columns")
drop_all_count = len(df.dropna())
drop_age_count = len(df.dropna(subset=["Age"]))
print("  dropna() rows retained:", drop_all_count)
print("  dropna(subset=['Age']) rows retained:", drop_age_count)
print("  Difference:", drop_age_count - drop_all_count, "rows saved by using subset=")


# ==============================================================================
# SECTION 11: SUMMARY TABLE
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 11: SUMMARY TABLE")
print("=" * 70)

summary = """
Operation                            | Syntax
-------------------------------------|-------------------------------------------
Filter by value                      | df[df["Col"] == value]
Filter by range                      | df[df["Col"].between(low, high)]
Filter by multiple values            | df[df["Col"].isin([v1, v2])]
Exclude values                       | df[~df["Col"].isin([v1, v2])]
String contains                      | df[df["Col"].str.contains("x", na=False)]
Combined AND                         | df[(cond1) & (cond2)]
Combined OR                          | df[(cond1) | (cond2)]
NOT condition                        | df[~(cond)]
Query string                         | df.query("Col > 30 and Sex == 'male'")
Select numeric columns               | df.select_dtypes(include=[np.number])
Select string columns                | df.select_dtypes(include=["object"])
Drop specific columns                | df.drop(columns=["Col1", "Col2"])
Drop rows by index                   | df.drop(index=[0, 1, 2])
Drop rows with any NaN               | df.dropna()
Drop rows with NaN in column         | df.dropna(subset=["Age"])
Drop duplicate rows                  | df.drop_duplicates()
Drop duplicates by column            | df.drop_duplicates(subset=["Name"])
Filter and copy safely               | df[mask].copy()
"""
print(summary)


# ==============================================================================
# SECTION 12: PRACTICE EXERCISES
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 12: PRACTICE EXERCISES")
print("=" * 70)

print("Exercise 1: Find all passengers with a Fare between 10 and 50")
print("who are NOT in third class, drop rows with missing Age")
ex1 = df[
    df["Fare"].between(10, 50) &
    ~df["Pclass"].isin([3])
].dropna(subset=["Age"])
print("  Count:", len(ex1))
print(ex1[["Name", "Pclass", "Fare", "Age"]].head())

print("\nExercise 2: Select only numeric columns, drop PassengerId and Survived")
ex2 = df.select_dtypes(include=[np.number]).drop(
    columns=["PassengerId", "Survived"], errors="ignore"
)
print("  Columns:", list(ex2.columns))
print(ex2.head())

print("\nExercise 3: Use query() to find first class survivors over 50")
ex3 = df.query("Pclass == 1 and Survived == 1 and Age > 50")
print("  Count:", len(ex3))
print(ex3[["Name", "Age", "Pclass", "Survived"]].head())

print("\nExercise 4: Drop columns Cabin and Ticket, then drop rows")
print("where both Age and Embarked are missing")
ex4 = (
    df
    .drop(columns=["Cabin", "Ticket"])
    .dropna(subset=["Age", "Embarked"], how="all")
)
print("  Shape:", ex4.shape)
print("  Columns:", list(ex4.columns))


# ==============================================================================
# NEXT LESSON PREVIEW
# ==============================================================================
print("\n" + "=" * 70)
print("NEXT LESSON PREVIEW")
print("=" * 70)

preview = """
Lesson 20: Data Cleaning - Missing Values, Duplicates, and Type Conversion

You will learn:
- Systematic approach to handling missing values
- Strategies for filling NaN (mean, median, mode, forward fill, interpolate)
- Detecting and removing duplicate records
- Converting column data types correctly
- Cleaning string columns
- Validating cleaned data with assertions
- Building a complete data cleaning function
"""
print(preview)

print("\n" + "=" * 70)
print("END OF LESSON 19")
print("=" * 70)
print("\nYou can now filter rows and columns using every major Pandas technique.")
print("You have also built reusable filter functions ready for production pipelines.")