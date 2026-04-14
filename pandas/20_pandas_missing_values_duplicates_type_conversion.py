"""
LESSON 20: DATA CLEANING - MISSING VALUES, DUPLICATES, AND TYPE CONVERSION
================================================================================

What You Will Learn:
- Systematic approach to identifying data quality issues
- Strategies for handling missing values (drop, fill, interpolate)
- Detecting and removing duplicate records
- Converting column data types correctly
- Cleaning string columns (whitespace, case, invalid characters)
- Validating cleaned data with assertions
- Building a complete data cleaning pipeline

Real World Usage:
- Preparing raw data exports for analysis
- Cleaning customer databases before CRM migration
- Preprocessing datasets for machine learning
- Standardizing data from multiple sources
- ETL pipeline data quality steps

Dataset Used:
Titanic passenger data with intentionally messy additions
URL: https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv

================================================================================
"""

import numpy as np
import pandas as pd

print("=" * 70)
print("LESSON 20: DATA CLEANING")
print("Missing Values, Duplicates, and Type Conversion")
print("=" * 70)


# ==============================================================================
# SECTION 1: LOAD DATA AND CREATE MESSY VERSION
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 1: LOAD DATA AND CREATE MESSY VERSION")
print("=" * 70)

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

print("Loading clean Titanic dataset from:")
print(url)

df_clean = pd.read_csv(url)
print("Original shape:", df_clean.shape)

# Create a messy version to practice cleaning
print("\nCreating intentionally messy dataset for practice...")

df = df_clean.copy()

# Add some duplicate rows
duplicates = df.sample(n=20, random_state=42)
df = pd.concat([df, duplicates], ignore_index=True)

# Add whitespace issues to Name column
df.loc[10:20, "Name"] = df.loc[10:20, "Name"].apply(lambda x: "  " + x + "  ")

# Add case inconsistency to Sex column
df.loc[50:70, "Sex"] = df.loc[50:70, "Sex"].str.upper()
df.loc[100:110, "Sex"] = df.loc[100:110, "Sex"].str.title()

# Add invalid values to Embarked
df.loc[200:205, "Embarked"] = "Unknown"
df.loc[300:303, "Embarked"] = ""

# Add some additional NaN values
np.random.seed(42)
random_indices = np.random.choice(df.index, size=30, replace=False)
df.loc[random_indices[:15], "Age"] = np.nan
df.loc[random_indices[15:], "Fare"] = np.nan

# Add a column with wrong dtype (numeric stored as string)
df["FareStr"] = df["Fare"].astype(str)
df.loc[df["FareStr"] == "nan", "FareStr"] = "missing"

# Add leading zeros to PassengerId stored as string
df["PassengerIdStr"] = df["PassengerId"].apply(lambda x: f"P{x:05d}")

print("Messy dataset created.")
print("New shape:", df.shape)
print("Columns:", list(df.columns))


# ==============================================================================
# SECTION 2: DATA QUALITY ASSESSMENT
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 2: DATA QUALITY ASSESSMENT")
print("=" * 70)

def assess_data_quality(data, name="DataFrame"):
    """
    Generate a comprehensive data quality report.
    Run this before and after cleaning to verify improvements.
    """
    print("\n" + "-" * 50)
    print(f"DATA QUALITY REPORT: {name}")
    print("-" * 50)

    print(f"\nShape: {data.shape[0]} rows x {data.shape[1]} columns")

    # Missing values
    print("\n1. MISSING VALUES:")
    missing = data.isnull().sum()
    missing_pct = (data.isnull().mean() * 100).round(2)
    missing_report = pd.DataFrame({
        "Missing Count": missing,
        "Missing %": missing_pct
    })
    missing_report = missing_report[missing_report["Missing Count"] > 0]
    if len(missing_report) > 0:
        print(missing_report)
    else:
        print("   No missing values found.")

    # Duplicates
    print("\n2. DUPLICATE ROWS:")
    dup_count = data.duplicated().sum()
    print(f"   Duplicate rows: {dup_count}")
    print(f"   Percentage: {dup_count / len(data) * 100:.2f}%")

    # Data types
    print("\n3. DATA TYPES:")
    print(data.dtypes)

    # Unique values for object columns
    print("\n4. UNIQUE VALUES (object columns):")
    for col in data.select_dtypes(include=["object"]).columns:
        unique_count = data[col].nunique()
        sample_values = data[col].dropna().unique()[:5]
        print(f"   {col}: {unique_count} unique values")
        print(f"      Sample: {list(sample_values)}")

    # Numeric column ranges
    print("\n5. NUMERIC COLUMN RANGES:")
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        col_min = data[col].min()
        col_max = data[col].max()
        print(f"   {col}: min={col_min}, max={col_max}")

    print("-" * 50)


# Run initial assessment
assess_data_quality(df, "Messy Titanic Data")


# ==============================================================================
# SECTION 3: HANDLING MISSING VALUES
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 3: HANDLING MISSING VALUES")
print("=" * 70)

# Work on a copy to preserve original messy data for comparison
df_working = df.copy()

# ------------------------------------------------------------------------------
# 3.1 Identifying Missing Values
# ------------------------------------------------------------------------------
print("\n--- 3.1 Identifying Missing Values ---")

print("Missing value counts:")
print(df_working.isnull().sum())

print("\nRows with any missing value:")
rows_with_missing = df_working.isnull().any(axis=1).sum()
print(f"   {rows_with_missing} rows ({rows_with_missing/len(df_working)*100:.1f}%)")

# Visualize missing pattern
print("\nMissing value pattern (first 10 columns, sample rows):")
sample_missing = df_working[df_working.isnull().any(axis=1)].head(5)
print(sample_missing.isnull().astype(int))

# ------------------------------------------------------------------------------
# 3.2 Strategy 1: Drop Rows with Missing Values
# ------------------------------------------------------------------------------
print("\n--- 3.2 Strategy 1: Drop Rows with Missing Values ---")

# Drop rows where critical columns are missing
print("Dropping rows where Age is missing:")
df_drop_age = df_working.dropna(subset=["Age"])
print(f"   Before: {len(df_working)} rows")
print(f"   After:  {len(df_drop_age)} rows")
print(f"   Dropped: {len(df_working) - len(df_drop_age)} rows")

# Drop rows where ANY value is missing (aggressive)
df_drop_all = df_working.dropna()
print("\nDropping rows where ANY column has missing value:")
print(f"   Before: {len(df_working)} rows")
print(f"   After:  {len(df_drop_all)} rows")
print(f"   Dropped: {len(df_working) - len(df_drop_all)} rows")

print("\nWhen to use dropna:")
print("   - When missing data is a small percentage (<5%)")
print("   - When the column is critical and cannot be imputed")
print("   - When you have plenty of data and can afford to lose rows")

# ------------------------------------------------------------------------------
# 3.3 Strategy 2: Fill with Constant Value
# ------------------------------------------------------------------------------
print("\n--- 3.3 Strategy 2: Fill with Constant Value ---")

# Fill Cabin with 'Unknown' since it is mostly missing
df_working["Cabin"] = df_working["Cabin"].fillna("Unknown")
cabin_unknown = (df_working["Cabin"] == "Unknown").sum()
print(f"Filled missing Cabin with 'Unknown': {cabin_unknown} values")

# Fill Embarked with most common value (mode)
embarked_mode = df_working["Embarked"].mode()[0]
df_working["Embarked"] = df_working["Embarked"].fillna(embarked_mode)
print(f"Filled missing Embarked with mode '{embarked_mode}'")

# Verify no missing Embarked
assert df_working["Embarked"].isnull().sum() == 0, "Embarked still has missing"
print("   Embarked missing count:", df_working["Embarked"].isnull().sum())

# ------------------------------------------------------------------------------
# 3.4 Strategy 3: Fill with Statistical Measures
# ------------------------------------------------------------------------------
print("\n--- 3.4 Strategy 3: Fill with Statistical Measures ---")

# Fill Age with median (robust to outliers)
age_median = df_working["Age"].median()
age_missing_before = df_working["Age"].isnull().sum()
df_working["Age"] = df_working["Age"].fillna(age_median)
print(f"Filled {age_missing_before} missing Age values with median: {age_median}")

# Fill Fare with mean
fare_mean = df_working["Fare"].mean()
fare_missing_before = df_working["Fare"].isnull().sum()
df_working["Fare"] = df_working["Fare"].fillna(fare_mean)
print(f"Filled {fare_missing_before} missing Fare values with mean: {fare_mean:.2f}")

# Verify
assert df_working["Age"].isnull().sum() == 0, "Age still has missing"
assert df_working["Fare"].isnull().sum() == 0, "Fare still has missing"
print("Age and Fare missing counts verified as 0")

# ------------------------------------------------------------------------------
# 3.5 Strategy 4: Group-Based Filling
# ------------------------------------------------------------------------------
print("\n--- 3.5 Strategy 4: Group-Based Filling ---")

# Demonstrate filling based on group statistics
# Reset to messy data for this example
df_group_demo = df.copy()

print("Fill Age based on Pclass and Sex median:")
print("Before filling - missing Age:", df_group_demo["Age"].isnull().sum())

# Calculate group medians
group_medians = df_group_demo.groupby(["Pclass", "Sex"])["Age"].median()
print("\nGroup medians:")
print(group_medians)

# Fill using transform
df_group_demo["Age"] = df_group_demo.groupby(["Pclass", "Sex"])["Age"].transform(
    lambda x: x.fillna(x.median())
)

# Fill any remaining with overall median
df_group_demo["Age"] = df_group_demo["Age"].fillna(df_group_demo["Age"].median())

print("\nAfter filling - missing Age:", df_group_demo["Age"].isnull().sum())
print("Group-based filling preserves more realistic age distributions")

# ------------------------------------------------------------------------------
# 3.6 Strategy 5: Forward Fill and Backward Fill
# ------------------------------------------------------------------------------
print("\n--- 3.6 Strategy 5: Forward Fill and Backward Fill ---")

# Useful for time series or ordered data
print("ffill: Forward fill (carry previous value forward)")
print("bfill: Backward fill (carry next value backward)")

# Create a simple time series example
ts_data = pd.DataFrame({
    "Date": pd.date_range("2024-01-01", periods=10),
    "Value": [1.0, np.nan, np.nan, 4.0, 5.0, np.nan, 7.0, np.nan, np.nan, 10.0]
})

print("\nOriginal time series:")
print(ts_data["Value"].tolist())

print("\nAfter forward fill:")
print(ts_data["Value"].ffill().tolist())

print("\nAfter backward fill:")
print(ts_data["Value"].bfill().tolist())

print("\nAfter forward then backward fill (fills all):")
print(ts_data["Value"].ffill().bfill().tolist())

# ------------------------------------------------------------------------------
# 3.7 Strategy 6: Interpolation
# ------------------------------------------------------------------------------
print("\n--- 3.7 Strategy 6: Interpolation ---")

print("Interpolation estimates missing values based on neighbors")

# Linear interpolation
ts_data["Interpolated"] = ts_data["Value"].interpolate(method="linear")
print("\nLinear interpolation:")
print(ts_data[["Value", "Interpolated"]])

print("\nWhen to use interpolation:")
print("   - Time series data with missing points")
print("   - Numeric data where neighbors provide good estimates")
print("   - NOT suitable for categorical data")


# ==============================================================================
# SECTION 4: HANDLING DUPLICATES
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 4: HANDLING DUPLICATES")
print("=" * 70)

# ------------------------------------------------------------------------------
# 4.1 Detecting Duplicates
# ------------------------------------------------------------------------------
print("\n--- 4.1 Detecting Duplicates ---")

# Check for exact duplicate rows
dup_count = df_working.duplicated().sum()
print(f"Total duplicate rows: {dup_count}")

# View duplicate rows
print("\nFirst few duplicate rows:")
duplicates = df_working[df_working.duplicated(keep=False)]
print(f"All rows involved in duplication: {len(duplicates)}")
print(duplicates[["PassengerId", "Name", "Age", "Ticket"]].head(10))

# Check duplicates based on specific columns
dup_name_count = df_working.duplicated(subset=["Name"]).sum()
print(f"\nDuplicate Names: {dup_name_count}")

dup_ticket_count = df_working.duplicated(subset=["Ticket"]).sum()
print(f"Duplicate Tickets: {dup_ticket_count}")
print("(Note: Same ticket number can be valid for family members)")

# ------------------------------------------------------------------------------
# 4.2 Removing Duplicates
# ------------------------------------------------------------------------------
print("\n--- 4.2 Removing Duplicates ---")

before_count = len(df_working)

# Remove exact duplicate rows, keep first occurrence
df_working = df_working.drop_duplicates(keep="first")

after_count = len(df_working)
print(f"Removed exact duplicates:")
print(f"   Before: {before_count} rows")
print(f"   After:  {after_count} rows")
print(f"   Removed: {before_count - after_count} rows")

# Verify no duplicates remain
assert df_working.duplicated().sum() == 0, "Duplicates still present"
print("Duplicate check passed: 0 duplicates remaining")

# ------------------------------------------------------------------------------
# 4.3 Handling Duplicates by Subset
# ------------------------------------------------------------------------------
print("\n--- 4.3 Handling Duplicates by Subset ---")

# Sometimes you want to dedupe based on a key column
# Example: Keep only first occurrence of each PassengerId
df_deduped_id = df_working.drop_duplicates(subset=["PassengerId"], keep="first")
print(f"After deduping on PassengerId:")
print(f"   Before: {len(df_working)} rows")
print(f"   After:  {len(df_deduped_id)} rows")

# Keep last instead of first
df_deduped_last = df_working.drop_duplicates(subset=["PassengerId"], keep="last")
print(f"   Keeping last: {len(df_deduped_last)} rows")

# Mark duplicates without removing (for review)
df_working["is_duplicate"] = df_working.duplicated(subset=["Name"], keep=False)
dup_names = df_working["is_duplicate"].sum()
print(f"\nRows with duplicate names (for review): {dup_names}")

# Clean up marker column
df_working = df_working.drop(columns=["is_duplicate"])


# ==============================================================================
# SECTION 5: DATA TYPE CONVERSION
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 5: DATA TYPE CONVERSION")
print("=" * 70)

# ------------------------------------------------------------------------------
# 5.1 Checking Current Data Types
# ------------------------------------------------------------------------------
print("\n--- 5.1 Checking Current Data Types ---")

print("Current data types:")
print(df_working.dtypes)

# ------------------------------------------------------------------------------
# 5.2 Converting Numeric Types
# ------------------------------------------------------------------------------
print("\n--- 5.2 Converting Numeric Types ---")

# Age should be integer (or nullable integer)
print("Age dtype before:", df_working["Age"].dtype)

# Convert to nullable integer (Int64 with capital I supports NaN)
df_working["Age"] = df_working["Age"].round().astype("Int64")
print("Age dtype after:", df_working["Age"].dtype)
print("Sample Age values:", df_working["Age"].head().tolist())

# Survived and Pclass should be integers
df_working["Survived"] = df_working["Survived"].astype("Int64")
df_working["Pclass"] = df_working["Pclass"].astype("Int64")
print("Survived dtype:", df_working["Survived"].dtype)
print("Pclass dtype:", df_working["Pclass"].dtype)

# ------------------------------------------------------------------------------
# 5.3 Converting String to Numeric
# ------------------------------------------------------------------------------
print("\n--- 5.3 Converting String to Numeric ---")

# FareStr column has numeric values stored as strings
print("FareStr sample values:", df_working["FareStr"].head(10).tolist())
print("FareStr dtype:", df_working["FareStr"].dtype)

# Use pd.to_numeric with errors='coerce' to handle invalid values
df_working["FareFromStr"] = pd.to_numeric(df_working["FareStr"], errors="coerce")

print("\nAfter pd.to_numeric:")
print("FareFromStr dtype:", df_working["FareFromStr"].dtype)
coerced_nan = df_working["FareFromStr"].isnull().sum()
print(f"Values coerced to NaN: {coerced_nan}")

# Fill coerced NaN with original Fare median
df_working["FareFromStr"] = df_working["FareFromStr"].fillna(df_working["Fare"].median())

# Drop the temporary column
df_working = df_working.drop(columns=["FareStr", "FareFromStr"])

# ------------------------------------------------------------------------------
# 5.4 Converting to Category Type
# ------------------------------------------------------------------------------
print("\n--- 5.4 Converting to Category Type ---")

# Categorical columns save memory and enable special operations
print("Sex dtype before:", df_working["Sex"].dtype)
print("Sex memory usage:", df_working["Sex"].memory_usage(deep=True), "bytes")

df_working["Sex"] = df_working["Sex"].astype("category")
print("\nSex dtype after:", df_working["Sex"].dtype)
print("Sex memory usage:", df_working["Sex"].memory_usage(deep=True), "bytes")
print("Sex categories:", df_working["Sex"].cat.categories.tolist())

# Convert other categorical columns
df_working["Embarked"] = df_working["Embarked"].astype("category")
df_working["Pclass"] = df_working["Pclass"].astype("category")

print("\nCategorical columns converted: Sex, Embarked, Pclass")

# ------------------------------------------------------------------------------
# 5.5 Converting to Datetime
# ------------------------------------------------------------------------------
print("\n--- 5.5 Converting to Datetime ---")

# Create an example date column
df_working["BookingDate"] = "2024-01-15"
df_working.loc[:100, "BookingDate"] = "2024-02-20"
df_working.loc[101:200, "BookingDate"] = "2024-03-10"

print("BookingDate dtype before:", df_working["BookingDate"].dtype)

df_working["BookingDate"] = pd.to_datetime(df_working["BookingDate"])
print("BookingDate dtype after:", df_working["BookingDate"].dtype)
print("Sample dates:", df_working["BookingDate"].head().tolist())

# Extract date components
df_working["BookingMonth"] = df_working["BookingDate"].dt.month
df_working["BookingDayOfWeek"] = df_working["BookingDate"].dt.day_name()
print("\nExtracted date components:")
print(df_working[["BookingDate", "BookingMonth", "BookingDayOfWeek"]].head())

# Clean up demo columns
df_working = df_working.drop(columns=["BookingDate", "BookingMonth", "BookingDayOfWeek"])


# ==============================================================================
# SECTION 6: CLEANING STRING COLUMNS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 6: CLEANING STRING COLUMNS")
print("=" * 70)

# ------------------------------------------------------------------------------
# 6.1 Removing Whitespace
# ------------------------------------------------------------------------------
print("\n--- 6.1 Removing Whitespace ---")

# Check for leading/trailing whitespace in Name
sample_names = df_working["Name"].head(20)
has_whitespace = sample_names.str.contains(r"^\s|\s$", regex=True, na=False)
print("Names with leading/trailing whitespace:", has_whitespace.sum())

# Strip whitespace from all string columns
string_columns = df_working.select_dtypes(include=["object"]).columns
for col in string_columns:
    df_working[col] = df_working[col].str.strip()

print(f"Stripped whitespace from columns: {list(string_columns)}")

# Verify
sample_names_after = df_working["Name"].head(20)
has_whitespace_after = sample_names_after.str.contains(r"^\s|\s$", regex=True, na=False)
print("Names with whitespace after cleaning:", has_whitespace_after.sum())

# ------------------------------------------------------------------------------
# 6.2 Standardizing Case
# ------------------------------------------------------------------------------
print("\n--- 6.2 Standardizing Case ---")

# Check unique values in Sex column
print("Unique Sex values before:", df_working["Sex"].unique())

# Need to convert back from category to string to modify
df_working["Sex"] = df_working["Sex"].astype(str)

# Standardize to lowercase
df_working["Sex"] = df_working["Sex"].str.lower()

print("Unique Sex values after:", df_working["Sex"].unique())

# Convert back to category
df_working["Sex"] = df_working["Sex"].astype("category")

# ------------------------------------------------------------------------------
# 6.3 Replacing Invalid Values
# ------------------------------------------------------------------------------
print("\n--- 6.3 Replacing Invalid Values ---")

# Check Embarked for invalid values
print("Unique Embarked values:", df_working["Embarked"].unique())

# Convert to string to clean
df_working["Embarked"] = df_working["Embarked"].astype(str)

# Replace invalid values
invalid_embarked = ["Unknown", "", "nan", "NaN"]
valid_embarked = ["S", "C", "Q"]

# Count invalid
invalid_count = df_working["Embarked"].isin(invalid_embarked).sum()
print(f"Invalid Embarked values: {invalid_count}")

# Replace with mode
embarked_mode = "S"  # Most common embarkation point
df_working.loc[~df_working["Embarked"].isin(valid_embarked), "Embarked"] = embarked_mode

print("Unique Embarked values after cleaning:", df_working["Embarked"].unique())

# Convert back to category
df_working["Embarked"] = df_working["Embarked"].astype("category")

# ------------------------------------------------------------------------------
# 6.4 Extracting Information from Strings
# ------------------------------------------------------------------------------
print("\n--- 6.4 Extracting Information from Strings ---")

# Extract title from Name
df_working["Title"] = df_working["Name"].str.extract(r",\s*([A-Za-z]+)\.", expand=False)

print("Extracted titles:")
print(df_working["Title"].value_counts())

# Standardize rare titles
title_mapping = {
    "Mlle": "Miss",
    "Ms": "Miss",
    "Mme": "Mrs",
    "Lady": "Mrs",
    "Countess": "Mrs",
    "Capt": "Officer",
    "Col": "Officer",
    "Don": "Mr",
    "Dr": "Dr",
    "Major": "Officer",
    "Rev": "Rev",
    "Sir": "Mr",
    "Jonkheer": "Mr",
    "Dona": "Mrs"
}

df_working["Title"] = df_working["Title"].replace(title_mapping)
print("\nTitles after standardization:")
print(df_working["Title"].value_counts())

# ------------------------------------------------------------------------------
# 6.5 Cleaning PassengerIdStr Column
# ------------------------------------------------------------------------------
print("\n--- 6.5 Extracting Numeric ID from String ---")

print("PassengerIdStr sample:", df_working["PassengerIdStr"].head().tolist())

# Extract numeric part
df_working["PassengerIdExtracted"] = (
    df_working["PassengerIdStr"]
    .str.replace("P", "", regex=False)
    .str.lstrip("0")
    .astype(int)
)

print("Extracted IDs:", df_working["PassengerIdExtracted"].head().tolist())

# Verify extraction matches original
match_count = (df_working["PassengerId"] == df_working["PassengerIdExtracted"]).sum()
print(f"Matching IDs: {match_count} / {len(df_working)}")

# Drop temporary columns
df_working = df_working.drop(columns=["PassengerIdStr", "PassengerIdExtracted"])


# ==============================================================================
# SECTION 7: DATA VALIDATION
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 7: DATA VALIDATION")
print("=" * 70)

# ------------------------------------------------------------------------------
# 7.1 Assertion-Based Validation
# ------------------------------------------------------------------------------
print("\n--- 7.1 Assertion-Based Validation ---")

def validate_cleaned_data(data):
    """
    Validate that cleaned data meets all quality requirements.
    Raises AssertionError if any check fails.
    """
    print("Running validation checks...")
    checks_passed = 0
    total_checks = 0

    # Check 1: No duplicate rows
    total_checks += 1
    dup_count = data.duplicated().sum()
    assert dup_count == 0, f"Found {dup_count} duplicate rows"
    print("   [PASS] No duplicate rows")
    checks_passed += 1

    # Check 2: No missing values in critical columns
    critical_columns = ["Survived", "Pclass", "Sex", "Age", "Fare"]
    for col in critical_columns:
        total_checks += 1
        missing = data[col].isnull().sum()
        assert missing == 0, f"Column {col} has {missing} missing values"
        print(f"   [PASS] No missing values in {col}")
        checks_passed += 1

    # Check 3: Valid value ranges
    total_checks += 1
    assert data["Age"].min() >= 0, "Negative age found"
    assert data["Age"].max() <= 120, "Age exceeds 120"
    print("   [PASS] Age in valid range [0, 120]")
    checks_passed += 1

    total_checks += 1
    assert data["Fare"].min() >= 0, "Negative fare found"
    print("   [PASS] Fare is non-negative")
    checks_passed += 1

    # Check 4: Valid categorical values
    total_checks += 1
    valid_sex = {"male", "female"}
    actual_sex = set(data["Sex"].unique())
    assert actual_sex.issubset(valid_sex), f"Invalid Sex values: {actual_sex - valid_sex}"
    print("   [PASS] Sex values are valid")
    checks_passed += 1

    total_checks += 1
    valid_embarked = {"S", "C", "Q"}
    actual_embarked = set(data["Embarked"].astype(str).unique())
    assert actual_embarked.issubset(valid_embarked), f"Invalid Embarked: {actual_embarked - valid_embarked}"
    print("   [PASS] Embarked values are valid")
    checks_passed += 1

    total_checks += 1
    valid_pclass = {1, 2, 3}
    actual_pclass = set(data["Pclass"].unique())
    assert actual_pclass.issubset(valid_pclass), f"Invalid Pclass: {actual_pclass - valid_pclass}"
    print("   [PASS] Pclass values are valid")
    checks_passed += 1

    # Check 5: Survived is binary
    total_checks += 1
    valid_survived = {0, 1}
    actual_survived = set(data["Survived"].unique())
    assert actual_survived.issubset(valid_survived), f"Invalid Survived: {actual_survived}"
    print("   [PASS] Survived is binary (0/1)")
    checks_passed += 1

    print(f"\nValidation complete: {checks_passed}/{total_checks} checks passed")
    return True


# Run validation
try:
    validate_cleaned_data(df_working)
    print("\nAll validation checks passed successfully")
except AssertionError as e:
    print(f"\nValidation failed: {e}")

# ------------------------------------------------------------------------------
# 7.2 Statistical Validation
# ------------------------------------------------------------------------------
print("\n--- 7.2 Statistical Validation ---")

print("Comparing statistics before and after cleaning:")
print("\nOriginal data statistics:")
print(df_clean[["Age", "Fare"]].describe())

print("\nCleaned data statistics:")
print(df_working[["Age", "Fare"]].describe())

# Check if distributions are reasonably preserved
age_diff = abs(df_clean["Age"].mean() - df_working["Age"].mean())
fare_diff = abs(df_clean["Fare"].mean() - df_working["Fare"].mean())

print(f"\nMean Age difference: {age_diff:.2f}")
print(f"Mean Fare difference: {fare_diff:.2f}")

if age_diff < 5 and fare_diff < 10:
    print("Statistical distributions reasonably preserved after cleaning")


# ==============================================================================
# SECTION 8: COMPLETE DATA CLEANING PIPELINE
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 8: COMPLETE DATA CLEANING PIPELINE")
print("=" * 70)

def clean_titanic_data(data, verbose=True):
    """
    Complete data cleaning pipeline for Titanic dataset.
    
    Parameters
    ----------
    data : pd.DataFrame
        Raw Titanic data
    verbose : bool
        Print progress messages
    
    Returns
    -------
    pd.DataFrame
        Cleaned data
    """
    if verbose:
        print("Starting data cleaning pipeline...")
        print(f"Input shape: {data.shape}")
    
    df = data.copy()
    
    # Step 1: Remove duplicates
    before = len(df)
    df = df.drop_duplicates()
    if verbose:
        print(f"Step 1: Removed {before - len(df)} duplicate rows")
    
    # Step 2: Clean string columns
    string_cols = df.select_dtypes(include=["object"]).columns
    for col in string_cols:
        df[col] = df[col].str.strip()
    if verbose:
        print(f"Step 2: Stripped whitespace from {len(string_cols)} columns")
    
    # Step 3: Standardize Sex column
    if "Sex" in df.columns:
        df["Sex"] = df["Sex"].str.lower()
        valid_sex = ["male", "female"]
        df.loc[~df["Sex"].isin(valid_sex), "Sex"] = np.nan
        df["Sex"] = df["Sex"].fillna(df["Sex"].mode()[0])
    if verbose:
        print("Step 3: Standardized Sex column")
    
    # Step 4: Clean Embarked column
    if "Embarked" in df.columns:
        valid_embarked = ["S", "C", "Q"]
        df.loc[~df["Embarked"].isin(valid_embarked), "Embarked"] = np.nan
        df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    if verbose:
        print("Step 4: Cleaned Embarked column")
    
    # Step 5: Fill missing Age with median by Pclass and Sex
    if "Age" in df.columns:
        df["Age"] = df.groupby(["Pclass", "Sex"])["Age"].transform(
            lambda x: x.fillna(x.median())
        )
        df["Age"] = df["Age"].fillna(df["Age"].median())
    if verbose:
        print("Step 5: Filled missing Age values")
    
    # Step 6: Fill missing Fare with median by Pclass
    if "Fare" in df.columns:
        df["Fare"] = df.groupby("Pclass")["Fare"].transform(
            lambda x: x.fillna(x.median())
        )
        df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    if verbose:
        print("Step 6: Filled missing Fare values")
    
    # Step 7: Fill Cabin with 'Unknown'
    if "Cabin" in df.columns:
        df["Cabin"] = df["Cabin"].fillna("Unknown")
    if verbose:
        print("Step 7: Filled missing Cabin values")
    
    # Step 8: Extract Title from Name
    if "Name" in df.columns:
        df["Title"] = df["Name"].str.extract(r",\s*([A-Za-z]+)\.", expand=False)
        title_mapping = {
            "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
            "Lady": "Mrs", "Countess": "Mrs", "Capt": "Officer",
            "Col": "Officer", "Don": "Mr", "Major": "Officer",
            "Rev": "Rev", "Sir": "Mr", "Jonkheer": "Mr", "Dona": "Mrs"
        }
        df["Title"] = df["Title"].replace(title_mapping)
        df["Title"] = df["Title"].fillna("Unknown")
    if verbose:
        print("Step 8: Extracted and standardized Title")
    
    # Step 9: Convert data types
    if "Survived" in df.columns:
        df["Survived"] = df["Survived"].astype("Int64")
    if "Pclass" in df.columns:
        df["Pclass"] = df["Pclass"].astype("category")
    if "Sex" in df.columns:
        df["Sex"] = df["Sex"].astype("category")
    if "Embarked" in df.columns:
        df["Embarked"] = df["Embarked"].astype("category")
    if verbose:
        print("Step 9: Converted data types")
    
    # Step 10: Drop unnecessary columns
    cols_to_drop = ["PassengerIdStr", "FareStr"]
    existing_to_drop = [c for c in cols_to_drop if c in df.columns]
    if existing_to_drop:
        df = df.drop(columns=existing_to_drop)
    if verbose:
        print(f"Step 10: Dropped {len(existing_to_drop)} unnecessary columns")
    
    if verbose:
        print(f"Output shape: {df.shape}")
        print("Data cleaning pipeline complete")
    
    return df


# Run the complete pipeline on original messy data
df_final = clean_titanic_data(df.copy())

# Final validation
print("\n" + "-" * 50)
print("FINAL DATA QUALITY CHECK")
print("-" * 50)
assess_data_quality(df_final, "Cleaned Titanic Data")


# ==============================================================================
# SECTION 9: COMMON PITFALLS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 9: COMMON PITFALLS")
print("=" * 70)

pitfalls = """
Pitfall 1: Filling NaN before understanding why it is missing
   - Missing values might indicate important information (e.g., no cabin = lower class)
   - Always investigate BEFORE choosing a fill strategy

Pitfall 2: Using mean for skewed distributions
   - Median is more robust for skewed data (like Fare)
   - Mean can be heavily influenced by outliers

Pitfall 3: Modifying data in place without copy
   - WRONG: df.fillna(0, inplace=True) in the middle of a pipeline
   - RIGHT: df = df.fillna(0) or work on explicit copies

Pitfall 4: Dropping too many rows with dropna()
   - dropna() without subset= drops rows with ANY missing value
   - This can eliminate most of your data

Pitfall 5: Forgetting that drop_duplicates uses ALL columns by default
   - Use subset= to specify which columns define uniqueness

Pitfall 6: Not validating after cleaning
   - Always check shape, missing counts, and value ranges
   - Use assertions to catch problems early

Pitfall 7: String comparison failures due to case/whitespace
   - Always strip() and lower() before comparing strings
   - Use .str accessor methods for string cleaning

Pitfall 8: Using inplace=True
   - Makes code harder to debug and chain
   - Reassignment (df = df.method()) is clearer
"""
print(pitfalls)


# ==============================================================================
# SECTION 10: SUMMARY TABLE
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 10: SUMMARY TABLE")
print("=" * 70)

summary = """
Operation                      | Syntax
-------------------------------|--------------------------------------------
Check missing values           | df.isnull().sum()
Drop rows with any NaN         | df.dropna()
Drop rows with NaN in column   | df.dropna(subset=["col"])
Fill NaN with value            | df["col"].fillna(value)
Fill NaN with mean             | df["col"].fillna(df["col"].mean())
Fill NaN by group              | df.groupby("g")["col"].transform(lambda x: x.fillna(x.median()))
Forward fill                   | df["col"].ffill()
Interpolate                    | df["col"].interpolate()
Check duplicates               | df.duplicated().sum()
Drop duplicates                | df.drop_duplicates()
Drop duplicates by column      | df.drop_duplicates(subset=["col"])
Convert to numeric             | pd.to_numeric(df["col"], errors="coerce")
Convert to category            | df["col"].astype("category")
Convert to datetime            | pd.to_datetime(df["col"])
Strip whitespace               | df["col"].str.strip()
Lowercase                      | df["col"].str.lower()
Replace values                 | df["col"].replace({"old": "new"})
Extract pattern                | df["col"].str.extract(r"pattern")
"""
print(summary)


# ==============================================================================
# SECTION 11: PRACTICE EXERCISES
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 11: PRACTICE EXERCISES")
print("=" * 70)

print("Exercise 1: Count missing values as percentage")
missing_pct = (df_clean.isnull().mean() * 100).round(2)
print("Missing value percentages:")
print(missing_pct[missing_pct > 0])

print("\nExercise 2: Fill Age with median by Pclass only")
df_ex2 = df_clean.copy()
df_ex2["Age"] = df_ex2.groupby("Pclass")["Age"].transform(
    lambda x: x.fillna(x.median())
)
print("Missing Age after fill:", df_ex2["Age"].isnull().sum())

print("\nExercise 3: Find rows where Name contains special characters")
special_char_mask = df_clean["Name"].str.contains(r"[^a-zA-Z,.\s\-\(\)\']", regex=True, na=False)
print("Rows with special characters in Name:", special_char_mask.sum())
if special_char_mask.sum() > 0:
    print(df_clean.loc[special_char_mask, "Name"].head())

print("\nExercise 4: Convert Pclass to ordered category")
df_ex4 = df_clean.copy()
df_ex4["Pclass"] = pd.Categorical(
    df_ex4["Pclass"],
    categories=[1, 2, 3],
    ordered=True
)
print("Pclass dtype:", df_ex4["Pclass"].dtype)
print("Is ordered:", df_ex4["Pclass"].cat.ordered)


# ==============================================================================
# SECTION 12: KEY TAKEAWAYS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 12: KEY TAKEAWAYS")
print("=" * 70)

takeaways = """
1. Always assess data quality BEFORE and AFTER cleaning
2. Understand WHY data is missing before choosing a fill strategy
3. Use median for skewed data, mode for categorical, mean for normal distributions
4. Group-based filling preserves more realistic distributions
5. Remove duplicates early in the pipeline
6. Standardize string columns: strip whitespace, consistent case
7. Convert dtypes appropriately: category for low-cardinality, datetime for dates
8. Validate cleaned data with assertions and statistical checks
9. Build reusable cleaning functions for consistent pipelines
10. Document all cleaning steps for reproducibility
"""
print(takeaways)


# ==============================================================================
# NEXT LESSON PREVIEW
# ==============================================================================
print("\n" + "=" * 70)
print("NEXT LESSON PREVIEW")
print("=" * 70)

preview = """
Lesson 21: Data Types and Conversion in Pandas

You will learn:
- Deep dive into Pandas dtypes (int64, float64, object, category, datetime64)
- Memory optimization through dtype selection
- Nullable integer types (Int64 vs int64)
- String dtype vs object dtype
- Working with datetime and timedelta
- Automatic type inference and its pitfalls
- Best practices for dtype management in production
"""
print(preview)

print("\n" + "=" * 70)
print("END OF LESSON 20")
print("=" * 70)
print("\nYou can now systematically clean messy datasets by handling missing values,")
print("removing duplicates, converting types, and validating the results.")