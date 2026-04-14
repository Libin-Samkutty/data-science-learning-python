"""
LESSON 33: FEATURE ENGINEERING BASICS
================================================================================

What You Will Learn:
- What feature engineering is and why it matters for ML
- Numeric transformations: log, power, binning, scaling, interactions
- Categorical encoding: label, one-hot, target, frequency encoding
- Date and time feature extraction
- Text-based feature extraction
- Feature crossing and polynomial features
- Handling skewed distributions
- Building a reusable feature engineering pipeline
- Validating features for ML readiness

Real World Usage:
- Preparing raw data for machine learning model training
- Creating business-meaningful derived variables
- Encoding categorical variables for tree-based and linear models
- Extracting temporal patterns from date columns
- Building consistent feature pipelines for training and inference

Dataset Used:
Titanic dataset (public, no login required)
URL: https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv

================================================================================
"""

import numpy as np
import pandas as pd
import re
import time

print("=" * 70)
print("LESSON 33: FEATURE ENGINEERING BASICS")
print("=" * 70)


# ==============================================================================
# SECTION 1: LOAD DATASET
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 1: LOAD DATASET")
print("=" * 70)

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

print("Loading Titanic dataset from:")
print(url)

df_raw = pd.read_csv(url)
df_raw.columns = [
    c.strip().lower().replace(" ", "_") for c in df_raw.columns
]

print(f"\nShape: {df_raw.shape}")
print(f"Columns: {list(df_raw.columns)}")
print("\nHead:")
print(df_raw.head())
print("\nDtypes:")
print(df_raw.dtypes)
print("\nMissing values:")
print(df_raw.isnull().sum())
print("\nTarget variable distribution (survived):")
print(df_raw["survived"].value_counts())
print(f"Survival rate: {df_raw['survived'].mean()*100:.1f}%")

# Working copy
df = df_raw.copy()


# ==============================================================================
# SECTION 2: WHAT IS FEATURE ENGINEERING
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 2: WHAT IS FEATURE ENGINEERING")
print("=" * 70)

explanation = """
FEATURE ENGINEERING is the process of creating new input variables
from raw data to improve model performance.

Raw data rarely has the optimal representation for learning.
Feature engineering bridges the gap between raw data and model input.

FEATURE TYPES:
  1. Numeric transformations   (log, power, scaling, binning)
  2. Categorical encodings     (label, one-hot, target, frequency)
  3. Date/time extractions     (hour, day of week, is_weekend)
  4. Text extractions          (word count, patterns, titles)
  5. Interaction features      (product, ratio of two columns)
  6. Aggregation features      (group means, counts, ranks)

IMPORTANT RULES:
  - Never use the target variable to create features on the full dataset
    (causes data leakage, except with cross-validated target encoding)
  - Apply the SAME transformations to training and test/inference data
  - Document every transformation for reproducibility
  - Validate feature distributions after creation
"""
print(explanation)


# ==============================================================================
# SECTION 3: NUMERIC TRANSFORMATIONS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 3: NUMERIC TRANSFORMATIONS")
print("=" * 70)

# Fill missing numerics for demonstration
df["age"]  = df["age"].fillna(df["age"].median())
df["fare"] = df["fare"].fillna(df["fare"].median())

# ------------------------------------------------------------------------------
# 3.1 Log Transform
# ------------------------------------------------------------------------------
print("\n--- 3.1 Log Transform ---")

explanation = """
Log transform compresses right-skewed distributions.
Use np.log1p (log(1+x)) to handle zero values safely.
Inverse: np.expm1 (exp(x)-1).

When to use:
  - Highly right-skewed numeric columns (skew > 1)
  - Values spanning multiple orders of magnitude (e.g. income, prices)
  - Linear models that assume normal distributions
"""
print(explanation)

print(f"Fare skewness before: {df['fare'].skew():.3f}")

df["fare_log"] = np.log1p(df["fare"])

print(f"Fare skewness after log: {df['fare_log'].skew():.3f}")
print(f"\nOriginal fare sample:  {df['fare'].head(5).tolist()}")
print(f"Log-transformed fare:  {df['fare_log'].head(5).round(4).tolist()}")

# Verify inverse
reconstructed = np.expm1(df["fare_log"])
assert np.allclose(df["fare"], reconstructed), "Log inverse failed"
print("Inverse verification passed: expm1(log1p(x)) == x")

# ------------------------------------------------------------------------------
# 3.2 Power Transform (Square Root, Box-Cox Inspired)
# ------------------------------------------------------------------------------
print("\n--- 3.2 Power Transform ---")

# Square root: milder than log, works for moderate skew
df["fare_sqrt"] = np.sqrt(df["fare"])

# Square: amplifies differences (use for left-skewed data)
df["age_squared"] = df["age"] ** 2

print(f"Fare skewness after sqrt: {df['fare_sqrt'].skew():.3f}")
print(f"Age skewness before:      {df['age'].skew():.3f}")
print(f"Age^2 skewness:           {df['age_squared'].skew():.3f}")

# ------------------------------------------------------------------------------
# 3.3 Binning (Discretization)
# ------------------------------------------------------------------------------
print("\n--- 3.3 Binning (Discretization) ---")

explanation = """
Binning converts continuous variables into discrete categories.

Two approaches:
  pd.cut():  Equal-width bins (by value range)
  pd.qcut(): Equal-frequency bins (same number of rows per bin)

When to use:
  - Non-linear relationships (tree models handle this internally)
  - Creating business-meaningful categories (age groups, price tiers)
  - Reducing the effect of outliers
"""
print(explanation)

# Equal-width bins
df["age_bin_equal"] = pd.cut(
    df["age"],
    bins=[0, 12, 18, 35, 60, 120],
    labels=["child", "teen", "young_adult", "adult", "senior"]
)

# Equal-frequency bins (quartiles)
df["fare_quartile"] = pd.qcut(
    df["fare"],
    q=4,
    labels=["Q1_low", "Q2_mid_low", "Q3_mid_high", "Q4_high"]
)

print("Age bins (equal-width):")
print(df["age_bin_equal"].value_counts().sort_index())

print("\nFare quartiles (equal-frequency):")
print(df["fare_quartile"].value_counts().sort_index())

# Verify each quartile has roughly same count
quartile_counts = df["fare_quartile"].value_counts()
expected_per_bin = len(df) / 4
max_deviation = abs(quartile_counts - expected_per_bin).max()
print(f"\nMax deviation from expected per quartile: {max_deviation:.0f} rows")

# ------------------------------------------------------------------------------
# 3.4 Min-Max Scaling
# ------------------------------------------------------------------------------
print("\n--- 3.4 Min-Max Scaling ---")

explanation = """
Scales values to [0, 1] range.
Formula: (x - min) / (max - min)

When to use:
  - Distance-based models (KNN, SVM)
  - Neural networks (input normalization)
  - When you need bounded feature values

Do NOT use when:
  - Tree-based models (they are scale-invariant)
  - Outliers are extreme (they compress the bulk of data)
"""
print(explanation)


def min_max_scale(series):
    """Scale a Series to [0, 1] range."""
    s_min = series.min()
    s_max = series.max()
    if s_max == s_min:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - s_min) / (s_max - s_min)


df["age_scaled"]  = min_max_scale(df["age"])
df["fare_scaled"] = min_max_scale(df["fare"])

print(f"Age  range: [{df['age'].min():.1f}, {df['age'].max():.1f}] -> [{df['age_scaled'].min():.4f}, {df['age_scaled'].max():.4f}]")
print(f"Fare range: [{df['fare'].min():.1f}, {df['fare'].max():.1f}] -> [{df['fare_scaled'].min():.4f}, {df['fare_scaled'].max():.4f}]")

# ------------------------------------------------------------------------------
# 3.5 Standard Scaling (Z-Score)
# ------------------------------------------------------------------------------
print("\n--- 3.5 Standard Scaling (Z-Score) ---")

explanation = """
Centers data at mean=0 and std=1.
Formula: (x - mean) / std

When to use:
  - Linear models, logistic regression
  - PCA and other algorithms that assume zero-centered data
  - When comparing features on different scales
"""
print(explanation)


def z_score_scale(series):
    """Standardize to mean=0, std=1."""
    mean = series.mean()
    std  = series.std()
    if std == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - mean) / std


df["age_zscore"]  = z_score_scale(df["age"])
df["fare_zscore"] = z_score_scale(df["fare"])

print(f"Age  z-score: mean={df['age_zscore'].mean():.6f}, std={df['age_zscore'].std():.6f}")
print(f"Fare z-score: mean={df['fare_zscore'].mean():.6f}, std={df['fare_zscore'].std():.6f}")


# ==============================================================================
# SECTION 4: CATEGORICAL ENCODING
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 4: CATEGORICAL ENCODING")
print("=" * 70)

# Fill missing categoricals for demonstration
df["embarked"] = df["embarked"].fillna("S")

# ------------------------------------------------------------------------------
# 4.1 Label Encoding (Ordinal Encoding)
# ------------------------------------------------------------------------------
print("\n--- 4.1 Label Encoding ---")

explanation = """
Maps each category to a unique integer.
Preserves memory, but implies an order that may not exist.

When to use:
  - Ordinal variables (education level, rating)
  - Tree-based models (they split on thresholds, order does not matter)

When NOT to use:
  - Linear models on nominal variables (model interprets as numeric order)
"""
print(explanation)

# Binary encoding (special case of label encoding)
sex_map = {"male": 0, "female": 1}
df["sex_encoded"] = df["sex"].map(sex_map)
print("Binary encoding of sex:")
print(df[["sex", "sex_encoded"]].drop_duplicates())

# Multi-class label encoding
embarked_map = {"S": 0, "C": 1, "Q": 2}
df["embarked_label"] = df["embarked"].map(embarked_map)
print("\nLabel encoding of embarked:")
print(df[["embarked", "embarked_label"]].drop_duplicates())

# Validate no unmapped values
assert df["sex_encoded"].isnull().sum() == 0, "Unmapped sex values"
assert df["embarked_label"].isnull().sum() == 0, "Unmapped embarked values"
print("\nNo unmapped values in label encoding.")

# ------------------------------------------------------------------------------
# 4.2 One-Hot Encoding (Dummy Variables)
# ------------------------------------------------------------------------------
print("\n--- 4.2 One-Hot Encoding ---")

explanation = """
Creates a binary column for each category value.
No implied order, but increases dimensionality.

When to use:
  - Nominal variables with low cardinality (<20 unique values)
  - Linear models and neural networks
  - When you want the model to treat each category independently

When NOT to use:
  - High cardinality columns (creates too many columns)
  - Tree models (label encoding is usually sufficient)

drop_first=True avoids the dummy variable trap (multicollinearity).
"""
print(explanation)

# One-hot encode embarked
embarked_dummies = pd.get_dummies(
    df["embarked"],
    prefix="embarked",
    drop_first=False,
    dtype=int
)

print("One-hot encoded embarked:")
print(embarked_dummies.head(5))
print(f"New columns: {list(embarked_dummies.columns)}")

# With drop_first to avoid multicollinearity
embarked_dummies_drop = pd.get_dummies(
    df["embarked"],
    prefix="embarked",
    drop_first=True,
    dtype=int
)

print("\nWith drop_first=True:")
print(embarked_dummies_drop.head(5))
print(f"Columns (one fewer): {list(embarked_dummies_drop.columns)}")

# Add to DataFrame
df = pd.concat([df, embarked_dummies], axis=1)

# One-hot encode pclass (it looks numeric but is actually ordinal/categorical)
pclass_dummies = pd.get_dummies(
    df["pclass"],
    prefix="class",
    dtype=int
)
df = pd.concat([df, pclass_dummies], axis=1)
print(f"\nOne-hot encoded pclass columns: {list(pclass_dummies.columns)}")

# ------------------------------------------------------------------------------
# 4.3 Frequency Encoding
# ------------------------------------------------------------------------------
print("\n--- 4.3 Frequency Encoding ---")

explanation = """
Replaces each category with how often it appears in the dataset.
Captures the 'commonness' of each category as a numeric signal.

When to use:
  - High cardinality columns where one-hot is not practical
  - Any model type (preserves monotonic relationship with frequency)
  - When popularity or frequency is a meaningful signal
"""
print(explanation)

# Extract title from name for a higher-cardinality example
df["title"] = df["name"].str.extract(r",\s*([A-Za-z]+)\.", expand=False)

title_freq = df["title"].value_counts(normalize=True)
df["title_freq_encoded"] = df["title"].map(title_freq)

print("Frequency encoding of title:")
freq_demo = (
    df[["title", "title_freq_encoded"]]
    .drop_duplicates()
    .sort_values("title_freq_encoded", ascending=False)
)
print(freq_demo.head(10).round(4))

# Verify: frequencies sum to 1
total_freq = df["title_freq_encoded"].sum()
expected   = len(df)  # Each row maps to its category's proportion, sum = N * sum(proportions) = N
print(f"\nFrequency encoding validation: sum of all row frequencies = {total_freq:.2f}")
print(f"Expected (n_rows * 1.0): {expected}")

# ------------------------------------------------------------------------------
# 4.4 Target Encoding (Mean Encoding)
# ------------------------------------------------------------------------------
print("\n--- 4.4 Target Encoding ---")

explanation = """
Replaces each category with the mean of the target variable for that category.
Very powerful but DANGEROUS: causes data leakage if not done correctly.

CORRECT APPROACH:
  - Compute target means ONLY on the training set
  - Apply the same mapping to test/validation sets
  - Or use cross-validated target encoding (leave-one-out)

For demonstration, we compute on the full dataset with a warning.
In production, always split before computing target statistics.
"""
print(explanation)

# WARNING: This is for demonstration only.
# In practice, compute on training data only.
print("WARNING: Computing target encoding on full dataset (demo only).")
print("In production, compute on training split only to avoid leakage.\n")

title_target_mean = df.groupby("title")["survived"].mean()
df["title_target_encoded"] = df["title"].map(title_target_mean)

print("Target encoding of title (mean survival rate per title):")
target_enc_demo = (
    df[["title", "title_target_encoded"]]
    .drop_duplicates()
    .sort_values("title_target_encoded", ascending=False)
)
print(target_enc_demo.head(10).round(4))

# Smoothed target encoding (reduces overfitting on rare categories)
global_mean = df["survived"].mean()
min_samples = 10

title_counts = df.groupby("title")["survived"].count()
title_means  = df.groupby("title")["survived"].mean()

# Bayesian smoothing formula: (count * group_mean + min_samples * global_mean) / (count + min_samples)
smoothed_target = (
    (title_counts * title_means + min_samples * global_mean) /
    (title_counts + min_samples)
)
df["title_target_smoothed"] = df["title"].map(smoothed_target)

print("\nSmoothed target encoding (reduces rare category overfitting):")
comparison = pd.DataFrame({
    "title": title_means.index,
    "count": title_counts.values,
    "raw_target_enc": title_means.values.round(4),
    "smoothed_enc": smoothed_target.values.round(4),
    "global_mean": global_mean
}).sort_values("count", ascending=False)
print(comparison.head(10).to_string(index=False))


# ==============================================================================
# SECTION 5: TEXT-BASED FEATURE EXTRACTION
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 5: TEXT-BASED FEATURE EXTRACTION")
print("=" * 70)

# ------------------------------------------------------------------------------
# 5.1 Extract Patterns from Name Column
# ------------------------------------------------------------------------------
print("\n--- 5.1 Features from Name Column ---")

# Title already extracted above
# Name length
df["name_length"] = df["name"].str.len()

# Word count
df["name_word_count"] = df["name"].str.split().str.len()

# Has parenthesised text (indicates maiden name)
df["has_maiden_name"] = df["name"].str.contains(r"\(", regex=True, na=False).astype(int)

# Has quotation marks (indicates nickname)
df["has_nickname"] = df["name"].str.contains(r'"', regex=True, na=False).astype(int)

print("Text features from name:")
print(df[["name", "title", "name_length", "name_word_count",
          "has_maiden_name", "has_nickname"]].head(8))

# Standardise rare titles
title_map = {
    "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
    "Lady": "Mrs", "Countess": "Mrs",
    "Capt": "Officer", "Col": "Officer", "Major": "Officer",
    "Don": "Mr", "Sir": "Mr", "Jonkheer": "Mr", "Dona": "Mrs",
    "Rev": "Rev", "Dr": "Dr"
}
df["title_clean"] = df["title"].replace(title_map)

print("\nCleaned title distribution:")
print(df["title_clean"].value_counts())

# ------------------------------------------------------------------------------
# 5.2 Features from Ticket Column
# ------------------------------------------------------------------------------
print("\n--- 5.2 Features from Ticket Column ---")

# Ticket prefix (letters before the number)
df["ticket_prefix"] = df["ticket"].str.extract(r"^([A-Za-z/\.\s]+)", expand=False)
df["ticket_prefix"] = df["ticket_prefix"].str.strip().str.upper()
df["ticket_prefix"] = df["ticket_prefix"].fillna("NONE")

# Ticket is numeric only (no prefix)
df["ticket_is_numeric"] = df["ticket"].str.match(r"^\d+$", na=False).astype(int)

# Shared tickets (same ticket number = travelling together)
ticket_counts = df["ticket"].value_counts()
df["ticket_group_size"] = df["ticket"].map(ticket_counts)

print("Ticket features:")
print(df[["ticket", "ticket_prefix", "ticket_is_numeric", "ticket_group_size"]].head(8))
print(f"\nTicket group size distribution:")
print(df["ticket_group_size"].value_counts().head(5))

# ------------------------------------------------------------------------------
# 5.3 Features from Cabin Column
# ------------------------------------------------------------------------------
print("\n--- 5.3 Features from Cabin Column ---")

# Cabin is mostly missing, but the missing pattern itself is a feature
df["cabin_known"] = df["cabin"].notna().astype(int)

# Extract deck letter (first character of cabin)
df["deck"] = df["cabin"].str.extract(r"^([A-Za-z])", expand=False)
df["deck"] = df["deck"].fillna("Unknown")

# Count of cabin numbers (some passengers have multiple cabins)
df["n_cabins"] = df["cabin"].str.split().str.len()
df["n_cabins"] = df["n_cabins"].fillna(0).astype(int)

print("Cabin features:")
print(df[["cabin", "cabin_known", "deck", "n_cabins"]].head(10))
print(f"\nCabin known vs survived:")
print(df.groupby("cabin_known")["survived"].agg(["mean", "count"]).round(3))
print(f"\nDeck distribution:")
print(df["deck"].value_counts())


# ==============================================================================
# SECTION 6: INTERACTION AND COMBINATION FEATURES
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 6: INTERACTION AND COMBINATION FEATURES")
print("=" * 70)

# ------------------------------------------------------------------------------
# 6.1 Arithmetic Interactions
# ------------------------------------------------------------------------------
print("\n--- 6.1 Arithmetic Interactions ---")

explanation = """
Interaction features capture relationships between two variables
that neither captures alone.

Common patterns:
  Product:    A * B (captures joint effect)
  Ratio:      A / B (captures relative magnitude)
  Difference: A - B (captures gap)
  Sum:        A + B (captures total)
"""
print(explanation)

# Family size
df["family_size"] = df["sibsp"] + df["parch"] + 1

# Is alone
df["is_alone"] = (df["family_size"] == 1).astype(int)

# Fare per person (fare divided by group size)
df["fare_per_person"] = df["fare"] / df["family_size"]

# Age times class (interaction: being old in third class is different from old in first)
df["age_x_class"] = df["age"] * df["pclass"]

# Fare per class (normalise fare within class context)
df["fare_per_class"] = df["fare"] / df["pclass"]

print("Interaction features:")
print(df[["family_size", "is_alone", "fare_per_person",
          "age_x_class", "fare_per_class"]].head(8).round(2))

# Validate: fare_per_person should always be > 0 when fare > 0
assert (df.loc[df["fare"] > 0, "fare_per_person"] > 0).all(), "fare_per_person has zeros"
print("\nfare_per_person validation passed.")

# Survival rate by family size
print("\nSurvival rate by family size:")
family_survival = (
    df.groupby("family_size")["survived"]
    .agg(["mean", "count"])
    .reset_index()
    .rename(columns={"mean": "survival_rate", "count": "n"})
)
family_survival["survival_rate"] = (family_survival["survival_rate"] * 100).round(1)
print(family_survival.to_string(index=False))

# ------------------------------------------------------------------------------
# 6.2 Polynomial Features
# ------------------------------------------------------------------------------
print("\n--- 6.2 Polynomial Features ---")

explanation = """
Create squared and interaction terms for numeric features.
Useful for linear models that cannot learn non-linear relationships.

For tree-based models, polynomial features are usually unnecessary
because trees can learn non-linear splits naturally.
"""
print(explanation)

# Squared terms
df["age_squared"]  = df["age"] ** 2
df["fare_squared"] = df["fare"] ** 2

# Cross term
df["age_fare_interaction"] = df["age"] * df["fare"]

print("Polynomial features:")
print(df[["age", "fare", "age_squared", "fare_squared", "age_fare_interaction"]].head(5).round(2))


# ==============================================================================
# SECTION 7: GROUP-BASED FEATURES (AGGREGATION FEATURES)
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 7: GROUP-BASED FEATURES (AGGREGATION FEATURES)")
print("=" * 70)

explanation = """
Add group-level statistics as features on each row.
Uses groupby().transform() to preserve the original shape.

These features give each row context about its group:
  - How does this passenger's fare compare to their class average?
  - Is this passenger older or younger than their title group?
"""
print(explanation)

# Class-level statistics
df["class_avg_fare"]    = df.groupby("pclass")["fare"].transform("mean")
df["class_median_fare"] = df.groupby("pclass")["fare"].transform("median")
df["class_std_fare"]    = df.groupby("pclass")["fare"].transform("std")
df["class_max_fare"]    = df.groupby("pclass")["fare"].transform("max")
df["class_fare_count"]  = df.groupby("pclass")["fare"].transform("count")

# Relative fare within class
df["fare_ratio_to_class"] = np.where(
    df["class_avg_fare"] > 0,
    df["fare"] / df["class_avg_fare"],
    1.0
)
df["fare_above_class_median"] = (df["fare"] > df["class_median_fare"]).astype(int)

# Rank within group
df["fare_rank_in_class"] = df.groupby("pclass")["fare"].rank(
    ascending=False, method="dense"
).astype(int)

# Title-level statistics
df["title_avg_age"] = df.groupby("title_clean")["age"].transform("mean")
df["age_vs_title_avg"] = df["age"] - df["title_avg_age"]

print("Group-based features:")
print(df[["pclass", "fare", "class_avg_fare", "fare_ratio_to_class",
          "fare_rank_in_class", "fare_above_class_median"]].head(10).round(2))

print("\nTitle-based age features:")
print(df[["title_clean", "age", "title_avg_age", "age_vs_title_avg"]].head(8).round(2))


# ==============================================================================
# SECTION 8: COMPLETE FEATURE ENGINEERING PIPELINE
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 8: COMPLETE FEATURE ENGINEERING PIPELINE")
print("=" * 70)


def engineer_titanic_features(df_in):
    """
    Complete feature engineering pipeline for Titanic dataset.

    Takes raw Titanic DataFrame and returns feature-enriched DataFrame.
    All transformations are documented and deterministic.

    Parameters
    ----------
    df_in : pd.DataFrame
        Raw Titanic data with original column names (lowercase)

    Returns
    -------
    pd.DataFrame with engineered features added
    """
    out = df_in.copy()

    # --- Missing value handling ---
    out["age"]      = out["age"].fillna(out["age"].median())
    out["fare"]     = out["fare"].fillna(out["fare"].median())
    out["embarked"] = out["embarked"].fillna("S")

    # --- Title extraction and cleaning ---
    out["title"] = out["name"].str.extract(r",\s*([A-Za-z]+)\.", expand=False)
    title_map = {
        "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
        "Lady": "Mrs", "Countess": "Mrs",
        "Capt": "Officer", "Col": "Officer", "Major": "Officer",
        "Don": "Mr", "Sir": "Mr", "Jonkheer": "Mr", "Dona": "Mrs",
        "Rev": "Rev", "Dr": "Dr"
    }
    out["title"] = out["title"].replace(title_map)
    out["title"] = out["title"].fillna("Unknown")

    # --- Numeric transforms ---
    out["fare_log"]  = np.log1p(out["fare"])
    out["age_scaled"] = z_score_scale(out["age"])

    # --- Binning ---
    out["age_group"] = pd.cut(
        out["age"],
        bins=[0, 12, 18, 35, 60, 120],
        labels=["child", "teen", "young_adult", "adult", "senior"]
    )

    # --- Family features ---
    out["family_size"]    = out["sibsp"] + out["parch"] + 1
    out["is_alone"]       = (out["family_size"] == 1).astype(int)
    out["fare_per_person"] = out["fare"] / out["family_size"]

    # --- Cabin features ---
    out["cabin_known"] = out["cabin"].notna().astype(int)
    out["deck"]        = out["cabin"].str.extract(r"^([A-Za-z])", expand=False)
    out["deck"]        = out["deck"].fillna("Unknown")
    out["n_cabins"]    = out["cabin"].str.split().str.len().fillna(0).astype(int)

    # --- Ticket features ---
    ticket_counts = out["ticket"].value_counts()
    out["ticket_group_size"]  = out["ticket"].map(ticket_counts)
    out["ticket_is_numeric"]  = out["ticket"].str.match(r"^\d+$", na=False).astype(int)

    # --- Name features ---
    out["name_length"]     = out["name"].str.len()
    out["has_maiden_name"] = out["name"].str.contains(r"\(", regex=True, na=False).astype(int)

    # --- Encodings ---
    out["sex_encoded"]      = out["sex"].map({"male": 0, "female": 1})
    out["embarked_encoded"] = out["embarked"].map({"S": 0, "C": 1, "Q": 2})

    # Frequency encoding for title
    title_freq = out["title"].value_counts(normalize=True)
    out["title_freq"] = out["title"].map(title_freq)

    # --- Group-based features ---
    out["class_avg_fare"]   = out.groupby("pclass")["fare"].transform("mean")
    out["fare_ratio_class"] = np.where(
        out["class_avg_fare"] > 0,
        out["fare"] / out["class_avg_fare"],
        1.0
    )
    out["title_avg_age"]    = out.groupby("title")["age"].transform("mean")
    out["age_vs_title"]     = out["age"] - out["title_avg_age"]

    # --- Interaction features ---
    out["age_x_class"]      = out["age"] * out["pclass"]
    out["fare_per_class"]   = out["fare"] / out["pclass"]

    return out


# Run the pipeline
print("Running complete feature engineering pipeline...")
start = time.perf_counter()
df_featured = engineer_titanic_features(df_raw)
elapsed = (time.perf_counter() - start) * 1000

print(f"Pipeline completed in {elapsed:.1f} ms")
print(f"Input shape:  {df_raw.shape}")
print(f"Output shape: {df_featured.shape}")
print(f"New columns:  {df_featured.shape[1] - df_raw.shape[1]}")

new_cols = [c for c in df_featured.columns if c not in df_raw.columns]
print(f"\nAll engineered features ({len(new_cols)}):")
for i, col in enumerate(new_cols, 1):
    dtype = df_featured[col].dtype
    nulls = df_featured[col].isnull().sum()
    print(f"  {i:2d}. {col:<30} dtype={str(dtype):<12} nulls={nulls}")


# ==============================================================================
# SECTION 9: FEATURE VALIDATION
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 9: FEATURE VALIDATION")
print("=" * 70)


def validate_features(df_in, target_col="survived"):
    """
    Validate engineered features for ML readiness.

    Checks:
      1. No infinite values
      2. No constant features (zero variance)
      3. Acceptable null rates
      4. No perfect correlation with target (possible leakage)
      5. No duplicate columns

    Parameters
    ----------
    df_in : pd.DataFrame
    target_col : str

    Returns
    -------
    dict with validation results
    """
    print("\nRunning feature validation...")

    numeric_cols = df_in.select_dtypes(include=[np.number]).columns.tolist()
    issues = []
    passed = 0
    total  = 0

    # Check 1: Infinite values
    total += 1
    inf_counts = {}
    for col in numeric_cols:
        n_inf = np.isinf(df_in[col]).sum()
        if n_inf > 0:
            inf_counts[col] = int(n_inf)

    if inf_counts:
        issues.append(f"Infinite values found in: {inf_counts}")
        print(f"  [FAIL] Infinite values: {inf_counts}")
    else:
        print(f"  [PASS] No infinite values in {len(numeric_cols)} numeric columns")
        passed += 1

    # Check 2: Constant features
    total += 1
    constant_cols = [c for c in numeric_cols if df_in[c].nunique() <= 1]
    if constant_cols:
        issues.append(f"Constant features: {constant_cols}")
        print(f"  [FAIL] Constant features (zero variance): {constant_cols}")
    else:
        print(f"  [PASS] No constant features")
        passed += 1

    # Check 3: Null rates
    total += 1
    high_null = []
    for col in numeric_cols:
        null_pct = df_in[col].isnull().mean() * 100
        if null_pct > 50:
            high_null.append((col, round(null_pct, 1)))
    if high_null:
        issues.append(f"High null features: {high_null}")
        print(f"  [FAIL] Features with >50% nulls: {high_null}")
    else:
        print(f"  [PASS] All numeric features have acceptable null rates")
        passed += 1

    # Check 4: Leakage check (perfect correlation with target)
    total += 1
    if target_col in df_in.columns and target_col in numeric_cols:
        feature_cols = [c for c in numeric_cols if c != target_col]
        target_corr  = df_in[feature_cols].corrwith(df_in[target_col]).abs()
        perfect_corr = target_corr[target_corr > 0.99]
        if len(perfect_corr) > 0:
            issues.append(f"Possible leakage (corr>0.99 with target): {perfect_corr.index.tolist()}")
            print(f"  [FAIL] Possible data leakage: {perfect_corr.index.tolist()}")
        else:
            print(f"  [PASS] No features have perfect correlation with target")
            passed += 1
    else:
        print(f"  [SKIP] Target column not numeric, skipping leakage check")

    # Check 5: Duplicate columns
    total += 1
    dup_cols = df_in.columns[df_in.columns.duplicated()].tolist()
    if dup_cols:
        issues.append(f"Duplicate column names: {dup_cols}")
        print(f"  [FAIL] Duplicate columns: {dup_cols}")
    else:
        print(f"  [PASS] No duplicate column names")
        passed += 1

    print(f"\nValidation summary: {passed}/{total} checks passed")
    if issues:
        print("Issues to address:")
        for issue in issues:
            print(f"  - {issue}")

    return {"passed": passed, "total": total, "issues": issues}


validation = validate_features(df_featured, target_col="survived")


# ==============================================================================
# SECTION 10: SELECTING FINAL FEATURE SET
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 10: SELECTING FINAL FEATURE SET")
print("=" * 70)

explanation = """
After engineering many features, select the subset that will be
passed to the model. Drop:
  - Raw text columns (name, ticket, cabin)
  - Original columns replaced by encoded versions
  - ID columns (passengerid)
  - Target variable (separate from features)
  - Intermediate columns used only for computation
"""
print(explanation)

# Define the final feature set
drop_from_features = [
    "passengerid", "survived",  # ID and target
    "name", "ticket", "cabin",  # Raw text
    "sex", "embarked",          # Replaced by encoded versions
]

feature_cols = [c for c in df_featured.columns if c not in drop_from_features]

# Separate numeric and categorical features
numeric_features = df_featured[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
categorical_features = df_featured[feature_cols].select_dtypes(include=["object", "category"]).columns.tolist()

print(f"Total feature columns: {len(feature_cols)}")
print(f"Numeric features ({len(numeric_features)}): {numeric_features}")
print(f"Categorical features ({len(categorical_features)}): {categorical_features}")

# Final feature matrix
X = df_featured[numeric_features].copy()
y = df_featured["survived"].copy()

print(f"\nFinal feature matrix X shape: {X.shape}")
print(f"Target vector y shape:        {y.shape}")
print(f"Feature matrix nulls:         {X.isnull().sum().sum()}")

# Fill any remaining nulls in feature matrix
X = X.fillna(0)
print(f"After fill: Feature matrix nulls: {X.isnull().sum().sum()}")

# Quick correlation with target
target_correlations = X.corrwith(y).abs().sort_values(ascending=False)
print("\nTop 15 features correlated with survival:")
print(target_correlations.head(15).round(4))


# ==============================================================================
# SECTION 11: COMMON PITFALLS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 11: COMMON PITFALLS")
print("=" * 70)

pitfalls = """
Pitfall 1: Target leakage in feature engineering
   - Computing target encoding on the entire dataset leaks information
   - Fix: Always compute encoding statistics on training data only

Pitfall 2: Applying different transforms to train and test
   - Scaling parameters (mean, std, min, max) must come from training set
   - Fix: Compute on train, apply to test using stored parameters

Pitfall 3: One-hot encoding creating mismatched columns
   - Test data may have categories not seen in training
   - Fix: Use pd.get_dummies after ensuring same categories, or use
     a fitted encoder that ignores unseen categories

Pitfall 4: Binning that creates empty bins on unseen data
   - Bins based on training quantiles may not cover test range
   - Fix: Use fixed bin edges, not quantile-based edges for production

Pitfall 5: Creating features from missing value patterns
   - cabin_known is valid, but be careful about leaking the target
   - Missingness can be correlated with the outcome

Pitfall 6: Too many features causing overfitting
   - Each feature adds noise; not all add signal
   - Fix: Use feature importance from a model to prune low-value features

Pitfall 7: Using string operations without .str accessor
   - WRONG: df["name"].contains("Mr") raises AttributeError
   - RIGHT: df["name"].str.contains("Mr")

Pitfall 8: Forgetting to handle new categories at inference time
   - A new title or deck value breaks label/frequency encoding
   - Fix: Map unknown values to a default (e.g., 0 or global mean)

Pitfall 9: Not validating feature distributions after creation
   - A feature with 99% zeros or all NaN adds no signal
   - Fix: Run validate_features() after engineering

Pitfall 10: Scaling before splitting
   - Fitting scaler on full data leaks test distribution info
   - Fix: Fit on train, transform both train and test
"""
print(pitfalls)


# ==============================================================================
# SECTION 12: SUMMARY TABLE
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 12: SUMMARY TABLE")
print("=" * 70)

summary = """
Technique                        | Syntax / Pattern
---------------------------------|--------------------------------------------------
Log transform                    | np.log1p(df["col"])
Square root transform            | np.sqrt(df["col"])
Min-max scaling                  | (x - min) / (max - min)
Z-score scaling                  | (x - mean) / std
Equal-width binning              | pd.cut(df["col"], bins=[...], labels=[...])
Equal-frequency binning          | pd.qcut(df["col"], q=4, labels=[...])
Label encoding                   | df["col"].map({"A": 0, "B": 1})
One-hot encoding                 | pd.get_dummies(df["col"], prefix="col")
Frequency encoding               | df["col"].map(df["col"].value_counts(normalize=True))
Target encoding                  | df["col"].map(df.groupby("col")["target"].mean())
Smoothed target encoding         | Bayesian blend of group mean and global mean
String extraction                | df["col"].str.extract(r"pattern")
String contains                  | df["col"].str.contains("text", na=False).astype(int)
Family/group features            | df["a"] + df["b"] + 1
Fare per person                  | df["fare"] / df["family_size"]
Interaction feature              | df["a"] * df["b"]
Group mean feature               | df.groupby("g")["col"].transform("mean")
Rank within group                | df.groupby("g")["col"].rank(method="dense")
Relative to group                | df["col"] / df.groupby("g")["col"].transform("mean")
Feature validation               | validate_features(df, target_col)
"""
print(summary)


# ==============================================================================
# SECTION 13: PRACTICE EXERCISES
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 13: PRACTICE EXERCISES")
print("=" * 70)

print("Exercise 1: Create a 'is_child' binary feature (age < 15)")
df_featured["is_child"] = (df_featured["age"] < 15).astype(int)
child_survival = df_featured.groupby("is_child")["survived"].mean()
print(f"  Child survival rate: {child_survival[1]*100:.1f}%")
print(f"  Adult survival rate: {child_survival[0]*100:.1f}%")

print("\nExercise 2: Frequency-encode the 'deck' column")
deck_freq = df_featured["deck"].value_counts(normalize=True)
df_featured["deck_freq"] = df_featured["deck"].map(deck_freq)
print("  Deck frequency encoding:")
print(df_featured[["deck", "deck_freq"]].drop_duplicates().sort_values("deck_freq", ascending=False))

print("\nExercise 3: Create fare quintile rank within each class")
df_featured["fare_quintile_in_class"] = df_featured.groupby("pclass")["fare"].transform(
    lambda x: pd.qcut(x, q=5, labels=False, duplicates="drop")
)
print("  Fare quintile within class:")
print(df_featured[["pclass", "fare", "fare_quintile_in_class"]].head(10))

print("\nExercise 4: Measure which feature has highest correlation with survived")
num_feats = df_featured.select_dtypes(include=[np.number]).columns
target_corr = df_featured[num_feats].corrwith(df_featured["survived"]).abs()
best_feature = target_corr.drop("survived", errors="ignore").idxmax()
best_corr = target_corr.drop("survived", errors="ignore").max()
print(f"  Best single feature predictor: {best_feature} (|r| = {best_corr:.4f})")


# ==============================================================================
# SECTION 14: KEY TAKEAWAYS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 14: KEY TAKEAWAYS")
print("=" * 70)

takeaways = """
1.  Feature engineering transforms raw data into model-ready inputs.
    Good features often matter more than model choice.

2.  Log transform (np.log1p) is the first tool for right-skewed numerics.

3.  Binning creates discrete categories from continuous values.
    Use pd.cut for fixed-width and pd.qcut for equal-frequency.

4.  Label encoding for ordinal variables and tree models.
    One-hot encoding for nominal variables and linear models.

5.  Frequency and target encoding handle high-cardinality categoricals.
    Always smooth target encoding and compute on training set only.

6.  Text columns are gold mines: titles, prefixes, lengths, patterns.

7.  Interaction features capture joint effects that single features miss.

8.  Group-based features (transform) give each row context about its group.

9.  Always validate features: check for inf, constants, nulls, and leakage.

10. Apply the SAME transformations to training and inference data.
    Store encoding maps and scaling parameters from training.

11. Document every feature: what it measures, how it was computed,
    and what raw columns it depends on.
"""
print(takeaways)


# ==============================================================================
# NEXT LESSON PREVIEW
# ==============================================================================
print("\n" + "=" * 70)
print("NEXT LESSON PREVIEW")
print("=" * 70)

preview = """
This is the final lesson in the curriculum.

You have now covered the complete progression:

NUMPY FOUNDATIONS (Lessons 1-16)
  Arrays, vectorization, broadcasting, aggregations, dtypes,
  missing values, copies vs views, masking, ufuncs, sorting,
  combining/splitting, performance, debugging

PANDAS CORE (Lessons 17-27)
  DataFrames, indexing, filtering, cleaning, dtypes, renaming,
  groupby, sorting, merging, dates/time series, apply/map/transform

ADVANCED PANDAS AND PRODUCTION (Lessons 28-33)
  Large datasets, validation, debugging pipelines,
  method chaining, EDA, feature engineering

You are now equipped to:
  - Load, clean, and explore any tabular dataset
  - Build reproducible data pipelines with validation
  - Engineer features for machine learning
  - Handle datasets larger than RAM
  - Debug and test data transformations systematically
"""
print(preview)

print("\n" + "=" * 70)
print("END OF LESSON 33 -- COURSE COMPLETE")
print("=" * 70)
print("\nCongratulations on completing the full NumPy and Pandas curriculum.")
print("You have the skills to handle real-world data engineering tasks")
print("from raw files to production-ready feature matrices.")