"""
LESSON 32: EXPLORATORY DATA ANALYSIS (EDA)
================================================================================

What You Will Learn:
- A systematic EDA workflow from first load to actionable insight
- Univariate analysis: distributions, outliers, missing patterns
- Bivariate analysis: correlations, group comparisons, crosstabs
- Multivariate analysis: pivot summaries, feature interactions
- EDA-specific Pandas patterns and helpers
- Building a reusable EDA report function
- Preparing findings clearly for stakeholder communication

Real World Usage:
- Understanding a new dataset before building any model
- Identifying data quality issues before they reach production
- Generating business insights from raw transaction data
- Preparing an EDA report for a data science project kickoff
- Detecting unexpected patterns that change analysis direction

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
import datetime

print("=" * 70)
print("LESSON 32: EXPLORATORY DATA ANALYSIS (EDA)")
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
df = df_raw.copy()

print(f"\nDataset: {dataset_name}")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")


# ==============================================================================
# SECTION 2: THE EDA WORKFLOW
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 2: THE EDA WORKFLOW")
print("=" * 70)

workflow = """
A systematic EDA workflow:

PHASE 1 — FIRST LOOK
  1. Shape, dtypes, memory usage
  2. Head, tail, random sample
  3. Missing value map
  4. Duplicate detection

PHASE 2 — UNIVARIATE ANALYSIS
  5. Numeric: distribution stats, quartiles, skewness, outliers
  6. Categorical: cardinality, value counts, rare categories

PHASE 3 — BIVARIATE ANALYSIS
  7. Numeric vs numeric: correlation, scatter patterns
  8. Categorical vs numeric: group means, medians, ranges
  9. Categorical vs categorical: crosstabs, frequency tables

PHASE 4 — MULTIVARIATE ANALYSIS
  10. Pivot tables across multiple dimensions
  11. Feature interaction signals
  12. Target variable analysis (if supervised)

PHASE 5 — SUMMARISE FINDINGS
  13. Key facts table
  14. Data quality issues
  15. Hypotheses for next steps
"""
print(workflow)


# ==============================================================================
# SECTION 3: PHASE 1 - FIRST LOOK
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 3: PHASE 1 - FIRST LOOK")
print("=" * 70)

# ------------------------------------------------------------------------------
# 3.1 Shape, Dtypes, Memory
# ------------------------------------------------------------------------------
print("\n--- 3.1 Shape, Dtypes, Memory ---")

print(f"Shape:         {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"Memory usage:  {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

print("\nColumn inventory:")
inventory = pd.DataFrame({
    "dtype":    df.dtypes,
    "non_null": df.count(),
    "null":     df.isnull().sum(),
    "null_pct": (df.isnull().mean() * 100).round(2),
    "nunique":  df.nunique(),
    "sample":   [df[c].dropna().iloc[0] if df[c].count() > 0 else None for c in df.columns]
})
print(inventory.to_string())

# ------------------------------------------------------------------------------
# 3.2 Head, Tail, Sample
# ------------------------------------------------------------------------------
print("\n--- 3.2 Head, Tail, Random Sample ---")

print("\nHead (first 3 rows):")
print(df.head(3))

print("\nTail (last 3 rows):")
print(df.tail(3))

print("\nRandom sample (3 rows, seed=42):")
print(df.sample(3, random_state=42))

# ------------------------------------------------------------------------------
# 3.3 Missing Value Map
# ------------------------------------------------------------------------------
print("\n--- 3.3 Missing Value Map ---")

missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)

if len(missing) == 0:
    print("No missing values found in any column.")
else:
    print("Columns with missing values:")
    missing_df = pd.DataFrame({
        "missing_count": missing,
        "missing_pct":   (missing / len(df) * 100).round(2),
        "dtype":         df[missing.index].dtypes
    })
    print(missing_df)

    # Categorise missing severity
    high_missing   = missing_df[missing_df["missing_pct"] > 50].index.tolist()
    medium_missing = missing_df[(missing_df["missing_pct"] > 10) & (missing_df["missing_pct"] <= 50)].index.tolist()
    low_missing    = missing_df[missing_df["missing_pct"] <= 10].index.tolist()

    print(f"\nHigh missing (>50%):    {high_missing}")
    print(f"Medium missing (10-50%): {medium_missing}")
    print(f"Low missing (<10%):      {low_missing}")

# ------------------------------------------------------------------------------
# 3.4 Duplicate Detection
# ------------------------------------------------------------------------------
print("\n--- 3.4 Duplicate Detection ---")

n_dup_rows = df.duplicated().sum()
print(f"Exact duplicate rows: {n_dup_rows} ({n_dup_rows/len(df)*100:.2f}%)")

if n_dup_rows > 0:
    print("Sample of duplicate rows:")
    print(df[df.duplicated(keep=False)].head(4))

# Check for duplicate values in potential key columns
potential_keys = [c for c in df.columns if "id" in c.lower()]
print(f"\nPotential key columns: {potential_keys}")
for key_col in potential_keys:
    dup_key = df[key_col].duplicated().sum()
    print(f"  {key_col}: {dup_key} duplicate values ({dup_key/len(df)*100:.2f}%)")


# ==============================================================================
# SECTION 4: PHASE 2 - UNIVARIATE ANALYSIS (NUMERIC)
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 4: PHASE 2 - UNIVARIATE ANALYSIS (NUMERIC)")
print("=" * 70)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"Numeric columns ({len(numeric_cols)}): {numeric_cols}")


# ------------------------------------------------------------------------------
# 4.1 Extended Descriptive Statistics
# ------------------------------------------------------------------------------
print("\n--- 4.1 Extended Descriptive Statistics ---")


def extended_describe(df_in, cols):
    """
    Extended numeric summary beyond pd.describe().
    Adds skewness, kurtosis, IQR, outlier counts, and coefficient of variation.

    Parameters
    ----------
    df_in : pd.DataFrame
    cols : list of str

    Returns
    -------
    pd.DataFrame with one column per numeric variable
    """
    rows = {}
    for col in cols:
        s = df_in[col].dropna()
        if len(s) == 0:
            continue

        q1   = s.quantile(0.25)
        q3   = s.quantile(0.75)
        iqr  = q3 - q1
        lo   = q1 - 1.5 * iqr
        hi   = q3 + 1.5 * iqr

        rows[col] = {
            "count":       len(s),
            "mean":        round(s.mean(), 4),
            "median":      round(s.median(), 4),
            "std":         round(s.std(), 4),
            "min":         round(s.min(), 4),
            "q1":          round(q1, 4),
            "q3":          round(q3, 4),
            "max":         round(s.max(), 4),
            "iqr":         round(iqr, 4),
            "skewness":    round(s.skew(), 4),
            "kurtosis":    round(s.kurt(), 4),
            "cv":          round(s.std() / s.mean(), 4) if s.mean() != 0 else np.nan,
            "outliers_iqr":int(((s < lo) | (s > hi)).sum()),
            "null_count":  int(df_in[col].isnull().sum()),
        }

    return pd.DataFrame(rows).T


ext_stats = extended_describe(df, numeric_cols)
print("Extended numeric summary:")
print(ext_stats.to_string())


# ------------------------------------------------------------------------------
# 4.2 Skewness Classification
# ------------------------------------------------------------------------------
print("\n--- 4.2 Skewness Classification ---")


def classify_skewness(skew_val):
    """Classify skewness into human-readable category."""
    if pd.isna(skew_val):
        return "unknown"
    if abs(skew_val) < 0.5:
        return "symmetric"
    if abs(skew_val) < 1.0:
        return "moderate_skew"
    return "heavy_skew"


skew_df = pd.DataFrame({
    "skewness":  ext_stats["skewness"].astype(float),
    "category":  ext_stats["skewness"].astype(float).map(classify_skewness)
})
print("Skewness of numeric columns:")
print(skew_df)

highly_skewed = skew_df[skew_df["category"] == "heavy_skew"].index.tolist()
print(f"\nHighly skewed columns (consider log transform): {highly_skewed}")


# ------------------------------------------------------------------------------
# 4.3 Outlier Summary
# ------------------------------------------------------------------------------
print("\n--- 4.3 Outlier Summary ---")

outlier_summary = ext_stats[["count", "outliers_iqr"]].copy()
outlier_summary["outlier_pct"] = (
    outlier_summary["outliers_iqr"] / outlier_summary["count"] * 100
).round(2)
outlier_summary = outlier_summary.sort_values("outlier_pct", ascending=False)
print("Outlier counts (IQR method):")
print(outlier_summary.to_string())

high_outlier_cols = outlier_summary[outlier_summary["outlier_pct"] > 5].index.tolist()
print(f"\nColumns with >5% outliers: {high_outlier_cols}")


# ------------------------------------------------------------------------------
# 4.4 Distribution Pattern per Column
# ------------------------------------------------------------------------------
print("\n--- 4.4 Distribution Patterns (Text Histogram) ---")


def text_histogram(series, bins=10, width=40, label=""):
    """
    Print a text-based histogram for a numeric Series.
    Useful when plotting libraries are not available.
    """
    clean = series.dropna()
    if len(clean) == 0:
        print(f"  {label}: no data")
        return

    counts, bin_edges = np.histogram(clean, bins=bins)
    max_count = counts.max()

    print(f"\n  {label or series.name}  (n={len(clean):,})")
    for i, count in enumerate(counts):
        bar_len = int(count / max_count * width) if max_count > 0 else 0
        bar     = "#" * bar_len
        lo      = bin_edges[i]
        hi      = bin_edges[i + 1]
        print(f"  [{lo:>9.2f}, {hi:>9.2f}): {bar:<{width}} {count:,}")


# Show for the first two numeric columns
for col in numeric_cols[:2]:
    text_histogram(df[col], bins=8, label=col)


# ==============================================================================
# SECTION 5: PHASE 2 - UNIVARIATE ANALYSIS (CATEGORICAL)
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 5: PHASE 2 - UNIVARIATE ANALYSIS (CATEGORICAL)")
print("=" * 70)

cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
print(f"Categorical columns ({len(cat_cols)}): {cat_cols}")


# ------------------------------------------------------------------------------
# 5.1 Cardinality Report
# ------------------------------------------------------------------------------
print("\n--- 5.1 Cardinality Report ---")

cardinality_df = pd.DataFrame({
    "nunique":       df[cat_cols].nunique(),
    "nunique_pct":   (df[cat_cols].nunique() / len(df) * 100).round(2),
    "null_count":    df[cat_cols].isnull().sum(),
    "null_pct":      (df[cat_cols].isnull().mean() * 100).round(2),
    "top_value":     df[cat_cols].apply(lambda x: x.value_counts().index[0] if x.count() > 0 else None),
    "top_value_pct": df[cat_cols].apply(
        lambda x: round(x.value_counts().iloc[0] / x.count() * 100, 2) if x.count() > 0 else None
    )
}).sort_values("nunique", ascending=False)

print("Categorical column cardinality:")
print(cardinality_df.to_string())

# Classify columns by cardinality
binary_cols    = cardinality_df[cardinality_df["nunique"] == 2].index.tolist()
low_card_cols  = cardinality_df[(cardinality_df["nunique"] > 2) & (cardinality_df["nunique"] <= 20)].index.tolist()
high_card_cols = cardinality_df[cardinality_df["nunique"] > 20].index.tolist()

print(f"\nBinary columns:        {binary_cols}")
print(f"Low cardinality (<20): {low_card_cols}")
print(f"High cardinality(>20): {high_card_cols}")


# ------------------------------------------------------------------------------
# 5.2 Value Counts for Low-Cardinality Columns
# ------------------------------------------------------------------------------
print("\n--- 5.2 Value Counts for Low-Cardinality Columns ---")

for col in low_card_cols[:4]:
    vc = df[col].value_counts(normalize=False)
    vc_pct = df[col].value_counts(normalize=True) * 100

    print(f"\n  {col} ({df[col].nunique()} unique values):")
    print(f"  {'Value':<25} {'Count':>8} {'Percent':>9}")
    print("  " + "-" * 45)
    for val, cnt in vc.items():
        pct = vc_pct[val]
        print(f"  {str(val):<25} {cnt:>8,} {pct:>8.1f}%")


# ------------------------------------------------------------------------------
# 5.3 Rare Category Detection
# ------------------------------------------------------------------------------
print("\n--- 5.3 Rare Category Detection ---")

RARE_THRESHOLD = 0.01  # Categories appearing in fewer than 1% of rows

print(f"Rare category threshold: {RARE_THRESHOLD*100:.0f}% of rows")

for col in low_card_cols[:3]:
    vc_pct  = df[col].value_counts(normalize=True)
    rare    = vc_pct[vc_pct < RARE_THRESHOLD].index.tolist()
    n_rare  = df[df[col].isin(rare)].shape[0]
    print(f"\n  {col}: {len(rare)} rare categories affecting {n_rare} rows")
    if rare:
        print(f"    Rare values: {rare}")


# ==============================================================================
# SECTION 6: PHASE 3 - BIVARIATE ANALYSIS (NUMERIC vs NUMERIC)
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 6: PHASE 3 - BIVARIATE ANALYSIS (NUMERIC vs NUMERIC)")
print("=" * 70)


# ------------------------------------------------------------------------------
# 6.1 Correlation Matrix
# ------------------------------------------------------------------------------
print("\n--- 6.1 Correlation Matrix ---")

if len(numeric_cols) >= 2:
    corr_matrix = df[numeric_cols].corr(method="pearson").round(3)
    print("Pearson Correlation Matrix:")
    print(corr_matrix)

    # Extract strong correlations (exclude diagonal)
    corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            col_a = corr_matrix.columns[i]
            col_b = corr_matrix.columns[j]
            r     = corr_matrix.iloc[i, j]
            corr_pairs.append({"col_a": col_a, "col_b": col_b, "pearson_r": r})

    corr_pairs_df = (
        pd.DataFrame(corr_pairs)
        .assign(abs_r=lambda x: x["pearson_r"].abs())
        .sort_values("abs_r", ascending=False)
    )

    print("\nAll pairs sorted by absolute correlation:")
    print(corr_pairs_df.to_string(index=False))

    strong_pos = corr_pairs_df[corr_pairs_df["pearson_r"] > 0.5]
    strong_neg = corr_pairs_df[corr_pairs_df["pearson_r"] < -0.5]

    print(f"\nStrong positive correlations (r > 0.5): {len(strong_pos)}")
    if len(strong_pos) > 0:
        print(strong_pos[["col_a", "col_b", "pearson_r"]].to_string(index=False))

    print(f"\nStrong negative correlations (r < -0.5): {len(strong_neg)}")
    if len(strong_neg) > 0:
        print(strong_neg[["col_a", "col_b", "pearson_r"]].to_string(index=False))


# ------------------------------------------------------------------------------
# 6.2 Scatter Pattern Summary (Text-Based)
# ------------------------------------------------------------------------------
print("\n--- 6.2 Joint Distribution Summary (Quintile Grid) ---")


def quintile_grid(df_in, col_x, col_y, n_bins=5):
    """
    Show average of col_y across quintiles of col_x.
    A text-based substitute for a scatter plot.
    """
    if col_x not in df_in.columns or col_y not in df_in.columns:
        return

    clean = df_in[[col_x, col_y]].dropna()
    if len(clean) < n_bins:
        return

    try:
        clean["x_bin"] = pd.qcut(clean[col_x], q=n_bins, duplicates="drop")
    except Exception:
        return

    summary = clean.groupby("x_bin", observed=True)[col_y].agg(["mean", "count"])

    print(f"\n  {col_y} mean across {col_x} quintiles:")
    print(f"  {'Bin':<30} {'Mean':>10} {'Count':>8}")
    print("  " + "-" * 52)
    for idx, row in summary.iterrows():
        print(f"  {str(idx):<30} {row['mean']:>10.2f} {int(row['count']):>8,}")


if dataset_name == "superstore":
    quintile_grid(df, "discount", "profit")
    quintile_grid(df, "quantity", "sales")
else:
    quintile_grid(df, "age", "fare")
    quintile_grid(df, "fare", "survived")


# ==============================================================================
# SECTION 7: PHASE 3 - BIVARIATE ANALYSIS (CATEGORICAL vs NUMERIC)
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 7: PHASE 3 - BIVARIATE ANALYSIS (CATEGORICAL vs NUMERIC)")
print("=" * 70)


# ------------------------------------------------------------------------------
# 7.1 Group Statistics
# ------------------------------------------------------------------------------
print("\n--- 7.1 Group Statistics ---")


def group_numeric_summary(df_in, group_col, value_cols, top_n=None):
    """
    Summarise numeric columns grouped by a categorical column.

    Parameters
    ----------
    df_in : pd.DataFrame
    group_col : str
    value_cols : list of str
    top_n : int or None
        Show only top N groups by count

    Returns
    -------
    pd.DataFrame
    """
    if group_col not in df_in.columns:
        return pd.DataFrame()

    value_cols = [c for c in value_cols if c in df_in.columns]
    if not value_cols:
        return pd.DataFrame()

    agg_dict = {}
    for col in value_cols:
        agg_dict[f"{col}_mean"]   = (col, "mean")
        agg_dict[f"{col}_median"] = (col, "median")
        agg_dict[f"{col}_std"]    = (col, "std")

    result = (
        df_in
        .groupby(group_col)
        .agg(
            count=(value_cols[0], "count"),
            **agg_dict
        )
        .round(2)
        .reset_index()
        .sort_values("count", ascending=False)
    )

    if top_n is not None:
        result = result.head(top_n)

    return result


if dataset_name == "superstore":
    show_numeric = [c for c in ["sales", "profit", "discount"] if c in df.columns]
    group_cols_to_show = [c for c in ["region", "category", "segment"] if c in df.columns]

    for gc in group_cols_to_show[:2]:
        print(f"\nGrouped by '{gc}':")
        gs = group_numeric_summary(df, gc, show_numeric)
        print(gs.to_string(index=False))

else:
    show_numeric = [c for c in ["fare", "age"] if c in df.columns]
    group_cols_to_show = [c for c in ["pclass", "sex", "embarked"] if c in df.columns]

    for gc in group_cols_to_show[:2]:
        print(f"\nGrouped by '{gc}':")
        gs = group_numeric_summary(df, gc, show_numeric)
        print(gs.to_string(index=False))


# ------------------------------------------------------------------------------
# 7.2 Distribution Comparison Across Groups (Text-Based)
# ------------------------------------------------------------------------------
print("\n--- 7.2 Distribution Comparison Across Groups ---")


def group_distribution_summary(df_in, group_col, value_col):
    """
    Print quartile distributions for each group.
    Reveals whether groups have similar or different shapes.
    """
    if group_col not in df_in.columns or value_col not in df_in.columns:
        return

    groups = df_in[group_col].dropna().unique()
    print(f"\n  {value_col} distribution by {group_col}:")
    print(f"  {'Group':<20} {'Min':>8} {'Q1':>8} {'Median':>8} {'Q3':>8} {'Max':>8} {'IQR':>8}")
    print("  " + "-" * 72)

    for grp in sorted(groups):
        sub = df_in[df_in[group_col] == grp][value_col].dropna()
        if len(sub) == 0:
            continue
        q1  = sub.quantile(0.25)
        q3  = sub.quantile(0.75)
        print(
            f"  {str(grp):<20} "
            f"{sub.min():>8.2f} {q1:>8.2f} {sub.median():>8.2f} "
            f"{q3:>8.2f} {sub.max():>8.2f} {q3-q1:>8.2f}"
        )


if dataset_name == "superstore":
    if "region" in df.columns and "sales" in df.columns:
        group_distribution_summary(df, "region", "sales")
    if "category" in df.columns and "profit" in df.columns:
        group_distribution_summary(df, "category", "profit")
else:
    if "pclass" in df.columns and "fare" in df.columns:
        group_distribution_summary(df, "pclass", "fare")
    if "sex" in df.columns and "age" in df.columns:
        group_distribution_summary(df, "sex", "age")


# ==============================================================================
# SECTION 8: PHASE 3 - BIVARIATE ANALYSIS (CATEGORICAL vs CATEGORICAL)
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 8: PHASE 3 - BIVARIATE ANALYSIS (CATEGORICAL vs CATEGORICAL)")
print("=" * 70)


# ------------------------------------------------------------------------------
# 8.1 Crosstab Frequency Table
# ------------------------------------------------------------------------------
print("\n--- 8.1 Crosstab Frequency Table ---")

if dataset_name == "superstore":
    cols_for_cross = [c for c in ["region", "category", "segment"] if c in df.columns]
    if len(cols_for_cross) >= 2:
        ct = pd.crosstab(df[cols_for_cross[0]], df[cols_for_cross[1]])
        print(f"Crosstab: {cols_for_cross[0]} vs {cols_for_cross[1]}")
        print(ct)

        ct_pct = pd.crosstab(
            df[cols_for_cross[0]],
            df[cols_for_cross[1]],
            normalize="index"
        ).round(3) * 100

        print(f"\nRow-normalised percentages:")
        print(ct_pct)

else:
    if "pclass" in df.columns and "survived" in df.columns:
        ct = pd.crosstab(df["pclass"], df["survived"])
        ct.columns = ["not_survived", "survived"] if len(ct.columns) == 2 else ct.columns
        print("Crosstab: pclass vs survived")
        print(ct)

        ct_pct = pd.crosstab(df["pclass"], df["survived"], normalize="index").round(3) * 100
        print("\nRow-normalised percentages:")
        print(ct_pct)

    if "sex" in df.columns and "survived" in df.columns:
        ct2 = pd.crosstab(df["sex"], df["survived"])
        ct2.columns = ["not_survived", "survived"] if len(ct2.columns) == 2 else ct2.columns
        print("\nCrosstab: sex vs survived")
        print(ct2)


# ------------------------------------------------------------------------------
# 8.2 Association Strength (Cramer's V)
# ------------------------------------------------------------------------------
print("\n--- 8.2 Cramer's V (Categorical Association Strength) ---")


def cramers_v(col_a, col_b, df_in):
    """
    Compute Cramer's V statistic for association between two categorical columns.
    V = 0 means no association, V = 1 means perfect association.

    Parameters
    ----------
    col_a : str
    col_b : str
    df_in : pd.DataFrame

    Returns
    -------
    float
    """
    ct = pd.crosstab(df_in[col_a], df_in[col_b])
    n  = ct.sum().sum()
    if n == 0:
        return np.nan

    chi2 = 0.0
    expected = np.outer(ct.sum(axis=1), ct.sum(axis=0)) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = float(np.nansum((ct.values - expected) ** 2 / np.where(expected == 0, np.nan, expected)))

    r, c = ct.shape
    denom = n * (min(r, c) - 1)
    if denom <= 0:
        return np.nan

    return float(np.sqrt(chi2 / denom))


cat_cols_for_assoc = [c for c in low_card_cols if c in df.columns][:5]

if len(cat_cols_for_assoc) >= 2:
    print(f"Computing Cramer's V for: {cat_cols_for_assoc}")
    assoc_results = []
    for i in range(len(cat_cols_for_assoc)):
        for j in range(i + 1, len(cat_cols_for_assoc)):
            ca = cat_cols_for_assoc[i]
            cb = cat_cols_for_assoc[j]
            try:
                v = cramers_v(ca, cb, df.dropna(subset=[ca, cb]))
                assoc_results.append({"col_a": ca, "col_b": cb, "cramers_v": round(v, 4)})
            except Exception:
                pass

    assoc_df = pd.DataFrame(assoc_results).sort_values("cramers_v", ascending=False)
    print("\nCramer's V between categorical columns:")
    print(assoc_df.to_string(index=False))

    strong_assoc = assoc_df[assoc_df["cramers_v"] > 0.3]
    print(f"\nStrong associations (V > 0.3): {len(strong_assoc)}")
    if len(strong_assoc) > 0:
        print(strong_assoc.to_string(index=False))


# ==============================================================================
# SECTION 9: PHASE 4 - MULTIVARIATE ANALYSIS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 9: PHASE 4 - MULTIVARIATE ANALYSIS")
print("=" * 70)


# ------------------------------------------------------------------------------
# 9.1 Pivot Table Analysis
# ------------------------------------------------------------------------------
print("\n--- 9.1 Pivot Table Analysis ---")

if dataset_name == "superstore":
    pivot_cols = [c for c in ["region", "category", "segment"] if c in df.columns]
    value_col  = "sales" if "sales" in df.columns else numeric_cols[0]

    if len(pivot_cols) >= 2:
        pivot = pd.pivot_table(
            df,
            values=value_col,
            index=pivot_cols[0],
            columns=pivot_cols[1],
            aggfunc="sum",
            fill_value=0,
            margins=True,
            margins_name="Total"
        ).round(0)

        print(f"Pivot table: Total {value_col} by {pivot_cols[0]} x {pivot_cols[1]}")
        print(pivot)

        if len(pivot_cols) >= 3:
            pivot2 = pd.pivot_table(
                df,
                values=value_col,
                index=pivot_cols[0],
                columns=pivot_cols[2],
                aggfunc="mean",
                fill_value=0
            ).round(2)
            print(f"\nPivot: Mean {value_col} by {pivot_cols[0]} x {pivot_cols[2]}")
            print(pivot2)

else:
    if all(c in df.columns for c in ["pclass", "sex", "survived"]):
        pivot = pd.pivot_table(
            df,
            values="survived",
            index="pclass",
            columns="sex",
            aggfunc="mean",
            fill_value=0,
            margins=True,
            margins_name="Overall"
        ).round(3)
        print("Pivot table: Survival rate by pclass and sex")
        print(pivot)

        pivot_fare = pd.pivot_table(
            df,
            values="fare",
            index="pclass",
            columns="survived",
            aggfunc=["mean", "count"],
            fill_value=0
        ).round(2)
        print("\nPivot: Fare by pclass and survival:")
        print(pivot_fare)


# ------------------------------------------------------------------------------
# 9.2 Feature Interaction Analysis
# ------------------------------------------------------------------------------
print("\n--- 9.2 Feature Interaction Analysis ---")


def interaction_summary(df_in, group_cols, value_col, top_n=10):
    """
    Summarise value_col across combinations of group_cols.
    Reveals interaction effects between categorical features.

    Parameters
    ----------
    df_in : pd.DataFrame
    group_cols : list of str
    value_col : str
    top_n : int

    Returns
    -------
    pd.DataFrame
    """
    group_cols = [c for c in group_cols if c in df_in.columns]
    if not group_cols or value_col not in df_in.columns:
        return pd.DataFrame()

    result = (
        df_in
        .groupby(group_cols)[value_col]
        .agg(["mean", "count", "std"])
        .reset_index()
        .rename(columns={"mean": f"{value_col}_mean", "std": f"{value_col}_std"})
        .sort_values(f"{value_col}_mean", ascending=False)
        .head(top_n)
    )
    return result


if dataset_name == "superstore":
    grp_cols = [c for c in ["region", "category"] if c in df.columns]
    v_col    = "profit" if "profit" in df.columns else numeric_cols[0]

    interaction = interaction_summary(df, grp_cols, v_col)
    print(f"Top combinations of {grp_cols} by {v_col} mean:")
    print(interaction.round(2).to_string(index=False))

else:
    grp_cols = [c for c in ["pclass", "sex"] if c in df.columns]
    v_col    = "survived"
    if v_col in df.columns:
        interaction = interaction_summary(df, grp_cols, v_col)
        print(f"Interaction of {grp_cols} on {v_col}:")
        print(interaction.round(3).to_string(index=False))


# ------------------------------------------------------------------------------
# 9.3 Target Variable Analysis (if applicable)
# ------------------------------------------------------------------------------
print("\n--- 9.3 Target Variable Analysis ---")

if dataset_name == "superstore":
    target = "profit"
    if target in df.columns:
        print(f"Target variable: {target}")
        print(f"\nOverall stats:")
        print(df[target].describe().round(2))

        profit_positive = (df[target] > 0).mean() * 100
        profit_negative = (df[target] < 0).mean() * 100
        profit_zero     = (df[target] == 0).mean() * 100

        print(f"\nProfitable orders:   {profit_positive:.1f}%")
        print(f"Loss-making orders:  {profit_negative:.1f}%")
        print(f"Break-even orders:   {profit_zero:.1f}%")

        # Which categories have the most losses?
        if "category" in df.columns:
            loss_by_cat = (
                df.groupby("category")[target]
                .apply(lambda x: (x < 0).sum())
                .reset_index()
            )
            loss_by_cat.columns = ["category", "loss_orders"]
            loss_by_cat["loss_pct"] = (
                loss_by_cat["loss_orders"] /
                df.groupby("category")[target].count().values * 100
            ).round(1)
            print(f"\nLoss-making orders by category:")
            print(loss_by_cat.to_string(index=False))

elif dataset_name == "titanic":
    target = "survived"
    if target in df.columns:
        print(f"Target variable: {target}")
        overall_rate = df[target].mean() * 100
        print(f"\nOverall survival rate: {overall_rate:.1f}%")

        print("\nSurvival rate by key features:")
        for cat_col in [c for c in ["pclass", "sex", "embarked"] if c in df.columns]:
            rates = (
                df.groupby(cat_col)[target]
                .agg(["mean", "count"])
                .reset_index()
                .rename(columns={"mean": "survival_rate", "count": "n"})
            )
            rates["survival_rate"] = (rates["survival_rate"] * 100).round(1)
            print(f"\n  By {cat_col}:")
            print(rates.to_string(index=False))


# ==============================================================================
# SECTION 10: REUSABLE EDA REPORT FUNCTION
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 10: REUSABLE EDA REPORT FUNCTION")
print("=" * 70)


def run_eda_report(df_in, dataset_label="dataset", target_col=None):
    """
    Generate a comprehensive EDA report for any DataFrame.

    Prints all key sections to stdout.
    Returns a summary dictionary for downstream use.

    Parameters
    ----------
    df_in : pd.DataFrame
    dataset_label : str
    target_col : str or None
        If provided, include target variable analysis

    Returns
    -------
    dict with EDA summary
    """
    ts      = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary = {"generated_at": ts, "dataset": dataset_label}

    sep = "=" * 60

    print(f"\n{sep}")
    print(f"EDA REPORT: {dataset_label}")
    print(f"Generated: {ts}")
    print(sep)

    # --- Overview ---
    print("\n[1] OVERVIEW")
    print(f"  Rows:       {df_in.shape[0]:,}")
    print(f"  Columns:    {df_in.shape[1]}")
    print(f"  Memory:     {df_in.memory_usage(deep=True).sum()/1024/1024:.2f} MB")
    print(f"  Duplicates: {df_in.duplicated().sum():,}")

    summary["n_rows"]      = df_in.shape[0]
    summary["n_cols"]      = df_in.shape[1]
    summary["n_duplicates"]= int(df_in.duplicated().sum())

    # --- Missing values ---
    print("\n[2] MISSING VALUES")
    missing = df_in.isnull().sum()
    missing = missing[missing > 0]
    summary["n_cols_with_missing"] = len(missing)

    if len(missing) == 0:
        print("  No missing values.")
    else:
        missing_pct = (df_in.isnull().mean() * 100).round(2)
        for col in missing.index:
            print(f"  {col:<25}: {missing[col]:>6,} ({missing_pct[col]:.1f}%)")

    # --- Numeric summary ---
    print("\n[3] NUMERIC COLUMNS")
    num_cols = df_in.select_dtypes(include=[np.number]).columns.tolist()
    summary["numeric_cols"] = num_cols
    print(f"  Count: {len(num_cols)}")

    if num_cols:
        desc = df_in[num_cols].describe().round(3)
        print(desc.to_string())

    # --- Categorical summary ---
    print("\n[4] CATEGORICAL COLUMNS")
    cat_cols_report = df_in.select_dtypes(include=["object", "category"]).columns.tolist()
    summary["categorical_cols"] = cat_cols_report
    print(f"  Count: {len(cat_cols_report)}")

    for col in cat_cols_report:
        n_unique = df_in[col].nunique()
        top_val  = df_in[col].value_counts().index[0] if df_in[col].count() > 0 else None
        top_pct  = (df_in[col].value_counts().iloc[0] / df_in[col].count() * 100
                    if df_in[col].count() > 0 else 0)
        print(f"  {col:<25}: {n_unique:>5} unique  top='{top_val}' ({top_pct:.1f}%)")

    # --- Correlations ---
    print("\n[5] TOP CORRELATIONS")
    if len(num_cols) >= 2:
        corr = df_in[num_cols].corr().abs()
        pairs = []
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                pairs.append({
                    "col_a": corr.columns[i],
                    "col_b": corr.columns[j],
                    "abs_r": corr.iloc[i, j]
                })
        top_corr = (
            pd.DataFrame(pairs)
            .sort_values("abs_r", ascending=False)
            .head(5)
        )
        print(top_corr.round(3).to_string(index=False))
        summary["top_correlations"] = top_corr.to_dict("records")

    # --- Target variable ---
    if target_col and target_col in df_in.columns:
        print(f"\n[6] TARGET VARIABLE: {target_col}")
        tc = df_in[target_col]
        if pd.api.types.is_numeric_dtype(tc):
            print(f"  Mean:   {tc.mean():.4f}")
            print(f"  Median: {tc.median():.4f}")
            print(f"  Std:    {tc.std():.4f}")
            print(f"  Nulls:  {tc.isnull().sum()}")
        else:
            print(f"  Value counts:")
            print(tc.value_counts().to_string())
        summary["target_col"] = target_col

    # --- Data quality issues ---
    print("\n[7] DATA QUALITY FLAGS")
    flags = []

    high_null_cols = df_in.columns[df_in.isnull().mean() > 0.5].tolist()
    if high_null_cols:
        flags.append(f"High missing (>50%): {high_null_cols}")

    dup_count = df_in.duplicated().sum()
    if dup_count > 0:
        flags.append(f"Duplicate rows: {dup_count}")

    for col in num_cols:
        if pd.api.types.is_numeric_dtype(df_in[col]):
            inf_count = np.isinf(df_in[col].dropna()).sum()
            if inf_count > 0:
                flags.append(f"Inf values in {col}: {inf_count}")

    if not flags:
        print("  No critical data quality issues found.")
    else:
        for f in flags:
            print(f"  WARNING: {f}")

    summary["quality_flags"] = flags

    print(f"\n{sep}")
    print("EDA REPORT COMPLETE")
    print(sep)

    return summary


target = "profit" if dataset_name == "superstore" else "survived"
eda_summary = run_eda_report(df_raw, dataset_label=dataset_name, target_col=target)

print("\nEDA summary dictionary keys:")
print(list(eda_summary.keys()))


# ==============================================================================
# SECTION 11: PREPARING FINDINGS FOR STAKEHOLDERS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 11: PREPARING FINDINGS FOR STAKEHOLDERS")
print("=" * 70)

explanation = """
After EDA, translate findings into clear business language.

Good stakeholder communication:
  - Leads with business impact, not technical details
  - Supports each claim with a specific number
  - Prioritises actionable findings over observations
  - Notes data quality issues clearly
  - Proposes concrete next steps

Template for EDA findings:
  1. Dataset overview (rows, columns, time period)
  2. Data quality issues found (missing, duplicates, outliers)
  3. Key patterns discovered (top factors, distributions)
  4. Surprises or unexpected findings
  5. Recommended next steps
"""
print(explanation)


def print_stakeholder_summary(df_in, dataset_label, target_col=None):
    """
    Print a business-friendly EDA summary.

    Parameters
    ----------
    df_in : pd.DataFrame
    dataset_label : str
    target_col : str or None
    """
    print(f"\n{'='*60}")
    print(f"FINDINGS SUMMARY: {dataset_label.upper()}")
    print("=" * 60)

    # 1. Dataset overview
    print("\n1. DATASET OVERVIEW")
    print(f"   Total records:   {df_in.shape[0]:,}")
    print(f"   Total variables: {df_in.shape[1]}")
    print(f"   Memory:          {df_in.memory_usage(deep=True).sum()/1024/1024:.1f} MB")

    # 2. Data quality
    print("\n2. DATA QUALITY")
    missing_cols = df_in.isnull().sum()
    missing_cols = missing_cols[missing_cols > 0]
    dup          = df_in.duplicated().sum()

    if len(missing_cols) == 0 and dup == 0:
        print("   No missing values or duplicates detected.")
    else:
        if dup > 0:
            print(f"   Duplicate rows found: {dup:,} ({dup/len(df_in)*100:.1f}%)")
        if len(missing_cols) > 0:
            worst_col  = missing_cols.idxmax()
            worst_pct  = missing_cols.max() / len(df_in) * 100
            print(f"   Columns with missing values: {len(missing_cols)}")
            print(f"   Worst column: '{worst_col}' ({worst_pct:.1f}% missing)")

    # 3. Key patterns
    print("\n3. KEY PATTERNS")
    num_cols = df_in.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols_local = df_in.select_dtypes(include=["object", "category"]).columns.tolist()

    for col in num_cols[:3]:
        skew = df_in[col].skew()
        print(f"   {col}: mean={df_in[col].mean():.2f}, "
              f"std={df_in[col].std():.2f}, "
              f"skew={skew:.2f} ({'right-skewed' if skew > 0.5 else 'left-skewed' if skew < -0.5 else 'symmetric'})")

    for col in cat_cols_local[:2]:
        top    = df_in[col].value_counts().index[0]
        top_pct = df_in[col].value_counts().iloc[0] / df_in[col].count() * 100
        print(f"   {col}: most common = '{top}' ({top_pct:.1f}%)")

    # 4. Target variable
    if target_col and target_col in df_in.columns:
        print(f"\n4. TARGET VARIABLE ({target_col})")
        tc = df_in[target_col].dropna()
        if pd.api.types.is_numeric_dtype(tc):
            print(f"   Mean: {tc.mean():.3f}  |  Median: {tc.median():.3f}  |  Std: {tc.std():.3f}")
            print(f"   Range: [{tc.min():.2f}, {tc.max():.2f}]")
        else:
            vc = tc.value_counts(normalize=True)
            print(f"   Distribution:")
            for val, pct in vc.items():
                print(f"     {val}: {pct*100:.1f}%")

    # 5. Next steps
    print("\n5. RECOMMENDED NEXT STEPS")
    missing_pct_max = (df_in.isnull().mean() * 100).max()
    if missing_pct_max > 50:
        print("   - Investigate columns with >50% missing; consider dropping")
    if missing_pct_max > 0:
        print("   - Impute missing values using appropriate strategy per column")
    if df_in.duplicated().sum() > 0:
        print("   - Remove or investigate duplicate rows")

    highly_skewed_local = [
        c for c in num_cols
        if abs(df_in[c].skew()) > 1.0
    ]
    if highly_skewed_local:
        print(f"   - Apply log transform to heavily skewed columns: {highly_skewed_local}")

    print("   - Proceed to feature engineering and model training")
    print("=" * 60)


print_stakeholder_summary(df_raw, dataset_name, target_col=target)


# ==============================================================================
# SECTION 12: COMMON PITFALLS IN EDA
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 12: COMMON PITFALLS IN EDA")
print("=" * 70)

pitfalls = """
Pitfall 1: Skipping EDA and jumping straight to modelling
   - Results in models built on dirty or misunderstood data
   - Always run EDA first, even on familiar datasets

Pitfall 2: Using mean without checking skewness
   - Mean is misleading for skewed distributions (use median)
   - Always report skewness alongside central tendency

Pitfall 3: Ignoring the tail of distributions
   - Outliers can dominate aggregations and break models
   - Always report percentiles and outlier counts

Pitfall 4: Assuming correlation implies causation
   - Strong Pearson r only tells you linear co-movement
   - Report correlation as a signal to investigate, not a conclusion

Pitfall 5: Not checking for data leakage
   - Features that are proxies for the target corrupt model evaluation
   - Check which features correlate very strongly with the target

Pitfall 6: Treating high-cardinality columns as categorical
   - Names, IDs, and free text need different treatment
   - Separate true categoricals from identifiers

Pitfall 7: Not recording EDA findings
   - Insights forgotten by next sprint
   - Always document what you found and what it implies

Pitfall 8: Running EDA on cleaned data only
   - You lose visibility of the original data problems
   - Run EDA on raw data first, then again after cleaning

Pitfall 9: Not segmenting analysis
   - Global stats can hide group-specific patterns
   - Always break down key metrics by relevant groups

Pitfall 10: Confusing null with zero
   - Missing value and zero are different things
   - Check if nulls have a systematic pattern (MCAR, MAR, MNAR)
"""
print(pitfalls)


# ==============================================================================
# SECTION 13: SUMMARY TABLE
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 13: SUMMARY TABLE")
print("=" * 70)

summary_table = """
EDA Task                                 | Pandas Pattern
-----------------------------------------|-------------------------------------------
Shape and dtypes overview                | df.shape, df.dtypes, df.memory_usage()
First look at data                       | df.head(), df.tail(), df.sample(n, seed)
Missing value count and pct              | df.isnull().sum(), df.isnull().mean()*100
Duplicate rows                           | df.duplicated().sum()
Numeric descriptive stats                | df.describe(), extended_describe(df, cols)
Skewness and kurtosis                    | df["col"].skew(), df["col"].kurt()
Outlier count (IQR method)               | custom extended_describe function
Value counts for categorical             | df["col"].value_counts(normalize=True)
Cardinality report                       | df[cat_cols].nunique()
Rare category detection                  | vc[vc < threshold].index
Correlation matrix                       | df.corr(method="pearson")
Group statistics                         | df.groupby("col").agg([...])
Distribution by group                    | group_distribution_summary (custom)
Crosstab frequency table                 | pd.crosstab(df["a"], df["b"])
Row-normalised crosstab                  | pd.crosstab(..., normalize="index")
Categorical association                  | cramers_v(col_a, col_b, df)
Pivot table multi-dimension              | pd.pivot_table(df, values, index, columns)
Feature interaction summary              | interaction_summary (custom)
Target variable analysis                 | groupby + mean, value_counts
Stakeholder-ready summary                | print_stakeholder_summary (custom)
"""
print(summary_table)


# ==============================================================================
# SECTION 14: PRACTICE EXERCISES
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 14: PRACTICE EXERCISES")
print("=" * 70)

print("Exercise 1: Identify the numeric column with the highest skewness")
num_skews = df_raw.select_dtypes(include=[np.number]).skew().abs().sort_values(ascending=False)
print(f"  Column with highest skewness: {num_skews.index[0]} (|skew| = {num_skews.iloc[0]:.3f})")

print("\nExercise 2: Find the categorical column with the most balanced distribution")
balance_scores = {}
for col in cat_cols[:5]:
    vc = df_raw[col].value_counts(normalize=True)
    balance_scores[col] = round(float(vc.std()), 4)

most_balanced = min(balance_scores, key=balance_scores.get)
print(f"  Most balanced column: {most_balanced} (std of proportions = {balance_scores[most_balanced]})")
print(f"  All balance scores: {balance_scores}")

print("\nExercise 3: Build a pair-wise missing value co-occurrence matrix")
missing_any = df_raw.isnull()
missing_cols_only = missing_any[missing_any.any(axis=1)].columns[missing_any.any()].tolist()

if len(missing_cols_only) >= 2:
    co_occur = pd.DataFrame(index=missing_cols_only, columns=missing_cols_only, dtype=float)
    for ca in missing_cols_only:
        for cb in missing_cols_only:
            co_occur.loc[ca, cb] = float(
                (missing_any[ca] & missing_any[cb]).sum()
            )
    print("  Missing value co-occurrence (both null at same row):")
    print(co_occur)
else:
    print("  Not enough columns with missing values for co-occurrence matrix.")

print("\nExercise 4: Top 5 most correlated column pairs")
if len(numeric_cols) >= 2:
    corr_full = df_raw[numeric_cols].corr().abs()
    pairs_ex = []
    for i in range(len(corr_full.columns)):
        for j in range(i + 1, len(corr_full.columns)):
            pairs_ex.append({
                "col_a": corr_full.columns[i],
                "col_b": corr_full.columns[j],
                "abs_r": corr_full.iloc[i, j]
            })
    top5 = pd.DataFrame(pairs_ex).sort_values("abs_r", ascending=False).head(5)
    print(top5.round(4).to_string(index=False))

print("\nExercise 5: Run full EDA report on a 20% sample")
sample_20pct = df_raw.sample(frac=0.2, random_state=42)
print(f"  Sample size: {len(sample_20pct):,} rows")
_ = run_eda_report(sample_20pct, dataset_label=f"{dataset_name}_sample_20pct", target_col=target)


# ==============================================================================
# SECTION 15: KEY TAKEAWAYS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 15: KEY TAKEAWAYS")
print("=" * 70)

takeaways = """
1.  EDA is MANDATORY before modelling or reporting. Always do it on raw data.

2.  First look: shape, dtypes, memory, head, nulls, duplicates.
    This catches 80% of data quality issues immediately.

3.  Univariate analysis: use extended_describe for skew, kurtosis,
    IQR outlier counts, and coefficient of variation.

4.  Bivariate: correlation matrix for numeric pairs,
    crosstab + Cramer's V for categorical pairs.

5.  Always segment numeric distributions by key categorical variables.
    Global averages hide important group-level patterns.

6.  Pivot tables are the fastest way to reveal multi-dimensional patterns.

7.  Build reusable EDA functions that work on any DataFrame.
    run_eda_report() is a template to extend for your organisation.

8.  Translate findings into business language before sharing.
    Stakeholders need conclusions, not correlation matrices.

9.  Record all EDA findings and hypotheses before moving to next step.

10. Run EDA again after cleaning to verify quality improvements.
"""
print(takeaways)


# ==============================================================================
# NEXT LESSON PREVIEW
# ==============================================================================
print("\n" + "=" * 70)
print("NEXT LESSON PREVIEW")
print("=" * 70)

preview = """
Lesson 33: Feature Engineering Basics

You will learn:
- What feature engineering is and why it matters for ML
- Numeric transformations: log, power, binning, interactions
- Categorical encoding: label, one-hot, target, frequency encoding
- Date/time feature extraction
- Feature crossing and polynomial features
- Handling skewed distributions
- Building a reusable feature engineering pipeline
- Validating features for ML readiness
"""
print(preview)

print("\n" + "=" * 70)
print("END OF LESSON 32")
print("=" * 70)
print("\nYou now have a complete, systematic EDA workflow you can apply")
print("to any dataset before analysis, modelling, or reporting.")

