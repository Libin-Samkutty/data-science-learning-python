"""
LESSON 29: DATA VALIDATION AND PIPELINE BUILDING
================================================================================

What You Will Learn:
- Writing assertion-based validation functions
- Validating schema, ranges, uniqueness, and referential integrity
- Building reusable validation decorators
- Structuring a complete data pipeline with validation at each step
- Logging validation results to track issues over time
- Handling validation failures gracefully (warn vs raise)
- Testing pipeline correctness with known inputs
- Method chaining for clean pipeline syntax

Real World Usage:
- Preventing bad data from entering production databases
- Catching upstream data source changes early
- Building auditable ETL pipelines with validation logs
- Ensuring ML training data meets quality standards
- Automated data quality reporting for business stakeholders
- Compliance and regulatory data validation

Dataset Used:
E-commerce orders dataset (public, no login required)
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
import traceback
from functools import wraps

print("=" * 70)
print("LESSON 29: DATA VALIDATION AND PIPELINE BUILDING")
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
    """Convert column name to snake_case."""
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
print("\nFirst 5 rows:")
print(df_raw.head())
print("\nData types:")
print(df_raw.dtypes)
print("\nMissing values:")
print(df_raw.isnull().sum())


# ==============================================================================
# SECTION 2: WHY VALIDATION MATTERS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 2: WHY VALIDATION MATTERS")
print("=" * 70)

explanation = """
DATA VALIDATION is the process of verifying that data meets
defined quality requirements before it is used.

Without validation:
  - Corrupt data silently propagates through pipelines
  - ML models train on garbage and produce garbage predictions
  - Reports show wrong numbers that stakeholders trust
  - Bugs surface far downstream, making them hard to trace

With validation:
  - Problems are caught at the SOURCE as early as possible
  - Pipeline failures have clear, informative error messages
  - Data quality history is tracked over time
  - Trust in data products increases across the organisation

VALIDATION CATEGORIES:
  1. Schema validation      Correct columns, correct dtypes
  2. Completeness           No unexpected missing values
  3. Range validation       Values within physically valid bounds
  4. Uniqueness             No unexpected duplicate records
  5. Referential integrity  Foreign keys match lookup tables
  6. Statistical checks     Distributions match expectations
  7. Business rules         Domain-specific correctness rules
"""
print(explanation)


# ==============================================================================
# SECTION 3: BASIC VALIDATION BUILDING BLOCKS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 3: BASIC VALIDATION BUILDING BLOCKS")
print("=" * 70)

# ------------------------------------------------------------------------------
# 3.1 ValidationResult - A Simple Result Object
# ------------------------------------------------------------------------------
print("\n--- 3.1 ValidationResult Object ---")


class ValidationResult:
    """
    Holds the result of a single validation check.

    Attributes
    ----------
    name : str
        Name of the validation check
    passed : bool
        Whether the check passed
    message : str
        Human-readable description
    severity : str
        'error' raises, 'warning' logs, 'info' always passes
    details : dict
        Optional extra context
    """

    def __init__(self, name, passed, message, severity="error", details=None):
        self.name      = name
        self.passed    = passed
        self.message   = message
        self.severity  = severity
        self.details   = details or {}
        self.timestamp = datetime.datetime.now().isoformat()

    def __repr__(self):
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] [{self.severity.upper()}] {self.name}: {self.message}"

    def to_dict(self):
        return {
            "name":      self.name,
            "passed":    self.passed,
            "message":   self.message,
            "severity":  self.severity,
            "details":   self.details,
            "timestamp": self.timestamp
        }


# Demonstrate
result_pass = ValidationResult(
    name     = "row_count_check",
    passed   = True,
    message  = "Row count 891 is within expected range [500, 10000]",
    severity = "error"
)
result_fail = ValidationResult(
    name     = "missing_values_check",
    passed   = False,
    message  = "Column 'age' has 177 missing values (19.9%), exceeds threshold 10%",
    severity = "warning",
    details  = {"column": "age", "missing_pct": 19.9, "threshold": 10.0}
)

print("ValidationResult examples:")
print(result_pass)
print(result_fail)
print("\nAs dict:")
print(result_fail.to_dict())

# ------------------------------------------------------------------------------
# 3.2 ValidationReport - Collection of Results
# ------------------------------------------------------------------------------
print("\n--- 3.2 ValidationReport Object ---")


class ValidationReport:
    """
    Collects multiple ValidationResult objects and provides summary.

    Parameters
    ----------
    pipeline_name : str
        Name of the pipeline being validated
    """

    def __init__(self, pipeline_name="unnamed_pipeline"):
        self.pipeline_name = pipeline_name
        self.results       = []
        self.start_time    = datetime.datetime.now()

    def add(self, result):
        """Add a ValidationResult."""
        self.results.append(result)
        return self

    def passed(self):
        """True only if all error-severity checks passed."""
        return all(
            r.passed for r in self.results if r.severity == "error"
        )

    def n_passed(self):
        return sum(1 for r in self.results if r.passed)

    def n_failed(self):
        return sum(1 for r in self.results if not r.passed)

    def failures(self):
        return [r for r in self.results if not r.passed]

    def error_failures(self):
        return [r for r in self.results if not r.passed and r.severity == "error"]

    def print_summary(self):
        """Print a formatted validation report."""
        elapsed = (datetime.datetime.now() - self.start_time).total_seconds()
        total = len(self.results)
        n_pass = self.n_passed()
        n_fail = self.n_failed()

        print(f"\n{'='*60}")
        print(f"VALIDATION REPORT: {self.pipeline_name}")
        print(f"{'='*60}")
        print(f"Total checks:  {total}")
        print(f"Passed:        {n_pass}")
        print(f"Failed:        {n_fail}")
        print(f"Overall status: {'PASS' if self.passed() else 'FAIL'}")
        print(f"Elapsed:       {elapsed:.3f}s")

        if n_fail > 0:
            print(f"\nFailed checks:")
            for r in self.failures():
                print(f"  {r}")
                if r.details:
                    for k, v in r.details.items():
                        print(f"    {k}: {v}")

        print("=" * 60)

    def to_dataframe(self):
        """Return all results as a DataFrame."""
        return pd.DataFrame([r.to_dict() for r in self.results])

    def raise_on_errors(self):
        """Raise ValueError if any error-severity check failed."""
        error_failures = self.error_failures()
        if error_failures:
            messages = "\n".join(str(r) for r in error_failures)
            raise ValueError(
                f"Validation failed with {len(error_failures)} error(s):\n{messages}"
            )


# Demonstrate
report = ValidationReport("demo_pipeline")
report.add(result_pass)
report.add(result_fail)
report.print_summary()


# ==============================================================================
# SECTION 4: INDIVIDUAL VALIDATION CHECKS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 4: INDIVIDUAL VALIDATION CHECKS")
print("=" * 70)

# ------------------------------------------------------------------------------
# 4.1 Schema Validation
# ------------------------------------------------------------------------------
print("\n--- 4.1 Schema Validation ---")


def check_schema(df, expected_columns, check_dtypes=None):
    """
    Validate DataFrame has expected columns and optional dtypes.

    Parameters
    ----------
    df : pd.DataFrame
    expected_columns : list of str
        Columns that must be present
    check_dtypes : dict or None
        {column_name: expected_dtype_string}

    Returns
    -------
    list of ValidationResult
    """
    results = []

    # Check all expected columns are present
    actual_cols  = set(df.columns)
    expected_set = set(expected_columns)
    missing_cols = expected_set - actual_cols
    extra_cols   = actual_cols - expected_set

    results.append(ValidationResult(
        name    = "required_columns_present",
        passed  = len(missing_cols) == 0,
        message = (
            "All required columns present"
            if len(missing_cols) == 0
            else f"Missing columns: {sorted(missing_cols)}"
        ),
        severity = "error",
        details  = {"missing": sorted(missing_cols), "extra": sorted(extra_cols)}
    ))

    # Check dtypes if specified
    if check_dtypes:
        for col, expected_dtype in check_dtypes.items():
            if col not in df.columns:
                continue
            actual_dtype = str(df[col].dtype)
            match = actual_dtype == expected_dtype
            results.append(ValidationResult(
                name    = f"dtype_{col}",
                passed  = match,
                message = (
                    f"{col}: dtype is {actual_dtype} (expected {expected_dtype})"
                    if not match
                    else f"{col}: dtype correct ({actual_dtype})"
                ),
                severity = "error",
                details  = {"column": col, "expected": expected_dtype, "actual": actual_dtype}
            ))

    return results


# Test schema validation
if dataset_name == "superstore":
    expected_cols = ["order_id", "customer_name", "category", "sales", "profit"]
    dtype_check   = {"sales": "float64", "profit": "float64"}
else:
    expected_cols = ["passengerid", "survived", "pclass", "name", "sex", "age"]
    dtype_check   = {"survived": "int64", "pclass": "int64"}

schema_results = check_schema(df_raw, expected_cols, dtype_check)
print("Schema validation results:")
for r in schema_results:
    print(f"  {r}")

# ------------------------------------------------------------------------------
# 4.2 Completeness Validation
# ------------------------------------------------------------------------------
print("\n--- 4.2 Completeness Validation ---")


def check_completeness(df, max_missing_pct=None, required_complete=None):
    """
    Validate missing value levels in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
    max_missing_pct : float or None
        Maximum allowed missing percentage for ALL columns (0-100)
    required_complete : list of str or None
        Columns that must have ZERO missing values

    Returns
    -------
    list of ValidationResult
    """
    results = []
    missing_pct = (df.isnull().mean() * 100).round(2)

    # Check overall missing threshold
    if max_missing_pct is not None:
        violations = missing_pct[missing_pct > max_missing_pct]
        results.append(ValidationResult(
            name    = "missing_values_threshold",
            passed  = len(violations) == 0,
            message = (
                f"All columns within {max_missing_pct}% missing threshold"
                if len(violations) == 0
                else f"{len(violations)} columns exceed {max_missing_pct}% threshold"
            ),
            severity = "warning",
            details  = violations.to_dict()
        ))

    # Check columns that must be 100% complete
    if required_complete:
        for col in required_complete:
            if col not in df.columns:
                continue
            n_missing = df[col].isnull().sum()
            results.append(ValidationResult(
                name     = f"complete_{col}",
                passed   = n_missing == 0,
                message  = (
                    f"{col}: fully complete"
                    if n_missing == 0
                    else f"{col}: {n_missing} missing values ({missing_pct[col]:.1f}%)"
                ),
                severity = "error",
                details  = {"column": col, "missing_count": int(n_missing)}
            ))

    return results


if dataset_name == "superstore":
    completeness_results = check_completeness(
        df_raw,
        max_missing_pct=5.0,
        required_complete=["order_id", "sales", "profit"]
    )
else:
    completeness_results = check_completeness(
        df_raw,
        max_missing_pct=25.0,
        required_complete=["passengerid", "survived"]
    )

print("Completeness validation results:")
for r in completeness_results:
    print(f"  {r}")

# ------------------------------------------------------------------------------
# 4.3 Range Validation
# ------------------------------------------------------------------------------
print("\n--- 4.3 Range Validation ---")


def check_ranges(df, range_rules):
    """
    Validate numeric columns are within expected bounds.

    Parameters
    ----------
    df : pd.DataFrame
    range_rules : dict
        {column_name: {"min": value, "max": value, "severity": "error"}}

    Returns
    -------
    list of ValidationResult
    """
    results = []

    for col, rules in range_rules.items():
        if col not in df.columns:
            results.append(ValidationResult(
                name     = f"range_{col}",
                passed   = False,
                message  = f"Column '{col}' not found for range check",
                severity = rules.get("severity", "error")
            ))
            continue

        col_data = df[col].dropna()
        violations = 0
        issues = []

        col_min = rules.get("min")
        col_max = rules.get("max")

        if col_min is not None:
            below = (col_data < col_min).sum()
            if below > 0:
                violations += below
                issues.append(f"{below} values below min ({col_min})")

        if col_max is not None:
            above = (col_data > col_max).sum()
            if above > 0:
                violations += above
                issues.append(f"{above} values above max ({col_max})")

        results.append(ValidationResult(
            name     = f"range_{col}",
            passed   = violations == 0,
            message  = (
                f"{col}: all {len(col_data)} values within range"
                if violations == 0
                else f"{col}: {violations} out-of-range values - " + ", ".join(issues)
            ),
            severity = rules.get("severity", "error"),
            details  = {
                "column":     col,
                "violations": violations,
                "actual_min": float(col_data.min()),
                "actual_max": float(col_data.max()),
                "rule_min":   col_min,
                "rule_max":   col_max
            }
        ))

    return results


if dataset_name == "superstore":
    range_rules = {
        "sales":    {"min": 0,    "max": 100000, "severity": "error"},
        "profit":   {"min": -5000,"max": 50000,  "severity": "warning"},
        "discount": {"min": 0,    "max": 1,      "severity": "error"},
        "quantity": {"min": 1,    "max": 100,    "severity": "warning"},
    }
else:
    range_rules = {
        "survived": {"min": 0, "max": 1,   "severity": "error"},
        "pclass":   {"min": 1, "max": 3,   "severity": "error"},
        "age":      {"min": 0, "max": 120, "severity": "warning"},
        "fare":     {"min": 0, "max": 600, "severity": "warning"},
    }

range_results = check_ranges(df_raw, range_rules)
print("Range validation results:")
for r in range_results:
    print(f"  {r}")

# ------------------------------------------------------------------------------
# 4.4 Uniqueness Validation
# ------------------------------------------------------------------------------
print("\n--- 4.4 Uniqueness Validation ---")


def check_uniqueness(df, unique_columns, allow_nulls=False):
    """
    Validate that specified columns have unique values.

    Parameters
    ----------
    df : pd.DataFrame
    unique_columns : list of str
        Columns or column combinations that should be unique
    allow_nulls : bool
        Whether to ignore NaN values in uniqueness check

    Returns
    -------
    list of ValidationResult
    """
    results = []

    for col_spec in unique_columns:
        # col_spec can be a string (single col) or list (composite key)
        if isinstance(col_spec, str):
            cols = [col_spec]
            name = col_spec
        else:
            cols = col_spec
            name = "+".join(col_spec)

        # Check columns exist
        missing = [c for c in cols if c not in df.columns]
        if missing:
            results.append(ValidationResult(
                name     = f"unique_{name}",
                passed   = False,
                message  = f"Columns not found: {missing}",
                severity = "error"
            ))
            continue

        subset_df = df[cols]
        if allow_nulls:
            subset_df = subset_df.dropna()

        dup_count = subset_df.duplicated().sum()

        results.append(ValidationResult(
            name     = f"unique_{name}",
            passed   = dup_count == 0,
            message  = (
                f"{name}: all values are unique"
                if dup_count == 0
                else f"{name}: {dup_count} duplicate rows found"
            ),
            severity = "error",
            details  = {
                "columns":    cols,
                "duplicates": int(dup_count),
                "total_rows": len(subset_df)
            }
        ))

    return results


if dataset_name == "superstore":
    unique_checks = ["order_id", ["order_id", "product_id"]]
else:
    unique_checks = ["passengerid", "name"]

uniqueness_results = check_uniqueness(df_raw, unique_checks, allow_nulls=True)
print("Uniqueness validation results:")
for r in uniqueness_results:
    print(f"  {r}")

# ------------------------------------------------------------------------------
# 4.5 Referential Integrity Validation
# ------------------------------------------------------------------------------
print("\n--- 4.5 Referential Integrity ---")


def check_referential_integrity(df, column, valid_values, severity="error"):
    """
    Validate that values in a column exist in a valid set.

    Parameters
    ----------
    df : pd.DataFrame
    column : str
        Column to validate
    valid_values : set or list
        The set of allowed values
    severity : str

    Returns
    -------
    ValidationResult
    """
    if column not in df.columns:
        return ValidationResult(
            name     = f"referential_{column}",
            passed   = False,
            message  = f"Column '{column}' not found",
            severity = severity
        )

    valid_set = set(valid_values)
    actual_values = set(df[column].dropna().unique())
    invalid_values = actual_values - valid_set

    return ValidationResult(
        name     = f"referential_{column}",
        passed   = len(invalid_values) == 0,
        message  = (
            f"{column}: all values in valid set"
            if len(invalid_values) == 0
            else f"{column}: {len(invalid_values)} invalid values found"
        ),
        severity = severity,
        details  = {
            "column":         column,
            "invalid_values": sorted(str(v) for v in invalid_values),
            "valid_count":    len(valid_set)
        }
    )


if dataset_name == "superstore":
    valid_regions = {"Central", "East", "South", "West"}
    ref_result = check_referential_integrity(df_raw, "region", valid_regions)
else:
    valid_ports = {"S", "C", "Q"}
    ref_result = check_referential_integrity(df_raw, "embarked", valid_ports)

print("Referential integrity result:")
print(f"  {ref_result}")

# ------------------------------------------------------------------------------
# 4.6 Statistical Distribution Checks
# ------------------------------------------------------------------------------
print("\n--- 4.6 Statistical Distribution Checks ---")


def check_statistics(df, stat_rules):
    """
    Validate statistical properties of numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
    stat_rules : dict
        {col: {"mean_min": x, "mean_max": x, "std_max": x, "severity": "warning"}}

    Returns
    -------
    list of ValidationResult
    """
    results = []

    for col, rules in stat_rules.items():
        if col not in df.columns:
            continue

        col_data = df[col].dropna()
        actual_mean = col_data.mean()
        actual_std  = col_data.std()
        actual_med  = col_data.median()
        issues = []

        if "mean_min" in rules and actual_mean < rules["mean_min"]:
            issues.append(f"mean {actual_mean:.2f} below minimum {rules['mean_min']}")
        if "mean_max" in rules and actual_mean > rules["mean_max"]:
            issues.append(f"mean {actual_mean:.2f} above maximum {rules['mean_max']}")
        if "std_max" in rules and actual_std > rules["std_max"]:
            issues.append(f"std {actual_std:.2f} above maximum {rules['std_max']}")
        if "median_min" in rules and actual_med < rules["median_min"]:
            issues.append(f"median {actual_med:.2f} below minimum {rules['median_min']}")

        results.append(ValidationResult(
            name     = f"stats_{col}",
            passed   = len(issues) == 0,
            message  = (
                f"{col}: statistics within expected ranges"
                if len(issues) == 0
                else f"{col}: " + "; ".join(issues)
            ),
            severity = rules.get("severity", "warning"),
            details  = {
                "column": col,
                "mean":   round(float(actual_mean), 4),
                "std":    round(float(actual_std), 4),
                "median": round(float(actual_med), 4)
            }
        ))

    return results


if dataset_name == "superstore":
    stat_rules = {
        "sales":    {"mean_min": 50,  "mean_max": 1000, "std_max": 1000, "severity": "warning"},
        "profit":   {"mean_min": -50, "mean_max": 500,  "severity": "warning"},
        "discount": {"mean_min": 0,   "mean_max": 0.5,  "severity": "warning"},
    }
else:
    stat_rules = {
        "age":  {"mean_min": 20, "mean_max": 50, "std_max": 20, "severity": "warning"},
        "fare": {"mean_min": 10, "mean_max": 100, "severity": "warning"},
    }

stat_results = check_statistics(df_raw, stat_rules)
print("Statistical validation results:")
for r in stat_results:
    print(f"  {r}")


# ==============================================================================
# SECTION 5: VALIDATION DECORATOR
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 5: VALIDATION DECORATOR")
print("=" * 70)

explanation = """
A decorator wraps a pipeline step function with automatic validation.

Benefits:
  - Validation runs automatically without cluttering business logic
  - Consistent validation behaviour across all pipeline steps
  - Elapsed time and status logged uniformly
  - Easy to enable/disable validation without modifying step code
"""
print(explanation)


def validate_step(
    input_checks=None,
    output_checks=None,
    step_name=None,
    on_failure="raise"
):
    """
    Decorator that wraps a pipeline step with pre/post validation.

    Parameters
    ----------
    input_checks : callable or None
        Function(df) -> list[ValidationResult] run BEFORE the step
    output_checks : callable or None
        Function(df) -> list[ValidationResult] run AFTER the step
    step_name : str or None
        Display name for logging
    on_failure : str
        "raise" to raise on errors, "warn" to print and continue

    Returns
    -------
    Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(df, *args, **kwargs):
            name = step_name or func.__name__
            print(f"\n[STEP] {name}")
            start = time.perf_counter()
            report = ValidationReport(name)

            # Pre-execution validation
            if input_checks is not None:
                pre_results = input_checks(df)
                for r in pre_results:
                    report.add(r)
                pre_failures = [r for r in pre_results if not r.passed and r.severity == "error"]
                if pre_failures:
                    report.print_summary()
                    if on_failure == "raise":
                        raise ValueError(f"[{name}] Input validation failed")
                    else:
                        print(f"  WARNING: Input validation failed, continuing anyway")

            # Execute the step
            result_df = func(df, *args, **kwargs)

            # Post-execution validation
            if output_checks is not None:
                post_results = output_checks(result_df)
                for r in post_results:
                    report.add(r)

            elapsed = time.perf_counter() - start
            n_pass = report.n_passed()
            n_fail = report.n_failed()

            print(f"  Status:  {'PASS' if report.passed() else 'FAIL'}")
            print(f"  Checks:  {n_pass} passed, {n_fail} failed")
            print(f"  Shape:   {df.shape} -> {result_df.shape}")
            print(f"  Elapsed: {elapsed*1000:.1f} ms")

            if not report.passed() and on_failure == "raise":
                report.print_summary()
                raise ValueError(f"[{name}] Output validation failed")

            return result_df

        wrapper._is_validated = True
        return wrapper

    return decorator


# Demonstrate decorator usage
def check_no_nulls_in_key(df):
    """Pre-check: key columns must not be null."""
    results = []
    if dataset_name == "superstore":
        key_cols = ["order_id", "sales"]
    else:
        key_cols = ["passengerid", "survived"]

    for col in key_cols:
        if col in df.columns:
            n_null = df[col].isnull().sum()
            results.append(ValidationResult(
                name     = f"no_null_{col}",
                passed   = n_null == 0,
                message  = (
                    f"{col}: no nulls"
                    if n_null == 0
                    else f"{col}: {n_null} nulls found"
                ),
                severity = "error"
            ))
    return results


def check_output_shape(df):
    """Post-check: output must have rows."""
    return [ValidationResult(
        name     = "output_has_rows",
        passed   = len(df) > 0,
        message  = f"Output has {len(df)} rows",
        severity = "error"
    )]


@validate_step(
    input_checks  = check_no_nulls_in_key,
    output_checks = check_output_shape,
    step_name     = "DemoStep: Filter Completed Orders",
    on_failure    = "warn"
)
def filter_valid_records(df):
    """Filter to valid records only."""
    if dataset_name == "superstore":
        return df[df["sales"] > 0].copy()
    else:
        return df[df["survived"].isin([0, 1])].copy()


result = filter_valid_records(df_raw)
print(f"\nStep output shape: {result.shape}")


# ==============================================================================
# SECTION 6: COMPLETE PIPELINE WITH VALIDATION
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 6: COMPLETE PIPELINE WITH VALIDATION")
print("=" * 70)


class DataPipeline:
    """
    A configurable data pipeline with built-in validation at each step.

    Design principles:
      1. Each step is a function: DataFrame -> DataFrame
      2. Validation runs before and after each step
      3. A report is produced for the entire pipeline
      4. Failures can halt or warn depending on severity
      5. Steps are composable and reusable

    Parameters
    ----------
    name : str
        Pipeline identifier
    on_failure : str
        "raise" stops pipeline on error, "warn" logs and continues
    """

    def __init__(self, name="pipeline", on_failure="raise"):
        self.name       = name
        self.on_failure = on_failure
        self.steps      = []
        self.report     = ValidationReport(name)
        self.history    = []

    def add_step(self, func, input_checks=None, output_checks=None, step_name=None):
        """Register a pipeline step with optional validation."""
        self.steps.append({
            "func":          func,
            "input_checks":  input_checks,
            "output_checks": output_checks,
            "name":          step_name or func.__name__
        })
        return self

    def run(self, df):
        """
        Execute all registered steps in order.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame

        Returns
        -------
        pd.DataFrame
            Transformed DataFrame after all steps
        """
        print(f"\n{'='*60}")
        print(f"PIPELINE START: {self.name}")
        print(f"Input shape: {df.shape}")
        print("=" * 60)

        current_df = df.copy()
        pipeline_start = time.perf_counter()

        for step_num, step in enumerate(self.steps, 1):
            step_name = step["name"]
            print(f"\n[Step {step_num}/{len(self.steps)}] {step_name}")
            step_start = time.perf_counter()

            step_report = ValidationReport(step_name)

            # Input validation
            if step["input_checks"] is not None:
                try:
                    in_results = step["input_checks"](current_df)
                    for r in in_results:
                        step_report.add(r)
                        self.report.add(r)
                    in_failures = step_report.error_failures()
                    if in_failures:
                        print(f"  INPUT VALIDATION FAILED: {len(in_failures)} error(s)")
                        for r in in_failures:
                            print(f"    {r}")
                        if self.on_failure == "raise":
                            raise ValueError(
                                f"Step '{step_name}' input validation failed"
                            )
                except ValueError:
                    raise
                except Exception as e:
                    print(f"  Input check error: {e}")

            # Execute step
            try:
                shape_before = current_df.shape
                current_df = step["func"](current_df)
                shape_after = current_df.shape
                elapsed = time.perf_counter() - step_start

                self.history.append({
                    "step":         step_num,
                    "name":         step_name,
                    "shape_before": shape_before,
                    "shape_after":  shape_after,
                    "elapsed_ms":   round(elapsed * 1000, 1),
                    "status":       "ok"
                })

                print(f"  Shape: {shape_before} -> {shape_after}")
                print(f"  Elapsed: {elapsed*1000:.1f} ms")

            except Exception as e:
                self.history.append({
                    "step":    step_num,
                    "name":    step_name,
                    "status":  "error",
                    "error":   str(e)
                })
                if self.on_failure == "raise":
                    raise
                else:
                    print(f"  STEP FAILED: {e}")
                    continue

            # Output validation
            if step["output_checks"] is not None:
                try:
                    out_results = step["output_checks"](current_df)
                    for r in out_results:
                        step_report.add(r)
                        self.report.add(r)
                    out_failures = [
                        r for r in out_results
                        if not r.passed and r.severity == "error"
                    ]
                    if out_failures:
                        print(f"  OUTPUT VALIDATION FAILED: {len(out_failures)} error(s)")
                        for r in out_failures:
                            print(f"    {r}")
                        if self.on_failure == "raise":
                            raise ValueError(
                                f"Step '{step_name}' output validation failed"
                            )
                except ValueError:
                    raise
                except Exception as e:
                    print(f"  Output check error: {e}")

        pipeline_elapsed = time.perf_counter() - pipeline_start
        print(f"\n{'='*60}")
        print(f"PIPELINE COMPLETE: {self.name}")
        print(f"Output shape: {current_df.shape}")
        print(f"Total elapsed: {pipeline_elapsed*1000:.1f} ms")
        print("=" * 60)

        return current_df

    def print_history(self):
        """Print step-by-step execution history."""
        print(f"\nPipeline History: {self.name}")
        print(f"{'Step':<5} {'Name':<35} {'Before':>10} {'After':>10} {'ms':>8} {'Status'}")
        print("-" * 80)
        for h in self.history:
            before = str(h.get("shape_before", "N/A"))
            after  = str(h.get("shape_after", "N/A"))
            elapsed = str(h.get("elapsed_ms", "N/A"))
            status = h.get("status", "unknown")
            print(
                f"{h['step']:<5} {h['name'][:34]:<35} "
                f"{before:>10} {after:>10} {elapsed:>8} {status}"
            )


# ==============================================================================
# SECTION 7: BUILD A REAL PIPELINE
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 7: BUILD A REAL PIPELINE FOR THE LOADED DATASET")
print("=" * 70)

df = df_raw.copy()

# Define pipeline steps

def step_drop_duplicates(df):
    """Remove exact duplicate rows."""
    return df.drop_duplicates()


def step_standardize_strings(df):
    """Strip whitespace and lowercase string columns."""
    result = df.copy()
    for col in result.select_dtypes(include=["object", "str"]).columns:
        result[col] = result[col].str.strip()
    return result


def step_fill_missing(df):
    """Fill missing values with appropriate defaults."""
    result = df.copy()
    numeric_cols = result.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if result[col].isnull().any():
            result[col] = result[col].fillna(result[col].median())
    object_cols = result.select_dtypes(include=["object", "str"]).columns
    for col in object_cols:
        if result[col].isnull().any():
            result[col] = result[col].fillna("Unknown")
    return result


def step_validate_ranges(df):
    """Clip values to valid physical ranges."""
    result = df.copy()
    if dataset_name == "superstore":
        if "sales" in result.columns:
            result["sales"] = result["sales"].clip(lower=0)
        if "discount" in result.columns:
            result["discount"] = result["discount"].clip(lower=0, upper=1)
        if "quantity" in result.columns:
            result["quantity"] = result["quantity"].clip(lower=1)
    else:
        if "age" in result.columns:
            result["age"] = result["age"].clip(lower=0, upper=120)
        if "fare" in result.columns:
            result["fare"] = result["fare"].clip(lower=0)
    return result


def step_optimize_dtypes(df):
    """Optimize numeric dtypes for memory efficiency."""
    result = df.copy()
    for col in result.select_dtypes(include=[np.number]).columns:
        if result[col].dtype == np.float64:
            result[col] = result[col].astype(np.float32)
    for col in result.select_dtypes(include=["object", "str"]).columns:
        if result[col].nunique() / len(result) < 0.5:
            result[col] = result[col].astype("category")
    return result


def step_add_features(df):
    """Engineer derived features."""
    result = df.copy()
    if dataset_name == "superstore":
        if "sales" in result.columns and "profit" in result.columns:
            result["profit_margin"] = np.where(
                result["sales"] > 0,
                result["profit"] / result["sales"],
                0.0
            )
        if "quantity" in result.columns and "sales" in result.columns:
            result["unit_sale_value"] = np.where(
                result["quantity"] > 0,
                result["sales"] / result["quantity"],
                0.0
            )
    else:
        if "sibsp" in result.columns and "parch" in result.columns:
            result["family_size"] = result["sibsp"] + result["parch"] + 1
            result["is_alone"]    = (result["family_size"] == 1).astype(int)
        if "fare" in result.columns:
            result["fare_log"] = np.log1p(result["fare"])
    return result


# Define validation check functions for each step

def check_input_has_data(df):
    return [ValidationResult(
        name     = "input_has_rows",
        passed   = len(df) > 0,
        message  = f"Input has {len(df)} rows",
        severity = "error"
    )]


def check_no_empty_result(df):
    return [ValidationResult(
        name     = "result_not_empty",
        passed   = len(df) > 0,
        message  = (
            f"Result has {len(df)} rows"
            if len(df) > 0
            else "Result is empty - all rows removed!"
        ),
        severity = "error"
    )]


def check_no_missing_after_fill(df):
    """Validate no missing values remain in numeric columns."""
    results = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        n_null = df[col].isnull().sum()
        results.append(ValidationResult(
            name     = f"filled_{col}",
            passed   = n_null == 0,
            message  = (
                f"{col}: no missing values"
                if n_null == 0
                else f"{col}: {n_null} missing values remain"
            ),
            severity = "error"
        ))
    return results


def check_derived_features_exist(df):
    """Validate that feature engineering created expected columns."""
    if dataset_name == "superstore":
        expected = ["profit_margin", "unit_sale_value"]
    else:
        expected = ["family_size", "is_alone", "fare_log"]

    results = []
    for col in expected:
        results.append(ValidationResult(
            name     = f"feature_exists_{col}",
            passed   = col in df.columns,
            message  = (
                f"Feature '{col}' created"
                if col in df.columns
                else f"Feature '{col}' missing from output"
            ),
            severity = "error"
        ))
    return results


# Build the pipeline
pipeline = DataPipeline(
    name       = f"{dataset_name}_cleaning_pipeline",
    on_failure = "warn"
)

pipeline.add_step(
    func          = step_drop_duplicates,
    input_checks  = check_input_has_data,
    output_checks = check_no_empty_result,
    step_name     = "Drop Duplicates"
)
pipeline.add_step(
    func      = step_standardize_strings,
    step_name = "Standardize Strings"
)
pipeline.add_step(
    func          = step_fill_missing,
    output_checks = check_no_missing_after_fill,
    step_name     = "Fill Missing Values"
)
pipeline.add_step(
    func      = step_validate_ranges,
    step_name = "Clip Out-of-Range Values"
)
pipeline.add_step(
    func      = step_optimize_dtypes,
    step_name = "Optimize Dtypes"
)
pipeline.add_step(
    func          = step_add_features,
    output_checks = check_derived_features_exist,
    step_name     = "Feature Engineering"
)

# Run the pipeline
df_clean = pipeline.run(df_raw)

# Print execution history
pipeline.print_history()

# Print validation report
pipeline.report.print_summary()

# Show result
print(f"\nFinal cleaned DataFrame:")
print(f"  Shape: {df_clean.shape}")
print(f"  Columns: {list(df_clean.columns)}")
print("\nFirst 5 rows:")
print(df_clean.head())


# ==============================================================================
# SECTION 8: METHOD CHAINING PATTERN
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 8: METHOD CHAINING PATTERN")
print("=" * 70)

explanation = """
Method chaining applies transformations in a readable linear sequence.
Each method returns a DataFrame so the next method can be called immediately.

Advantages:
  - Reads like a list of instructions (top to bottom)
  - No intermediate variable names to track
  - Easy to comment out or reorder steps
  - Works naturally with Pandas pipe()

df.pipe(step1)
  .pipe(step2)
  .pipe(step3)
  ...
  .pipe(stepN)
"""
print(explanation)


def clean_and_validate(df):
    """Step 1: clean."""
    return df.dropna(thresh=int(len(df.columns) * 0.5))


def add_computed_columns(df):
    """Step 2: add features."""
    result = df.copy()
    if dataset_name == "superstore" and "sales" in result.columns:
        result["high_value"] = result["sales"] > result["sales"].quantile(0.75)
    elif dataset_name == "titanic" and "fare" in result.columns:
        result["high_value"] = result["fare"] > result["fare"].quantile(0.75)
    return result


def filter_to_analysis_set(df):
    """Step 3: filter."""
    if dataset_name == "superstore" and "sales" in df.columns:
        return df[df["sales"] > 0].copy()
    elif dataset_name == "titanic" and "survived" in df.columns:
        return df[df["survived"].isin([0, 1])].copy()
    return df.copy()


def log_shape(df, label=""):
    """Utility: log current shape without transforming."""
    print(f"  [pipe log] {label}: {df.shape}")
    return df


# Clean pipeline using .pipe()
print("Method chaining with .pipe():")
df_piped = (
    df_raw
    .pipe(log_shape, "after load")
    .pipe(clean_and_validate)
    .pipe(log_shape, "after clean")
    .pipe(add_computed_columns)
    .pipe(log_shape, "after features")
    .pipe(filter_to_analysis_set)
    .pipe(log_shape, "final")
)

print(f"\nFinal piped DataFrame shape: {df_piped.shape}")
print("Columns:")
print(list(df_piped.columns))


# ==============================================================================
# SECTION 9: TESTING PIPELINE WITH KNOWN INPUTS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 9: TESTING PIPELINE WITH KNOWN INPUTS")
print("=" * 70)

explanation = """
Testing a pipeline means:
  1. Create small DataFrames with KNOWN properties
  2. Run the pipeline (or a step) on them
  3. Assert the output matches exactly what you expect

This catches regressions when pipeline logic changes.
Always test:
  - Normal happy-path input
  - Edge cases (empty DataFrame, all nulls, single row)
  - Invalid input (wrong dtypes, out-of-range values)
"""
print(explanation)


def run_pipeline_tests():
    """Run a suite of pipeline validation tests."""
    test_results = []

    def record(test_name, passed, message=""):
        test_results.append({"test": test_name, "passed": passed, "msg": message})
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {test_name}" + (f": {message}" if message else ""))

    print("\nRunning pipeline tests...")

    # Test 1: step_drop_duplicates removes duplicates
    df_with_dupes = df_raw.head(10).copy()
    df_with_dupes = pd.concat([df_with_dupes, df_with_dupes.head(3)], ignore_index=True)
    result = step_drop_duplicates(df_with_dupes)
    expected_rows = len(df_raw.head(10))
    record(
        "drop_duplicates_removes_exact_dupes",
        len(result) == expected_rows,
        f"Expected {expected_rows}, got {len(result)}"
    )

    # Test 2: step_drop_duplicates preserves columns
    record(
        "drop_duplicates_preserves_columns",
        list(result.columns) == list(df_with_dupes.columns)
    )

    # Test 3: step_fill_missing eliminates nulls in numeric cols
    df_with_nulls = df_raw.head(20).copy()
    numeric_col   = df_with_nulls.select_dtypes(include=[np.number]).columns[0]
    df_with_nulls.iloc[0, df_with_nulls.columns.get_loc(numeric_col)] = np.nan
    result_filled = step_fill_missing(df_with_nulls)
    nulls_after   = result_filled[numeric_col].isnull().sum()
    record(
        "fill_missing_clears_numeric_nulls",
        nulls_after == 0,
        f"Nulls remaining: {nulls_after}"
    )

    # Test 4: step_fill_missing does not increase rows
    record(
        "fill_missing_preserves_row_count",
        len(result_filled) == len(df_with_nulls)
    )

    # Test 5: step_validate_ranges clips values
    if dataset_name == "titanic" and "age" in df_raw.columns:
        df_bad_ages = df_raw.head(10).copy()
        df_bad_ages["age"] = 999
        result_clipped = step_validate_ranges(df_bad_ages)
        max_age = result_clipped["age"].max()
        record(
            "validate_ranges_clips_age_to_120",
            max_age <= 120,
            f"Max age after clip: {max_age}"
        )

    if dataset_name == "superstore" and "discount" in df_raw.columns:
        df_bad_discount = df_raw.head(10).copy()
        df_bad_discount["discount"] = 5.0
        result_clipped = step_validate_ranges(df_bad_discount)
        max_discount = result_clipped["discount"].max()
        record(
            "validate_ranges_clips_discount_to_1",
            max_discount <= 1.0,
            f"Max discount after clip: {max_discount}"
        )

    # Test 6: step_add_features creates new columns
    result_features = step_add_features(df_raw.head(20).copy())
    if dataset_name == "superstore":
        record(
            "add_features_creates_profit_margin",
            "profit_margin" in result_features.columns
        )
    else:
        record(
            "add_features_creates_family_size",
            "family_size" in result_features.columns
        )

    # Test 7: step_add_features does not drop existing columns
    original_cols = set(df_raw.columns)
    result_cols   = set(result_features.columns)
    record(
        "add_features_preserves_original_columns",
        original_cols.issubset(result_cols),
        f"Missing: {original_cols - result_cols}"
    )

    # Test 8: Empty DataFrame edge case
    df_empty = df_raw.head(0).copy()
    result_empty = step_drop_duplicates(df_empty)
    record(
        "drop_duplicates_handles_empty_df",
        len(result_empty) == 0
    )

    # Test 9: Single row edge case
    df_single = df_raw.head(1).copy()
    result_single = step_fill_missing(df_single)
    record(
        "fill_missing_handles_single_row",
        len(result_single) == 1
    )

    # Test 10: ValidationResult pass/fail logic
    r_pass = ValidationResult("test", True, "ok", "error")
    r_fail = ValidationResult("test", False, "fail", "error")
    report_test = ValidationReport("test")
    report_test.add(r_pass)
    report_test.add(r_fail)
    record(
        "validation_report_detects_failure",
        not report_test.passed()
    )

    # Summary
    passed = sum(1 for r in test_results if r["passed"])
    total  = len(test_results)
    print(f"\n  Test summary: {passed}/{total} passed")
    assert passed == total, f"Tests failed: {total - passed} failures"
    return test_results


test_summary = run_pipeline_tests()


# ==============================================================================
# SECTION 10: LOGGING VALIDATION RESULTS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 10: LOGGING VALIDATION RESULTS")
print("=" * 70)

explanation = """
In production, validation results should be persisted for:
  - Tracking data quality trends over time
  - Alerting when quality degrades
  - Auditing pipeline runs for compliance
  - Debugging intermittent issues

The simplest approach is writing ValidationReport to a CSV or JSON.
This lesson demonstrates writing to an in-memory DataFrame (simulating disk write).
"""
print(explanation)


class ValidationLogger:
    """
    Logs validation results to an append-only DataFrame.
    In production this would write to a database or file.
    """

    def __init__(self):
        self.log_df = pd.DataFrame(columns=[
            "run_id", "pipeline_name", "step", "check_name",
            "passed", "severity", "message", "timestamp"
        ])
        self.run_counter = 0

    def log_report(self, report, step_name=""):
        """Append all results from a ValidationReport."""
        self.run_counter += 1
        run_id = f"run_{self.run_counter:04d}"

        new_rows = []
        for r in report.results:
            new_rows.append({
                "run_id":        run_id,
                "pipeline_name": report.pipeline_name,
                "step":          step_name,
                "check_name":    r.name,
                "passed":        r.passed,
                "severity":      r.severity,
                "message":       r.message,
                "timestamp":     r.timestamp
            })

        if new_rows:
            new_df       = pd.DataFrame(new_rows)
            self.log_df  = pd.concat([self.log_df, new_df], ignore_index=True)

        return run_id

    def get_failure_rate(self, pipeline_name=None):
        """Compute failure rate by check name."""
        df_log = self.log_df.copy()
        if pipeline_name:
            df_log = df_log[df_log["pipeline_name"] == pipeline_name]

        if len(df_log) == 0:
            return pd.DataFrame()

        return df_log.groupby("check_name").agg(
            total_runs   = ("passed", "count"),
            pass_count   = ("passed", "sum"),
            fail_count   = ("passed", lambda x: (~x).sum()),
        ).assign(
            pass_rate = lambda x: (x["pass_count"] / x["total_runs"] * 100).round(1)
        ).sort_values("fail_count", ascending=False)

    def print_recent(self, n=10):
        """Print the most recent log entries."""
        recent = self.log_df.tail(n)
        print(f"\nLast {len(recent)} validation log entries:")
        print(recent[["run_id", "check_name", "passed", "severity", "message"]].to_string(index=False))


# Demonstrate logger
logger = ValidationLogger()

# Simulate multiple pipeline runs
for run_num in range(3):
    run_report = ValidationReport(f"{dataset_name}_pipeline")
    schema_r = check_schema(df_raw, expected_cols)
    for r in schema_r:
        run_report.add(r)
    completeness_r = check_completeness(df_raw, max_missing_pct=30.0)
    for r in completeness_r:
        run_report.add(r)
    run_id = logger.log_report(run_report, step_name="quality_check")
    print(f"Logged run: {run_id}")

print(f"\nTotal log entries: {len(logger.log_df)}")
logger.print_recent(8)

print("\nFailure rate by check:")
print(logger.get_failure_rate())


# ==============================================================================
# SECTION 11: COMMON PITFALLS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 11: COMMON PITFALLS")
print("=" * 70)

pitfalls = """
Pitfall 1: Validating AFTER the problem has propagated
   - Validate at the ENTRY POINT of each pipeline step
   - The earlier you catch a problem, the cheaper it is to fix

Pitfall 2: Raising an exception from inside a transform step
   - Always separate validation logic from transformation logic
   - Validation functions return ValidationResult, never raise directly

Pitfall 3: Using assert statements in production validation
   - assert is disabled when Python runs with -O flag
   - Use explicit if-checks and raise ValueError explicitly

Pitfall 4: Checking only for NaN, not for sentinel values
   - Many systems use -999, 0, 'N/A', 'null' as missing indicators
   - Always inspect sample data before writing range rules

Pitfall 5: Forgetting that validation can silently pass on empty data
   - isnull().sum() == 0 on an empty DataFrame always passes
   - Always check row count first before other validations

Pitfall 6: Overly strict validation breaks on legitimate data drift
   - Data distributions change over time
   - Use warning severity for statistical checks, error for structural checks
   - Review and update thresholds periodically

Pitfall 7: Not logging validation results
   - Without logs you cannot detect gradual quality degradation
   - Always persist validation reports with timestamps

Pitfall 8: Validating a copy when you modified the original
   - df_clean = df.copy(); some_step(df)  <- validates df, not df_clean
   - Always validate the same object you will use downstream

Pitfall 9: Composing pipeline steps that assume specific column names
   - Each step should validate its required columns exist before using them
   - Use check_schema at the start of every step that has requirements

Pitfall 10: Testing only the happy path
   - Always test edge cases: empty DataFrame, all nulls, out-of-range values
   - Test that validation FAILS correctly, not just that it passes
"""
print(pitfalls)


# ==============================================================================
# SECTION 12: SUMMARY TABLE
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 12: SUMMARY TABLE")
print("=" * 70)

summary = """
Pattern                          | Implementation
---------------------------------|---------------------------------------------------
Check required columns exist     | check_schema(df, expected_columns)
Check missing value threshold    | check_completeness(df, max_missing_pct=5)
Check numeric ranges             | check_ranges(df, {col: {min:0, max:100}})
Check column uniqueness          | check_uniqueness(df, ["id_col"])
Check valid categorical values   | check_referential_integrity(df, col, valid_set)
Check statistical properties     | check_statistics(df, {col: {mean_min:0}})
Collect and report all checks    | ValidationReport + ValidationResult
Auto-validate pipeline steps     | @validate_step(input_checks, output_checks)
Structured step-by-step pipeline | DataPipeline.add_step().run(df)
Method chaining                  | df.pipe(step1).pipe(step2).pipe(step3)
Test with known inputs           | Create mini-DataFrame, assert exact output
Log validation history           | ValidationLogger.log_report(report)
Warn vs raise on failure         | on_failure='warn' or on_failure='raise'
"""
print(summary)


# ==============================================================================
# SECTION 13: PRACTICE EXERCISES
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 13: PRACTICE EXERCISES")
print("=" * 70)

print("Exercise 1: Write a validation check that detects constant columns")


def check_no_constant_columns(df, severity="warning"):
    """Flag columns where all non-null values are identical."""
    results = []
    for col in df.columns:
        n_unique = df[col].nunique()
        results.append(ValidationResult(
            name     = f"not_constant_{col}",
            passed   = n_unique > 1,
            message  = (
                f"{col}: {n_unique} unique values (OK)"
                if n_unique > 1
                else f"{col}: constant column (only 1 unique value)"
            ),
            severity = severity
        ))
    return [r for r in results if not r.passed]


constant_check = check_no_constant_columns(df_raw)
print(f"  Constant columns found: {len(constant_check)}")
for r in constant_check:
    print(f"  {r}")

print("\nExercise 2: Build a one-step validation pipeline for the loaded data")
quick_pipe = DataPipeline("quick_check", on_failure="warn")
quick_pipe.add_step(
    func          = step_drop_duplicates,
    input_checks  = check_input_has_data,
    output_checks = check_no_empty_result,
    step_name     = "Remove Duplicates"
)
quick_pipe.add_step(
    func      = step_fill_missing,
    step_name = "Fill Nulls"
)
quick_result = quick_pipe.run(df_raw)
print(f"  Quick pipeline output shape: {quick_result.shape}")

print("\nExercise 3: Verify that fill_missing preserves values that are not null")
df_test = df_raw.head(5).copy()
numeric_col = df_test.select_dtypes(include=[np.number]).columns[0]
original_values = df_test[numeric_col].dropna().tolist()
filled = step_fill_missing(df_test)
preserved = filled[numeric_col].tolist()[:len(original_values)]
match = all(
    abs(a - b) < 0.001
    for a, b in zip(original_values, preserved)
    if not pd.isna(a)
)
print(f"  Non-null values preserved: {match}")

print("\nExercise 4: Count warnings vs errors in pipeline report")
report_df = pipeline.report.to_dataframe()
if len(report_df) > 0:
    severity_counts = report_df["severity"].value_counts()
    print(f"  Validation check severities:")
    print(severity_counts)
    pass_rate = report_df["passed"].mean() * 100
    print(f"  Overall pass rate: {pass_rate:.1f}%")


# ==============================================================================
# SECTION 14: KEY TAKEAWAYS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 14: KEY TAKEAWAYS")
print("=" * 70)

takeaways = """
1.  Validate at every pipeline entry and exit point, not just at the end

2.  Separate validation logic from transformation logic completely
    Validation functions return results; they do not transform data

3.  Use severity levels: error halts the pipeline, warning logs and continues

4.  Collect all validation results into a report before deciding to raise
    This gives you a complete picture of all problems at once

5.  Persist validation logs with timestamps to track data quality over time

6.  Test pipeline steps with known inputs including edge cases

7.  Use on_failure='warn' during development, 'raise' in production

8.  Validate schema first: if columns are missing all other checks are invalid

9.  Statistical thresholds should be warnings not errors because
    data distributions legitimately change over time

10. Method chaining with .pipe() produces the most readable pipelines

11. Always check row count before and after every filtering step

12. Build small, single-responsibility step functions that are
    easy to test, reorder, and replace independently
"""
print(takeaways)


# ==============================================================================
# NEXT LESSON PREVIEW
# ==============================================================================
print("\n" + "=" * 70)
print("NEXT LESSON PREVIEW")
print("=" * 70)

preview = """
Lesson 30: Debugging Data Pipelines

You will learn:
- Systematic debugging workflow for data pipelines
- Inspecting intermediate outputs at every step
- Tracing NaN and incorrect value propagation
- Detecting silent shape changes
- Common Pandas pitfalls that cause incorrect output
- Writing diagnostic helper functions
- Reproducing bugs with minimal examples
- Building a self-auditing pipeline with checkpoints
"""
print(preview)

print("\n" + "=" * 70)
print("END OF LESSON 29")
print("=" * 70)
print("\nYou can now build production-grade data pipelines with validation")
print("at every step, comprehensive reporting, logging, and test coverage.")