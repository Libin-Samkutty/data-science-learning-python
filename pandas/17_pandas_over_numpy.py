"""
LESSON 17: TRANSITION TO PANDAS - WHY PANDAS OVER NUMPY FOR REAL DATA
================================================================================

What You Will Learn:
- Limitations of NumPy for real-world tabular data
- How Pandas builds on top of NumPy
- Series vs DataFrame fundamentals
- Loading real CSV data directly from a public URL
- Basic DataFrame inspection and exploration methods
- Direct comparison between NumPy and Pandas approaches

Real World Usage:
- Cleaning and exploring customer datasets
- Preparing features for machine learning models
- Aggregating sales or sensor data
- Building reproducible data pipelines in production

Dataset Used:
Titanic passenger survival data (classic introductory dataset)
URL: https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv
This CSV contains mixed data types, missing values, and labeled columns.

================================================================================
"""

import numpy as np
import pandas as pd
import urllib.request

print("=" * 70)
print("LESSON 17: TRANSITION TO PANDAS - WHY PANDAS OVER NUMPY FOR REAL DATA")
print("=" * 70)


# ==============================================================================
# SECTION 1: WHY PANDAS OVER NUMPY FOR REAL DATA
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 1: WHY PANDAS OVER NUMPY FOR REAL DATA")
print("=" * 70)

explanation = """
NumPy is excellent for numerical arrays but has limitations with real data:

1. Homogeneous data types only (all elements same dtype)
2. No column labels - you must remember what index 3 means
3. Poor support for missing values (NaN works but is cumbersome)
4. No built-in alignment when combining datasets
5. Difficult to handle mixed string and numeric data

Pandas builds directly on NumPy but adds:
- Heterogeneous columns (int, float, string, datetime in one table)
- Labeled axes (column names and index labels)
- Built-in handling of missing data
- Powerful grouping, merging, and reshaping tools
- Seamless integration with NumPy (DataFrames contain NumPy arrays)

We will use the Titanic dataset to demonstrate these differences.
"""
print(explanation)


# ==============================================================================
# SECTION 2: LOADING REAL DATA WITH PANDAS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 2: LOADING REAL DATA WITH PANDAS")
print("=" * 70)

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

print("Loading dataset from public URL:")
print(url)

# pandas.read_csv can read directly from URL
df = pd.read_csv(url)

print("\nData loaded successfully.")
print("Type of object:", type(df))
print("Shape (rows, columns):", df.shape)


# ==============================================================================
# SECTION 3: DATAFRAME VS NUMPY ARRAY
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 3: DATAFRAME VS NUMPY ARRAY")
print("=" * 70)

# Convert to numpy for comparison
data_array = df.to_numpy()

print("Pandas DataFrame shape:", df.shape)
print("NumPy array shape:    ", data_array.shape)
print("DataFrame column names:", list(df.columns))
print("\nFirst 3 rows as DataFrame:")
print(df.head(3))
print("\nFirst 3 rows as NumPy array:")
print(data_array[:3])

print("\nKey difference:")
print("- DataFrame has column labels and mixed types")
print("- NumPy array is homogeneous and unlabeled")


# ==============================================================================
# SECTION 4: PANDAS SERIES FUNDAMENTALS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 4: PANDAS SERIES FUNDAMENTALS")
print("=" * 70)

# A Series is a 1-dimensional labeled array (like a labeled NumPy array)
age_series = df["Age"]

print("Extracted Age column as Series:")
print("Type:", type(age_series))
print("Shape:", age_series.shape)
print("Index labels (first 5):", age_series.index[:5].tolist())
print("\nFirst 5 values:")
print(age_series.head())

print("\nSeries contains:")
print("- Values (backed by NumPy array):", type(age_series.values))
print("- Index labels")
print("- Name:", age_series.name)

# Series can be created directly
example_series = pd.Series([25, 30, 45, 22], index=["Alice", "Bob", "Charlie", "David"], name="Age")
print("\nManually created Series with custom index:")
print(example_series)


# ==============================================================================
# SECTION 5: DATAFRAME FUNDAMENTALS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 5: DATAFRAME FUNDAMENTALS")
print("=" * 70)

print("DataFrame is a 2D labeled structure with columns of different types.")
print("Each column is a Series that shares the same index.")

print("\nColumn data types:")
print(df.dtypes)

print("\nBasic DataFrame inspection methods:")
print("\n1. df.head() - first 5 rows")
print(df.head())

print("\n2. df.info() - summary of structure and memory")
print(df.info())

print("\n3. df.describe() - statistical summary of numeric columns")
print(df.describe())

print("\n4. df.columns - column labels")
print(list(df.columns))

print("\n5. df.index - row labels (default is integer range)")
print(df.index)


# ==============================================================================
# SECTION 6: HANDLING MISSING VALUES - PANDAS VS NUMPY
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 6: HANDLING MISSING VALUES - PANDAS VS NUMPY")
print("=" * 70)

# Count missing values per column
missing_counts = df.isnull().sum()
print("Missing values per column:")
print(missing_counts)

print("\nPercentage of missing values:")
print((df.isnull().mean() * 100).round(2))

# In NumPy we would need to be more manual
array_version = df.select_dtypes(include=[np.number]).to_numpy().astype(float)
nan_count_numpy = np.isnan(array_version).sum(axis=0)
print("\nEquivalent count using NumPy (numeric columns only):")
print(nan_count_numpy[:6])  # Only first few for illustration

print("\nPandas makes missing value detection and handling much easier.")


# ==============================================================================
# SECTION 7: LABEL BASED ACCESS VS POSITIONAL ACCESS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 7: LABEL BASED ACCESS VS POSITIONAL ACCESS")
print("=" * 70)

print("Pandas allows selecting data by column NAME instead of position.")

# Select specific columns by name (very readable)
subset = df[["Name", "Age", "Sex", "Survived"]]

print("Selected columns by name:")
print(subset.head(3))

# Compare to NumPy (requires knowing column positions)
# Assume columns: 0=PassengerId, 3=Name, 4=Sex, 5=Age, 1=Survived (actual order varies)
numpy_subset = data_array[:5, [3, 5, 4, 1]]
print("\nEquivalent NumPy selection requires remembering column indices:")
print(numpy_subset)

print("\nAdvantage of Pandas: code is self-documenting with column names.")


# ==============================================================================
# SECTION 8: REAL WORLD EXAMPLE - BASIC EXPLORATORY ANALYSIS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 8: REAL WORLD EXAMPLE - BASIC EXPLORATORY ANALYSIS")
print("=" * 70)

print("In real projects we often start with these questions:")
print("1. How many passengers survived?")
print("2. What was the average age?")
print("3. How many were male vs female?")

# Using Pandas (clean and readable)
survival_rate = df["Survived"].mean() * 100
print("\nSurvival rate: {:.2f}%".format(survival_rate))

avg_age = df["Age"].mean()
print("Average age: {:.1f} years".format(avg_age))

gender_counts = df["Sex"].value_counts()
print("\nGender distribution:")
print(gender_counts)

print("\nEquivalent operations in pure NumPy would require:")
print("- Mapping column indices")
print("- Manual handling of string columns")
print("- More code for grouping")


# ==============================================================================
# SECTION 9: PANDAS IS BUILT ON NUMPY
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 9: PANDAS IS BUILT ON NUMPY")
print("=" * 70)

print("Under the hood, each DataFrame column is a NumPy array.")
print("We can access the raw NumPy array with .values or .to_numpy()")

age_values = df["Age"].values
print("Type of df['Age'].values:", type(age_values))
print("Age array shape:", age_values.shape)
print("First 10 ages:", age_values[:10])

print("\nWe can use NumPy functions directly on Pandas objects:")
print("NumPy mean of Age column:", np.mean(df["Age"]))

print("\nBest practice:")
print("- Use Pandas for data loading, cleaning, selection, and grouping")
print("- Drop down to NumPy for heavy numerical computation when needed")
print("- Convert with .values only at the final step if required by ML library")


# ==============================================================================
# SECTION 10: COMMON TRANSITION PITFALLS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 10: COMMON TRANSITION PITFALLS")
print("=" * 70)

print("Pitfall 1: Forgetting that Pandas indices are labels, not just positions")
print("Pitfall 2: Mixing up .iloc (position) and .loc (label)")
print("Pitfall 3: Assuming all columns are numeric like in NumPy")
print("Pitfall 4: Not handling missing values explicitly")
print("Pitfall 5: Using loops instead of vectorized Pandas/NumPy operations")

print("\nExample of good practice - validation after loading:")
print("Check shape:", df.shape)
print("Check columns:", list(df.columns))
print("Check dtypes:\n", df.dtypes)
print("Check for missing data:\n", df.isnull().sum())


# ==============================================================================
# SECTION 11: SUMMARY TABLE - NUMPY VS PANDAS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 11: SUMMARY TABLE - NUMPY VS PANDAS")
print("=" * 70)

summary = """
Operation                | NumPy Approach                     | Pandas Approach
-------------------------|------------------------------------|-----------------------
Data Loading             | np.loadtxt() or np.genfromtxt()    | pd.read_csv()
Column Selection         | array[:, 2] (by position)          | df["Age"] (by name)
Missing Value Check      | np.isnan(array).sum()              | df.isnull().sum()
Summary Statistics       | np.mean(array, axis=0)             | df.describe()
Filtering Rows           | array[array[:, 4] > 30]            | df[df["Age"] > 30]
Grouping                 | Manual loops or advanced indexing  | df.groupby("Sex").mean()
Merging Datasets         | np.concatenate() with careful alignment | pd.merge() or pd.concat()
Readability              | Good for math, poor for business data | Excellent for labeled tabular data
"""
print(summary)


# ==============================================================================
# SECTION 12: PRACTICE EXERCISES
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 12: PRACTICE EXERCISES")
print("=" * 70)

print("\nExercise 1: Basic Inspection")
print("1. Print the number of unique values in the 'Pclass' column")
print("2. Calculate the survival rate for females only")
print("Solution code is shown below:")

unique_pclass = df["Pclass"].nunique()
print("Unique Pclass values:", unique_pclass)

female_survival = df[df["Sex"] == "female"]["Survived"].mean()
print("Female survival rate:", round(female_survival * 100, 2), "%")

print("\nExercise 2: Compare Memory Usage")
numpy_mem = data_array.nbytes / 1024
pandas_mem = df.memory_usage(deep=True).sum() / 1024
print("NumPy array memory (KB):", round(numpy_mem, 2))
print("Pandas DataFrame memory (KB):", round(pandas_mem, 2))
print("Note: Pandas uses more memory due to labels and object columns.")


# ==============================================================================
# SECTION 13: KEY TAKEAWAYS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 13: KEY TAKEAWAYS")
print("=" * 70)

takeaways = """
1. Use NumPy for pure numerical computation and performance critical sections.
2. Use Pandas when data has labels, mixed types, or missing values.
3. Pandas DataFrames and Series are built on NumPy arrays.
4. Always inspect data immediately after loading (shape, dtypes, missing values).
5. Column names make code more readable and less error-prone than numeric indices.
6. Start exploratory work in Pandas, drop to NumPy only when necessary for speed.
7. The transition is natural: many Pandas operations return NumPy arrays.
"""
print(takeaways)


# ==============================================================================
# NEXT LESSON PREVIEW
# ==============================================================================
print("\n" + "=" * 70)
print("NEXT LESSON PREVIEW")
print("=" * 70)

preview = """
Lesson 18: Pandas Indexing and Selection (loc, iloc, and Boolean Filtering)

You will learn:
- Label-based selection with .loc
- Position-based selection with .iloc
- Boolean masking for filtering rows
- Combining conditions safely
- Setting and resetting index
- Common indexing pitfalls and how to avoid them

This lesson will teach you how to efficiently extract exactly the data you need.
"""
print(preview)

print("\n" + "=" * 70)
print("END OF LESSON 17")
print("=" * 70)
print("\nYou have successfully transitioned from NumPy arrays to Pandas DataFrames.")
print("You can now load, inspect, and explore real tabular datasets.")