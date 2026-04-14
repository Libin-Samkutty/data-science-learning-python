"""
LESSON 18: PANDAS INDEXING AND SELECTION
================================================================================

What You Will Learn:
- Label-based selection with .loc
- Position-based selection with .iloc
- Single column and multi-column selection
- Boolean masking for filtering rows
- Combining multiple conditions safely
- Setting and resetting the index
- Common indexing mistakes and how to avoid them

Real World Usage:
- Extracting specific customer segments from a dataset
- Selecting feature columns for a machine learning model
- Filtering transactions above a threshold
- Auditing specific rows by ID or label

Dataset Used:
Titanic passenger survival data
URL: https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv

================================================================================
"""

import numpy as np
import pandas as pd

print("=" * 70)
print("LESSON 18: PANDAS INDEXING AND SELECTION")
print("=" * 70)


# ==============================================================================
# SECTION 1: LOAD AND VALIDATE THE DATASET
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 1: LOAD AND VALIDATE THE DATASET")
print("=" * 70)

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

print("Loading Titanic dataset from:")
print(url)

df = pd.read_csv(url)

# Always validate immediately after loading
print("\nValidation checks:")
print("  Shape          :", df.shape)
print("  Columns        :", list(df.columns))
print("  Dtypes         :\n", df.dtypes)
print("\nMissing values per column:")
print(df.isnull().sum())

print("\nFirst 5 rows:")
print(df.head())


# ==============================================================================
# SECTION 2: SINGLE COLUMN SELECTION
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 2: SINGLE COLUMN SELECTION")
print("=" * 70)

# There are two syntaxes for selecting a single column

# Method 1: bracket notation (always works, preferred)
age_bracket = df["Age"]
print("Bracket notation - df['Age']:")
print("  Type  :", type(age_bracket))
print("  Shape :", age_bracket.shape)
print("  Head  :", age_bracket.head().tolist())

# Method 2: dot notation (only works when column name has no spaces or
# conflicts with DataFrame method names - use with caution)
age_dot = df.Age
print("\nDot notation - df.Age:")
print("  Type  :", type(age_dot))
print("  Equal to bracket:", age_bracket.equals(age_dot))

print("\nBest Practice:")
print("  Always use bracket notation df['column'].")
print("  Dot notation fails if column name has spaces or matches a method name.")
print("  Example: df['count'] works, df.count would call the count() method.")


# ==============================================================================
# SECTION 3: MULTIPLE COLUMN SELECTION
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 3: MULTIPLE COLUMN SELECTION")
print("=" * 70)

# Pass a list of column names inside double brackets
subset = df[["Name", "Sex", "Age", "Survived"]]

print("Selecting multiple columns with a list:")
print("  df[['Name', 'Sex', 'Age', 'Survived']]")
print("  Type  :", type(subset))
print("  Shape :", subset.shape)
print(subset.head())

# Common mistake: single brackets with multiple columns
print("\nCommon mistake - single brackets with multiple columns:")
try:
    wrong = df["Name", "Age"]
except KeyError as e:
    print("  KeyError:", e)
    print("  Fix: use double brackets df[['Name', 'Age']]")

# Selecting columns by position is also possible via .iloc
# Covered in detail in Section 5


# ==============================================================================
# SECTION 4: ROW SELECTION WITH loc (LABEL BASED)
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 4: ROW SELECTION WITH loc (LABEL BASED)")
print("=" * 70)

explanation = """
.loc is LABEL based.

Syntax:
  df.loc[row_label]
  df.loc[row_label, column_label]
  df.loc[start_label:end_label]           (end IS included)
  df.loc[[label1, label2], [col1, col2]]

By default the Titanic DataFrame index is 0, 1, 2, ... (integer labels).
So df.loc[5] and df.iloc[5] return the same row here.
The difference matters when the index is NOT a default integer range.
"""
print(explanation)

# Select single row by label
print("Single row by label - df.loc[0]:")
print(df.loc[0])

# Select row range by label (end label IS included in loc)
print("\nRow range - df.loc[0:3] (includes row 3):")
print(df.loc[0:3, ["Name", "Age", "Survived"]])

# Select specific rows and columns
print("\nSpecific rows and columns - df.loc[[0, 5, 10], ['Name', 'Pclass', 'Survived']]:")
print(df.loc[[0, 5, 10], ["Name", "Pclass", "Survived"]])

# Select all rows, specific columns
print("\nAll rows, specific columns - df.loc[:, ['Age', 'Fare']]:")
print(df.loc[:, ["Age", "Fare"]].head())


# ==============================================================================
# SECTION 5: ROW SELECTION WITH iloc (POSITION BASED)
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 5: ROW SELECTION WITH iloc (POSITION BASED)")
print("=" * 70)

explanation = """
.iloc is INTEGER POSITION based.

Syntax:
  df.iloc[row_position]
  df.iloc[row_position, col_position]
  df.iloc[start:end]                      (end is NOT included, like Python slicing)
  df.iloc[[pos1, pos2], [col_pos1, col_pos2]]

This is equivalent to NumPy array indexing.
Use when you need to select by row/column number, not by name.
"""
print(explanation)

# Select single row by position
print("Single row by position - df.iloc[0]:")
print(df.iloc[0])

# Select row range by position (end is NOT included)
print("\nRow range - df.iloc[0:4] (excludes row 4, includes 0,1,2,3):")
print(df.iloc[0:4][["Name", "Age", "Survived"]])

# Select specific rows and columns by position
print("\nSpecific rows and columns by position - df.iloc[[0, 5, 10], [3, 4, 1]]:")
print(df.iloc[[0, 5, 10], [3, 4, 1]])

# Select last 3 rows
print("\nLast 3 rows - df.iloc[-3:]:")
print(df.iloc[-3:][["Name", "Age", "Survived"]])

# Select every other row
print("\nEvery other row (first 10) - df.iloc[0:10:2]:")
print(df.iloc[0:10:2][["Name", "Age"]])


# ==============================================================================
# SECTION 6: loc vs iloc SIDE BY SIDE COMPARISON
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 6: loc vs iloc SIDE BY SIDE COMPARISON")
print("=" * 70)

# Create a small DataFrame with a non-default index to show the difference clearly
sample = df[["Name", "Age", "Survived"]].copy()
sample.index = [100, 101, 102, 103, 104] + list(range(105, 991))

print("DataFrame with custom index starting at 100:")
print(sample.head())

print("\ndf.loc[100] - row with LABEL 100 (first row):")
print(sample.loc[100])

print("\ndf.iloc[0] - row at POSITION 0 (also first row):")
print(sample.iloc[0])

print("\ndf.loc[103] - row with LABEL 103 (fourth row):")
print(sample.loc[103])

print("\ndf.iloc[3] - row at POSITION 3 (also fourth row):")
print(sample.iloc[3])

print("\nKey rule:")
print("  .loc  uses the INDEX LABEL  (what you see in the leftmost column)")
print("  .iloc uses the ROW NUMBER   (0, 1, 2, ... regardless of index)")


# ==============================================================================
# SECTION 7: BOOLEAN MASKING - FILTERING ROWS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 7: BOOLEAN MASKING - FILTERING ROWS")
print("=" * 70)

explanation = """
Boolean masking works exactly like NumPy boolean indexing.
A condition on a column produces a True/False Series.
Passing that mask to the DataFrame returns only the True rows.
"""
print(explanation)

# Single condition: passengers who survived
survived_mask = df["Survived"] == 1
print("Boolean mask - df['Survived'] == 1 (first 5 values):")
print(survived_mask.head())
print("Type:", type(survived_mask))
print("True count:", survived_mask.sum())

# Apply the mask
survivors = df[survived_mask]
print("\nFiltered DataFrame (survivors only):")
print("  Shape:", survivors.shape)
print(survivors[["Name", "Age", "Sex", "Survived"]].head())

# Shorthand: condition directly inside brackets
first_class = df[df["Pclass"] == 1]
print("\nFirst class passengers - df[df['Pclass'] == 1]:")
print("  Shape:", first_class.shape)

# Numeric comparison
older_passengers = df[df["Age"] > 60]
print("\nPassengers older than 60 - df[df['Age'] > 60]:")
print("  Shape:", older_passengers.shape)
print(older_passengers[["Name", "Age", "Sex", "Survived"]].head())


# ==============================================================================
# SECTION 8: COMBINING CONDITIONS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 8: COMBINING CONDITIONS")
print("=" * 70)

explanation = """
Combine conditions using:
  &   AND  (both conditions must be True)
  |   OR   (at least one condition must be True)
  ~   NOT  (inverts the condition)

CRITICAL: Always wrap each condition in parentheses.
Operator precedence in Python means & and | bind tighter than == < > etc.
Without parentheses you will get incorrect results or errors.
"""
print(explanation)

# AND condition: female survivors
female_survivors = df[(df["Sex"] == "female") & (df["Survived"] == 1)]
print("Female survivors - (df['Sex']=='female') & (df['Survived']==1):")
print("  Count:", len(female_survivors))
print(female_survivors[["Name", "Age", "Sex", "Survived"]].head())

# OR condition: first class OR survived
first_class_or_survived = df[(df["Pclass"] == 1) | (df["Survived"] == 1)]
print("\nFirst class OR survived:")
print("  Count:", len(first_class_or_survived))

# NOT condition: did not survive
not_survived = df[~(df["Survived"] == 1)]
print("\nDid NOT survive - ~(df['Survived'] == 1):")
print("  Count:", len(not_survived))

# Three conditions combined: young female survivors in first class
vip_survivors = df[
    (df["Sex"] == "female") &
    (df["Survived"] == 1) &
    (df["Pclass"] == 1)
]
print("\nFemale first-class survivors:")
print("  Count:", len(vip_survivors))
print(vip_survivors[["Name", "Age", "Pclass", "Survived"]].head())

# Dangerous example: what happens without parentheses
print("\nDanger - without parentheses:")
try:
    wrong = df[df["Sex"] == "female" & df["Survived"] == 1]
except TypeError as e:
    print("  TypeError:", e)
    print("  Fix: always wrap each condition in parentheses")


# ==============================================================================
# SECTION 9: isin, between, and str FILTERS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 9: isin, between, AND str FILTERS")
print("=" * 70)

# isin: filter for rows where value is in a list
port_filter = df[df["Embarked"].isin(["C", "Q"])]
print("Passengers who embarked at Cherbourg or Queenstown (isin):")
print("  Shape:", port_filter.shape)
print(port_filter[["Name", "Embarked"]].head())

# between: inclusive range filter (works for numeric and dates)
mid_age = df[df["Age"].between(20, 30)]
print("\nPassengers aged 20 to 30 (between):")
print("  Shape:", mid_age.shape)
print(mid_age[["Name", "Age"]].head())

# str accessor: filter on string content
# .str methods work on object (string) columns
has_mrs = df[df["Name"].str.contains("Mrs", na=False)]
print("\nPassengers whose name contains 'Mrs' (str.contains):")
print("  Shape:", has_mrs.shape)
print(has_mrs[["Name", "Sex", "Survived"]].head())

# Case insensitive search
has_master = df[df["Name"].str.contains("master", case=False, na=False)]
print("\nNames containing 'master' (case insensitive):")
print("  Count:", len(has_master))
print(has_master[["Name", "Age"]].head())


# ==============================================================================
# SECTION 10: SETTING AND RESETTING THE INDEX
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 10: SETTING AND RESETTING THE INDEX")
print("=" * 70)

explanation = """
By default the DataFrame index is 0, 1, 2, ...
Setting a meaningful column as the index makes label-based selection cleaner.
This is common when working with data that has a natural unique identifier.
"""
print(explanation)

# Set PassengerId as index
df_indexed = df.set_index("PassengerId")
print("DataFrame with PassengerId as index:")
print(df_indexed.head(3)[["Name", "Age", "Survived"]])
print("\nNew index:", df_indexed.index[:5].tolist())

# Now .loc uses PassengerId
print("\nSelect passenger with ID 5 using .loc[5]:")
print(df_indexed.loc[5][["Name", "Age", "Survived"]])

# Select a range by PassengerId
print("\nSelect passengers with IDs 3 to 6 using .loc[3:6]:")
print(df_indexed.loc[3:6][["Name", "Age", "Survived"]])

# Reset index: bring the index back as a column
df_reset = df_indexed.reset_index()
print("\nAfter reset_index():")
print(df_reset.head(3)[["PassengerId", "Name", "Age"]])
print("Index is back to default:", df_reset.index[:5].tolist())

# Setting index without keeping old index
df_name_indexed = df.set_index("Name")
print("\nWith Name as index, select by passenger name:")
print(df_name_indexed.loc["Braund, Mr. Owen Harris"][["Age", "Pclass", "Survived"]])


# ==============================================================================
# SECTION 11: SELECTING WITH loc AND BOOLEAN MASK TOGETHER
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 11: SELECTING WITH loc AND BOOLEAN MASK TOGETHER")
print("=" * 70)

explanation = """
.loc also accepts a boolean Series as the row selector.
This lets you filter rows AND select specific columns in one clean expression.
This is the most common pattern in production code.
"""
print(explanation)

# Filter rows and select columns in one step
result = df.loc[df["Survived"] == 1, ["Name", "Age", "Sex", "Pclass"]]
print("Survivors with selected columns - df.loc[df['Survived']==1, ['Name','Age','Sex','Pclass']]:")
print("  Shape:", result.shape)
print(result.head())

# More complex example
result2 = df.loc[
    (df["Pclass"] == 1) & (df["Age"] > 40),
    ["Name", "Age", "Fare", "Survived"]
]
print("\nFirst class passengers over 40:")
print("  Shape:", result2.shape)
print(result2.head())

# Using loc to update values (safe for assignment, unlike chained indexing)
df_copy = df.copy()
df_copy.loc[df_copy["Age"].isnull(), "Age"] = df_copy["Age"].median()
print("\nFilled missing Age values using loc (safe assignment):")
print("  Missing Age before:", df["Age"].isnull().sum())
print("  Missing Age after:", df_copy["Age"].isnull().sum())


# ==============================================================================
# SECTION 12: COMMON INDEXING PITFALLS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 12: COMMON INDEXING PITFALLS")
print("=" * 70)

# Pitfall 1: Chained indexing (do not do this for assignment)
print("Pitfall 1: Chained Indexing for Assignment")
print("  WRONG : df['Survived'][0] = 1")
print("  WHY   : Creates a temporary copy, changes may not apply to original")
print("  RIGHT : df.loc[0, 'Survived'] = 1")

# Demonstrate the correct pattern
df_copy2 = df.copy()
df_copy2.loc[0, "Survived"] = 0
print("  Correct assignment with loc:")
print("  df_copy2.loc[0, 'Survived'] =", df_copy2.loc[0, "Survived"])

# Pitfall 2: Slice end behavior difference between loc and iloc
print("\nPitfall 2: Slice End Behavior")
print("  .loc[0:3]  includes row 3 (label inclusive)")
print("  .iloc[0:3] excludes row 3 (position exclusive, like Python)")
print("  loc  result rows:", df.loc[0:3].index.tolist())
print("  iloc result rows:", df.iloc[0:3].index.tolist())

# Pitfall 3: Selecting single row returns Series, not DataFrame
print("\nPitfall 3: Single Row Selection Returns Series")
row_series = df.loc[0]
row_df = df.loc[[0]]
print("  df.loc[0]   type:", type(row_series))
print("  df.loc[[0]] type:", type(row_df))
print("  Use double brackets df.loc[[0]] if you need a DataFrame back")

# Pitfall 4: Boolean mask from different DataFrame
print("\nPitfall 4: Misaligned Boolean Mask")
mask = df["Age"] > 30
df_copy3 = df.copy().reset_index(drop=True)
print("  Mask and DataFrame share same index - safe to apply")
print("  Alwasy ensure your mask comes from the same DataFrame being filtered")

# Pitfall 5: Forgetting na=False in string filters
print("\nPitfall 5: String Filter Without na=False")
print("  WRONG: df[df['Embarked'].str.contains('C')]")
print("         This raises an error when NaN values exist in the column")
print("  RIGHT: df[df['Embarked'].str.contains('C', na=False)]")


# ==============================================================================
# SECTION 13: REAL WORLD EXAMPLE - PASSENGER AUDIT
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 13: REAL WORLD EXAMPLE - PASSENGER AUDIT")
print("=" * 70)

print("Scenario: Generate a report of high-fare passengers who did not survive")

# Step 1: calculate fare threshold (75th percentile)
fare_threshold = df["Fare"].quantile(0.75)
print("\nFare 75th percentile threshold:", round(fare_threshold, 2))

# Step 2: filter using combined conditions
audit_df = df.loc[
    (df["Fare"] > fare_threshold) & (df["Survived"] == 0),
    ["PassengerId", "Name", "Pclass", "Sex", "Age", "Fare", "Survived"]
]

print("High-fare passengers who did not survive:")
print("  Count:", len(audit_df))
print(audit_df.head(10))

# Step 3: summary statistics on this group
print("\nSummary of this group:")
print("  Average age :", round(audit_df["Age"].mean(), 1))
print("  Average fare:", round(audit_df["Fare"].mean(), 2))
print("  Gender split:\n", audit_df["Sex"].value_counts())
print("  By class:\n", audit_df["Pclass"].value_counts())

# Validate
assert audit_df["Survived"].sum() == 0, "All should be non-survivors"
assert all(audit_df["Fare"] > fare_threshold), "All fares should exceed threshold"
print("\nValidation passed: all rows are non-survivors above fare threshold")


# ==============================================================================
# SECTION 14: SUMMARY TABLE
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 14: SUMMARY TABLE")
print("=" * 70)

summary = """
Operation                        | Syntax
---------------------------------|----------------------------------------------
Single column                    | df["Age"]
Multiple columns                 | df[["Age", "Sex"]]
Row by label                     | df.loc[5]
Row by position                  | df.iloc[5]
Row range by label               | df.loc[2:5]          (end inclusive)
Row range by position            | df.iloc[2:5]         (end exclusive)
Row + column by label            | df.loc[2, "Age"]
Row + column by position         | df.iloc[2, 4]
Boolean filter                   | df[df["Age"] > 30]
Combined conditions              | df[(cond1) & (cond2)]
NOT condition                    | df[~(cond)]
Filter with column select        | df.loc[mask, ["Col1", "Col2"]]
Value in list                    | df[df["Embarked"].isin(["C", "Q"])]
Numeric range                    | df[df["Age"].between(20, 30)]
String contains                  | df[df["Name"].str.contains("Mrs", na=False)]
Set index                        | df.set_index("PassengerId")
Reset index                      | df.reset_index()
Safe assignment                  | df.loc[row_mask, "Col"] = value
"""
print(summary)


# ==============================================================================
# SECTION 15: PRACTICE EXERCISES
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 15: PRACTICE EXERCISES")
print("=" * 70)

print("Exercise 1: Select columns Name, Age, Fare for passengers in 2nd class")
result_ex1 = df.loc[df["Pclass"] == 2, ["Name", "Age", "Fare"]]
print("  Shape:", result_ex1.shape)
print(result_ex1.head())

print("\nExercise 2: Find all passengers between ages 18 and 25 who survived")
result_ex2 = df[df["Age"].between(18, 25) & (df["Survived"] == 1)]
print("  Count:", len(result_ex2))
print(result_ex2[["Name", "Age", "Survived"]].head())

print("\nExercise 3: Select every 50th passenger using iloc")
result_ex3 = df.iloc[::50][["Name", "Age", "Pclass"]]
print(result_ex3)

print("\nExercise 4: Set PassengerId as index, select IDs 10 to 15")
df_ex4 = df.set_index("PassengerId")
result_ex4 = df_ex4.loc[10:15, ["Name", "Age", "Survived"]]
print(result_ex4)


# ==============================================================================
# NEXT LESSON PREVIEW
# ==============================================================================
print("\n" + "=" * 70)
print("NEXT LESSON PREVIEW")
print("=" * 70)

preview = """
Lesson 19: Filtering Rows and Columns

You will learn:
- Advanced row filtering patterns
- Filtering columns by dtype
- Dropping rows and columns
- Filtering with query() method
- Dealing with missing values during filtering
- Building reusable filter functions for data pipelines
"""
print(preview)

print("\n" + "=" * 70)
print("END OF LESSON 18")
print("=" * 70)
print("\nYou can now select any subset of a DataFrame using labels, positions,")
print("and boolean conditions. These are the most used operations in daily data work.")