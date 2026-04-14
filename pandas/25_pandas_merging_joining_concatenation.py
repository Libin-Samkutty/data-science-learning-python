"""
LESSON 25: MERGING, JOINING, AND CONCATENATION
================================================================================

What You Will Learn:
- pd.merge() for database-style joins (inner, left, right, outer, cross)
- pd.concat() for stacking DataFrames vertically and horizontally
- join() as a shorthand for index-based merges
- Handling duplicate columns after merge
- Validating merge results with indicator and validate parameters
- Detecting and debugging merge problems
- Many-to-many join pitfalls
- Real world multi-table data assembly pipeline

Real World Usage:
- Combining customer data with transaction records
- Merging product catalog with inventory and sales data
- Assembling features from multiple sources for machine learning
- Combining monthly reports into annual datasets
- Joining lookup tables for data enrichment

Datasets Used:
We will create realistic relational datasets simulating:
- Customers table
- Orders table
- Products table
- Regions table (lookup)

================================================================================
"""

import numpy as np
import pandas as pd

print("=" * 70)
print("LESSON 25: MERGING, JOINING, AND CONCATENATION")
print("=" * 70)


# ==============================================================================
# SECTION 1: CREATE REALISTIC RELATIONAL DATASETS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 1: CREATE REALISTIC RELATIONAL DATASETS")
print("=" * 70)

np.random.seed(42)

# Customers table
customers = pd.DataFrame({
    "customer_id":   [1, 2, 3, 4, 5, 6, 7, 8],
    "customer_name": ["Alice", "Bob", "Charlie", "Diana",
                      "Edward", "Fiona", "George", "Hannah"],
    "region_code":   ["EAST", "WEST", "EAST", "NORTH",
                      "SOUTH", "WEST", "EAST", "NORTH"],
    "signup_date":   pd.to_datetime([
        "2022-01-15", "2022-03-20", "2022-02-10", "2022-05-01",
        "2022-04-12", "2022-06-30", "2022-07-05", "2022-08-18"
    ]),
    "is_premium":    [True, False, True, True, False, True, False, False]
})

# Orders table (some customers have multiple orders, some have none)
orders = pd.DataFrame({
    "order_id":     [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
    "customer_id":  [1, 1, 2, 3, 3, 3, 5, 6, 9, 10],  # 9, 10 are unknown customers
    "product_id":   ["P01", "P02", "P01", "P03", "P01", "P02", "P03", "P01", "P02", "P01"],
    "order_date":   pd.to_datetime([
        "2023-01-10", "2023-02-15", "2023-01-20", "2023-03-05",
        "2023-03-10", "2023-04-01", "2023-05-15", "2023-06-20",
        "2023-07-01", "2023-07-15"
    ]),
    "quantity":     [2, 1, 3, 1, 2, 1, 4, 2, 1, 3],
    "unit_price":   [25.00, 45.00, 25.00, 100.00, 25.00,
                     45.00, 100.00, 25.00, 45.00, 25.00]
})

# Products table
products = pd.DataFrame({
    "product_id":   ["P01", "P02", "P03", "P04"],
    "product_name": ["Widget", "Gadget", "Device", "Tool"],
    "category":     ["Electronics", "Electronics", "Hardware", "Hardware"],
    "cost":         [10.00, 20.00, 50.00, 15.00]
})

# Regions lookup table
regions = pd.DataFrame({
    "region_code": ["EAST", "WEST", "NORTH", "SOUTH"],
    "region_name": ["Eastern Region", "Western Region",
                    "Northern Region", "Southern Region"],
    "manager":     ["John Smith", "Jane Doe", "Bob Wilson", "Mary Johnson"]
})

# Monthly summaries (for concatenation examples)
jan_sales = pd.DataFrame({
    "product_id":   ["P01", "P02", "P03"],
    "month":        ["2023-01", "2023-01", "2023-01"],
    "total_sales":  [1500.00, 2200.00, 800.00]
})

feb_sales = pd.DataFrame({
    "product_id":   ["P01", "P02", "P03"],
    "month":        ["2023-02", "2023-02", "2023-02"],
    "total_sales":  [1800.00, 1900.00, 1200.00]
})

mar_sales = pd.DataFrame({
    "product_id":   ["P01", "P02", "P04"],  # Note: P03 missing, P04 added
    "month":        ["2023-03", "2023-03", "2023-03"],
    "total_sales":  [2100.00, 2500.00, 600.00]
})

print("CUSTOMERS TABLE:")
print(customers)
print(f"\nShape: {customers.shape}")

print("\nORDERS TABLE:")
print(orders)
print(f"\nShape: {orders.shape}")

print("\nPRODUCTS TABLE:")
print(products)
print(f"\nShape: {products.shape}")

print("\nREGIONS LOOKUP TABLE:")
print(regions)
print(f"\nShape: {regions.shape}")


# ==============================================================================
# SECTION 2: MERGE BASICS - JOIN TYPES
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 2: MERGE BASICS - JOIN TYPES")
print("=" * 70)

explanation = """
pd.merge() combines two DataFrames based on common columns (keys).
It works exactly like SQL JOIN operations.

JOIN TYPES (how parameter):
  'inner'  Only rows where key exists in BOTH tables (intersection)
  'left'   All rows from LEFT table, matching rows from RIGHT
  'right'  All rows from RIGHT table, matching rows from LEFT
  'outer'  All rows from BOTH tables (union)
  'cross'  Cartesian product (every left row paired with every right row)

Default is inner join.
"""
print(explanation)

# ------------------------------------------------------------------------------
# 2.1 Inner Join
# ------------------------------------------------------------------------------
print("\n--- 2.1 Inner Join ---")

inner_result = pd.merge(
    orders,
    customers,
    on="customer_id",
    how="inner"
)

print("INNER JOIN: orders + customers on customer_id")
print(f"Orders rows:    {len(orders)}")
print(f"Customers rows: {len(customers)}")
print(f"Result rows:    {len(inner_result)}")
print("\nResult (only orders from known customers):")
print(inner_result[["order_id", "customer_id", "customer_name", "region_code"]])

# Explain the row count
orders_with_known = orders[orders["customer_id"].isin(customers["customer_id"])]
print(f"\nRows dropped: {len(orders) - len(inner_result)}")
print("Orders 109, 110 dropped because customer_id 9, 10 not in customers table")

# ------------------------------------------------------------------------------
# 2.2 Left Join
# ------------------------------------------------------------------------------
print("\n--- 2.2 Left Join ---")

left_result = pd.merge(
    orders,
    customers,
    on="customer_id",
    how="left"
)

print("LEFT JOIN: orders + customers on customer_id")
print(f"Result rows: {len(left_result)}")
print("\nResult (all orders, NaN for unknown customers):")
print(left_result[["order_id", "customer_id", "customer_name", "region_code"]])

# Show which rows have NaN from the right table
missing_customer = left_result[left_result["customer_name"].isnull()]
print(f"\nOrders without matching customer: {len(missing_customer)}")
print(missing_customer[["order_id", "customer_id"]])

# ------------------------------------------------------------------------------
# 2.3 Right Join
# ------------------------------------------------------------------------------
print("\n--- 2.3 Right Join ---")

right_result = pd.merge(
    orders,
    customers,
    on="customer_id",
    how="right"
)

print("RIGHT JOIN: orders + customers on customer_id")
print(f"Result rows: {len(right_result)}")
print("\nResult (all customers, NaN for those without orders):")
print(right_result[["order_id", "customer_id", "customer_name"]])

# Customers without orders
no_orders = right_result[right_result["order_id"].isnull()]
print(f"\nCustomers without any orders: {len(no_orders)}")
print(no_orders[["customer_id", "customer_name"]])

# ------------------------------------------------------------------------------
# 2.4 Outer Join
# ------------------------------------------------------------------------------
print("\n--- 2.4 Outer Join (Full Outer) ---")

outer_result = pd.merge(
    orders,
    customers,
    on="customer_id",
    how="outer"
)

print("OUTER JOIN: orders + customers on customer_id")
print(f"Result rows: {len(outer_result)}")
print("\nResult (all orders AND all customers):")
print(outer_result[["order_id", "customer_id", "customer_name"]])

# Summary
print("\nOuter join preserves:")
print(f"  Orders without customers: {outer_result['customer_name'].isnull().sum()}")
print(f"  Customers without orders: {outer_result['order_id'].isnull().sum()}")

# ------------------------------------------------------------------------------
# 2.5 Cross Join
# ------------------------------------------------------------------------------
print("\n--- 2.5 Cross Join (Cartesian Product) ---")

# Small example to avoid huge output
small_left = pd.DataFrame({"size": ["S", "M", "L"]})
small_right = pd.DataFrame({"color": ["Red", "Blue"]})

cross_result = pd.merge(small_left, small_right, how="cross")

print("CROSS JOIN: sizes x colors")
print(f"Left rows:   {len(small_left)}")
print(f"Right rows:  {len(small_right)}")
print(f"Result rows: {len(cross_result)} (3 x 2 = 6)")
print("\nResult (every combination):")
print(cross_result)

print("\nUse case: Generate all possible product variants")


# ==============================================================================
# SECTION 3: MERGE ON DIFFERENT COLUMN NAMES
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 3: MERGE ON DIFFERENT COLUMN NAMES")
print("=" * 70)

# ------------------------------------------------------------------------------
# 3.1 left_on and right_on
# ------------------------------------------------------------------------------
print("\n--- 3.1 Using left_on and right_on ---")

# Rename column in one table to simulate different names
orders_renamed = orders.rename(columns={"customer_id": "cust_id"})

print("Orders table has 'cust_id', customers table has 'customer_id'")

# Merge with different column names
merged = pd.merge(
    orders_renamed,
    customers,
    left_on="cust_id",
    right_on="customer_id",
    how="inner"
)

print("\nMerge with left_on='cust_id', right_on='customer_id':")
print(merged[["order_id", "cust_id", "customer_id", "customer_name"]].head(5))

print("\nNote: Both key columns appear in the result")
print("Drop one if redundant:")
merged_clean = merged.drop(columns=["customer_id"])
print(merged_clean[["order_id", "cust_id", "customer_name"]].head(3))

# ------------------------------------------------------------------------------
# 3.2 Merge on Index
# ------------------------------------------------------------------------------
print("\n--- 3.2 Merge on Index ---")

# Set customer_id as index
customers_indexed = customers.set_index("customer_id")
print("Customers with customer_id as index:")
print(customers_indexed.head(3))

# Merge: left table column with right table index
merged_idx = pd.merge(
    orders,
    customers_indexed,
    left_on="customer_id",
    right_index=True,
    how="inner"
)

print("\nMerge with right_index=True:")
print(merged_idx[["order_id", "customer_id", "customer_name"]].head(5))


# ==============================================================================
# SECTION 4: HANDLING DUPLICATE COLUMNS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 4: HANDLING DUPLICATE COLUMNS")
print("=" * 70)

# ------------------------------------------------------------------------------
# 4.1 Automatic Suffixes
# ------------------------------------------------------------------------------
print("\n--- 4.1 Automatic Suffixes ---")

# Both tables have a date column
orders_with_date = orders.copy()
customers_with_date = customers.copy()
customers_with_date = customers_with_date.rename(
    columns={"signup_date": "date"}
)
orders_with_date = orders_with_date.rename(
    columns={"order_date": "date"}
)

print("Both tables have a 'date' column")
print("Orders date: order date")
print("Customers date: signup date")

merged_dup = pd.merge(
    orders_with_date,
    customers_with_date,
    on="customer_id",
    how="inner"
)

print("\nMerge result with default suffixes:")
print(list(merged_dup.columns))
print(merged_dup[["order_id", "customer_id", "date_x", "date_y"]].head(3))

print("\ndate_x = from left (orders), date_y = from right (customers)")

# ------------------------------------------------------------------------------
# 4.2 Custom Suffixes
# ------------------------------------------------------------------------------
print("\n--- 4.2 Custom Suffixes ---")

merged_custom = pd.merge(
    orders_with_date,
    customers_with_date,
    on="customer_id",
    how="inner",
    suffixes=("_order", "_signup")
)

print("With suffixes=('_order', '_signup'):")
print(list(merged_custom.columns))
print(merged_custom[["order_id", "date_order", "date_signup"]].head(3))


# ==============================================================================
# SECTION 5: VALIDATING MERGES
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 5: VALIDATING MERGES")
print("=" * 70)

# ------------------------------------------------------------------------------
# 5.1 The indicator Parameter
# ------------------------------------------------------------------------------
print("\n--- 5.1 The indicator Parameter ---")

explanation = """
indicator=True adds a '_merge' column showing where each row came from:
  'left_only'   Row exists only in left table
  'right_only'  Row exists only in right table
  'both'        Row has match in both tables

This is essential for debugging and validating merge results.
"""
print(explanation)

merged_ind = pd.merge(
    orders,
    customers,
    on="customer_id",
    how="outer",
    indicator=True
)

print("Outer join with indicator=True:")
print(merged_ind[["order_id", "customer_id", "customer_name", "_merge"]])

# Summarize merge results
print("\nMerge summary:")
print(merged_ind["_merge"].value_counts())

# Find unmatched records
left_only = merged_ind[merged_ind["_merge"] == "left_only"]
right_only = merged_ind[merged_ind["_merge"] == "right_only"]

print(f"\nOrders without matching customer: {len(left_only)}")
print(left_only[["order_id", "customer_id"]])

print(f"\nCustomers without any orders: {len(right_only)}")
print(right_only[["customer_id", "customer_name"]])

# ------------------------------------------------------------------------------
# 5.2 The validate Parameter
# ------------------------------------------------------------------------------
print("\n--- 5.2 The validate Parameter ---")

explanation = """
validate parameter checks the expected relationship type:
  'one_to_one'   Each key appears at most once in both tables
  'one_to_many'  Each key appears at most once in left, any times in right
  'many_to_one'  Each key appears any times in left, at most once in right
  'many_to_many' No validation (default behavior)

If validation fails, merge raises a MergeError.
Use this to catch data quality issues early.
"""
print(explanation)

# Valid one-to-many: one customer can have many orders
print("Validating orders (many) to customers (one):")
try:
    validated = pd.merge(
        orders,
        customers,
        on="customer_id",
        how="inner",
        validate="many_to_one"
    )
    print("  Validation passed: many_to_one is correct")
except pd.errors.MergeError as e:
    print(f"  Validation failed: {e}")

# Invalid: customers should be unique, but let us test one_to_one
print("\nTrying to validate as one_to_one (will fail if customer has multiple orders):")
try:
    validated = pd.merge(
        orders,
        customers,
        on="customer_id",
        how="inner",
        validate="one_to_one"
    )
    print("  Validation passed")
except pd.errors.MergeError as e:
    print(f"  Validation failed: {e}")

# ------------------------------------------------------------------------------
# 5.3 Checking Row Count After Merge
# ------------------------------------------------------------------------------
print("\n--- 5.3 Checking Row Count After Merge ---")

def validate_merge_count(left_df, right_df, result_df, how, on):
    """Validate merge result row count based on join type."""
    left_count = len(left_df)
    right_count = len(right_df)
    result_count = len(result_df)

    print(f"  Left rows:   {left_count}")
    print(f"  Right rows:  {right_count}")
    print(f"  Result rows: {result_count}")

    if how == "inner":
        assert result_count <= min(left_count, right_count) or \
               result_count >= 0, "Inner join row count suspicious"
        print("  Inner join: result <= min(left, right) in one-to-one")

    elif how == "left":
        # Left join: result should be >= left if there are multiple matches
        # but at minimum equal to left keys
        assert result_count >= 0, "Left join produced negative rows"
        print("  Left join: preserves all left rows (may increase with duplicates)")

    elif how == "outer":
        assert result_count >= max(left_count, right_count), \
            "Outer join should have at least max(left, right) rows"
        print("  Outer join: result >= max(left, right)")

    return True

print("Validating inner join:")
inner = pd.merge(orders, customers, on="customer_id", how="inner")
validate_merge_count(orders, customers, inner, "inner", "customer_id")


# ==============================================================================
# SECTION 6: MANY-TO-MANY JOINS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 6: MANY-TO-MANY JOINS")
print("=" * 70)

# ------------------------------------------------------------------------------
# 6.1 Understanding Many-to-Many
# ------------------------------------------------------------------------------
print("\n--- 6.1 Understanding Many-to-Many Joins ---")

explanation = """
A many-to-many join occurs when the key appears multiple times
in BOTH tables. The result is a Cartesian product of matching rows.

Example:
  Left table has 3 rows with key='A'
  Right table has 2 rows with key='A'
  Result has 3 x 2 = 6 rows for key='A'

This can cause unexpected row explosion and is often a data modeling error.
"""
print(explanation)

# Create many-to-many example
left_mm = pd.DataFrame({
    "key": ["A", "A", "B"],
    "left_val": [1, 2, 3]
})

right_mm = pd.DataFrame({
    "key": ["A", "A", "B", "B"],
    "right_val": [10, 20, 30, 40]
})

result_mm = pd.merge(left_mm, right_mm, on="key", how="inner")

print("Left table:")
print(left_mm)
print("\nRight table:")
print(right_mm)
print("\nMany-to-many join result:")
print(result_mm)

print(f"\nRow explosion: {len(left_mm)} x {len(right_mm)} combinations = {len(result_mm)} rows")
print("Key 'A': 2 left rows x 2 right rows = 4 result rows")
print("Key 'B': 1 left row x 2 right rows = 2 result rows")

# ------------------------------------------------------------------------------
# 6.2 Detecting Accidental Many-to-Many
# ------------------------------------------------------------------------------
print("\n--- 6.2 Detecting Accidental Many-to-Many ---")

def check_key_uniqueness(df, key_col, table_name):
    """Check if key column has duplicates."""
    dup_count = df[key_col].duplicated().sum()
    is_unique = dup_count == 0

    print(f"  {table_name}['{key_col}']: ", end="")
    if is_unique:
        print("UNIQUE")
    else:
        print(f"HAS DUPLICATES ({dup_count} duplicate rows)")

    return is_unique

print("Checking key uniqueness before merge:")
left_unique = check_key_uniqueness(orders, "customer_id", "orders")
right_unique = check_key_uniqueness(customers, "customer_id", "customers")

if not left_unique and not right_unique:
    print("\nWARNING: Many-to-many join will occur!")
elif not left_unique:
    print("\nJoin type: many-to-one (expected for orders to customers)")
elif not right_unique:
    print("\nJoin type: one-to-many")
else:
    print("\nJoin type: one-to-one")


# ==============================================================================
# SECTION 7: CONCATENATION WITH pd.concat()
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 7: CONCATENATION WITH pd.concat()")
print("=" * 70)

# ------------------------------------------------------------------------------
# 7.1 Vertical Concatenation (Stacking Rows)
# ------------------------------------------------------------------------------
print("\n--- 7.1 Vertical Concatenation (Stacking Rows) ---")

explanation = """
pd.concat() stacks DataFrames vertically (axis=0) or horizontally (axis=1).

Vertical concatenation (axis=0):
  - Combines rows from multiple DataFrames
  - All DataFrames should have the same columns (ideally)
  - Missing columns get NaN values

Use case: Combining monthly data files into an annual dataset.
"""
print(explanation)

print("January sales:")
print(jan_sales)
print("\nFebruary sales:")
print(feb_sales)
print("\nMarch sales:")
print(mar_sales)

# Concatenate vertically
quarterly_sales = pd.concat(
    [jan_sales, feb_sales, mar_sales],
    axis=0,
    ignore_index=True
)

print("\nConcatenated quarterly sales (ignore_index=True):")
print(quarterly_sales)

# Note: ignore_index=True resets the index to 0, 1, 2, ...
# Without it, original indices are preserved (may cause duplicates)

quarterly_sales_keep_idx = pd.concat(
    [jan_sales, feb_sales, mar_sales],
    axis=0,
    ignore_index=False
)

print("\nWithout ignore_index (original indices preserved):")
print(quarterly_sales_keep_idx)
print(f"Index values: {quarterly_sales_keep_idx.index.tolist()}")

# ------------------------------------------------------------------------------
# 7.2 Adding Keys to Track Source
# ------------------------------------------------------------------------------
print("\n--- 7.2 Adding Keys to Track Source ---")

quarterly_with_keys = pd.concat(
    [jan_sales, feb_sales, mar_sales],
    axis=0,
    keys=["jan", "feb", "mar"]
)

print("Concatenated with keys (creates MultiIndex):")
print(quarterly_with_keys)
print(f"\nIndex type: {type(quarterly_with_keys.index)}")

# Access specific month
print("\nAccess February data with .loc['feb']:")
print(quarterly_with_keys.loc["feb"])

# Flatten the MultiIndex
quarterly_flat = quarterly_with_keys.reset_index(level=0)
quarterly_flat = quarterly_flat.rename(columns={"level_0": "source_month"})
print("\nFlattened with source column:")
print(quarterly_flat)

# ------------------------------------------------------------------------------
# 7.3 Handling Mismatched Columns
# ------------------------------------------------------------------------------
print("\n--- 7.3 Handling Mismatched Columns ---")

# Create tables with different columns
df_a = pd.DataFrame({
    "id": [1, 2],
    "value": [100, 200],
    "extra_a": ["x", "y"]
})

df_b = pd.DataFrame({
    "id": [3, 4],
    "value": [300, 400],
    "extra_b": ["z", "w"]
})

print("DataFrame A:")
print(df_a)
print("\nDataFrame B:")
print(df_b)

# Default: outer join on columns (all columns included)
concat_outer = pd.concat([df_a, df_b], axis=0, ignore_index=True)
print("\nConcat with join='outer' (default):")
print(concat_outer)
print("Missing columns filled with NaN")

# Inner join: only common columns
concat_inner = pd.concat([df_a, df_b], axis=0, ignore_index=True, join="inner")
print("\nConcat with join='inner':")
print(concat_inner)
print("Only common columns kept")

# ------------------------------------------------------------------------------
# 7.4 Horizontal Concatenation (Side by Side)
# ------------------------------------------------------------------------------
print("\n--- 7.4 Horizontal Concatenation (Side by Side) ---")

# Create tables to join side by side
demographics = pd.DataFrame({
    "age": [25, 30, 35],
    "gender": ["F", "M", "F"]
}, index=[0, 1, 2])

scores = pd.DataFrame({
    "math": [90, 85, 95],
    "science": [88, 92, 90]
}, index=[0, 1, 2])

print("Demographics:")
print(demographics)
print("\nScores:")
print(scores)

# Horizontal concat (axis=1)
combined = pd.concat([demographics, scores], axis=1)
print("\nHorizontal concatenation (axis=1):")
print(combined)

# Index alignment matters!
scores_misaligned = pd.DataFrame({
    "math": [90, 85, 95],
    "science": [88, 92, 90]
}, index=[0, 1, 3])  # Note: index 3 instead of 2

combined_misaligned = pd.concat([demographics, scores_misaligned], axis=1)
print("\nWith misaligned indices:")
print(combined_misaligned)
print("NaN appears where indices do not match")


# ==============================================================================
# SECTION 8: JOIN() METHOD
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 8: JOIN() METHOD")
print("=" * 70)

explanation = """
df.join() is a convenience method for merging on indices.
It is equivalent to pd.merge() with left_index=True, right_index=True.

Syntax: left_df.join(right_df, how='left')

Default is left join.
The right DataFrame's index is used to match the left DataFrame's index.
"""
print(explanation)

# Create indexed DataFrames
customers_idx = customers.set_index("customer_id")[["customer_name", "region_code"]]
orders_idx = orders.set_index("customer_id")[["order_id", "quantity"]]

print("Customers (indexed by customer_id):")
print(customers_idx.head())

print("\nOrders (indexed by customer_id):")
print(orders_idx.head())

# Join on index
joined = customers_idx.join(orders_idx, how="inner")
print("\nInner join using .join():")
print(joined)

# With left join (default)
joined_left = customers_idx.join(orders_idx, how="left")
print("\nLeft join (customers without orders get NaN):")
print(joined_left)

# lsuffix and rsuffix for overlapping columns
# Create overlap
orders_with_name = orders_idx.copy()
orders_with_name["name"] = "Order Name"
customers_with_name = customers_idx.copy()
customers_with_name["name"] = "Customer Name"

joined_overlap = customers_with_name.join(
    orders_with_name,
    how="inner",
    lsuffix="_cust",
    rsuffix="_order"
)
print("\nJoin with overlapping 'name' column:")
print(joined_overlap.head())


# ==============================================================================
# SECTION 9: MULTI-TABLE JOINS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 9: MULTI-TABLE JOINS")
print("=" * 70)

print("Joining orders + customers + products + regions")

# Step 1: Orders + Customers
step1 = pd.merge(
    orders,
    customers,
    on="customer_id",
    how="left",
    suffixes=("", "_cust")
)
print(f"\nStep 1: Orders + Customers = {len(step1)} rows")

# Step 2: Add Products
step2 = pd.merge(
    step1,
    products,
    on="product_id",
    how="left"
)
print(f"Step 2: + Products = {len(step2)} rows")

# Step 3: Add Regions
step3 = pd.merge(
    step2,
    regions,
    on="region_code",
    how="left"
)
print(f"Step 3: + Regions = {len(step3)} rows")

# Final result
print("\nFinal merged table:")
final_cols = [
    "order_id", "customer_name", "product_name",
    "quantity", "unit_price", "region_name"
]
print(step3[final_cols])

# Calculate total revenue
step3["total_price"] = step3["quantity"] * step3["unit_price"]
step3["profit"] = step3["total_price"] - (step3["quantity"] * step3["cost"])

print("\nWith calculated columns:")
print(step3[["order_id", "product_name", "quantity",
             "total_price", "cost", "profit"]])


# ==============================================================================
# SECTION 10: COMMON MERGE PROBLEMS AND SOLUTIONS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 10: COMMON MERGE PROBLEMS AND SOLUTIONS")
print("=" * 70)

# ------------------------------------------------------------------------------
# 10.1 Problem: More Rows Than Expected After Merge
# ------------------------------------------------------------------------------
print("\n--- 10.1 Problem: More Rows Than Expected ---")

print("Symptom: Merged table has more rows than the left table")
print("Cause: Duplicate keys in the right table (many-to-many join)")
print("\nSolution 1: Check for duplicates before merge")

def diagnose_row_explosion(left, right, key):
    """Diagnose potential row explosion from duplicates."""
    left_dups = left[key].duplicated().sum()
    right_dups = right[key].duplicated().sum()

    print(f"  Left duplicates on '{key}': {left_dups}")
    print(f"  Right duplicates on '{key}': {right_dups}")

    if left_dups > 0 and right_dups > 0:
        print("  WARNING: Many-to-many join will cause row explosion!")
        return False
    return True

print("\nDiagnosing orders + customers merge:")
diagnose_row_explosion(orders, customers, "customer_id")

# ------------------------------------------------------------------------------
# 10.2 Problem: Missing Data After Merge
# ------------------------------------------------------------------------------
print("\n--- 10.2 Problem: Missing Data After Merge ---")

print("Symptom: Fewer rows than expected or unexpected NaN values")
print("Cause: Key values do not match between tables")
print("\nSolution: Use indicator and analyze unmatched rows")

# Find orders that will not match
unmatched_orders = orders[~orders["customer_id"].isin(customers["customer_id"])]
print(f"\nOrders that will not match: {len(unmatched_orders)}")
print(unmatched_orders[["order_id", "customer_id"]])

# Find customers that will not match
unmatched_customers = customers[~customers["customer_id"].isin(orders["customer_id"])]
print(f"\nCustomers that will not match: {len(unmatched_customers)}")
print(unmatched_customers[["customer_id", "customer_name"]])

# ------------------------------------------------------------------------------
# 10.3 Problem: Key Type Mismatch
# ------------------------------------------------------------------------------
print("\n--- 10.3 Problem: Key Type Mismatch ---")

print("Symptom: Merge returns no matches when matches should exist")
print("Cause: Key column has different dtype in each table (int vs string)")

# Create example
left_str = pd.DataFrame({
    "id": ["1", "2", "3"],  # String
    "value": [10, 20, 30]
})

right_int = pd.DataFrame({
    "id": [1, 2, 3],  # Integer
    "name": ["A", "B", "C"]
})

print(f"\nLeft 'id' dtype: {left_str['id'].dtype}")
print(f"Right 'id' dtype: {right_int['id'].dtype}")

# Merge with type mismatch (newer pandas raises ValueError instead of returning 0 rows)
try:
    merged_mismatch = pd.merge(left_str, right_int, on="id", how="inner")
    print(f"\nMerge result: {len(merged_mismatch)} rows (should be 3!)")
except ValueError as e:
    print(f"\nMerge raised ValueError: {e}")
    print("(Newer pandas enforces dtype compatibility — 0 matches would result)")

# Solution: convert types before merge
left_str["id"] = left_str["id"].astype(int)
merged_fixed = pd.merge(left_str, right_int, on="id", how="inner")
print(f"After type fix: {len(merged_fixed)} rows")
print(merged_fixed)

# ------------------------------------------------------------------------------
# 10.4 Problem: Duplicate Column Names
# ------------------------------------------------------------------------------
print("\n--- 10.4 Problem: Duplicate Column Names ---")

print("Symptom: Columns named 'value_x' and 'value_y' appear")
print("Cause: Both tables have columns with the same name (not keys)")
print("\nSolution: Use suffixes parameter or rename before merge")

# Already demonstrated in Section 4


# ==============================================================================
# SECTION 11: REAL WORLD PIPELINE - ORDER ANALYSIS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 11: REAL WORLD PIPELINE - ORDER ANALYSIS")
print("=" * 70)

def build_order_analysis_table(orders_df, customers_df, products_df, regions_df):
    """
    Build a complete order analysis table from multiple sources.

    Returns a denormalized table suitable for reporting and analysis.
    """
    print("Building order analysis table...")

    # Step 1: Validate inputs
    print("  Step 1: Validating inputs")
    assert "customer_id" in orders_df.columns, "Missing customer_id in orders"
    assert "product_id" in orders_df.columns, "Missing product_id in orders"
    assert "customer_id" in customers_df.columns, "Missing customer_id in customers"
    assert "product_id" in products_df.columns, "Missing product_id in products"

    # Step 2: Check for key issues
    print("  Step 2: Checking key relationships")
    orders_without_customer = ~orders_df["customer_id"].isin(customers_df["customer_id"])
    orders_without_product = ~orders_df["product_id"].isin(products_df["product_id"])

    if orders_without_customer.any():
        print(f"    WARNING: {orders_without_customer.sum()} orders have unknown customers")
    if orders_without_product.any():
        print(f"    WARNING: {orders_without_product.sum()} orders have unknown products")

    # Step 3: Merge orders with customers
    print("  Step 3: Merging orders with customers")
    result = pd.merge(
        orders_df,
        customers_df,
        on="customer_id",
        how="left",
        validate="many_to_one"
    )

    # Step 4: Merge with products
    print("  Step 4: Merging with products")
    result = pd.merge(
        result,
        products_df,
        on="product_id",
        how="left",
        validate="many_to_one"
    )

    # Step 5: Merge with regions
    print("  Step 5: Merging with regions")
    result = pd.merge(
        result,
        regions_df,
        on="region_code",
        how="left",
        validate="many_to_one"
    )

    # Step 6: Calculate metrics
    print("  Step 6: Calculating metrics")
    result["total_price"] = result["quantity"] * result["unit_price"]
    result["total_cost"] = result["quantity"] * result["cost"]
    result["profit"] = result["total_price"] - result["total_cost"]
    result["profit_margin"] = (result["profit"] / result["total_price"] * 100).round(2)

    # Step 7: Validate output
    print("  Step 7: Validating output")
    assert len(result) == len(orders_df), "Row count changed unexpectedly"
    assert result["total_price"].isnull().sum() == 0, "Missing total_price"

    print(f"  Complete: {len(result)} rows, {len(result.columns)} columns")

    return result


# Build the analysis table
analysis_table = build_order_analysis_table(orders, customers, products, regions)

print("\nFinal Order Analysis Table:")
display_cols = [
    "order_id", "customer_name", "product_name", "category",
    "quantity", "total_price", "profit", "profit_margin", "region_name"
]
print(analysis_table[display_cols])

# Generate summary reports
print("\n" + "-" * 50)
print("SUMMARY REPORTS FROM MERGED DATA")
print("-" * 50)

# Revenue by region
print("\nRevenue by Region:")
by_region = analysis_table.groupby("region_name").agg(
    total_revenue = ("total_price", "sum"),
    total_profit  = ("profit", "sum"),
    order_count   = ("order_id", "count")
).round(2)
print(by_region)

# Revenue by product category
print("\nRevenue by Category:")
by_category = analysis_table.groupby("category").agg(
    total_revenue  = ("total_price", "sum"),
    avg_margin     = ("profit_margin", "mean"),
    units_sold     = ("quantity", "sum")
).round(2)
print(by_category)

# Top customers
print("\nTop Customers by Revenue:")
by_customer = analysis_table.groupby("customer_name").agg(
    total_spent = ("total_price", "sum"),
    orders      = ("order_id", "count")
).sort_values("total_spent", ascending=False).round(2)
print(by_customer.head(5))


# ==============================================================================
# SECTION 12: PERFORMANCE CONSIDERATIONS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 12: PERFORMANCE CONSIDERATIONS")
print("=" * 70)

performance_notes = """
MERGE PERFORMANCE:
  1. Merge is O(n + m) for sorted keys, O(n log n) otherwise
  2. Index-based joins are faster than column-based joins
  3. Smaller table should generally be on the right in left join

OPTIMIZATION TIPS:
  1. Sort both DataFrames by the key before merging large datasets
  2. Use categorical dtype for string keys to reduce memory
  3. Filter rows before merging to reduce input size
  4. Use validate parameter to catch data issues early
  5. For very large datasets, consider chunked processing

CONCAT PERFORMANCE:
  1. Concatenating many small DataFrames is slow
  2. Collect DataFrames in a list, then concat once at the end
  3. Bad:  for df in dfs: result = pd.concat([result, df])
  4. Good: result = pd.concat(dfs)  # All at once

MEMORY TIPS:
  1. Drop unnecessary columns before merging
  2. Use appropriate dtypes (int32 vs int64, category vs object)
  3. For huge datasets, consider dask or vaex for out-of-core processing
"""
print(performance_notes)


# ==============================================================================
# SECTION 13: COMMON PITFALLS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 13: COMMON PITFALLS")
print("=" * 70)

pitfalls = """
Pitfall 1: Accidental many-to-many join causing row explosion
   - Always check key uniqueness before merge
   - Use validate parameter to enforce expected relationship

Pitfall 2: Key dtype mismatch (int vs string)
   - Merge silently returns no matches
   - Check dtypes before merge: df["col"].dtype

Pitfall 3: Forgetting that merge does not modify in place
   - result = pd.merge(a, b, ...) is required
   - a.merge(b) also returns a new DataFrame

Pitfall 4: Losing rows in inner join unexpectedly
   - Use indicator=True to see what gets dropped
   - Consider left/outer join if all rows matter

Pitfall 5: Duplicate column names causing confusion
   - Use explicit suffixes parameter
   - Or rename columns before merge

Pitfall 6: Wrong join type for the use case
   - left join when you want all customers, even without orders
   - inner join when you only want matched records

Pitfall 7: Concatenating DataFrames with mismatched columns
   - Use join='inner' to keep only common columns
   - Or align columns before concatenation

Pitfall 8: Index issues after concatenation
   - Use ignore_index=True to reset to 0, 1, 2, ...
   - Or use keys parameter to create hierarchical index

Pitfall 9: Assuming merge order does not matter
   - left vs right join are not symmetric
   - Table order matters for join type and suffix application
"""
print(pitfalls)


# ==============================================================================
# SECTION 14: SUMMARY TABLE
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 14: SUMMARY TABLE")
print("=" * 70)

summary = """
Operation                              | Syntax
---------------------------------------|---------------------------------------------
Inner join on column                   | pd.merge(a, b, on="key", how="inner")
Left join                              | pd.merge(a, b, on="key", how="left")
Right join                             | pd.merge(a, b, on="key", how="right")
Outer join                             | pd.merge(a, b, on="key", how="outer")
Cross join                             | pd.merge(a, b, how="cross")
Join on different column names         | pd.merge(a, b, left_on="x", right_on="y")
Join on index                          | pd.merge(a, b, left_index=True, right_on="y")
Join with indicator                    | pd.merge(a, b, on="key", indicator=True)
Validate join relationship             | pd.merge(a, b, on="key", validate="many_to_one")
Custom suffixes                        | pd.merge(a, b, on="key", suffixes=("_l","_r"))
Vertical concat (stack rows)           | pd.concat([a, b], axis=0, ignore_index=True)
Horizontal concat (side by side)       | pd.concat([a, b], axis=1)
Concat with source tracking            | pd.concat([a, b], keys=["src_a", "src_b"])
Concat only common columns             | pd.concat([a, b], join="inner")
Index-based join                       | a.join(b, how="left")
Index join with suffix                 | a.join(b, lsuffix="_l", rsuffix="_r")
"""
print(summary)


# ==============================================================================
# SECTION 15: PRACTICE EXERCISES
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 15: PRACTICE EXERCISES")
print("=" * 70)

print("Exercise 1: Find customers who have never placed an order")
ex1 = pd.merge(
    customers,
    orders[["customer_id"]].drop_duplicates(),
    on="customer_id",
    how="left",
    indicator=True
)
no_orders = ex1[ex1["_merge"] == "left_only"]
print("  Customers with no orders:")
print(no_orders[["customer_id", "customer_name"]].to_string(index=False))

print("\nExercise 2: Calculate total quantity ordered per product")
ex2 = pd.merge(orders, products, on="product_id", how="inner")
qty_by_product = ex2.groupby("product_name")["quantity"].sum().reset_index()
print("  Total quantity per product:")
print(qty_by_product.to_string(index=False))

print("\nExercise 3: Concatenate monthly sales and find total by product")
all_sales = pd.concat([jan_sales, feb_sales, mar_sales], ignore_index=True)
total_by_product = all_sales.groupby("product_id")["total_sales"].sum().reset_index()
print("  Total sales across all months:")
print(total_by_product.to_string(index=False))

print("\nExercise 4: Join orders with products and find orders with unknown products")
ex4 = pd.merge(
    orders,
    products,
    on="product_id",
    how="left",
    indicator=True
)
unknown_products = ex4[ex4["_merge"] == "left_only"]
print(f"  Orders with unknown product_id: {len(unknown_products)}")
if len(unknown_products) > 0:
    print(unknown_products[["order_id", "product_id"]])
else:
    print("  All orders have valid product_id")


# ==============================================================================
# SECTION 16: KEY TAKEAWAYS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 16: KEY TAKEAWAYS")
print("=" * 70)

takeaways = """
1.  Use pd.merge() for database-style joins on columns
2.  Default join is inner - use how= to specify left/right/outer
3.  Always check key uniqueness to avoid many-to-many explosions
4.  Use indicator=True to debug and validate merge results
5.  Use validate= to enforce expected relationship type
6.  Handle duplicate column names with suffixes parameter
7.  Use pd.concat() for stacking rows or columns
8.  Use ignore_index=True in concat to reset row numbers
9.  Use keys= in concat to track source of each row
10. Use .join() for quick index-based merges
11. Check dtypes match before merging to avoid silent failures
12. Filter and reduce data before merging large datasets
13. Build multi-table joins step by step with validation
14. Always validate row counts after merge operations
"""
print(takeaways)


# ==============================================================================
# NEXT LESSON PREVIEW
# ==============================================================================
print("\n" + "=" * 70)
print("NEXT LESSON PREVIEW")
print("=" * 70)

preview = """
Lesson 26: Working with Dates and Time Series

You will learn:
- Creating and parsing datetime columns
- Extracting date components (year, month, day, weekday)
- Date arithmetic and time deltas
- Resampling time series data (daily to monthly, etc.)
- Rolling windows and moving averages
- Handling time zones
- Shifting and lagging time series data
- Real world time series analysis examples
"""
print(preview)

print("\n" + "=" * 70)
print("END OF LESSON 25")
print("=" * 70)
print("\nYou can now merge DataFrames with confidence, validate join results,")
print("concatenate tables, and build multi-table analysis pipelines.")