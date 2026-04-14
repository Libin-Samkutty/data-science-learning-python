import numpy as np

# ============================================================
# PART 1: Basic Sorting
# ============================================================

print("="*60)
print("PART 1: BASIC SORTING")
print("="*60)

# Simple array sorting
scores = np.array([85, 92, 78, 95, 88, 72, 90, 83])
print(f"Original scores: {scores}")

# Sort ascending (default)
sorted_scores = np.sort(scores)
print(f"\nSorted (ascending): {sorted_scores}")

# Sort descending
sorted_desc = np.sort(scores)[::-1]
print(f"Sorted (descending): {sorted_desc}")

# Important: np.sort() returns a COPY
print(f"\nOriginal unchanged: {scores}")

# In-place sorting with .sort() method
scores_copy = scores.copy()
scores_copy.sort()  # Modifies in-place
print(f"After in-place sort: {scores_copy}")

# Sorting different data types
print("\n--- Sorting Different Types ---")
names = np.array(['Charlie', 'Alice', 'Bob', 'Diana'])
print(f"Original names: {names}")
print(f"Sorted names: {np.sort(names)}")

floats = np.array([3.14, 1.41, 2.72, 1.62])
print(f"\nOriginal floats: {floats}")
print(f"Sorted floats: {np.sort(floats)}")

# ============================================================
# PART 2: Sorting Multi-Dimensional Arrays
# ============================================================

print("\n" + "="*60)
print("PART 2: SORTING MULTI-DIMENSIONAL ARRAYS")
print("="*60)

# 2D array: 4 students × 3 subjects
grades = np.array([
    [85, 90, 78],  # Student 0
    [92, 88, 95],  # Student 1
    [78, 82, 80],  # Student 2
    [88, 85, 87]   # Student 3
])

print("Original grades (4 students × 3 subjects):")
print(grades)

# Sort along axis=1 (within each row - each student's grades)
print("\n--- Sort Each Row (axis=1) ---")
sorted_rows = np.sort(grades, axis=1)
print("Each student's grades sorted:")
print(sorted_rows)

# Sort along axis=0 (within each column - each subject's grades)
print("\n--- Sort Each Column (axis=0) ---")
sorted_cols = np.sort(grades, axis=0)
print("Each subject's grades sorted:")
print(sorted_cols)

# Sort flattened (axis=None)
print("\n--- Sort Flattened ---")
sorted_flat = np.sort(grades, axis=None)
print(f"All grades sorted: {sorted_flat}")

# ============================================================
# PART 3: Indirect Sorting (argsort)
# ============================================================

print("\n" + "="*60)
print("PART 3: INDIRECT SORTING (argsort)")
print("="*60)

print("""
np.argsort() returns the INDICES that would sort the array.
This is extremely useful when you need to:
- Sort one array based on another
- Keep track of original positions
- Create rankings
""")

scores = np.array([85, 92, 78, 95, 88])
students = np.array(['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'])

print(f"Scores: {scores}")
print(f"Students: {students}")

# Get sorting indices
sort_indices = np.argsort(scores)
print(f"\nSort indices (ascending): {sort_indices}")
print("Explanation: index 2 (78) is smallest, then 0 (85), etc.")

# Use indices to sort both arrays together
print("\n--- Sorting Related Arrays Together ---")
sorted_scores = scores[sort_indices]
sorted_students = students[sort_indices]

print("Sorted by score (ascending):")
for student, score in zip(sorted_students, sorted_scores):
    print(f"  {student}: {score}")

# Descending order
desc_indices = np.argsort(scores)[::-1]
print("\nTop performers (descending):")
for i, idx in enumerate(desc_indices[:3], 1):
    print(f"  {i}. {students[idx]}: {scores[idx]}")

# ============================================================
# PART 4: Ranking Data
# ============================================================

print("\n" + "="*60)
print("PART 4: RANKING DATA")
print("="*60)

scores = np.array([85, 92, 78, 95, 88, 78])
print(f"Scores: {scores}")

# Method 1: Simple ranking using argsort twice
print("\n--- Method 1: argsort Twice ---")
# argsort gives positions; argsort again gives ranks
temp = np.argsort(scores)  # [2, 5, 0, 4, 1, 3]
ranks_asc = np.argsort(temp) + 1  # +1 for 1-based ranking
print(f"Ranks (ascending, 1=lowest): {ranks_asc}")

# Descending rank (1 = highest)
temp = np.argsort(-scores)
ranks_desc = np.argsort(temp) + 1
print(f"Ranks (descending, 1=highest): {ranks_desc}")

# Method 2: Using searchsorted for ranking
print("\n--- Method 2: Percentile Ranking ---")
sorted_scores = np.sort(scores)
percentile_rank = np.searchsorted(sorted_scores, scores) / len(scores) * 100
print(f"Percentile ranks: {percentile_rank.round(1)}")

# Display rankings
print("\n--- Score Rankings ---")
students = ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank']
for student, score, rank in zip(students, scores, ranks_desc):
    print(f"  {student}: Score={score}, Rank={rank}")

# ============================================================
# PART 5: Partial Sorting (partition)
# ============================================================

print("\n" + "="*60)
print("PART 5: PARTIAL SORTING (partition)")
print("="*60)

print("""
np.partition() is FASTER than full sorting when you only need:
- Top N or bottom N elements
- The element at a specific position if sorted

It rearranges elements so that:
- Elements smaller than kth are to the left
- Elements larger than kth are to the right
- Element at position k is correct
""")

np.random.seed(42)
data = np.random.randint(1, 100, 15)
print(f"Original data: {data}")

# Partition around the 3rd smallest (index 2)
k = 2
partitioned = np.partition(data, k)
print(f"\nPartitioned at k={k}:")
print(f"Result: {partitioned}")
print(f"Element at position {k}: {partitioned[k]} (would be 3rd smallest if sorted)")
print(f"Elements to left (smaller): {partitioned[:k]}")
print(f"Elements to right (larger): {partitioned[k+1:]}")

# Verify with full sort
print(f"\nFully sorted: {np.sort(data)}")
print(f"3rd smallest: {np.sort(data)[k]}")

# Real-world: Get top 5 scores efficiently
print("\n--- Real-World: Top 5 Scores ---")
scores = np.random.randint(50, 100, 1000)
print(f"Processing {len(scores)} scores...")

# Method 1: Full sort (slower for large arrays)
import time
start = time.time()
top5_sort = np.sort(scores)[-5:][::-1]
sort_time = time.time() - start

# Method 2: Partition (faster)
start = time.time()
# Partition to get top 5 at the end
partitioned = np.partition(scores, -5)
top5_partition = np.sort(partitioned[-5:])[::-1]
partition_time = time.time() - start

print(f"Top 5 (full sort): {top5_sort}")
print(f"Top 5 (partition): {top5_partition}")
print(f"Results match: {np.array_equal(top5_sort, top5_partition)}")

# argpartition - get indices of top N
print("\n--- argpartition: Get Indices of Top N ---")
top5_indices = np.argpartition(scores, -5)[-5:]
top5_indices_sorted = top5_indices[np.argsort(scores[top5_indices])[::-1]]
print(f"Indices of top 5 scores: {top5_indices_sorted}")
print(f"Top 5 scores: {scores[top5_indices_sorted]}")

# ============================================================
# PART 6: Searching - Finding Elements
# ============================================================

print("\n" + "="*60)
print("PART 6: SEARCHING - FINDING ELEMENTS")
print("="*60)

data = np.array([10, 25, 30, 15, 45, 30, 20, 35, 30])
print(f"Data: {data}")

# np.where() - Find indices where condition is true
print("\n--- np.where() - Conditional Search ---")
indices = np.where(data == 30)
print(f"Indices where value is 30: {indices[0]}")
print(f"Values at those indices: {data[indices]}")

indices_gt_25 = np.where(data > 25)
print(f"Indices where value > 25: {indices_gt_25[0]}")
print(f"Values > 25: {data[indices_gt_25]}")

# np.argmax() and np.argmin()
print("\n--- argmax/argmin - Find Extremes ---")
print(f"Index of maximum: {np.argmax(data)} (value: {data[np.argmax(data)]})")
print(f"Index of minimum: {np.argmin(data)} (value: {data[np.argmin(data)]})")

# For 2D arrays
matrix = np.array([[10, 25, 30], [15, 45, 20], [35, 30, 40]])
print(f"\nMatrix:\n{matrix}")
print(f"Index of max (flattened): {np.argmax(matrix)}")
print(f"Index of max per column: {np.argmax(matrix, axis=0)}")
print(f"Index of max per row: {np.argmax(matrix, axis=1)}")

# Unravel index to get row, col from flattened index
flat_idx = np.argmax(matrix)
row_idx, col_idx = np.unravel_index(flat_idx, matrix.shape)
print(f"Max value {matrix.max()} is at row {row_idx}, col {col_idx}")

# np.nonzero() - Find non-zero elements
print("\n--- np.nonzero() - Find Non-Zero Elements ---")
sparse = np.array([0, 3, 0, 0, 7, 2, 0, 1])
print(f"Array: {sparse}")
nonzero_idx = np.nonzero(sparse)
print(f"Non-zero indices: {nonzero_idx[0]}")
print(f"Non-zero values: {sparse[nonzero_idx]}")

# ============================================================
# PART 7: Binary Search (searchsorted)
# ============================================================

print("\n" + "="*60)
print("PART 7: BINARY SEARCH (searchsorted)")
print("="*60)

print("""
np.searchsorted() finds insertion points in a SORTED array.
It uses binary search, so it's very efficient: O(log n)

Returns the index where values should be inserted to maintain order.
""")

# Sorted array
sorted_arr = np.array([10, 20, 30, 40, 50, 60, 70])
print(f"Sorted array: {sorted_arr}")

# Find insertion point for single value
value = 35
idx = np.searchsorted(sorted_arr, value)
print(f"\nInsertion point for {value}: index {idx}")
print(f"  Values before: {sorted_arr[:idx]}")
print(f"  Values after: {sorted_arr[idx:]}")

# Multiple values at once
values = np.array([15, 35, 55, 75])
indices = np.searchsorted(sorted_arr, values)
print(f"\nInsertion points for {values}: {indices}")

# 'left' vs 'right' side
sorted_with_dups = np.array([10, 20, 20, 20, 30, 40])
print(f"\nArray with duplicates: {sorted_with_dups}")
print(f"searchsorted(20, side='left'): {np.searchsorted(sorted_with_dups, 20, side='left')}")
print(f"searchsorted(20, side='right'): {np.searchsorted(sorted_with_dups, 20, side='right')}")

# Real-world: Binning data
print("\n--- Real-World: Binning/Bucketing Data ---")
ages = np.array([5, 15, 25, 35, 45, 55, 65, 75, 85])
bins = np.array([0, 18, 35, 50, 65, 100])
labels = ['Child', 'Young Adult', 'Adult', 'Middle Age', 'Senior']

bin_indices = np.searchsorted(bins, ages, side='right') - 1
bin_indices = np.clip(bin_indices, 0, len(labels) - 1)

print(f"Ages: {ages}")
print(f"Bins: {bins}")
print(f"Bin indices: {bin_indices}")
print("\nAge groups:")
for age, idx in zip(ages, bin_indices):
    print(f"  Age {age}: {labels[idx]}")

# ============================================================
# PART 8: Unique Values and Counts
# ============================================================

print("\n" + "="*60)
print("PART 8: UNIQUE VALUES AND COUNTS")
print("="*60)

# Basic unique
data = np.array([3, 1, 2, 3, 1, 4, 2, 3, 1, 5])
print(f"Data: {data}")

unique_vals = np.unique(data)
print(f"\nUnique values: {unique_vals}")

# Unique with counts
unique_vals, counts = np.unique(data, return_counts=True)
print(f"\n--- Unique with Counts ---")
for val, count in zip(unique_vals, counts):
    print(f"  Value {val}: appears {count} times")

# Unique with indices
print("\n--- Unique with Indices ---")
unique_vals, first_indices = np.unique(data, return_index=True)
print(f"First occurrence indices: {first_indices}")
for val, idx in zip(unique_vals, first_indices):
    print(f"  Value {val}: first appears at index {idx}")

# Inverse indices (reconstruct original from unique)
unique_vals, inverse = np.unique(data, return_inverse=True)
print(f"\nInverse indices: {inverse}")
print(f"Reconstructed: {unique_vals[inverse]}")
print(f"Matches original: {np.array_equal(data, unique_vals[inverse])}")

# Real-world: Categorical data analysis
print("\n--- Real-World: Survey Responses ---")
responses = np.array(['Yes', 'No', 'Maybe', 'Yes', 'Yes', 'No', 
                      'Maybe', 'Yes', 'No', 'Yes', 'No', 'No'])
print(f"Responses: {responses}")

values, counts = np.unique(responses, return_counts=True)
total = len(responses)

print("\nResponse Distribution:")
for val, count in zip(values, counts):
    pct = count / total * 100
    bar = '█' * int(pct / 5)
    print(f"  {val:6}: {count:2} ({pct:5.1f}%) {bar}")

# ============================================================
# PART 9: Set Operations
# ============================================================

print("\n" + "="*60)
print("PART 9: SET OPERATIONS")
print("="*60)

# Two arrays
A = np.array([1, 2, 3, 4, 5])
B = np.array([3, 4, 5, 6, 7])

print(f"Set A: {A}")
print(f"Set B: {B}")

# Union (all unique elements from both)
union = np.union1d(A, B)
print(f"\nUnion (A ∪ B): {union}")

# Intersection (elements in both)
intersection = np.intersect1d(A, B)
print(f"Intersection (A ∩ B): {intersection}")

# Difference (elements in A but not in B)
diff_A = np.setdiff1d(A, B)
print(f"Difference (A - B): {diff_A}")

diff_B = np.setdiff1d(B, A)
print(f"Difference (B - A): {diff_B}")

# Symmetric difference (elements in either but not both)
sym_diff = np.setxor1d(A, B)
print(f"Symmetric difference (A △ B): {sym_diff}")

# Check membership
print(f"\n--- Membership Testing ---")
test_values = np.array([1, 3, 6, 9])
print(f"Test values: {test_values}")
print(f"In A: {np.isin(test_values, A)}")
print(f"In B: {np.isin(test_values, B)}")

# Real-world: Customer analysis
print("\n--- Real-World: Customer Analysis ---")
customers_jan = np.array([101, 102, 103, 104, 105, 106])
customers_feb = np.array([103, 104, 105, 107, 108, 109])

print(f"January customers: {customers_jan}")
print(f"February customers: {customers_feb}")

retained = np.intersect1d(customers_jan, customers_feb)
churned = np.setdiff1d(customers_jan, customers_feb)
new_customers = np.setdiff1d(customers_feb, customers_jan)
all_customers = np.union1d(customers_jan, customers_feb)

print(f"\nRetained customers: {retained} ({len(retained)} customers)")
print(f"Churned customers: {churned} ({len(churned)} customers)")
print(f"New customers: {new_customers} ({len(new_customers)} customers)")
print(f"Total unique customers: {all_customers} ({len(all_customers)} customers)")

retention_rate = len(retained) / len(customers_jan) * 100
churn_rate = len(churned) / len(customers_jan) * 100
print(f"\nRetention rate: {retention_rate:.1f}%")
print(f"Churn rate: {churn_rate:.1f}%")

# ============================================================
# PART 10: Real-World Example - Student Leaderboard
# ============================================================

print("\n" + "="*60)
print("PART 10: REAL-WORLD - STUDENT LEADERBOARD")
print("="*60)

# Simulated student data
np.random.seed(42)
n_students = 20

student_ids = np.arange(1001, 1001 + n_students)
names = np.array([f"Student_{i}" for i in range(1, n_students + 1)])
math_scores = np.random.randint(50, 100, n_students)
science_scores = np.random.randint(50, 100, n_students)
english_scores = np.random.randint(50, 100, n_students)

# Calculate total and average
total_scores = math_scores + science_scores + english_scores
avg_scores = total_scores / 3

print("Student Scores Summary:")
print(f"{'ID':<6} {'Name':<12} {'Math':<6} {'Science':<8} {'English':<8} {'Total':<6} {'Avg':<6}")
print("-" * 60)
for i in range(5):  # Show first 5
    print(f"{student_ids[i]:<6} {names[i]:<12} {math_scores[i]:<6} {science_scores[i]:<8} "
          f"{english_scores[i]:<8} {total_scores[i]:<6} {avg_scores[i]:<6.1f}")
print("... (showing 5 of 20 students)")

# Create leaderboard (sorted by total score)
print("\n--- Top 5 Leaderboard ---")
rank_indices = np.argsort(total_scores)[::-1]  # Descending

print(f"{'Rank':<6} {'Name':<12} {'Total':<8} {'Average':<8}")
print("-" * 40)
for rank, idx in enumerate(rank_indices[:5], 1):
    print(f"{rank:<6} {names[idx]:<12} {total_scores[idx]:<8} {avg_scores[idx]:<8.1f}")

# Subject-wise toppers
print("\n--- Subject Toppers ---")
subjects = ['Math', 'Science', 'English']
scores_by_subject = [math_scores, science_scores, english_scores]

for subject, scores in zip(subjects, scores_by_subject):
    top_idx = np.argmax(scores)
    print(f"{subject}: {names[top_idx]} ({scores[top_idx]})")

# Grade distribution
print("\n--- Grade Distribution ---")
def assign_grade(avg):
    if avg >= 90: return 'A'
    elif avg >= 80: return 'B'
    elif avg >= 70: return 'C'
    elif avg >= 60: return 'D'
    else: return 'F'

grades = np.array([assign_grade(avg) for avg in avg_scores])
unique_grades, grade_counts = np.unique(grades, return_counts=True)

# Sort grades in order A, B, C, D, F
grade_order = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'F': 4}
sort_idx = np.argsort([grade_order[g] for g in unique_grades])
unique_grades = unique_grades[sort_idx]
grade_counts = grade_counts[sort_idx]

print(f"{'Grade':<8} {'Count':<8} {'Percentage':<12} {'Bar'}")
print("-" * 45)
for grade, count in zip(unique_grades, grade_counts):
    pct = count / n_students * 100
    bar = '█' * int(pct / 5)
    print(f"{grade:<8} {count:<8} {pct:>5.1f}%       {bar}")

# ============================================================
# PART 11: Real-World Example - Inventory Management
# ============================================================

print("\n" + "="*60)
print("PART 11: REAL-WORLD - INVENTORY MANAGEMENT")
print("="*60)

# Product inventory
np.random.seed(123)
n_products = 15

product_ids = np.array([f"PRD{str(i).zfill(3)}" for i in range(1, n_products + 1)])
categories = np.array(['Electronics', 'Clothing', 'Food', 'Electronics', 'Clothing',
                       'Food', 'Electronics', 'Food', 'Clothing', 'Food',
                       'Electronics', 'Clothing', 'Food', 'Electronics', 'Clothing'])
quantities = np.random.randint(0, 100, n_products)
prices = np.random.uniform(10, 500, n_products).round(2)
sales_last_month = np.random.randint(0, 50, n_products)

print("Inventory Sample:")
print(f"{'ID':<8} {'Category':<12} {'Qty':<6} {'Price':<10} {'Sales':<6}")
print("-" * 45)
for i in range(5):
    print(f"{product_ids[i]:<8} {categories[i]:<12} {quantities[i]:<6} "
          f"${prices[i]:<9.2f} {sales_last_month[i]:<6}")
print("...")

# Analysis 1: Low stock items (quantity < 20)
print("\n--- Low Stock Alert (qty < 20) ---")
low_stock_mask = quantities < 20
low_stock_indices = np.where(low_stock_mask)[0]

if len(low_stock_indices) > 0:
    print(f"{'ID':<8} {'Category':<12} {'Qty':<6} {'Action'}")
    print("-" * 40)
    for idx in low_stock_indices:
        action = "URGENT" if quantities[idx] < 10 else "Reorder"
        print(f"{product_ids[idx]:<8} {categories[idx]:<12} {quantities[idx]:<6} {action}")
else:
    print("All products are well stocked!")

# Analysis 2: Category breakdown
print("\n--- Category Analysis ---")
unique_cats, cat_counts = np.unique(categories, return_counts=True)

print(f"{'Category':<15} {'Products':<10} {'Total Qty':<12} {'Avg Price':<12} {'Total Value'}")
print("-" * 65)
for cat in unique_cats:
    mask = categories == cat
    count = mask.sum()
    total_qty = quantities[mask].sum()
    avg_price = prices[mask].mean()
    total_value = (quantities[mask] * prices[mask]).sum()
    print(f"{cat:<15} {count:<10} {total_qty:<12} ${avg_price:<11.2f} ${total_value:,.2f}")

# Analysis 3: Top selling products
print("\n--- Top 5 Selling Products ---")
top_sellers_idx = np.argsort(sales_last_month)[::-1][:5]

print(f"{'Rank':<6} {'ID':<8} {'Category':<12} {'Sales':<8} {'Revenue'}")
print("-" * 50)
for rank, idx in enumerate(top_sellers_idx, 1):
    revenue = sales_last_month[idx] * prices[idx]
    print(f"{rank:<6} {product_ids[idx]:<8} {categories[idx]:<12} "
          f"{sales_last_month[idx]:<8} ${revenue:,.2f}")

# Analysis 4: Dead stock (no sales and high quantity)
print("\n--- Dead Stock Analysis ---")
dead_stock_mask = (sales_last_month == 0) & (quantities > 30)
dead_stock_idx = np.where(dead_stock_mask)[0]

if len(dead_stock_idx) > 0:
    print("Products with no sales but high inventory:")
    total_dead_value = 0
    for idx in dead_stock_idx:
        value = quantities[idx] * prices[idx]
        total_dead_value += value
        print(f"  {product_ids[idx]}: {quantities[idx]} units = ${value:,.2f}")
    print(f"\nTotal dead stock value: ${total_dead_value:,.2f}")
else:
    print("No significant dead stock found.")

# ============================================================
# PART 12: Sorting Stability and Complex Sorts
# ============================================================

print("\n" + "="*60)
print("PART 12: SORTING STABILITY AND COMPLEX SORTS")
print("="*60)

print("""
Sorting Algorithms in NumPy:
- 'quicksort': Default, fast but NOT stable
- 'mergesort': Stable sort (equal elements keep original order)
- 'heapsort': O(n log n) worst case, NOT stable
- 'stable': Alias for mergesort

Stable sorting is important when sorting by multiple criteria.
""")

# Demonstrate stable vs unstable sort
data = np.array([3, 1, 2, 1, 3, 2])
original_indices = np.arange(len(data))
print(f"Data: {data}")
print(f"Original indices: {original_indices}")

# Get sort indices
quick_idx = np.argsort(data, kind='quicksort')
stable_idx = np.argsort(data, kind='stable')

print(f"\nQuicksort indices: {quick_idx}")
print(f"Stable sort indices: {stable_idx}")
print("Note: For equal values (1s, 2s, 3s), stable preserves original order")

# Multi-key sorting using lexsort
print("\n--- Multi-Key Sorting with np.lexsort() ---")
# Sort by department, then by salary (within department)
departments = np.array(['Sales', 'IT', 'Sales', 'IT', 'HR', 'Sales', 'IT', 'HR'])
salaries = np.array([50000, 60000, 55000, 65000, 45000, 52000, 70000, 48000])
names_emp = np.array(['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank', 'Grace', 'Henry'])

print(f"Names: {names_emp}")
print(f"Departments: {departments}")
print(f"Salaries: {salaries}")

# lexsort sorts by LAST key first, then by earlier keys
# So we pass (salary, department) to sort by department first, then salary
sort_idx = np.lexsort((salaries, departments))

print("\nSorted by Department, then Salary:")
print(f"{'Name':<10} {'Department':<12} {'Salary':<10}")
print("-" * 35)
for idx in sort_idx:
    print(f"{names_emp[idx]:<10} {departments[idx]:<12} ${salaries[idx]:,}")

# ============================================================
# PART 13: Performance Comparison
# ============================================================

print("\n" + "="*60)
print("PART 13: PERFORMANCE COMPARISON")
print("="*60)

import time

# Create large array
np.random.seed(42)
large_array = np.random.randint(0, 1000000, 1000000)

# Test sorting algorithms
print("Sorting 1 million integers:")
print("-" * 40)

for kind in ['quicksort', 'mergesort', 'heapsort']:
    arr_copy = large_array.copy()
    start = time.time()
    sorted_arr = np.sort(arr_copy, kind=kind)
    elapsed = time.time() - start
    print(f"{kind:12}: {elapsed:.4f} seconds")

# Partition vs full sort for top-k
print("\n--- Finding Top 100 Elements ---")

start = time.time()
top100_sort = np.sort(large_array)[-100:]
sort_time = time.time() - start

start = time.time()
partitioned = np.partition(large_array, -100)
top100_partition = np.sort(partitioned[-100:])
partition_time = time.time() - start

print(f"Full sort:  {sort_time:.4f} seconds")
print(f"Partition:  {partition_time:.4f} seconds")
print(f"Speedup:    {sort_time/partition_time:.2f}x")

# ============================================================
# PART 14: Summary and Reference
# ============================================================

print("\n" + "="*60)
print("PART 14: SUMMARY AND REFERENCE")
print("="*60)

print("""
SORTING FUNCTIONS:
  np.sort(arr)           - Returns sorted copy
  arr.sort()             - In-place sort
  np.argsort(arr)        - Indices that would sort
  np.lexsort(keys)       - Sort by multiple keys
  np.partition(arr, k)   - Partial sort (k-th element)
  np.argpartition(arr,k) - Indices for partial sort

SEARCHING FUNCTIONS:
  np.where(condition)    - Indices where condition is True
  np.argmax(arr)         - Index of maximum
  np.argmin(arr)         - Index of minimum
  np.searchsorted(a, v)  - Binary search in sorted array
  np.nonzero(arr)        - Indices of non-zero elements
  np.flatnonzero(arr)    - Flat indices of non-zero

UNIQUE AND COUNTING:
  np.unique(arr)                    - Unique sorted values
  np.unique(arr, return_counts=True) - With counts
  np.unique(arr, return_index=True)  - With first indices
  np.unique(arr, return_inverse=True)- With reconstruction

SET OPERATIONS:
  np.union1d(a, b)       - Union
  np.intersect1d(a, b)   - Intersection
  np.setdiff1d(a, b)     - Difference (a - b)
  np.setxor1d(a, b)      - Symmetric difference
  np.isin(a, b)          - Membership test
  np.in1d(a, b)          - Same as isin (older)

SORTING ALGORITHMS:
  'quicksort'  - Fast, O(n log n) avg, NOT stable
  'mergesort'  - O(n log n), stable
  'heapsort'   - O(n log n) worst, NOT stable
  'stable'     - Alias for mergesort

TIPS:
  - Use argsort to sort related arrays together
  - Use partition when you only need top/bottom k
  - Use searchsorted for binning/bucketing
  - Use stable sort for multi-key sorting
  - Use lexsort for explicit multi-key sorting
""")

print("\n" + "="*60)
print("LESSON 13 COMPLETE!")
print("="*60)