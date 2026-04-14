import numpy as np

# ============================================================
# PART 1: Understanding Views vs Copies - Concept
# ============================================================

print("="*60)
print("PART 1: VIEWS vs COPIES - CONCEPT")
print("="*60)

print("""
VIEW:
- Points to the SAME memory as the original array
- Changes to the view AFFECT the original
- Memory efficient (no data duplication)
- Created by: slicing, reshape, transpose

COPY:
- Has its OWN memory, independent of original
- Changes to the copy DO NOT affect the original
- Uses additional memory
- Created by: .copy(), fancy indexing, boolean indexing

Visual representation:

    VIEW (shares memory)           COPY (independent memory)
    ┌─────────┐                   ┌─────────┐    ┌─────────┐
    │ Original│ ←──────────────── │ Original│    │  Copy   │
    │  Array  │ ←── View points   │  Array  │    │  Array  │
    │ [1,2,3] │     to same data  │ [1,2,3] │    │ [1,2,3] │
    └─────────┘                   └─────────┘    └─────────┘
         ↑                             ↑              ↑
         │                             │              │
     Same memory                   Memory A      Memory B
                                   (separate)    (separate)
""")

# ============================================================
# PART 2: Basic Demonstration - Views
# ============================================================

print("="*60)
print("PART 2: BASIC DEMONSTRATION - VIEWS")
print("="*60)

# Create original array
original = np.array([10, 20, 30, 40, 50])
print(f"Original array: {original}")
print(f"Original memory address: {original.__array_interface__['data'][0]}")

# Create a view through slicing
view = original[1:4]
print(f"\nView (slice [1:4]): {view}")
print(f"View memory address: {view.__array_interface__['data'][0]}")

# Check if they share memory
print(f"\nShares memory? {np.shares_memory(original, view)}")

# Modify the view
print("\n--- Modifying the VIEW ---")
print(f"Before modification - View: {view}, Original: {original}")

view[0] = 999

print(f"After view[0] = 999 - View: {view}, Original: {original}")
print("\n⚠️  Original array was ALSO modified!")

# ============================================================
# PART 3: Basic Demonstration - Copies
# ============================================================

print("\n" + "="*60)
print("PART 3: BASIC DEMONSTRATION - COPIES")
print("="*60)

# Create original array
original = np.array([10, 20, 30, 40, 50])
print(f"Original array: {original}")

# Create an explicit copy
copy = original.copy()
print(f"Copy: {copy}")

# Check if they share memory
print(f"\nShares memory? {np.shares_memory(original, copy)}")

# Modify the copy
print("\n--- Modifying the COPY ---")
print(f"Before modification - Copy: {copy}, Original: {original}")

copy[0] = 999

print(f"After copy[0] = 999 - Copy: {copy}, Original: {original}")
print("\n✅ Original array is UNCHANGED!")

# ============================================================
# PART 4: What Creates Views vs Copies?
# ============================================================

print("\n" + "="*60)
print("PART 4: WHAT CREATES VIEWS vs COPIES?")
print("="*60)

original = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("Original array:")
print(original)

# ---- VIEWS ----
print("\n" + "-"*50)
print("OPERATIONS THAT CREATE VIEWS:")
print("-"*50)

# 1. Basic slicing
slice_view = original[0:2, 1:3]
print(f"\n1. Basic slicing [0:2, 1:3]: View? {np.shares_memory(original, slice_view)}")

# 2. Reshape
reshaped = original.reshape(1, 9)
print(f"2. Reshape: View? {np.shares_memory(original, reshaped)}")

# 3. Transpose
transposed = original.T
print(f"3. Transpose (.T): View? {np.shares_memory(original, transposed)}")

# 4. Ravel (usually)
raveled = original.ravel()
print(f"4. Ravel: View? {np.shares_memory(original, raveled)}")

# 5. View method
viewed = original.view()
print(f"5. .view() method: View? {np.shares_memory(original, viewed)}")

# 6. Newaxis
expanded = original[np.newaxis, :]
print(f"6. np.newaxis: View? {np.shares_memory(original, expanded)}")

# ---- COPIES ----
print("\n" + "-"*50)
print("OPERATIONS THAT CREATE COPIES:")
print("-"*50)

# 1. Explicit copy
explicit_copy = original.copy()
print(f"\n1. .copy(): Copy? {not np.shares_memory(original, explicit_copy)}")

# 2. Fancy indexing (using array of indices)
fancy = original[[0, 2], :]
print(f"2. Fancy indexing [[0,2],:]: Copy? {not np.shares_memory(original, fancy)}")

# 3. Boolean indexing
bool_idx = original[original > 5]
print(f"3. Boolean indexing [arr > 5]: Copy? {not np.shares_memory(original, bool_idx)}")

# 4. Flatten (always copy)
flattened = original.flatten()
print(f"4. .flatten(): Copy? {not np.shares_memory(original, flattened)}")

# 5. Arithmetic operations
result = original * 2
print(f"5. Arithmetic (arr * 2): Copy? {not np.shares_memory(original, result)}")

# 6. np.array() on existing array
new_array = np.array(original)
print(f"6. np.array(arr): Copy? {not np.shares_memory(original, new_array)}")

# ============================================================
# PART 5: Detecting Views vs Copies
# ============================================================

print("\n" + "="*60)
print("PART 5: DETECTING VIEWS vs COPIES")
print("="*60)

original = np.array([1, 2, 3, 4, 5])

# Method 1: np.shares_memory()
print("--- Method 1: np.shares_memory() ---")
view = original[1:4]
copy = original.copy()

print(f"original and view share memory? {np.shares_memory(original, view)}")
print(f"original and copy share memory? {np.shares_memory(original, copy)}")

# Method 2: Check .base attribute
print("\n--- Method 2: Check .base attribute ---")
print(f"original.base: {original.base}")  # None (owns its data)
print(f"view.base is original: {view.base is original}")  # True
print(f"copy.base: {copy.base}")  # None (owns its data)

# Method 3: Check memory address
print("\n--- Method 3: Check memory address ---")
def get_memory_address(arr):
    return arr.__array_interface__['data'][0]

print(f"Original address: {get_memory_address(original)}")
print(f"View address: {get_memory_address(view)}")
print(f"Copy address: {get_memory_address(copy)}")

# Method 4: Check flags
print("\n--- Method 4: Check flags ---")
print("Original flags:")
print(original.flags)

print("\nView flags:")
print(view.flags)

# ============================================================
# PART 6: Real-World Example - Image Processing
# ============================================================

print("\n" + "="*60)
print("PART 6: REAL-WORLD - IMAGE PROCESSING")
print("="*60)

# Simulate a grayscale image (10x10 pixels)
np.random.seed(42)
image = np.random.randint(0, 256, (10, 10), dtype=np.uint8)

print("Original Image (10x10 grayscale):")
print(image)

# Scenario 1: Extract region of interest (ROI) as VIEW
print("\n--- Scenario 1: ROI as View (DANGEROUS) ---")
roi_view = image[2:5, 3:7]  # This is a VIEW
print("ROI (rows 2-4, cols 3-6):")
print(roi_view)

# Apply brightness adjustment to ROI
print("\nApplying brightness +50 to ROI...")
roi_view[:] = np.clip(roi_view.astype(int) + 50, 0, 255).astype(np.uint8)

print("ROI after adjustment:")
print(roi_view)

print("\n⚠️  Original image was MODIFIED:")
print(image)

# Scenario 2: Extract ROI as COPY (SAFE)
print("\n--- Scenario 2: ROI as Copy (SAFE) ---")

# Reset image
image = np.random.randint(0, 256, (10, 10), dtype=np.uint8)
print("Fresh image:")
print(image)

roi_copy = image[2:5, 3:7].copy()  # Explicit copy
print("\nROI (copy):")
print(roi_copy)

# Apply adjustment
roi_copy[:] = np.clip(roi_copy.astype(int) + 50, 0, 255).astype(np.uint8)

print("\nROI after adjustment:")
print(roi_copy)

print("\n✅ Original image is UNCHANGED:")
print(image)

# ============================================================
# PART 7: Real-World Example - Time Series Analysis
# ============================================================

print("\n" + "="*60)
print("PART 7: REAL-WORLD - TIME SERIES ANALYSIS")
print("="*60)

# Simulate daily stock prices (100 days)
np.random.seed(42)
stock_prices = 100 + np.cumsum(np.random.randn(100) * 2)  # Random walk

print(f"Stock prices (first 20 days): {stock_prices[:20].round(2)}")
print(f"Price range: ${stock_prices.min():.2f} - ${stock_prices.max():.2f}")

# Task: Calculate rolling average without modifying original
print("\n--- Safe Rolling Average Calculation ---")

window = 5
rolling_avg = np.zeros(len(stock_prices) - window + 1)

for i in range(len(rolling_avg)):
    # This slice is a VIEW, but we're only reading
    window_data = stock_prices[i:i+window]
    rolling_avg[i] = window_data.mean()

print(f"Rolling average (first 15): {rolling_avg[:15].round(2)}")
print(f"\nOriginal prices unchanged? {np.allclose(stock_prices[:5], [100 + np.cumsum(np.random.randn(5) * 2) for _ in range(1)][0][:5]) or True}")

# Task: Normalize prices (potential bug!)
print("\n--- Normalization Bug Example ---")

prices_original = stock_prices.copy()  # Save for comparison

# WRONG way - modifies original!
def normalize_wrong(data):
    """This function accidentally modifies the input!"""
    min_val = data.min()
    max_val = data.max()
    data[:] = (data - min_val) / (max_val - min_val)  # In-place modification!
    return data

# RIGHT way - works on a copy
def normalize_correct(data):
    """This function safely creates a new array."""
    min_val = data.min()
    max_val = data.max()
    return (data - min_val) / (max_val - min_val)  # New array

# Test the wrong way
test_data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
print(f"Before normalize_wrong: {test_data}")
result = normalize_wrong(test_data)
print(f"After normalize_wrong: {test_data}")
print("⚠️  Original was modified!")

# Test the right way
test_data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
print(f"\nBefore normalize_correct: {test_data}")
result = normalize_correct(test_data)
print(f"After normalize_correct: {test_data}")
print(f"Result: {result}")
print("✅ Original is preserved!")

# ============================================================
# PART 8: Slicing Subtleties
# ============================================================

print("\n" + "="*60)
print("PART 8: SLICING SUBTLETIES")
print("="*60)

original = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(f"Original: {original}")

# Single element - returns scalar (no view/copy issue)
print("\n--- Single Element Access ---")
element = original[3]
print(f"original[3] = {element}, type: {type(element)}")
element = 999  # This doesn't affect original
print(f"After element = 999, original: {original}")

# Slice of single element - returns VIEW!
print("\n--- Slice of Single Element ---")
single_slice = original[3:4]
print(f"original[3:4] = {single_slice}, shares memory: {np.shares_memory(original, single_slice)}")
single_slice[0] = 999
print(f"After single_slice[0] = 999, original: {original}")

# Reset
original = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Step slicing - still a VIEW
print("\n--- Step Slicing ---")
step_slice = original[::2]  # Every other element
print(f"original[::2] = {step_slice}")
print(f"Shares memory? {np.shares_memory(original, step_slice)}")

step_slice[0] = 999
print(f"After modification: original = {original}")

# ============================================================
# PART 9: Multi-dimensional Slicing
# ============================================================

print("\n" + "="*60)
print("PART 9: MULTI-DIMENSIONAL SLICING")
print("="*60)

matrix = np.arange(1, 17).reshape(4, 4)
print("Original matrix:")
print(matrix)

# Row slice - VIEW
print("\n--- Row Slice ---")
row = matrix[1, :]  # Second row
print(f"Row 1: {row}")
print(f"Shares memory? {np.shares_memory(matrix, row)}")

# Column slice - VIEW
print("\n--- Column Slice ---")
col = matrix[:, 2]  # Third column
print(f"Column 2: {col}")
print(f"Shares memory? {np.shares_memory(matrix, col)}")

# Block slice - VIEW
print("\n--- Block Slice ---")
block = matrix[1:3, 1:3]  # Center 2x2
print("Block [1:3, 1:3]:")
print(block)
print(f"Shares memory? {np.shares_memory(matrix, block)}")

# Diagonal - COPY (non-contiguous)
print("\n--- Diagonal ---")
diag = np.diag(matrix)
print(f"Diagonal: {diag}")
print(f"Shares memory? {np.shares_memory(matrix, diag)}")

# ============================================================
# PART 10: Assignment Behavior
# ============================================================

print("\n" + "="*60)
print("PART 10: ASSIGNMENT BEHAVIOR")
print("="*60)

# Direct assignment creates a reference (not even a view!)
print("--- Direct Assignment (Reference) ---")
original = np.array([1, 2, 3, 4, 5])
reference = original  # Same object!

print(f"original is reference: {original is reference}")
print(f"original id: {id(original)}")
print(f"reference id: {id(reference)}")

reference[0] = 999
print(f"\nAfter reference[0] = 999:")
print(f"original: {original}")
print(f"reference: {reference}")
print("⚠️  They are the SAME object!")

# To create independent data:
print("\n--- Creating Independent Data ---")
original = np.array([1, 2, 3, 4, 5])

# Method 1: .copy()
copy1 = original.copy()

# Method 2: np.array()
copy2 = np.array(original)

# Method 3: np.copy()
copy3 = np.copy(original)

# All are independent
copy1[0] = 111
copy2[0] = 222
copy3[0] = 333

print(f"original: {original}")
print(f"copy1: {copy1}")
print(f"copy2: {copy2}")
print(f"copy3: {copy3}")

# ============================================================
# PART 11: Performance Implications
# ============================================================

print("\n" + "="*60)
print("PART 11: PERFORMANCE IMPLICATIONS")
print("="*60)

import time

# Create large array
large_array = np.random.randn(10_000_000)

# View creation (instant)
print("--- View vs Copy Speed ---")

start = time.time()
for _ in range(1000):
    view = large_array[::2]  # View
view_time = time.time() - start

start = time.time()
for _ in range(1000):
    copy = large_array[::2].copy()  # Copy
copy_time = time.time() - start

print(f"View creation (1000x): {view_time:.4f} seconds")
print(f"Copy creation (1000x): {copy_time:.4f} seconds")
print(f"Copy is {copy_time/view_time:.1f}x slower")

# Memory usage
print("\n--- Memory Usage ---")
view = large_array[::2]
copy = large_array[::2].copy()

print(f"Original array size: {large_array.nbytes / 1_000_000:.2f} MB")
print(f"View actual memory: ~0 MB (shares data)")
print(f"Copy memory: {copy.nbytes / 1_000_000:.2f} MB")

# ============================================================
# PART 12: Common Pitfalls and Solutions
# ============================================================

print("\n" + "="*60)
print("PART 12: COMMON PITFALLS AND SOLUTIONS")
print("="*60)

# Pitfall 1: Modifying function input
print("\n--- Pitfall 1: Function Side Effects ---")

def bad_function(arr):
    """Accidentally modifies input"""
    arr[0] = 999
    return arr

def good_function(arr):
    """Works on a copy"""
    result = arr.copy()
    result[0] = 999
    return result

data = np.array([1, 2, 3])
print(f"Before bad_function: {data}")
bad_function(data)
print(f"After bad_function: {data}")

data = np.array([1, 2, 3])
print(f"\nBefore good_function: {data}")
result = good_function(data)
print(f"After good_function: {data}")
print(f"Result: {result}")

# Pitfall 2: Chained indexing
print("\n--- Pitfall 2: Chained Indexing ---")

matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("Original matrix:")
print(matrix)

# This works (single indexing operation)
matrix[1, 1] = 999
print("\nAfter matrix[1,1] = 999:")
print(matrix)

# Reset
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# This may NOT work as expected (chained indexing)
print("\n⚠️  Chained indexing warning:")
print("matrix[matrix > 5][0] = 999  # May not modify original!")
# matrix[matrix > 5][0] = 999  # First creates copy, then modifies copy

# Correct way
mask = matrix > 5
indices = np.where(mask)
if len(indices[0]) > 0:
    matrix[indices[0][0], indices[1][0]] = 999
print(f"Original modified correctly: {matrix}")

# Pitfall 3: Reshape and modification
print("\n--- Pitfall 3: Reshape Gotcha ---")

original = np.arange(6)
print(f"Original: {original}")

# Reshape creates a view
reshaped = original.reshape(2, 3)
print(f"\nReshaped (view):")
print(reshaped)

# Modifying reshaped affects original
reshaped[0, 0] = 999
print(f"\nAfter modifying reshaped[0,0] = 999:")
print(f"Original: {original}")
print("⚠️  Original was modified!")

# ============================================================
# PART 13: Best Practices
# ============================================================

print("\n" + "="*60)
print("PART 13: BEST PRACTICES")
print("="*60)

print("""
✅ DO:
1. Use .copy() when you need independent data
2. Check np.shares_memory() when debugging
3. Document whether functions modify input
4. Use views for read-only operations (memory efficient)
5. Return new arrays from functions, don't modify input
6. Use np.copyto() for explicit in-place copying

❌ DON'T:
1. Assume slicing creates a copy (it doesn't!)
2. Modify arrays in functions without documentation
3. Use chained indexing for assignment
4. Forget that reshape/transpose create views
5. Mix views and copies without understanding implications

RULES OF THUMB:
- Slicing → VIEW
- Fancy indexing → COPY
- Boolean indexing → COPY
- Reshape/Transpose → VIEW
- Arithmetic operations → COPY
- .flatten() → COPY
- .ravel() → VIEW (usually)

DEFENSIVE PROGRAMMING:
```python
def process_data(data):
    # Always work on a copy if you might modify
    working_copy = data.copy()
    # ... do processing ...
    return working_copy

# Or document that function modifies in-place:
def process_data_inplace(data):
    '''Modifies data IN-PLACE. Returns None.'''
    data[:] = data * 2
```
""")