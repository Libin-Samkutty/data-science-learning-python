import numpy as np

# ============================================================
# PART 1: Broadcasting Basics - What Is It?
# ============================================================

print("="*60)
print("PART 1: WHAT IS BROADCASTING?")
print("="*60)

print("""
Broadcasting is NumPy's way of performing operations on arrays
of DIFFERENT SHAPES without explicitly copying data.

Instead of manually expanding smaller arrays to match larger ones,
NumPy does it automatically and efficiently.
""")

# Simple example: Add scalar to array
arr = np.array([1, 2, 3, 4, 5])
print("Array:", arr)
print("Add 10:", arr + 10)  # 10 is "broadcast" to [10, 10, 10, 10, 10]

print("\nWhat happens under the hood:")
print("  arr:     [1, 2, 3, 4, 5]")
print("  scalar:  10")
print("  → 10 is conceptually expanded to [10, 10, 10, 10, 10]")
print("  → element-wise addition: [11, 12, 13, 14, 15]")
print("  → but NO ACTUAL COPYING happens!")

# Without broadcasting, you'd need to do:
arr_manual = np.array([1, 2, 3, 4, 5])
scalar_expanded = np.full(5, 10)
result_manual = arr_manual + scalar_expanded
print("\nManual (no broadcasting):", result_manual)
print("Broadcasting (automatic):", arr + 10)

# ============================================================
# PART 2: The Three Broadcasting Rules
# ============================================================

print("\n" + "="*60)
print("PART 2: THE THREE BROADCASTING RULES")
print("="*60)

print("""
NumPy broadcasting follows three rules:

RULE 1: If arrays have different number of dimensions,
        pad the smaller shape with 1s on the LEFT.
        
RULE 2: Arrays are compatible if, for each dimension,
        the sizes are either EQUAL or one of them is 1.
        
RULE 3: Where sizes are 1, stretch that dimension
        to match the other array.

Let's see examples of each rule...
""")

# RULE 1: Dimension padding
print("\n--- RULE 1: Dimension Padding ---")
a = np.array([1, 2, 3])      # Shape: (3,)
b = np.array([[10], [20]])   # Shape: (2, 1)

print(f"a shape: {a.shape} → interpreted as (1, 3)")
print(f"b shape: {b.shape}")
print("\nAfter padding, shapes align for comparison:")
print("  a: (1, 3)")
print("  b: (2, 1)")

result = a + b
print(f"\nResult shape: {result.shape}")
print("Result:")
print(result)
print("\nExplanation:")
print("  a is stretched to (2, 3): [[1, 2, 3], [1, 2, 3]]")
print("  b is stretched to (2, 3): [[10, 10, 10], [20, 20, 20]]")
print("  Then element-wise addition")

# RULE 2: Compatibility check
print("\n--- RULE 2: Compatibility Check ---")

# Compatible shapes
print("\nCompatible shape examples:")
print("  (3,) and (3,)     ✅ Same shape")
print("  (8, 1) and (1, 6) ✅ Different but compatible")
print("  (5, 4) and (1, 4) ✅ First dimension 5 vs 1")
print("  (3, 1, 5) and (1, 4, 5) ✅ Match or 1 in each dimension")

# Incompatible shapes
print("\nIncompatible shape examples:")
print("  (3,) and (4,)     ❌ Different and neither is 1")
print("  (2, 3) and (3, 2) ❌ Dimensions don't align")
print("  (3, 5) and (4, 5) ❌ First dimension 3 vs 4 (neither is 1)")

# RULE 3: Stretching
print("\n--- RULE 3: Stretching ---")
x = np.array([[1], [2], [3]])  # Shape: (3, 1)
y = np.array([10, 20, 30, 40]) # Shape: (4,) → (1, 4)

print(f"x shape: {x.shape}")
print(f"y shape: {y.shape} → (1, 4) after padding")

result = x + y
print(f"\nResult shape: {result.shape}")
print("x stretched to (3, 4):")
print("[[1, 1, 1, 1],")
print(" [2, 2, 2, 2],")
print(" [3, 3, 3, 3]]")
print("\ny stretched to (3, 4):")
print("[[10, 20, 30, 40],")
print(" [10, 20, 30, 40],")
print(" [10, 20, 30, 40]]")
print("\nResult (x + y):")
print(result)

# ============================================================
# PART 3: Common Broadcasting Patterns
# ============================================================

print("\n" + "="*60)
print("PART 3: COMMON BROADCASTING PATTERNS")
print("="*60)

# Pattern 1: Scalar with Array
print("\n--- Pattern 1: Scalar with Array ---")
data = np.array([10, 20, 30, 40])
print("Data:", data)
print("Multiply by 2:", data * 2)
print("Add 100:", data + 100)
print("Divide by 10:", data / 10)

# Pattern 2: 1D array with 2D array (rows)
print("\n--- Pattern 2: Add 1D to Each Row of 2D ---")
matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
row_vector = np.array([10, 20, 30])

print("Matrix (3x3):")
print(matrix)
print("\nRow vector (1x3):", row_vector)

result = matrix + row_vector
print("\nResult (add row vector to each row):")
print(result)

# Pattern 3: 1D array with 2D array (columns)
print("\n--- Pattern 3: Add 1D to Each Column of 2D ---")
col_vector = np.array([100, 200, 300])  # Shape: (3,)
col_vector_reshaped = col_vector[:, np.newaxis]  # Shape: (3, 1)

print("Column vector reshaped (3x1):")
print(col_vector_reshaped)

result = matrix + col_vector_reshaped
print("\nResult (add column vector to each column):")
print(result)

# Pattern 4: Outer product-like operation
print("\n--- Pattern 4: Outer Product Pattern ---")
a = np.array([1, 2, 3])[:, np.newaxis]  # (3, 1)
b = np.array([10, 20, 30, 40])          # (4,) → (1, 4)

print("a (3x1):")
print(a)
print("\nb (1x4):", b)

result = a * b
print(f"\nResult shape: {result.shape}")
print("Result (outer product):")
print(result)

# ============================================================
# PART 4: Real-World Example - Data Normalization
# ============================================================

print("\n" + "="*60)
print("PART 4: REAL-WORLD - DATA NORMALIZATION")
print("="*60)

# Student test scores: 5 students × 4 subjects
np.random.seed(42)
scores = np.random.randint(60, 100, (5, 4))

print("Student Scores (5 students × 4 subjects):")
print("Subjects: Math, Physics, Chemistry, Biology")
print(scores)

# Normalize by subject (each column)
# Z-score normalization: (x - mean) / std

# Calculate mean and std for each subject (column)
subject_means = scores.mean(axis=0)  # Shape: (4,)
subject_stds = scores.std(axis=0)    # Shape: (4,)

print(f"\nSubject means: {subject_means}")
print(f"Subject stds: {subject_stds}")

# Broadcasting: (5, 4) - (4,) and (5, 4) / (4,)
normalized_scores = (scores - subject_means) / subject_stds

print("\nNormalized scores (z-scores):")
print(normalized_scores.round(2))

# Verify normalization
print("\nVerification (should be ~0 and ~1):")
print(f"Mean per subject: {normalized_scores.mean(axis=0).round(3)}")
print(f"Std per subject: {normalized_scores.std(axis=0).round(3)}")

# Alternative: Normalize by student (each row)
print("\n--- Normalize by Student (Row-wise) ---")
student_means = scores.mean(axis=1)  # Shape: (5,)
student_stds = scores.std(axis=1)    # Shape: (5,)

print(f"Student means: {student_means}")
print(f"Student stds: {student_stds}")

# Need to reshape to (5, 1) to broadcast with (5, 4)
student_means_col = student_means[:, np.newaxis]  # (5, 1)
student_stds_col = student_stds[:, np.newaxis]    # (5, 1)

normalized_by_student = (scores - student_means_col) / student_stds_col

print("\nNormalized by student:")
print(normalized_by_student.round(2))

# ============================================================
# PART 5: Real-World Example - Sales Analysis
# ============================================================

print("\n" + "="*60)
print("PART 5: REAL-WORLD - QUARTERLY SALES ANALYSIS")
print("="*60)

# Sales data: 4 products × 4 quarters (in $1000s)
sales = np.array([
    [120, 135, 145, 160],  # Product A
    [80, 85, 90, 95],      # Product B
    [200, 210, 220, 230],  # Product C
    [50, 55, 58, 62]       # Product D
])

print("Quarterly Sales (4 products × 4 quarters, $1000s):")
print("      Q1   Q2   Q3   Q4")
print(sales)

# Scenario 1: Apply quarterly growth targets
# Q1: baseline, Q2: +5%, Q3: +10%, Q4: +15%
growth_factors = np.array([1.0, 1.05, 1.10, 1.15])  # Shape: (4,)

print(f"\nGrowth targets: {growth_factors}")

# Broadcasting: (4, 4) * (4,)
targets = sales[:, 0:1] * growth_factors  # Take Q1 as baseline

print("\nTargets based on Q1 with growth factors:")
print("      Q1   Q2   Q3   Q4")
print(targets.round(0))

# Compare actual vs target
performance = (sales / targets - 1) * 100  # Percentage difference

print("\nPerformance vs Target (%):")
print("      Q1   Q2   Q3   Q4")
print(performance.round(1))

# Scenario 2: Apply different discount rates per product
discount_rates = np.array([0.10, 0.15, 0.05, 0.20])  # Shape: (4,)
discount_rates_col = discount_rates[:, np.newaxis]   # Shape: (4, 1)

print(f"\nDiscount rates per product: {discount_rates}")

# Broadcasting: (4, 4) * (4, 1)
discounted_sales = sales * (1 - discount_rates_col)

print("\nSales after product-specific discounts:")
print("      Q1   Q2   Q3   Q4")
print(discounted_sales.round(0))

# Scenario 3: Compare each quarter against Q1 (baseline)
q1_baseline = sales[:, 0:1]  # Shape: (4, 1) - keep dimensions

print("\nQ1 Baseline (reshaped to 4x1):")
print(q1_baseline)

# Broadcasting: (4, 4) / (4, 1)
growth_vs_q1 = (sales / q1_baseline - 1) * 100

print("\nGrowth vs Q1 (%):")
print("      Q1   Q2   Q3   Q4")
print(growth_vs_q1.round(1))

# ============================================================
# PART 6: Real-World Example - Image Processing
# ============================================================

print("\n" + "="*60)
print("PART 6: REAL-WORLD - IMAGE PROCESSING")
print("="*60)

# Simulate a small grayscale image (8x8 pixels, values 0-255)
np.random.seed(42)
image = np.random.randint(0, 256, (8, 8))

print("Original Image (8x8, grayscale 0-255):")
print(image)

# Operation 1: Increase brightness (add scalar)
brightness_increase = 50
brightened = np.clip(image + brightness_increase, 0, 255)

print(f"\nBrightened (+{brightness_increase}):")
print(brightened)

# Operation 2: Adjust contrast (multiply by scalar)
contrast_factor = 1.5
contrasted = np.clip(image * contrast_factor, 0, 255).astype(int)

print(f"\nContrast adjusted (×{contrast_factor}):")
print(contrasted)

# Operation 3: Apply gradient (different brightness per row)
# Top rows darker, bottom rows lighter
row_gradient = np.linspace(-50, 50, 8)[:, np.newaxis]  # Shape: (8, 1)

print("\nRow gradient (8x1):")
print(row_gradient.astype(int))

gradient_applied = np.clip(image + row_gradient, 0, 255).astype(int)

print("\nImage with vertical gradient:")
print(gradient_applied)

# Operation 4: Apply different adjustments per column
col_adjustments = np.array([0, -20, -10, 0, 10, 20, 30, 40])  # Shape: (8,)

print("\nColumn adjustments (1x8):", col_adjustments)

col_adjusted = np.clip(image + col_adjustments, 0, 255).astype(int)

print("\nImage with column-specific adjustments:")
print(col_adjusted)

# Simulate RGB image (8x8x3)
print("\n--- RGB Image Example ---")
rgb_image = np.random.randint(0, 256, (8, 8, 3))

print(f"RGB Image shape: {rgb_image.shape} (height × width × channels)")

# Adjust only the red channel (channel 0)
red_boost = np.array([50, 0, 0])  # Shape: (3,)

print("Red boost:", red_boost)

rgb_boosted = np.clip(rgb_image + red_boost, 0, 255).astype(int)

print(f"\nAfter red boost, first pixel:")
print(f"Original: R={rgb_image[0,0,0]}, G={rgb_image[0,0,1]}, B={rgb_image[0,0,2]}")
print(f"Boosted:  R={rgb_boosted[0,0,0]}, G={rgb_boosted[0,0,1]}, B={rgb_boosted[0,0,2]}")

# ============================================================
# PART 7: Broadcasting Shape Compatibility Tester
# ============================================================

print("\n" + "="*60)
print("PART 7: SHAPE COMPATIBILITY CHECKER")
print("="*60)

def check_broadcast_compatibility(shape1, shape2):
    """Check if two shapes can be broadcast together"""
    # Pad shorter shape with 1s on the left
    ndim = max(len(shape1), len(shape2))
    shape1_padded = (1,) * (ndim - len(shape1)) + shape1
    shape2_padded = (1,) * (ndim - len(shape2)) + shape2
    
    print(f"\nShape 1: {shape1} → padded: {shape1_padded}")
    print(f"Shape 2: {shape2} → padded: {shape2_padded}")
    
    # Check each dimension
    compatible = True
    result_shape = []
    
    for i, (s1, s2) in enumerate(zip(shape1_padded, shape2_padded)):
        if s1 == s2 or s1 == 1 or s2 == 1:
            result_shape.append(max(s1, s2))
            print(f"  Dimension {i}: {s1} vs {s2} → {max(s1, s2)} ✅")
        else:
            print(f"  Dimension {i}: {s1} vs {s2} → ❌ INCOMPATIBLE")
            compatible = False
            break
    
    if compatible:
        print(f"✅ Compatible! Result shape: {tuple(result_shape)}")
    else:
        print("❌ Incompatible shapes - cannot broadcast")
    
    return compatible

# Test cases
print("\n--- Test Case 1 ---")
check_broadcast_compatibility((3, 4), (4,))

print("\n--- Test Case 2 ---")
check_broadcast_compatibility((8, 1, 6), (7, 1, 5))

print("\n--- Test Case 3 ---")
check_broadcast_compatibility((5, 4), (1,))

print("\n--- Test Case 4 (incompatible) ---")
check_broadcast_compatibility((3,), (4,))

print("\n--- Test Case 5 (incompatible) ---")
check_broadcast_compatibility((2, 3), (3, 2))

# ============================================================
# PART 8: Common Broadcasting Mistakes
# ============================================================

print("\n" + "="*60)
print("PART 8: COMMON BROADCASTING MISTAKES")
print("="*60)

# Mistake 1: Wrong dimension for row vs column operations
print("\n--- Mistake 1: Row vs Column Confusion ---")
matrix = np.array([[1, 2, 3], [4, 5, 6]])
vector = np.array([10, 20])

print("Matrix (2x3):")
print(matrix)
print(f"\nVector (2,): {vector}")

# This will fail - trying to add (2,) to (2, 3)
try:
    result = matrix + vector
    print("This shouldn't print")
except ValueError as e:
    print(f"❌ Error: {e}")
    print("   Vector shape (2,) treated as (1, 2), incompatible with (2, 3)")

# Correct: Reshape vector to (2, 1)
vector_col = vector[:, np.newaxis]
print(f"\nVector reshaped to (2, 1):")
print(vector_col)
result = matrix + vector_col
print("✅ Result:")
print(result)

# Mistake 2: Accidental broadcasting when you wanted element-wise
print("\n--- Mistake 2: Unintended Broadcasting ---")
a = np.array([[1, 2, 3]])    # Shape: (1, 3)
b = np.array([[10], [20]])   # Shape: (2, 1)

print("a (1x3):", a)
print("b (2x1):")
print(b)

# This broadcasts to (2, 3) - might not be intended!
result = a + b
print("\nResult (2x3) - was this intended?")
print(result)

# If you wanted element-wise, shapes must match exactly
print("\n💡 To prevent accidental broadcasting, use assertions:")
print("   assert a.shape == b.shape, 'Shapes must match'")

# Mistake 3: Forgetting to preserve dimensions
print("\n--- Mistake 3: Dimension Loss ---")
data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
col_means = data.mean(axis=0)

print("Data (3x3):")
print(data)
print(f"\nColumn means shape: {col_means.shape}")  # (3,) not (1, 3)

# This works due to broadcasting, but loses clarity
centered = data - col_means
print("\nCentered (works but shape is implicit):")
print(centered)

# Better: Keep dimensions explicit
col_means_kept = data.mean(axis=0, keepdims=True)
print(f"\nColumn means with keepdims: {col_means_kept.shape}")
centered_explicit = data - col_means_kept
print("Centered (explicit shape):")
print(centered_explicit)

# ============================================================
# PART 9: Broadcasting Performance Benefits
# ============================================================

print("\n" + "="*60)
print("PART 9: BROADCASTING PERFORMANCE")
print("="*60)

import time

# Create large matrices
np.random.seed(42)
large_matrix = np.random.randn(1000, 500)
row_vector = np.random.randn(500)

print(f"Matrix shape: {large_matrix.shape}")
print(f"Vector shape: {row_vector.shape}")

# Method 1: Manual expansion (SLOW - wastes memory)
print("\n--- Method 1: Manual Expansion ---")
start = time.time()
row_expanded = np.tile(row_vector, (1000, 1))  # Duplicate 1000 times
result_manual = large_matrix + row_expanded
time_manual = time.time() - start
print(f"Time: {time_manual:.4f} seconds")
print(f"Memory for expanded vector: {row_expanded.nbytes / 1_000_000:.2f} MB")

# Method 2: Broadcasting (FAST - no copy)
print("\n--- Method 2: Broadcasting ---")
start = time.time()
result_broadcast = large_matrix + row_vector  # Automatic broadcasting
time_broadcast = time.time() - start
print(f"Time: {time_broadcast:.4f} seconds")
print(f"Memory for vector: {row_vector.nbytes / 1_000:.2f} KB")
print(f"\nSpeedup: {time_manual/time_broadcast:.1f}x")
print(f"Memory saved: {(row_expanded.nbytes - row_vector.nbytes) / 1_000_000:.2f} MB")

# Verify results match
print(f"\nResults match: {np.allclose(result_manual, result_broadcast)}")

# ============================================================
# PART 10: Broadcasting Quick Reference
# ============================================================

print("\n" + "="*60)
print("PART 10: BROADCASTING QUICK REFERENCE")
print("="*60)

print("""
BROADCASTING RULES (Summary):
1. Pad smaller shape with 1s on the LEFT
2. Dimensions are compatible if equal OR one is 1
3. Stretch dimension of size 1 to match the other

COMMON PATTERNS:

| Operation          | Array 1   | Array 2   | Result    |
|--------------------|-----------|-----------|-----------|
| Scalar + Array     | ()        | (n,)      | (n,)      |
| Row to Matrix      | (n,)      | (m, n)    | (m, n)    |
| Column to Matrix   | (m, 1)    | (m, n)    | (m, n)    |
| Outer Product      | (m, 1)    | (1, n)    | (m, n)    |
| 3D Color Channel   | (h, w, 3) | (3,)      | (h, w, 3) |

TIPS:
✅ Use np.newaxis or reshape to add dimensions
✅ Use keepdims=True in aggregations to preserve dimensions
✅ Check shapes before operations to catch errors early
✅ Broadcasting is memory-efficient (no copying)
✅ Broadcasting is fast (vectorized operations)

COMMON MISTAKES:
❌ Forgetting to reshape 1D arrays for column operations
❌ Assuming dimension alignment (check with .shape)
❌ Losing dimensions in aggregations (use keepdims=True)
❌ Confusing (n,) with (n, 1) or (1, n)
""")

# ============================================================
# PART 11: Visual Shape Transformation Examples
# ============================================================

print("\n" + "="*60)
print("PART 11: SHAPE TRANSFORMATION CHEAT SHEET")
print("="*60)

arr = np.array([1, 2, 3])
print(f"Original array: {arr}, shape: {arr.shape}")

print("\nTransformations:")
print(f"  arr[np.newaxis, :]  → shape {arr[np.newaxis, :].shape}  (row vector)")
print(f"  arr[:, np.newaxis]  → shape {arr[:, np.newaxis].shape}  (column vector)")
print(f"  arr.reshape(1, -1)  → shape {arr.reshape(1, -1).shape}  (row vector)")
print(f"  arr.reshape(-1, 1)  → shape {arr.reshape(-1, 1).shape}  (column vector)")

print("\n✅ Use these to control broadcasting direction!")