import numpy as np

# ============================================================
# PART 1: Understanding Array Dimensions
# ============================================================

# 1D array - a vector (like a single row or column)
vector = np.array([10, 20, 30, 40, 50])
print("=== 1D ARRAY (Vector) ===")
print("Data:", vector)
print("Shape:", vector.shape)      # (5,) - 5 elements in 1 dimension
print("Number of dimensions:", vector.ndim)  # 1
print("Total elements:", vector.size)        # 5
print("Data type:", vector.dtype)

# 2D array - a matrix (like a spreadsheet)
matrix = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
])
print("\n=== 2D ARRAY (Matrix) ===")
print("Data:\n", matrix)
print("Shape:", matrix.shape)      # (3, 4) - 3 rows, 4 columns
print("Number of dimensions:", matrix.ndim)  # 2
print("Total elements:", matrix.size)        # 12 (3 × 4)

# 3D array - a cube (like multiple spreadsheets stacked)
# Real example: RGB image (height × width × color_channels)
cube = np.array([
    [[1, 2], [3, 4], [5, 6]],
    [[7, 8], [9, 10], [11, 12]]
])
print("\n=== 3D ARRAY (Cube) ===")
print("Data:\n", cube)
print("Shape:", cube.shape)        # (2, 3, 2)
print("Number of dimensions:", cube.ndim)  # 3
print("Total elements:", cube.size)        # 12 (2 × 3 × 2)

# ============================================================
# PART 2: Real-World Example - Sales Data Structure
# ============================================================

print("\n" + "="*60)
print("REAL-WORLD EXAMPLE: Regional Sales Data")
print("="*60)

# Sales data: 4 regions × 3 months
# Rows = regions (North, South, East, West)
# Columns = months (Jan, Feb, Mar)
sales = np.array([
    [50000, 55000, 60000],  # North
    [45000, 48000, 52000],  # South
    [60000, 62000, 65000],  # East
    [40000, 43000, 47000]   # West
])

print("\nSales Matrix (4 regions × 3 months):")
print(sales)
print(f"Shape: {sales.shape} - {sales.shape[0]} regions, {sales.shape[1]} months")

# Access individual elements
print(f"\nNorth region, February sales: ₹{sales[0, 1]:,}")
print(f"Total sales for all regions in March: ₹{sales[:, 2].sum():,}")
print(f"Average sales for South region: ₹{sales[1, :].mean():,.0f}")

# ============================================================
# PART 3: Reshaping - Changing Array Structure
# ============================================================

print("\n" + "="*60)
print("RESHAPING ARRAYS")
print("="*60)

# Original 1D array
original = np.arange(12)  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
print("\nOriginal 1D array:", original)
print("Shape:", original.shape)  # (12,)

# Reshape to 2D: 3 rows × 4 columns
reshaped_3x4 = original.reshape(3, 4)
print("\nReshaped to 3×4:")
print(reshaped_3x4)
print("Shape:", reshaped_3x4.shape)

# Reshape to 2D: 4 rows × 3 columns (different arrangement)
reshaped_4x3 = original.reshape(4, 3)
print("\nReshaped to 4×3:")
print(reshaped_4x3)
print("Shape:", reshaped_4x3.shape)

# Reshape to 3D: 2 × 2 × 3
reshaped_3d = original.reshape(2, 2, 3)
print("\nReshaped to 3D (2×2×3):")
print(reshaped_3d)
print("Shape:", reshaped_3d.shape)

# ============================================================
# PART 4: Automatic Dimension Inference with -1
# ============================================================

print("\n" + "="*60)
print("AUTOMATIC RESHAPING WITH -1")
print("="*60)

# Use -1 to let NumPy calculate one dimension automatically
data = np.arange(24)

# "Make it 4 rows, figure out columns automatically"
auto_reshape = data.reshape(4, -1)
print("\nReshape(4, -1) - 4 rows, auto columns:")
print(auto_reshape)
print("Shape:", auto_reshape.shape)  # (4, 6) - NumPy calculated 6 columns

# "Make it 3 columns, figure out rows automatically"
auto_reshape2 = data.reshape(-1, 3)
print("\nReshape(-1, 3) - auto rows, 3 columns:")
print(auto_reshape2)
print("Shape:", auto_reshape2.shape)  # (8, 3) - NumPy calculated 8 rows

# Real-world use: batch processing
# You have 1000 data points, want batches of 50
num_samples = 1000
batch_size = 50
dummy_data = np.arange(num_samples)
batched = dummy_data.reshape(-1, batch_size)  # Auto-calculate number of batches
print(f"\n{num_samples} samples reshaped to batches of {batch_size}:")
print(f"Shape: {batched.shape} - {batched.shape[0]} batches")

# ============================================================
# PART 5: Flattening - Converting to 1D
# ============================================================

print("\n" + "="*60)
print("FLATTENING ARRAYS")
print("="*60)

matrix_2d = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
print("Original 2D array:")
print(matrix_2d)

# Method 1: flatten() - creates a COPY
flattened_copy = matrix_2d.flatten()
print("\nFlattened (copy):", flattened_copy)
print("Shape:", flattened_copy.shape)

# Method 2: ravel() - returns a VIEW (no copy, faster)
flattened_view = matrix_2d.ravel()
print("\nRaveled (view):", flattened_view)
print("Shape:", flattened_view.shape)

# Method 3: reshape to -1
flattened_reshape = matrix_2d.reshape(-1)
print("\nReshape(-1):", flattened_reshape)
print("Shape:", flattened_reshape.shape)

# Demonstrate difference between copy and view
print("\n--- Copy vs View Demonstration ---")
original_matrix = np.array([[1, 2], [3, 4]])
copy_version = original_matrix.flatten()
view_version = original_matrix.ravel()

# Modify the flattened arrays
copy_version[0] = 999
view_version[0] = 888

print("Original after modifying copy:", original_matrix)  # Unchanged
print("Original after modifying view:", original_matrix)  # Changed!

# ============================================================
# PART 6: Adding and Removing Dimensions
# ============================================================

print("\n" + "="*60)
print("ADDING/REMOVING DIMENSIONS")
print("="*60)

# Start with 1D array
arr_1d = np.array([1, 2, 3, 4, 5])
print("Original 1D array:", arr_1d)
print("Shape:", arr_1d.shape)  # (5,)

# Add dimension using np.newaxis
arr_2d_row = arr_1d[np.newaxis, :]  # Add dimension at axis 0 (row vector)
print("\nRow vector (1×5):")
print(arr_2d_row)
print("Shape:", arr_2d_row.shape)  # (1, 5)

arr_2d_col = arr_1d[:, np.newaxis]  # Add dimension at axis 1 (column vector)
print("\nColumn vector (5×1):")
print(arr_2d_col)
print("Shape:", arr_2d_col.shape)  # (5, 1)

# Alternative: reshape
row_reshape = arr_1d.reshape(1, -1)
col_reshape = arr_1d.reshape(-1, 1)
print("\nUsing reshape for row:", row_reshape.shape)
print("Using reshape for column:", col_reshape.shape)

# Remove unnecessary dimensions with squeeze()
arr_with_extra_dims = np.array([[[1, 2, 3]]])  # Shape (1, 1, 3)
print(f"\nArray with extra dimensions: shape {arr_with_extra_dims.shape}")
squeezed = arr_with_extra_dims.squeeze()
print(f"After squeeze: {squeezed}, shape {squeezed.shape}")

# ============================================================
# PART 7: Real-World Example - Image Data Reshaping
# ============================================================

print("\n" + "="*60)
print("REAL-WORLD: IMAGE DATA RESHAPING")
print("="*60)

# Simulate a small grayscale image (8×8 pixels)
# In reality, images are loaded as flattened arrays or 2D matrices
np.random.seed(42)
image_flat = np.random.randint(0, 256, 64)  # 64 pixel values (0-255)
print(f"Flattened image data (64 pixels): {image_flat[:10]}... (showing first 10)")
print(f"Shape: {image_flat.shape}")

# Reshape to 2D image (8×8)
image_2d = image_flat.reshape(8, 8)
print(f"\nReshaped to 8×8 image:")
print(image_2d)
print(f"Shape: {image_2d.shape}")

# Simulate RGB color image (8×8 pixels, 3 color channels)
rgb_image_flat = np.random.randint(0, 256, 192)  # 8×8×3 = 192 values
rgb_image = rgb_image_flat.reshape(8, 8, 3)
print(f"\nRGB image shape: {rgb_image.shape} (height × width × channels)")
print(f"Pixel at position (0,0): R={rgb_image[0,0,0]}, G={rgb_image[0,0,1]}, B={rgb_image[0,0,2]}")

# Machine learning batch: 10 images of 8×8×3
batch_size = 10
ml_batch = np.random.randint(0, 256, (batch_size, 8, 8, 3))
print(f"\nML batch shape: {ml_batch.shape} (batch × height × width × channels)")

# ============================================================
# PART 8: Transpose - Swapping Dimensions
# ============================================================

print("\n" + "="*60)
print("TRANSPOSING ARRAYS")
print("="*60)

# Original matrix
matrix = np.array([
    [1, 2, 3],
    [4, 5, 6]
])
print("Original matrix (2×3):")
print(matrix)
print("Shape:", matrix.shape)

# Transpose - rows become columns, columns become rows
transposed = matrix.T
print("\nTransposed matrix (3×2):")
print(transposed)
print("Shape:", transposed.shape)

# Real-world: converting data format
# Data as rows (3 students, 4 subjects)
students_scores = np.array([
    [85, 90, 78, 92],  # Student 1
    [88, 85, 91, 87],  # Student 2
    [92, 88, 85, 90]   # Student 3
])
print("\nScores by student (3 students × 4 subjects):")
print(students_scores)

# Transpose to get scores by subject (4 subjects × 3 students)
subjects_scores = students_scores.T
print("\nScores by subject (4 subjects × 3 students):")
print(subjects_scores)
print(f"Average score in subject 1: {subjects_scores[0].mean():.1f}")

# ============================================================
# PART 9: Common Reshaping Patterns
# ============================================================

print("\n" + "="*60)
print("COMMON RESHAPING PATTERNS")
print("="*60)

# Pattern 1: Time series to matrix (rolling window)
# Daily sales for 30 days -> 10 rows of 3-day periods
daily_sales = np.random.randint(100, 500, 30)
print("Daily sales (30 days):", daily_sales[:10], "...")

# Reshape to analyze weekly patterns (not exactly weeks, but 6 periods of 5 days)
weekly_view = daily_sales.reshape(6, 5)
print("\nWeekly view (6 periods × 5 days):")
print(weekly_view)
print(f"Average sales in first period: {weekly_view[0].mean():.0f}")

# Pattern 2: Feature matrix for ML
# 100 samples, each with 5 features
num_samples = 100
num_features = 5
features = np.random.randn(num_samples, num_features)
print(f"\nFeature matrix shape: {features.shape} (samples × features)")
print(f"First sample: {features[0]}")

# Pattern 3: Flatten for processing, then reshape back
data_2d = np.array([[1, 2, 3], [4, 5, 6]])
print("\nOriginal 2D data:")
print(data_2d)

# Process as 1D (e.g., apply complex transformation)
flat = data_2d.ravel()
processed = flat * 2 + 10  # Some operation
print("Processed (flat):", processed)

# Reshape back to original structure
result_2d = processed.reshape(data_2d.shape)
print("Reshaped back to 2D:")
print(result_2d)