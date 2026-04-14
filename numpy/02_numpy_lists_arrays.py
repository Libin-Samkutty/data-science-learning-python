import numpy as np
import urllib.request

# ============================================================
# PART 1: Creating Arrays from Python Lists
# ============================================================

# 1D array (vector)
simple_array = np.array([10, 20, 30, 40, 50])
print("1D Array:", simple_array)
print("Shape:", simple_array.shape)  # (5,) means 1D with 5 elements
print("Dimensions:", simple_array.ndim)  # 1 dimension

# 2D array (matrix) - nested lists
matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
print("\n2D Array:\n", matrix)
print("Shape:", matrix.shape)  # (3, 3) means 3 rows, 3 columns
print("Dimensions:", matrix.ndim)  # 2 dimensions

# 3D array (cube of data) - useful for images, video
cube = np.array([
    [[1, 2], [3, 4]],
    [[5, 6], [7, 8]]
])
print("\n3D Array:\n", cube)
print("Shape:", cube.shape)  # (2, 2, 2)
print("Dimensions:", cube.ndim)  # 3 dimensions

# ============================================================
# PART 2: Creating Arrays with Built-in Functions
# ============================================================

# Zeros - initialize with zeros (common for accumulation)
zeros = np.zeros(5)
print("\nZeros (1D):", zeros)

zeros_matrix = np.zeros((3, 4))  # 3 rows, 4 columns - note the TUPLE
print("Zeros (2D):\n", zeros_matrix)

# Ones - initialize with ones
ones = np.ones((2, 3))
print("\nOnes (2D):\n", ones)

# Full - initialize with a specific value
# Real-world: setting default temperature, price, etc.
default_temp = np.full((7,), 20.0)  # 7 days, default 20°C
print("\nDefault temperatures for a week:", default_temp)

# Empty - fastest, but contains garbage values (use carefully!)
# Real-world: when you'll immediately overwrite all values
empty = np.empty(3)
print("\nEmpty array (random garbage):", empty)

# ============================================================
# PART 3: Range-Based Arrays (Critical for Data Work)
# ============================================================

# arange - like Python's range(), but returns array
# Syntax: np.arange(start, stop, step)
sequence = np.arange(0, 10, 2)  # 0, 2, 4, 6, 8 (stop is exclusive)
print("\nArange (0 to 10, step 2):", sequence)

# Common use: indices for large datasets
indices = np.arange(1000)  # 0, 1, 2, ..., 999
print("First 10 indices:", indices[:10])

# linspace - evenly spaced values between start and stop
# Syntax: np.linspace(start, stop, num_points)
# Real-world: time series, plotting, interpolation
time_points = np.linspace(0, 10, 5)  # 5 values from 0 to 10 (INCLUSIVE)
print("\nLinspace (0 to 10, 5 points):", time_points)

# Example: 24-hour timeline with measurements every 4 hours
hours = np.linspace(0, 24, 7)  # 0, 4, 8, 12, 16, 20, 24
print("Hourly measurements:", hours)

# ============================================================
# PART 4: Random Arrays (Testing, Simulation, Sampling)
# ============================================================

# ALWAYS set seed for reproducibility in production code
np.random.seed(42)

# Random floats between 0 and 1
random_floats = np.random.random(5)
print("\nRandom floats [0, 1):", random_floats)

# Random integers
# Syntax: np.random.randint(low, high, size)
dice_rolls = np.random.randint(1, 7, 10)  # Simulate 10 dice rolls
print("10 dice rolls:", dice_rolls)

# Random from normal distribution (Gaussian)
# Real-world: simulating measurement errors, natural phenomena
# Syntax: np.random.normal(mean, std_dev, size)
heights = np.random.normal(170, 10, 1000)  # 1000 heights, mean=170cm, std=10cm
print(f"\nSimulated heights - Mean: {heights.mean():.1f}cm, Std: {heights.std():.1f}cm")

# Random choice - sampling from existing data
cities = np.array(['Delhi', 'Mumbai', 'Bangalore', 'Chennai'])
sampled_cities = np.random.choice(cities, 10)  # Random sample of 10
print("\nRandomly sampled cities:", sampled_cities)

# ============================================================
# PART 5: Identity and Diagonal Matrices (Linear Algebra)
# ============================================================

# Identity matrix - 1s on diagonal, 0s elsewhere
identity = np.eye(4)
print("\n4x4 Identity Matrix:\n", identity)

# Diagonal matrix - custom values on diagonal
diagonal_values = np.array([1, 2, 3])
diag_matrix = np.diag(diagonal_values)
print("\nDiagonal Matrix:\n", diag_matrix)

# ============================================================
# PART 6: Loading REAL Data from CSV into Arrays
# ============================================================

# Download Iris dataset
url = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv'
filename = 'dataset/iris.csv'

print("\n--- Loading Real Dataset (Iris) ---")
print(f"Downloading from: {url}")

# Download the file
urllib.request.urlretrieve(url, filename)
print(f"✓ Downloaded to {filename}")

# Load CSV data - skip header row, use only numeric columns
# Iris CSV structure: sepal_length, sepal_width, petal_length, petal_width, species
iris_data = np.genfromtxt(
    filename,
    delimiter=',',
    skip_header=1,  # Skip column names
    usecols=(0, 1, 2, 3),  # Only numeric columns (skip 'species' for now)
    dtype=float
)

print(f"\nIris dataset shape: {iris_data.shape}")
print(f"Dimensions: {iris_data.ndim}D array")
print(f"Total elements: {iris_data.size}")

# Show first 5 rows
print("\nFirst 5 flowers (sepal_length, sepal_width, petal_length, petal_width):")
print(iris_data[:5])

# Basic statistics
print(f"\nAverage sepal length: {iris_data[:, 0].mean():.2f} cm")
print(f"Max petal length: {iris_data[:, 2].max():.2f} cm")

# ============================================================
# PART 7: Creating Arrays from Existing Arrays
# ============================================================

original = np.array([1, 2, 3, 4, 5])

# zeros_like, ones_like - same shape as another array
zeros_copy = np.zeros_like(original)
ones_copy = np.ones_like(original)

print("\n--- Creating from Existing Arrays ---")
print("Original:", original)
print("Zeros with same shape:", zeros_copy)
print("Ones with same shape:", ones_copy)

# Real-world use: creating result arrays that match input dimensions
sales_data = np.array([100, 200, 150, 300])
profit_margin = 0.2
profits = np.zeros_like(sales_data, dtype=float)  # Prepare results array
profits = sales_data * profit_margin  # Calculate profits
print("\nSales:", sales_data)
print("Profits (20% margin):", profits)

# ============================================================
# PART 8: Data Type Specification (Important for Memory)
# ============================================================

# Default integer
default_int = np.array([1, 2, 3])
print(f"\nDefault int dtype: {default_int.dtype}")  # Usually int64

# Specify smaller integer (save memory)
small_int = np.array([1, 2, 3], dtype=np.int8)  # -128 to 127
print(f"int8 dtype: {small_int.dtype}")
print(f"Memory: {small_int.nbytes} bytes vs {default_int.nbytes} bytes")

# Floats
float_array = np.array([1.1, 2.2, 3.3], dtype=np.float32)
print(f"\nFloat32 dtype: {float_array.dtype}")

# Boolean
bool_array = np.array([True, False, True])
print(f"Bool dtype: {bool_array.dtype}")