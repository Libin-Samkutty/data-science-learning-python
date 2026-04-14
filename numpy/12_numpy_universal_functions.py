import numpy as np

# ============================================================
# PART 1: What Are Universal Functions (ufuncs)?
# ============================================================

print("="*60)
print("PART 1: WHAT ARE UNIVERSAL FUNCTIONS (ufuncs)?")
print("="*60)

print("""
Universal Functions (ufuncs) are NumPy functions that:
1. Operate ELEMENT-WISE on arrays
2. Are implemented in compiled C code (FAST!)
3. Support broadcasting
4. Can output to pre-allocated arrays
5. Have special methods: reduce, accumulate, outer

Regular Python math → Loops (slow)
NumPy ufuncs → Vectorized operations (fast!)
""")

# Demonstrate the difference
import time

# Create large array
large_array = np.random.randn(1_000_000)

# Method 1: Python loop with math module
import math
print("--- Performance Comparison ---")

start = time.time()
result_loop = [math.sqrt(abs(x)) for x in large_array]
loop_time = time.time() - start

# Method 2: NumPy ufunc
start = time.time()
result_ufunc = np.sqrt(np.abs(large_array))
ufunc_time = time.time() - start

print(f"Python loop: {loop_time:.4f} seconds")
print(f"NumPy ufunc: {ufunc_time:.4f} seconds")
print(f"Speedup: {loop_time/ufunc_time:.1f}x faster!")

# ============================================================
# PART 2: Basic Arithmetic Operations
# ============================================================

print("\n" + "="*60)
print("PART 2: BASIC ARITHMETIC OPERATIONS")
print("="*60)

a = np.array([10, 20, 30, 40, 50])
b = np.array([2, 4, 5, 8, 10])

print(f"a = {a}")
print(f"b = {b}")

# Basic operations (all are ufuncs!)
print("\n--- Basic Arithmetic ---")
print(f"a + b = {a + b}  (np.add)")
print(f"a - b = {a - b}  (np.subtract)")
print(f"a * b = {a * b}  (np.multiply)")
print(f"a / b = {a / b}  (np.divide)")
print(f"a // b = {a // b}  (np.floor_divide)")
print(f"a % b = {a % b}  (np.mod - remainder)")
print(f"a ** b = {a ** b}  (np.power)")

# Equivalent function calls
print("\n--- Equivalent Function Calls ---")
print(f"np.add(a, b) = {np.add(a, b)}")
print(f"np.multiply(a, b) = {np.multiply(a, b)}")

# Unary operations
print("\n--- Unary Operations ---")
c = np.array([-5, -2, 0, 3, 7])
print(f"c = {c}")
print(f"np.abs(c) = {np.abs(c)}  (absolute value)")
print(f"np.negative(c) = {np.negative(c)}  (negation)")
print(f"np.sign(c) = {np.sign(c)}  (sign: -1, 0, or 1)")
print(f"np.reciprocal(a.astype(float)) = {np.reciprocal(a.astype(float))}  (1/x)")

# ============================================================
# PART 3: Power and Exponential Functions
# ============================================================

print("\n" + "="*60)
print("PART 3: POWER AND EXPONENTIAL FUNCTIONS")
print("="*60)

x = np.array([1, 2, 3, 4, 5])
print(f"x = {x}")

print("\n--- Power Functions ---")
print(f"np.square(x) = {np.square(x)}  (x²)")
print(f"np.sqrt(x) = {np.sqrt(x)}  (√x)")
print(f"np.cbrt(x) = {np.cbrt(x)}  (∛x - cube root)")
print(f"np.power(x, 3) = {np.power(x, 3)}  (x³)")

print("\n--- Exponential Functions ---")
print(f"np.exp(x) = {np.exp(x)}  (eˣ)")
print(f"np.exp2(x) = {np.exp2(x)}  (2ˣ)")
print(f"np.expm1(x) = {np.expm1(x)}  (eˣ - 1, more accurate for small x)")

print("\n--- Logarithmic Functions ---")
y = np.array([1, 2, 10, 100, 1000])
print(f"y = {y}")
print(f"np.log(y) = {np.log(y)}  (natural log, ln)")
print(f"np.log2(y) = {np.log2(y)}  (log base 2)")
print(f"np.log10(y) = {np.log10(y)}  (log base 10)")
print(f"np.log1p(x) = {np.log1p(x)}  (ln(1+x), more accurate for small x)")

# Real-world: Compound interest
print("\n--- Real-World: Compound Interest ---")
principal = 10000
rates = np.array([0.03, 0.05, 0.07, 0.10])  # 3%, 5%, 7%, 10%
years = 10

# Future value: FV = P * e^(r*t) for continuous compounding
future_values = principal * np.exp(rates * years)
print(f"Principal: ${principal:,}")
print(f"Rates: {rates * 100}%")
print(f"Years: {years}")
print(f"Future values (continuous compounding):")
for rate, fv in zip(rates, future_values):
    print(f"  {rate*100:.0f}%: ${fv:,.2f}")

# ============================================================
# PART 4: Trigonometric Functions
# ============================================================

print("\n" + "="*60)
print("PART 4: TRIGONOMETRIC FUNCTIONS")
print("="*60)

# Angles in degrees
degrees = np.array([0, 30, 45, 60, 90, 180, 270, 360])
print(f"Degrees: {degrees}")

# Convert to radians (trig functions expect radians)
radians = np.deg2rad(degrees)
print(f"Radians: {radians.round(4)}")

print("\n--- Basic Trig Functions ---")
print(f"sin: {np.sin(radians).round(4)}")
print(f"cos: {np.cos(radians).round(4)}")
print(f"tan: {np.tan(radians).round(4)}")

print("\n--- Inverse Trig Functions ---")
values = np.array([-1, -0.5, 0, 0.5, 1])
print(f"Values: {values}")
print(f"arcsin (degrees): {np.rad2deg(np.arcsin(values)).round(2)}")
print(f"arccos (degrees): {np.rad2deg(np.arccos(values)).round(2)}")

print("\n--- Hyperbolic Functions ---")
x = np.array([-2, -1, 0, 1, 2])
print(f"x = {x}")
print(f"sinh(x) = {np.sinh(x).round(4)}")
print(f"cosh(x) = {np.cosh(x).round(4)}")
print(f"tanh(x) = {np.tanh(x).round(4)}")

# Real-world: Wave generation
print("\n--- Real-World: Wave Generation ---")
# Generate 1 second of a 440 Hz sine wave (A note)
sample_rate = 44100  # samples per second
duration = 0.01  # 10 milliseconds for display
frequency = 440  # Hz

t = np.linspace(0, duration, int(sample_rate * duration))
wave = np.sin(2 * np.pi * frequency * t)

print(f"Generating {frequency} Hz sine wave")
print(f"Samples: {len(t)}")
print(f"First 10 values: {wave[:10].round(4)}")
print(f"Wave range: [{wave.min():.4f}, {wave.max():.4f}]")

# ============================================================
# PART 5: Rounding and Precision Functions
# ============================================================

print("\n" + "="*60)
print("PART 5: ROUNDING AND PRECISION FUNCTIONS")
print("="*60)

values = np.array([-2.7, -1.5, -0.3, 0.5, 1.2, 2.5, 3.7])
print(f"Values: {values}")

print("\n--- Rounding Functions ---")
print(f"np.round(values): {np.round(values)}  (round to nearest)")
print(f"np.floor(values): {np.floor(values)}  (round down)")
print(f"np.ceil(values): {np.ceil(values)}  (round up)")
print(f"np.trunc(values): {np.trunc(values)}  (truncate toward zero)")
print(f"np.rint(values): {np.rint(values)}  (round to nearest integer)")

print("\n--- Rounding to Decimals ---")
precise = np.array([3.14159265, 2.71828183, 1.41421356])
print(f"Original: {precise}")
print(f"Round to 2 decimals: {np.round(precise, 2)}")
print(f"Round to 4 decimals: {np.round(precise, 4)}")

print("\n--- Special Cases: Banker's Rounding ---")
# NumPy uses "round half to even" (banker's rounding)
edge_cases = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
print(f"Edge cases: {edge_cases}")
print(f"Rounded: {np.round(edge_cases)}")
print("Note: 0.5→0, 1.5→2, 2.5→2, 3.5→4 (rounds to even)")

print("\n--- Modulo and Remainder ---")
a = np.array([10, 11, 12, 13, 14, 15])
print(f"a = {a}")
print(f"a % 3 = {a % 3}  (modulo)")
print(f"np.fmod(a, 3) = {np.fmod(a, 3)}  (same for positive)")

# Difference with negative numbers
neg = np.array([-7])
print(f"\nFor -7 % 3:")
print(f"  np.mod(-7, 3) = {np.mod(-7, 3)}  (Python style)")
print(f"  np.fmod(-7, 3) = {np.fmod(-7, 3)}  (C style)")

# ============================================================
# PART 6: Comparison and Logical ufuncs
# ============================================================

print("\n" + "="*60)
print("PART 6: COMPARISON AND LOGICAL UFUNCS")
print("="*60)

a = np.array([1, 5, 3, 8, 2])
b = np.array([2, 3, 3, 6, 7])

print(f"a = {a}")
print(f"b = {b}")

print("\n--- Comparison ufuncs ---")
print(f"np.greater(a, b): {np.greater(a, b)}  (a > b)")
print(f"np.greater_equal(a, b): {np.greater_equal(a, b)}  (a >= b)")
print(f"np.less(a, b): {np.less(a, b)}  (a < b)")
print(f"np.less_equal(a, b): {np.less_equal(a, b)}  (a <= b)")
print(f"np.equal(a, b): {np.equal(a, b)}  (a == b)")
print(f"np.not_equal(a, b): {np.not_equal(a, b)}  (a != b)")

print("\n--- Element-wise Min/Max ---")
print(f"np.maximum(a, b): {np.maximum(a, b)}  (element-wise max)")
print(f"np.minimum(a, b): {np.minimum(a, b)}  (element-wise min)")

print("\n--- Logical ufuncs ---")
x = np.array([True, True, False, False])
y = np.array([True, False, True, False])
print(f"x = {x}")
print(f"y = {y}")
print(f"np.logical_and(x, y): {np.logical_and(x, y)}")
print(f"np.logical_or(x, y): {np.logical_or(x, y)}")
print(f"np.logical_xor(x, y): {np.logical_xor(x, y)}")
print(f"np.logical_not(x): {np.logical_not(x)}")

# ============================================================
# PART 7: Special Mathematical Functions
# ============================================================

print("\n" + "="*60)
print("PART 7: SPECIAL MATHEMATICAL FUNCTIONS")
print("="*60)

print("--- Mathematical Constants ---")
print(f"π (pi): {np.pi}")
print(f"e (Euler's number): {np.e}")
print(f"∞ (infinity): {np.inf}")
print(f"NaN (Not a Number): {np.nan}")

print("\n--- Absolute Value and Sign ---")
complex_arr = np.array([3+4j, 1-1j, -2+0j])
print(f"Complex array: {complex_arr}")
print(f"np.abs (magnitude): {np.abs(complex_arr)}")  # √(real² + imag²)
print(f"np.angle (radians): {np.angle(complex_arr)}")
print(f"np.angle (degrees): {np.rad2deg(np.angle(complex_arr))}")

print("\n--- Special Functions ---")
# Clip values to range
data = np.array([-5, -2, 0, 3, 7, 15])
print(f"Data: {data}")
print(f"np.clip(data, 0, 10): {np.clip(data, 0, 10)}")

# Hypotenuse (useful for distance calculations)
x = np.array([3, 5, 8])
y = np.array([4, 12, 15])
print(f"\nx = {x}, y = {y}")
print(f"np.hypot(x, y) = {np.hypot(x, y)}  (√(x² + y²))")

# Factorial (for integers only, use scipy for large)
print(f"\nFactorials of [1,2,3,4,5]: {[math.factorial(i) for i in range(1,6)]}")

# ============================================================
# PART 8: ufunc Methods - reduce, accumulate, outer
# ============================================================

print("\n" + "="*60)
print("PART 8: UFUNC METHODS - reduce, accumulate, outer")
print("="*60)

print("""
ufuncs have special methods:
- reduce: Apply operation across array, returning single value
- accumulate: Apply operation cumulatively, keeping intermediate results
- outer: Apply operation to all pairs of elements
""")

arr = np.array([1, 2, 3, 4, 5])
print(f"Array: {arr}")

# reduce - collapse array using operation
print("\n--- reduce (collapse to single value) ---")
print(f"np.add.reduce(arr) = {np.add.reduce(arr)}  (sum: 1+2+3+4+5)")
print(f"np.multiply.reduce(arr) = {np.multiply.reduce(arr)}  (product: 1*2*3*4*5)")
print(f"np.maximum.reduce(arr) = {np.maximum.reduce(arr)}  (max)")
print(f"np.minimum.reduce(arr) = {np.minimum.reduce(arr)}  (min)")

# accumulate - cumulative operation
print("\n--- accumulate (keep intermediate results) ---")
print(f"np.add.accumulate(arr) = {np.add.accumulate(arr)}  (cumsum)")
print(f"np.multiply.accumulate(arr) = {np.multiply.accumulate(arr)}  (cumprod)")
print(f"np.maximum.accumulate(arr) = {np.maximum.accumulate(arr)}  (running max)")

# outer - all pairs
print("\n--- outer (all pairs) ---")
a = np.array([1, 2, 3])
b = np.array([10, 20, 30])
print(f"a = {a}")
print(f"b = {b}")
print(f"np.add.outer(a, b):")
print(np.add.outer(a, b))
print(f"\nnp.multiply.outer(a, b):")
print(np.multiply.outer(a, b))

# Real-world: Distance matrix
print("\n--- Real-World: Creating Distance Differences ---")
locations = np.array([0, 5, 12, 20])  # 1D positions
print(f"Locations: {locations}")
distances = np.abs(np.subtract.outer(locations, locations))
print(f"Distance matrix:")
print(distances)

# ============================================================
# PART 9: Real-World Example - Projectile Motion
# ============================================================

print("\n" + "="*60)
print("PART 9: REAL-WORLD - PROJECTILE MOTION")
print("="*60)

# Physics simulation: Projectile motion
# y(t) = y0 + v0*sin(θ)*t - 0.5*g*t²
# x(t) = x0 + v0*cos(θ)*t

g = 9.81  # gravity (m/s²)
v0 = 50   # initial velocity (m/s)
angles_deg = np.array([15, 30, 45, 60, 75])  # launch angles

print(f"Initial velocity: {v0} m/s")
print(f"Launch angles: {angles_deg}°")
print(f"Gravity: {g} m/s²")

# Convert to radians
angles_rad = np.deg2rad(angles_deg)

# Calculate flight time (when y returns to 0)
# 0 = v0*sin(θ)*t - 0.5*g*t²
# t = 2*v0*sin(θ)/g
flight_times = 2 * v0 * np.sin(angles_rad) / g

# Calculate horizontal range
# range = v0*cos(θ)*t = v0²*sin(2θ)/g
ranges = v0**2 * np.sin(2 * angles_rad) / g

# Calculate maximum height
# max_height = (v0*sin(θ))² / (2*g)
max_heights = (v0 * np.sin(angles_rad))**2 / (2 * g)

print("\n--- Projectile Analysis ---")
print(f"{'Angle':>6} {'Flight Time':>12} {'Range':>10} {'Max Height':>12}")
print("-" * 44)
for angle, t, r, h in zip(angles_deg, flight_times, ranges, max_heights):
    print(f"{angle:>5}° {t:>10.2f} s {r:>9.2f} m {h:>10.2f} m")

# Find optimal angle for maximum range
optimal_idx = np.argmax(ranges)
print(f"\nOptimal angle for max range: {angles_deg[optimal_idx]}°")
print(f"Maximum range: {ranges[optimal_idx]:.2f} m")

# Generate trajectory for 45° launch
print("\n--- Trajectory for 45° Launch ---")
theta = np.deg2rad(45)
t_flight = 2 * v0 * np.sin(theta) / g
t = np.linspace(0, t_flight, 20)

x = v0 * np.cos(theta) * t
y = v0 * np.sin(theta) * t - 0.5 * g * t**2

print(f"Time points: {len(t)}")
print(f"x positions: {x.round(1)}")
print(f"y positions: {y.round(1)}")

# ============================================================
# PART 10: Real-World Example - Financial Calculations
# ============================================================

print("\n" + "="*60)
print("PART 10: REAL-WORLD - FINANCIAL CALCULATIONS")
print("="*60)

# Stock returns analysis
np.random.seed(42)
n_days = 252  # Trading days in a year

# Simulate daily returns (log-normal distribution)
daily_returns = np.random.normal(0.0005, 0.02, n_days)  # ~12% annual, 2% daily std

print(f"Simulated {n_days} trading days")
print(f"Daily returns (first 10): {daily_returns[:10].round(4)}")

# Calculate cumulative returns
# If r1, r2, r3 are daily returns, cumulative = (1+r1)*(1+r2)*(1+r3) - 1
cumulative_multiplier = np.cumprod(1 + daily_returns)
cumulative_returns = cumulative_multiplier - 1

print(f"\n--- Return Analysis ---")
print(f"Total return: {cumulative_returns[-1]*100:.2f}%")
print(f"Average daily return: {daily_returns.mean()*100:.4f}%")
print(f"Daily volatility (std): {daily_returns.std()*100:.4f}%")
print(f"Annualized volatility: {daily_returns.std() * np.sqrt(252) * 100:.2f}%")

# Risk metrics
print(f"\n--- Risk Metrics ---")
# Maximum drawdown
peak = np.maximum.accumulate(cumulative_multiplier)
drawdown = (cumulative_multiplier - peak) / peak
max_drawdown = drawdown.min()
print(f"Maximum drawdown: {max_drawdown*100:.2f}%")

# Value at Risk (VaR) - 95% confidence
var_95 = np.percentile(daily_returns, 5)
print(f"95% VaR (daily): {var_95*100:.2f}%")
print(f"  Interpretation: 95% of days, loss won't exceed {-var_95*100:.2f}%")

# Sharpe Ratio (assuming 2% risk-free rate)
risk_free_daily = 0.02 / 252
sharpe = (daily_returns.mean() - risk_free_daily) / daily_returns.std() * np.sqrt(252)
print(f"Sharpe Ratio (annualized): {sharpe:.2f}")

# Log returns (used in quantitative finance)
print(f"\n--- Log Returns ---")
prices = 100 * cumulative_multiplier  # Starting price = 100
log_returns = np.log(prices[1:] / prices[:-1])
print(f"Log returns (first 10): {log_returns[:10].round(4)}")
print(f"Sum of log returns: {log_returns.sum():.4f}")
print(f"Total return from log: {(np.exp(log_returns.sum()) - 1)*100:.2f}%")

# ============================================================
# PART 11: Real-World Example - Geographic Distance
# ============================================================

print("\n" + "="*60)
print("PART 11: REAL-WORLD - GEOGRAPHIC DISTANCE")
print("="*60)

# Calculate distances between cities using Haversine formula
# Haversine: d = 2r * arcsin(√(sin²(Δlat/2) + cos(lat1)*cos(lat2)*sin²(Δlon/2)))

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points on Earth (in km)"""
    R = 6371  # Earth's radius in km
    
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(np.deg2rad, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c

# Major cities (lat, lon)
cities = {
    'New York': (40.7128, -74.0060),
    'London': (51.5074, -0.1278),
    'Tokyo': (35.6762, 139.6503),
    'Sydney': (-33.8688, 151.2093),
    'Mumbai': (19.0760, 72.8777)
}

print("City Coordinates:")
for city, (lat, lon) in cities.items():
    print(f"  {city}: ({lat:.4f}°, {lon:.4f}°)")

# Calculate distance matrix
print("\n--- Distance Matrix (km) ---")
city_names = list(cities.keys())
n_cities = len(city_names)
distance_matrix = np.zeros((n_cities, n_cities))

for i, city1 in enumerate(city_names):
    for j, city2 in enumerate(city_names):
        lat1, lon1 = cities[city1]
        lat2, lon2 = cities[city2]
        distance_matrix[i, j] = haversine_distance(lat1, lon1, lat2, lon2)

# Print matrix
print(f"{'':>12}", end='')
for name in city_names:
    print(f"{name[:8]:>10}", end='')
print()

for i, name in enumerate(city_names):
    print(f"{name:>12}", end='')
    for j in range(n_cities):
        print(f"{distance_matrix[i,j]:>10.0f}", end='')
    print()

# Find closest and farthest pairs
mask = np.triu(np.ones((n_cities, n_cities), dtype=bool), k=1)
distances_upper = distance_matrix[mask]
pairs_upper = [(city_names[i], city_names[j]) 
               for i in range(n_cities) for j in range(i+1, n_cities)]

min_idx = np.argmin(distances_upper)
max_idx = np.argmax(distances_upper)

print(f"\nClosest cities: {pairs_upper[min_idx][0]} ↔ {pairs_upper[min_idx][1]}")
print(f"  Distance: {distances_upper[min_idx]:.0f} km")

print(f"\nFarthest cities: {pairs_upper[max_idx][0]} ↔ {pairs_upper[max_idx][1]}")
print(f"  Distance: {distances_upper[max_idx]:.0f} km")

# ============================================================
# PART 12: Creating Custom ufuncs
# ============================================================

print("\n" + "="*60)
print("PART 12: CREATING CUSTOM UFUNCS")
print("="*60)

# Method 1: np.frompyfunc (simple but returns object dtype)
print("--- Method 1: np.frompyfunc ---")

def custom_sigmoid(x):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-x))

# This already works element-wise because it uses numpy functions
x = np.array([-2, -1, 0, 1, 2])
print(f"x = {x}")
print(f"sigmoid(x) = {custom_sigmoid(x).round(4)}")

# Method 2: np.vectorize (more control, still object dtype)
print("\n--- Method 2: np.vectorize ---")

def classify_score(score):
    """Classify a score into grade categories"""
    if score >= 90:
        return 'A'
    elif score >= 80:
        return 'B'
    elif score >= 70:
        return 'C'
    elif score >= 60:
        return 'D'
    else:
        return 'F'

# Vectorize the function
vectorized_classify = np.vectorize(classify_score)

scores = np.array([95, 82, 78, 65, 55, 91, 73])
grades = vectorized_classify(scores)
print(f"Scores: {scores}")
print(f"Grades: {grades}")

# Method 3: Use NumPy operations directly (BEST for performance)
print("\n--- Method 3: Direct NumPy Operations (Fastest) ---")

def relu_slow(x):
    """ReLU using vectorize - slow"""
    return np.vectorize(lambda v: max(0, v))(x)

def relu_fast(x):
    """ReLU using NumPy operations - fast"""
    return np.maximum(x, 0)

x = np.array([-2, -1, 0, 1, 2])
print(f"x = {x}")
print(f"ReLU(x) = {relu_fast(x)}")

# Performance comparison
large_x = np.random.randn(100_000)

start = time.time()
result1 = relu_slow(large_x)
slow_time = time.time() - start

start = time.time()
result2 = relu_fast(large_x)
fast_time = time.time() - start

print(f"\nPerformance (100k elements):")
print(f"  np.vectorize: {slow_time:.4f} seconds")
print(f"  np.maximum: {fast_time:.4f} seconds")
print(f"  Speedup: {slow_time/fast_time:.1f}x")

# ============================================================
# PART 13: Output Arrays and In-Place Operations
# ============================================================

print("\n" + "="*60)
print("PART 13: OUTPUT ARRAYS AND IN-PLACE OPERATIONS")
print("="*60)

# Pre-allocated output array (saves memory allocation)
print("--- Using Output Array ---")
a = np.array([1, 2, 3, 4, 5], dtype=float)
b = np.array([2, 3, 4, 5, 6], dtype=float)
result = np.empty_like(a)

np.multiply(a, b, out=result)
print(f"a = {a}")
print(f"b = {b}")
print(f"result = {result}")

# In-place operations
print("\n--- In-Place Operations ---")
data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
print(f"Original: {data}")

# These modify the array in-place
np.sqrt(data, out=data)
print(f"After sqrt (in-place): {data}")

# Using operators with in-place
data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
data += 10  # In-place addition
print(f"After += 10: {data}")

data *= 2  # In-place multiplication
print(f"After *= 2: {data}")

# Performance benefit
print("\n--- Memory Efficiency ---")
large_a = np.random.randn(1_000_000)
large_b = np.random.randn(1_000_000)
output = np.empty_like(large_a)

# Without pre-allocation (creates new array)
start = time.time()
for _ in range(100):
    result = large_a + large_b
no_prealloc_time = time.time() - start

# With pre-allocation (reuses memory)
start = time.time()
for _ in range(100):
    np.add(large_a, large_b, out=output)
prealloc_time = time.time() - start

print(f"Without pre-allocation: {no_prealloc_time:.4f} seconds")
print(f"With pre-allocation: {prealloc_time:.4f} seconds")
print(f"Speedup: {no_prealloc_time/prealloc_time:.2f}x")

# ============================================================
# PART 14: Complete ufunc Reference
# ============================================================

print("\n" + "="*60)
print("PART 14: UFUNC REFERENCE GUIDE")
print("="*60)

print("""
ARITHMETIC:
  np.add, np.subtract, np.multiply, np.divide
  np.floor_divide, np.mod, np.power, np.negative
  np.abs, np.sign, np.reciprocal

EXPONENTS & LOGS:
  np.exp, np.exp2, np.expm1
  np.log, np.log2, np.log10, np.log1p
  np.sqrt, np.square, np.cbrt, np.power

TRIGONOMETRIC:
  np.sin, np.cos, np.tan
  np.arcsin, np.arccos, np.arctan, np.arctan2
  np.sinh, np.cosh, np.tanh
  np.arcsinh, np.arccosh, np.arctanh
  np.deg2rad, np.rad2deg, np.hypot

ROUNDING:
  np.round, np.floor, np.ceil, np.trunc, np.rint
  np.fix (toward zero)

COMPARISON:
  np.greater, np.greater_equal, np.less, np.less_equal
  np.equal, np.not_equal
  np.maximum, np.minimum, np.fmax, np.fmin

LOGICAL:
  np.logical_and, np.logical_or, np.logical_xor, np.logical_not

SPECIAL:
  np.clip, np.sign, np.copysign
  np.isnan, np.isinf, np.isfinite
  np.nan_to_num

UFUNC METHODS:
  ufunc.reduce()     - Reduce to single value
  ufunc.accumulate() - Cumulative results
  ufunc.outer()      - All pairs operation
  ufunc.at()         - Unbuffered in-place operation

CONSTANTS:
  np.pi, np.e, np.inf, np.nan
  np.euler_gamma (Euler-Mascheroni constant)
""")

print("\n" + "="*60)
print("LESSON 12 COMPLETE!")
print("="*60)