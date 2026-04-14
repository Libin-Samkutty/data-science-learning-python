import numpy as np
import urllib.request

# ============================================================
# PART 1: Boolean Mask Basics
# ============================================================

print("="*60)
print("PART 1: BOOLEAN MASK BASICS")
print("="*60)

print("""
A boolean mask is an array of True/False values that matches
the shape of your data array. It's used to select elements
that meet certain conditions.

Condition → Boolean Mask → Apply to Array → Filtered Result
""")

# Simple example
data = np.array([10, 25, 30, 15, 45, 50, 20, 35])
print(f"Data: {data}")

# Create a boolean mask
mask = data > 25
print(f"\nCondition: data > 25")
print(f"Boolean mask: {mask}")
print(f"Mask dtype: {mask.dtype}")

# Apply the mask to filter data
filtered = data[mask]
print(f"\nFiltered result: {filtered}")
print(f"Count: {len(filtered)} values meet the condition")

# One-liner (most common pattern)
print(f"\nOne-liner: data[data > 25] = {data[data > 25]}")

# ============================================================
# PART 2: Comparison Operators
# ============================================================

print("\n" + "="*60)
print("PART 2: COMPARISON OPERATORS")
print("="*60)

scores = np.array([65, 72, 85, 90, 55, 78, 92, 88, 45, 100])
print(f"Test scores: {scores}")

# All comparison operators
print("\n--- All Comparison Operators ---")
print(f"scores > 80:  {scores > 80}")
print(f"scores >= 80: {scores >= 80}")
print(f"scores < 60:  {scores < 60}")
print(f"scores <= 60: {scores <= 60}")
print(f"scores == 85: {scores == 85}")
print(f"scores != 85: {scores != 85}")

# Using masks to filter
print("\n--- Filtering with Masks ---")
print(f"Scores above 80: {scores[scores > 80]}")
print(f"Failing scores (< 60): {scores[scores < 60]}")
print(f"Perfect scores (== 100): {scores[scores == 100]}")

# Counting
print("\n--- Counting with Masks ---")
passing = scores >= 60
print(f"Number passing (>= 60): {passing.sum()}")
print(f"Number failing (< 60): {(~passing).sum()}")
print(f"Pass rate: {passing.mean() * 100:.1f}%")

# ============================================================
# PART 3: Combining Conditions (AND, OR, NOT)
# ============================================================

print("\n" + "="*60)
print("PART 3: COMBINING CONDITIONS (AND, OR, NOT)")
print("="*60)

print("""
⚠️  IMPORTANT: Use bitwise operators, not Python keywords!
   AND: &  (not 'and')
   OR:  |  (not 'or')
   NOT: ~  (not 'not')
   
   Always use PARENTHESES around each condition!
""")

ages = np.array([15, 22, 35, 45, 18, 62, 28, 55, 12, 40])
print(f"Ages: {ages}")

# AND condition (&)
print("\n--- AND Condition (&) ---")
# Working adults (18-65)
working_age = (ages >= 18) & (ages <= 65)
print(f"Working age (18-65): {working_age}")
print(f"Working age people: {ages[working_age]}")

# OR condition (|)
print("\n--- OR Condition (|) ---")
# Youth or senior
youth_or_senior = (ages < 18) | (ages > 60)
print(f"Youth (<18) or Senior (>60): {youth_or_senior}")
print(f"Youth or seniors: {ages[youth_or_senior]}")

# NOT condition (~)
print("\n--- NOT Condition (~) ---")
# Not working age
not_working_age = ~working_age
print(f"Not working age: {not_working_age}")
print(f"Non-working age people: {ages[not_working_age]}")

# Complex conditions
print("\n--- Complex Conditions ---")
salaries = np.array([30000, 45000, 75000, 120000, 35000, 90000, 55000, 150000, 25000, 80000])
print(f"Ages: {ages}")
print(f"Salaries: {salaries}")

# High earners (>= 70000) who are young (< 40)
young_high_earners = (salaries >= 70000) & (ages < 40)
print(f"\nYoung high earners (salary >= 70k AND age < 40):")
print(f"Mask: {young_high_earners}")
print(f"Ages: {ages[young_high_earners]}")
print(f"Salaries: {salaries[young_high_earners]}")

# Either high earner OR senior
high_or_senior = (salaries >= 70000) | (ages > 60)
print(f"\nHigh earner OR senior:")
print(f"Count: {high_or_senior.sum()}")

# ============================================================
# PART 4: Common Pitfall - Operator Precedence
# ============================================================

print("\n" + "="*60)
print("PART 4: COMMON PITFALL - OPERATOR PRECEDENCE")
print("="*60)

data = np.array([10, 20, 30, 40, 50])

# WRONG - missing parentheses
print("--- Common Mistake ---")
print("WRONG: data > 15 & data < 45")
try:
    result = data > 15 & data < 45  # & has higher precedence than >
    print(f"This would give unexpected results!")
except Exception as e:
    print(f"Error: {e}")

# CORRECT - with parentheses
print("\nCORRECT: (data > 15) & (data < 45)")
result = (data > 15) & (data < 45)
print(f"Result: {result}")
print(f"Filtered: {data[result]}")

# ============================================================
# PART 5: Using np.where() - Conditional Selection
# ============================================================

print("\n" + "="*60)
print("PART 5: np.where() - CONDITIONAL SELECTION")
print("="*60)

print("""
np.where() has two main uses:
1. np.where(condition) - Returns indices where condition is True
2. np.where(condition, x, y) - Returns x where True, y where False
""")

scores = np.array([65, 72, 85, 90, 55, 78, 92, 88, 45, 100])
print(f"Scores: {scores}")

# Usage 1: Get indices
print("\n--- Usage 1: Get Indices ---")
high_score_indices = np.where(scores >= 80)
print(f"Indices where score >= 80: {high_score_indices}")
print(f"Scores at those indices: {scores[high_score_indices]}")

# Usage 2: Conditional value selection
print("\n--- Usage 2: Conditional Value Selection ---")
# Create grade labels: 'Pass' if >= 60, else 'Fail'
pass_fail = np.where(scores >= 60, 'Pass', 'Fail')
print(f"Pass/Fail: {pass_fail}")

# Create numeric grades
grade_points = np.where(scores >= 90, 4.0,
               np.where(scores >= 80, 3.0,
               np.where(scores >= 70, 2.0,
               np.where(scores >= 60, 1.0, 0.0))))
print(f"Grade points: {grade_points}")

# Usage 3: Replace values conditionally
print("\n--- Usage 3: Conditional Replacement ---")
prices = np.array([10.0, 25.0, 5.0, 30.0, 8.0, 40.0])
print(f"Original prices: {prices}")

# Apply 20% discount to items over $20
discounted = np.where(prices > 20, prices * 0.8, prices)
print(f"After discount (>$20 items): {discounted}")

# Cap values at maximum
capped = np.where(prices > 25, 25, prices)
print(f"Capped at $25: {capped}")

# ============================================================
# PART 6: Real-World Example - Iris Dataset
# ============================================================

print("\n" + "="*60)
print("PART 6: REAL-WORLD - IRIS DATASET")
print("="*60)

# Download Iris dataset
url = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv'
filename = 'iris.csv'

print(f"Downloading Iris dataset...")
urllib.request.urlretrieve(url, filename)
print(f"✓ Downloaded")

# Load the data
# Columns: sepal_length, sepal_width, petal_length, petal_width, species
data = np.genfromtxt(filename, delimiter=',', skip_header=1, 
                     usecols=(0, 1, 2, 3), dtype=float)
species = np.genfromtxt(filename, delimiter=',', skip_header=1, 
                        usecols=(4,), dtype=str)

print(f"\nDataset shape: {data.shape}")
print(f"Columns: sepal_length, sepal_width, petal_length, petal_width")
print(f"Species: {np.unique(species)}")

# Extract columns for clarity
sepal_length = data[:, 0]
sepal_width = data[:, 1]
petal_length = data[:, 2]
petal_width = data[:, 3]

# Basic statistics
print("\n--- Basic Statistics ---")
print(f"Sepal length range: {sepal_length.min():.1f} - {sepal_length.max():.1f} cm")
print(f"Petal length range: {petal_length.min():.1f} - {petal_length.max():.1f} cm")

# Filter by species
print("\n--- Filtering by Species ---")
setosa_mask = species == 'setosa'
versicolor_mask = species == 'versicolor'
virginica_mask = species == 'virginica'

print(f"Setosa count: {setosa_mask.sum()}")
print(f"Versicolor count: {versicolor_mask.sum()}")
print(f"Virginica count: {virginica_mask.sum()}")

# Compare species characteristics
print("\n--- Species Comparison ---")
print("Average petal length:")
print(f"  Setosa: {petal_length[setosa_mask].mean():.2f} cm")
print(f"  Versicolor: {petal_length[versicolor_mask].mean():.2f} cm")
print(f"  Virginica: {petal_length[virginica_mask].mean():.2f} cm")

# Filter by measurements
print("\n--- Filtering by Measurements ---")

# Large flowers (sepal_length > 6 AND petal_length > 5)
large_flowers = (sepal_length > 6) & (petal_length > 5)
print(f"Large flowers (sepal > 6cm AND petal > 5cm): {large_flowers.sum()}")
print(f"Species of large flowers: {np.unique(species[large_flowers])}")

# Small petals (petal_length < 2)
small_petals = petal_length < 2
print(f"\nSmall petals (< 2cm): {small_petals.sum()}")
print(f"All small petals are: {np.unique(species[small_petals])}")

# Wide sepals (sepal_width > 3.5)
wide_sepals = sepal_width > 3.5
print(f"\nWide sepals (> 3.5cm): {wide_sepals.sum()}")

# Identify potential Virginica (large petal)
print("\n--- Species Prediction Based on Measurements ---")
likely_virginica = (petal_length > 5) & (petal_width > 1.8)
print(f"Likely Virginica (petal_length > 5 AND petal_width > 1.8):")
print(f"  Predicted count: {likely_virginica.sum()}")
print(f"  Actual Virginica: {(species[likely_virginica] == 'virginica').sum()}")
print(f"  Accuracy: {(species[likely_virginica] == 'virginica').mean() * 100:.1f}%")

# ============================================================
# PART 7: Real-World Example - E-commerce Transactions
# ============================================================

print("\n" + "="*60)
print("PART 7: REAL-WORLD - E-COMMERCE TRANSACTIONS")
print("="*60)

# Simulate e-commerce data
np.random.seed(42)
n_transactions = 1000

transaction_ids = np.arange(1, n_transactions + 1)
amounts = np.random.gamma(50, 2, n_transactions)  # Skewed distribution
quantities = np.random.randint(1, 10, n_transactions)
customer_ages = np.random.randint(18, 75, n_transactions)
is_member = np.random.choice([True, False], n_transactions, p=[0.3, 0.7])
categories = np.random.choice(['Electronics', 'Clothing', 'Books', 'Home', 'Food'], 
                              n_transactions)

print(f"Transaction count: {n_transactions}")
print(f"Sample data (first 5):")
print(f"  Amounts: {amounts[:5].round(2)}")
print(f"  Quantities: {quantities[:5]}")
print(f"  Customer ages: {customer_ages[:5]}")
print(f"  Is member: {is_member[:5]}")
print(f"  Categories: {categories[:5]}")

# Analysis 1: High-value transactions
print("\n--- Analysis 1: High-Value Transactions ---")
high_value_threshold = np.percentile(amounts, 90)
high_value = amounts >= high_value_threshold
print(f"High-value threshold (90th percentile): ${high_value_threshold:.2f}")
print(f"High-value transactions: {high_value.sum()}")
print(f"Total high-value revenue: ${amounts[high_value].sum():,.2f}")
print(f"% of total revenue: {amounts[high_value].sum() / amounts.sum() * 100:.1f}%")

# Analysis 2: Member vs Non-member
print("\n--- Analysis 2: Member vs Non-member ---")
member_amounts = amounts[is_member]
non_member_amounts = amounts[~is_member]
print(f"Members: {is_member.sum()}")
print(f"  Average order: ${member_amounts.mean():.2f}")
print(f"  Total revenue: ${member_amounts.sum():,.2f}")
print(f"Non-members: {(~is_member).sum()}")
print(f"  Average order: ${non_member_amounts.mean():.2f}")
print(f"  Total revenue: ${non_member_amounts.sum():,.2f}")

# Analysis 3: Age segments
print("\n--- Analysis 3: Age Segments ---")
young = customer_ages < 30
middle = (customer_ages >= 30) & (customer_ages < 50)
senior = customer_ages >= 50

segments = [
    ("Young (< 30)", young),
    ("Middle (30-49)", middle),
    ("Senior (50+)", senior)
]

for name, mask in segments:
    count = mask.sum()
    avg_amount = amounts[mask].mean()
    total = amounts[mask].sum()
    print(f"{name}: {count} customers, avg ${avg_amount:.2f}, total ${total:,.2f}")

# Analysis 4: Complex filtering
print("\n--- Analysis 4: Complex Filtering ---")

# High-value member transactions in Electronics
complex_filter = (amounts >= 100) & is_member & (categories == 'Electronics')
print(f"High-value member Electronics orders: {complex_filter.sum()}")
print(f"Total revenue: ${amounts[complex_filter].sum():,.2f}")

# Bulk purchases by young customers
bulk_young = (quantities >= 5) & (customer_ages < 30)
print(f"\nBulk purchases (qty >= 5) by young customers: {bulk_young.sum()}")
print(f"Average amount: ${amounts[bulk_young].mean():.2f}")

# Analysis 5: Category breakdown
print("\n--- Analysis 5: Category Analysis ---")
unique_categories = np.unique(categories)

for cat in unique_categories:
    cat_mask = categories == cat
    count = cat_mask.sum()
    total_rev = amounts[cat_mask].sum()
    avg_amount = amounts[cat_mask].mean()
    member_pct = is_member[cat_mask].mean() * 100
    print(f"{cat:12} - Count: {count:3}, Revenue: ${total_rev:8,.2f}, "
          f"Avg: ${avg_amount:6.2f}, Members: {member_pct:.1f}%")

# ============================================================
# PART 8: np.select() - Multiple Conditions
# ============================================================

print("\n" + "="*60)
print("PART 8: np.select() - MULTIPLE CONDITIONS")
print("="*60)

print("""
np.select() is better than nested np.where() for multiple conditions.
Syntax: np.select(conditions_list, choices_list, default=value)
""")

# Assign customer tiers based on purchase amount
print("--- Customer Tier Assignment ---")
print(f"Transaction amounts (first 20): {amounts[:20].round(2)}")

# Define conditions
conditions = [
    amounts >= 150,                         # Premium
    (amounts >= 100) & (amounts < 150),     # Gold
    (amounts >= 50) & (amounts < 100),      # Silver
    amounts < 50                            # Bronze
]

# Define corresponding values
choices = ['Premium', 'Gold', 'Silver', 'Bronze']

# Apply
tiers = np.select(conditions, choices, default='Unknown')

print(f"\nAssigned tiers (first 20): {tiers[:20]}")

# Count tiers
print("\n--- Tier Distribution ---")
for tier in ['Premium', 'Gold', 'Silver', 'Bronze']:
    tier_mask = tiers == tier
    count = tier_mask.sum()
    pct = count / len(tiers) * 100
    avg_amount = amounts[tier_mask].mean()
    print(f"{tier:8} - Count: {count:3} ({pct:5.1f}%), Avg amount: ${avg_amount:.2f}")

# Assign discount based on tier and membership
print("\n--- Conditional Discount Assignment ---")
discount_conditions = [
    (tiers == 'Premium') & is_member,       # 20% discount
    (tiers == 'Premium') & ~is_member,      # 15% discount
    (tiers == 'Gold') & is_member,          # 15% discount
    (tiers == 'Gold') & ~is_member,         # 10% discount
    is_member,                               # 5% discount
]
discount_choices = [0.20, 0.15, 0.15, 0.10, 0.05]

discounts = np.select(discount_conditions, discount_choices, default=0.0)

print(f"Sample discounts: {discounts[:20]}")
print(f"\nDiscount distribution:")
for disc in [0.20, 0.15, 0.10, 0.05, 0.0]:
    count = (discounts == disc).sum()
    pct = count / len(discounts) * 100
    print(f"  {disc*100:5.1f}%: {count:3} transactions ({pct:.1f}%)")

# Calculate discounted amounts
final_amounts = amounts * (1 - discounts)
savings = amounts - final_amounts

print(f"\nTotal savings from discounts: ${savings.sum():,.2f}")
print(f"Average savings per transaction: ${savings.mean():.2f}")

# ============================================================
# PART 9: Advanced Masking Techniques
# ============================================================

print("\n" + "="*60)
print("PART 9: ADVANCED MASKING TECHNIQUES")
print("="*60)

# Technique 1: np.isin() - Check membership in set
print("--- Technique 1: np.isin() ---")
premium_categories = ['Electronics', 'Home']
is_premium_cat = np.isin(categories, premium_categories)
print(f"Premium categories: {premium_categories}")
print(f"Transactions in premium categories: {is_premium_cat.sum()}")
print(f"Revenue from premium categories: ${amounts[is_premium_cat].sum():,.2f}")

# Technique 2: np.logical_and(), np.logical_or() - Reduce multiple masks
print("\n--- Technique 2: Reduce Multiple Masks ---")
# AND across multiple conditions
conditions_list = [
    amounts > 50,
    is_member,
    customer_ages < 40
]

combined_and = np.logical_and.reduce(conditions_list)
print(f"All conditions met: {combined_and.sum()}")

# OR across multiple conditions
combined_or = np.logical_or.reduce(conditions_list)
print(f"At least one condition met: {combined_or.sum()}")

# Technique 3: Using masks for assignment
print("\n--- Technique 3: Mask-Based Assignment ---")
status = np.full(n_transactions, 'Regular', dtype='<U20')
status[is_member] = 'Member'
status[high_value] = 'VIP'
status[high_value & is_member] = 'VIP Member'

print(f"Status distribution:")
for s in np.unique(status):
    count = (status == s).sum()
    print(f"  {s}: {count}")

# Technique 4: Percentile-based filtering
print("\n--- Technique 4: Percentile-Based Filtering ---")
p25 = np.percentile(amounts, 25)
p75 = np.percentile(amounts, 75)
iqr_mask = (amounts >= p25) & (amounts <= p75)

print(f"25th percentile: ${p25:.2f}")
print(f"75th percentile: ${p75:.2f}")
print(f"Within IQR (middle 50%): {iqr_mask.sum()}")

# Outlier detection
lower_bound = p25 - 1.5 * (p75 - p25)
upper_bound = p75 + 1.5 * (p75 - p25)
outliers = (amounts < lower_bound) | (amounts > upper_bound)
print(f"\nOutliers (IQR method): {outliers.sum()}")
print(f"Outlier values: {amounts[outliers].round(2)}")

# ============================================================
# PART 10: Real-World Example - Air Quality Monitoring
# ============================================================

print("\n" + "="*60)
print("PART 10: REAL-WORLD - AIR QUALITY MONITORING")
print("="*60)

# Simulate hourly air quality readings for 30 days
np.random.seed(42)
hours = 24 * 30  # 720 hours

# PM2.5 levels (μg/m³) - with some spikes
pm25 = np.random.gamma(15, 2, hours)
# Add pollution spikes
pm25[100:110] = np.random.uniform(100, 150, 10)  # Industrial event
pm25[500:520] = np.random.uniform(80, 120, 20)   # Weather event

# Temperature (°C)
temperature = 20 + 10 * np.sin(np.linspace(0, 30*2*np.pi, hours) / 24) + np.random.randn(hours) * 3

# Humidity (%)
humidity = 60 + 20 * np.cos(np.linspace(0, 30*2*np.pi, hours) / 24) + np.random.randn(hours) * 5
humidity = np.clip(humidity, 20, 100)

print(f"Air quality data: {hours} hourly readings over 30 days")
print(f"\nPM2.5 (μg/m³): min={pm25.min():.1f}, max={pm25.max():.1f}, mean={pm25.mean():.1f}")
print(f"Temperature (°C): min={temperature.min():.1f}, max={temperature.max():.1f}")
print(f"Humidity (%): min={humidity.min():.1f}, max={humidity.max():.1f}")

# Air Quality Index categories
print("\n--- Air Quality Classification ---")
good = pm25 <= 12
moderate = (pm25 > 12) & (pm25 <= 35)
unhealthy_sensitive = (pm25 > 35) & (pm25 <= 55)
unhealthy = (pm25 > 55) & (pm25 <= 150)
very_unhealthy = (pm25 > 150) & (pm25 <= 250)
hazardous = pm25 > 250

print("AQI Categories:")
print(f"  Good (0-12): {good.sum()} hours ({good.mean()*100:.1f}%)")
print(f"  Moderate (12-35): {moderate.sum()} hours ({moderate.mean()*100:.1f}%)")
print(f"  Unhealthy for Sensitive (35-55): {unhealthy_sensitive.sum()} hours")
print(f"  Unhealthy (55-150): {unhealthy.sum()} hours")
print(f"  Very Unhealthy (150-250): {very_unhealthy.sum()} hours")
print(f"  Hazardous (250+): {hazardous.sum()} hours")

# Health alerts
print("\n--- Health Alerts ---")
alert_threshold = 55  # Unhealthy level
alerts = pm25 > alert_threshold
alert_indices = np.where(alerts)[0]

print(f"Hours with health alerts: {alerts.sum()}")
if len(alert_indices) > 0:
    print(f"First alert at hour: {alert_indices[0]} (Day {alert_indices[0]//24 + 1})")
    print(f"Max PM2.5 during alerts: {pm25[alerts].max():.1f} μg/m³")
    
    # Consecutive alert hours
    alert_diff = np.diff(alert_indices)
    consecutive = np.sum(alert_diff == 1)
    print(f"Consecutive alert hours: {consecutive}")

# Weather correlation
print("\n--- Weather During Poor Air Quality ---")
poor_air = pm25 > 35
print(f"During poor air quality (PM2.5 > 35):")
print(f"  Average temperature: {temperature[poor_air].mean():.1f}°C")
print(f"  Average humidity: {humidity[poor_air].mean():.1f}%")
print(f"During good air quality (PM2.5 <= 12):")
print(f"  Average temperature: {temperature[good].mean():.1f}°C")
print(f"  Average humidity: {humidity[good].mean():.1f}%")

# Critical conditions: Poor air + extreme weather
print("\n--- Critical Conditions ---")
hot_and_poor = (pm25 > 35) & (temperature > 30)
humid_and_poor = (pm25 > 35) & (humidity > 80)

print(f"Hot (>30°C) AND poor air quality: {hot_and_poor.sum()} hours")
print(f"Humid (>80%) AND poor air quality: {humid_and_poor.sum()} hours")

# ============================================================
# PART 11: Performance Considerations
# ============================================================

print("\n" + "="*60)
print("PART 11: PERFORMANCE CONSIDERATIONS")
print("="*60)

import time

# Create large array
large_array = np.random.randn(10_000_000)

# Method 1: Boolean indexing (vectorized)
print("--- Method 1: Boolean Indexing (Vectorized) ---")
start = time.time()
result1 = large_array[large_array > 0]
time1 = time.time() - start
print(f"Time: {time1:.4f} seconds")
print(f"Result count: {len(result1):,}")

# Method 2: np.where() + indexing
print("\n--- Method 2: np.where() + Indexing ---")
start = time.time()
indices = np.where(large_array > 0)[0]
result2 = large_array[indices]
time2 = time.time() - start
print(f"Time: {time2:.4f} seconds")

# Method 3: Python loop (SLOW - don't do this!)
print("\n--- Method 3: Python Loop (DON'T DO THIS!) ---")
start = time.time()
result3 = []
for x in large_array[:100000]:  # Only 100k to save time
    if x > 0:
        result3.append(x)
result3 = np.array(result3)
time3 = time.time() - start
print(f"Time for 100k elements: {time3:.4f} seconds")
print(f"Estimated time for 10M: {time3 * 100:.2f} seconds")

print(f"\n✅ Boolean indexing is fastest!")
print(f"   Speedup vs loop: ~{(time3 * 100) / time1:.0f}x")

# ============================================================
# PART 12: Best Practices Summary
# ============================================================

print("\n" + "="*60)
print("PART 12: BEST PRACTICES SUMMARY")
print("="*60)

print("""
✅ DO:
1. Use parentheses around each condition: (a > 5) & (b < 10)
2. Use bitwise operators: & | ~ (not: and or not)
3. Use np.isin() for checking membership in sets
4. Use np.select() for multiple conditions
5. Use np.where() for conditional value assignment
6. Combine masks with & | ~ for complex filtering
7. Store reusable masks in variables

❌ DON'T:
1. Forget parentheses: a > 5 & b < 10 (WRONG!)
2. Use Python keywords: and, or, not (won't work element-wise)
3. Use loops for filtering (very slow)
4. Create unnecessary intermediate arrays
5. Ignore NaN handling in masks

COMMON PATTERNS:

# Filtering
filtered = arr[arr > threshold]

# Counting
count = (arr > threshold).sum()

# Percentage
pct = (arr > threshold).mean() * 100

# Conditional assignment
result = np.where(condition, value_if_true, value_if_false)

# Multiple conditions
result = np.select([cond1, cond2, cond3], [val1, val2, val3], default)

# Membership test
mask = np.isin(arr, [val1, val2, val3])

# Get indices
indices = np.where(condition)[0]

# Complex filter
mask = (cond1) & (cond2) | ~(cond3)
result = arr[mask]
""")

# Clean up
import os
if os.path.exists('iris.csv'):
    os.remove('iris.csv')

print("\n" + "="*60)
print("LESSON 11 COMPLETE!")
print("="*60)