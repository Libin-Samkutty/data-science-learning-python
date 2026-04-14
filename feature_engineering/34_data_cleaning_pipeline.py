# ============================================================
# CAPSTONE: Complete Data Cleaning Pipeline
# ============================================================
# Bringing together EVERYTHING from all 5 phases
# ============================================================

import pandas as pd
import numpy as np

print("=" * 60)
print("CAPSTONE: End-to-End Data Cleaning Pipeline")
print("=" * 60)

# Load real dataset
url = "https://raw.githubusercontent.com/erkansirin78/datasets/master/AB_NYC_2019.csv"
df = pd.read_csv(url)
original_shape = df.shape
print(f"Raw data: {df.shape[0]} rows x {df.shape[1]} columns")

# ============================================================
# PHASE 1: Missing Data
# ============================================================
print("\n" + "=" * 60)
print("PHASE 1: Handle Missing Data")
print("=" * 60)

# Report
missing_report = (df.isnull().sum() / len(df) * 100).round(2)
print("Missing percentages:")
print(missing_report[missing_report > 0])

# Drop rows with missing critical fields
df = df.dropna(subset=['name', 'host_name'])

# Domain-knowledge imputation
df['reviews_per_month'] = df['reviews_per_month'].fillna(0.0)

# Create missingness flag for last_review (don't impute dates)
df['has_reviews'] = df['last_review'].notna().astype(int)

# Handle zero-price (likely invalid)
print(f"\nZero-price listings: {(df['price'] == 0).sum()}")
df.loc[df['price'] == 0, 'price'] = np.nan  # Mark as missing
# Impute with group median
df['price'] = df.groupby(['neighbourhood_group', 'room_type'])['price'].transform(
    lambda x: x.fillna(x.median())
)

print(f"[OK] Missing data handled. Remaining nulls: {df.isnull().sum().sum()}")

# ============================================================
# PHASE 2: Scaling (Prepare numeric features)
# ============================================================
print("\n" + "=" * 60)
print("PHASE 2: Scale Numeric Features")
print("=" * 60)

numeric_features = ['price', 'minimum_nights', 'number_of_reviews',
                    'reviews_per_month', 'calculated_host_listings_count',
                    'availability_365']

# Z-score standardization
for col in numeric_features:
    mean = df[col].mean()
    std = df[col].std()
    df[f'{col}_scaled'] = (df[col] - mean) / std if std > 0 else 0

scaled_cols = [f'{col}_scaled' for col in numeric_features]
print(f"[OK] {len(numeric_features)} features standardized")
print(f"   Mean ~ 0: {all(abs(df[scaled_cols].mean()) < 1e-10)}")

# ============================================================
# PHASE 3: Parse Dates
# ============================================================
print("\n" + "=" * 60)
print("PHASE 3: Parse Date Fields")
print("=" * 60)

df['last_review'] = pd.to_datetime(df['last_review'], errors='coerce')
df['review_year'] = df['last_review'].dt.year
df['review_month'] = df['last_review'].dt.month

print(f"[OK] Dates parsed. Type: {df['last_review'].dtype}")
valid_dates = df['last_review'].notna().sum()
print(f"   Valid dates: {valid_dates}, NaT: {df['last_review'].isna().sum()}")

# ============================================================
# PHASE 4: Check Encoding
# ============================================================
print("\n" + "=" * 60)
print("PHASE 4: Verify Text Encoding")
print("=" * 60)

# Check for encoding artifacts
text_cols = df.select_dtypes(include=['object', 'str']).columns
for col in text_cols:
    artifacts = df[col].astype(str).str.contains(r'[Ã¢â€™Â]', regex=True, na=False).sum()
    if artifacts > 0:
        print(f"  [WARN] '{col}': {artifacts} rows with encoding artifacts")
    else:
        print(f"  [OK] '{col}': no encoding issues")

# ============================================================
# PHASE 5: Standardize Text
# ============================================================
print("\n" + "=" * 60)
print("PHASE 5: Standardize Text Data")
print("=" * 60)

# Clean neighbourhood_group
df['neighbourhood_group'] = df['neighbourhood_group'].str.strip().str.title()
print(f"Neighbourhood groups: {df['neighbourhood_group'].unique()}")

# Clean room_type
df['room_type'] = df['room_type'].str.strip().str.lower()
print(f"Room types: {df['room_type'].unique()}")

# Clean name field
df['name'] = df['name'].str.strip()

# ============================================================
# FINAL VALIDATION
# ============================================================
print("\n" + "=" * 60)
print("[CHECK] FINAL VALIDATION REPORT")
print("=" * 60)

checks = {
    'No missing in critical columns': (
        df[['name', 'host_name', 'price']].isnull().sum().sum() == 0
    ),
    'Dates are datetime type': (
        str(df['last_review'].dtype).startswith('datetime64')
    ),
    'Scaled features have mean ~ 0': (
        all(abs(df[scaled_cols].mean()) < 1e-10)
    ),
    'No zero prices': (
        (df['price'] > 0).all() if df['price'].notna().all() else True
    ),
    'Consistent room types': (
        df['room_type'].nunique() <= 4
    ),
    'Consistent neighbourhoods': (
        df['neighbourhood_group'].nunique() <= 6
    ),
    'reviews_per_month >= 0': (
        (df['reviews_per_month'] >= 0).all()
    ),
}

all_passed = True
for check_name, passed in checks.items():
    status = "[OK] PASS" if passed else "[FAIL] FAIL"
    if not passed:
        all_passed = False
    print(f"  {status}: {check_name}")

print(f"\n{'=' * 60}")
print(f"Original: {original_shape[0]} rows x {original_shape[1]} columns")
print(f"Cleaned:  {df.shape[0]} rows x {df.shape[1]} columns")
print(f"Data retained: {(df.shape[0]/original_shape[0]*100):.1f}%")
print(f"New features added: {df.shape[1] - original_shape[1]}")
print(f"Overall: {'[OK] ALL CHECKS PASSED' if all_passed else '[FAIL] SOME CHECKS FAILED'}")
print(f"{'=' * 60}")

# Save clean data
# df.to_csv('airbnb_cleaned.csv', index=False, encoding='utf-8')
# print("\n[SAVE] Clean dataset saved to 'airbnb_cleaned.csv'")
