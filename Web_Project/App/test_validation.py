import pandas as pd
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_processor import DataProcessor

# Get parent directory for sample data
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Test with insufficient rows
print("="*60)
print("TEST 1: Insufficient rows (only 3 rows)")
print("="*60)
df1 = pd.DataFrame({'A': [1,2,3], 'B': [4,5,6]})
print(f'Dataset: {len(df1)} rows, {len(df1.columns)} columns')
errors1 = DataProcessor.validate_data(df1)
if errors1:
    print("Validation FAILED with errors:")
    for e in errors1:
        print(f"  ❌ {e}")
else:
    print("✅ Validation PASSED")

# Test with valid data
print("\n" + "="*60)
print("TEST 2: Valid dataset (20 rows)")
print("="*60)
df2 = pd.read_csv(os.path.join(parent_dir, 'sample_data', 'housing.csv'))
print(f'Dataset: {len(df2)} rows, {len(df2.columns)} columns')
errors2 = DataProcessor.validate_data(df2)
if errors2:
    print("Validation FAILED with errors:")
    for e in errors2:
        print(f"  ❌ {e}")
else:
    print("✅ Validation PASSED")

# Test with insufficient numeric columns
print("\n" + "="*60)
print("TEST 3: Insufficient numeric columns (text only)")
print("="*60)
df3 = pd.DataFrame({'Name': ['Alice', 'Bob', 'Charlie']*5, 'City': ['NYC', 'LA', 'SF']*5})
print(f'Dataset: {len(df3)} rows, {len(df3.columns)} columns')
print(f'Numeric columns: {df3.select_dtypes(include=["number"]).columns.tolist()}')
errors3 = DataProcessor.validate_data(df3)
if errors3:
    print("Validation FAILED with errors:")
    for e in errors3:
        print(f"  ❌ {e}")
else:
    print("✅ Validation PASSED")

print("\n" + "="*60)
print("All tests completed!")
print("="*60)
