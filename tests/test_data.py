from src.data_get import load_clean_holdings

print("=== Testing data_get.py ===")
try:
    df = load_clean_holdings("../BUFC_May_2025_Allocations.xlsx")
    print(f"Loaded {len(df)} holdings successfully")
    print("Columns:", df.columns.tolist())
except Exception as e:
    print(f"Failed to load data: {e}")