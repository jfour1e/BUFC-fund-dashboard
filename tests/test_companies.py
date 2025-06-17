from src.companies import companies, sector_designations

print("=== Testing companies.py ===")
print(f"Number of companies: {len(companies)}")
print(f"First 3 sectors: {list(sector_designations.items())[:3]}")

if isinstance(companies, list) and isinstance(sector_designations, dict):
    print("companies.py looks good!")
else:
    print("Check companies.py for errors")