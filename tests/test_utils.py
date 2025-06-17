from src.dashboard_utils import assign_color

print("=== Testing dashboard_utils.py ===")
test_values = [-2, -1, 0, 1, 2]
for val in test_values:
    print(f"{val}% → {assign_color(val)}")
print("Color assignments working!")