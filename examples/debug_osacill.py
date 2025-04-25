import pandas as pd

# Load the CSV file
filename = "oscillator_8bit.csv"  # Replace with your actual file path
df = pd.read_csv(filename)

# Find rows where |u_sr| > 50 or |v_sr| > 50
large_sr_rows = df[(df['u_sr'].abs() > 1.0) | (df['v_sr'].abs() > 1.0)]

# Print matching rows
if large_sr_rows.empty:
    print("No rows found with |u_sr| or |v_sr| greater than 50.")
else:
    print("Rows with large u_sr or v_sr (> 50 in magnitude):")
    print(large_sr_rows)
