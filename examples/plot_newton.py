import pandas as pd

import matplotlib.pyplot as plt

# Load the CSV file
csv_file = 'newton_8bit.csv'  # Replace with the path to your CSV file
data = pd.read_csv(csv_file)

# Ensure the CSV has at least 4 columns
if data.shape[1] < 4:
    raise ValueError("The CSV file must have at least 4 columns.")

# Extract columns
x = data.iloc[:, 0]
y1 = data.iloc[:, 1]
y2 = data.iloc[:, 2]
y3 = data.iloc[:, 3]

# Get column names for labels
x_label = data.columns[0]
y1_label = data.columns[1]
y2_label = data.columns[2]
y3_label = data.columns[3]

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(x, y1, label=y1_label)
plt.plot(x, y2, label=y2_label)
plt.plot(x, y3, label=y3_label)

# Add labels, legend, and title
plt.xlabel(x_label)
plt.ylabel("Values")
plt.title("Plot of Functions")
plt.legend()

# Show the plot
plt.grid()
plt.tight_layout()
plt.show()