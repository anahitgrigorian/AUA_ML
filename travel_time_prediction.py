import numpy as np
import matplotlib.pyplot as plt

file_path = '/home/agrigoryan/Desktop/AUA_ML/Gasprice.txt'
data = np.loadtxt(file_path, delimiter=',')

# Extract columns from data
miles_driven = data[:, 0]
gallons_used = data[:, 1]
price_per_gallon = data[:, 2]
total_cost = data[:, 3]

# Perform calculations 
cost_per_mile = total_cost / miles_driven

# Print summary statistics
print(f"Total Data Points: {len(data)}")
print(f"Average Miles Driven: {np.mean(miles_driven):.2f}")
print(f"Average Gallons Used: {np.mean(gallons_used):.2f}")
print(f"Average Price per Gallon: {np.mean(price_per_gallon):.2f}")
print(f"Average Total Cost: {np.mean(total_cost):.2f}")
print(f"Average Cost per Mile: {np.mean(cost_per_mile):.2f}")

# Plot results
plt.figure(figsize=(10, 6))

# Scatter plot for Miles Driven vs Total Cost
plt.subplot(2, 1, 1)
plt.scatter(miles_driven, total_cost, color='blue')
plt.title('Miles Driven vs Total Cost')
plt.xlabel('Miles Driven')
plt.ylabel('Total Cost')

# Scatter plot for Cost per Mile vs Price per Gallon
plt.subplot(2, 1, 2)
plt.scatter(price_per_gallon, cost_per_mile, color='green')
plt.title('Price per Gallon vs Cost per Mile')
plt.xlabel('Price per Gallon')
plt.ylabel('Cost per Mile')

# Show the plots
plt.tight_layout()
plt.show()
