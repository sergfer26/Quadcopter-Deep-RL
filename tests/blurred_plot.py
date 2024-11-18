import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# Generate some sample data
np.random.seed(0)
data = np.random.normal(loc=0, scale=1, size=1000)

# Compute the first quartile (Q1), median (Q2), and third quartile (Q3)
Q1 = np.percentile(data, 25)
Q3 = np.percentile(data, 75)
IQR = Q3 - Q1

# Define the x-axis values
x_values = np.linspace(0, 1, len(data))

# Create an array to represent the IQR range
iqr_data = np.where((data >= Q1) & (data <= Q3), data, np.nan)

# Apply Gaussian filter to blur the IQR plot
blurred_iqr_data = gaussian_filter(iqr_data, sigma=5, mode='constant')

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x_values, data, color='lightgray', label='Data', alpha=0.6)
plt.plot(x_values, blurred_iqr_data, color='blue',
         label='Blurred IQR', linewidth=3)

# Highlight the IQR region
plt.fill_between(x_values, Q1, Q3, color='blue', alpha=0.3, label='IQR Region')

# Add labels and legend
plt.title('Interquartile Range (IQR) with Blurred Style')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()
