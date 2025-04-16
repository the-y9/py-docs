#%% 
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display, clear_output
import time
import math

# Create some data for plotting
x = np.linspace(0, 10, 100)
y1 = np.sin(np.tan(x))
y2 = np.cos(math.pi*(np.sin(x/2)))
y3 = np.tan(x)

# Create the figure and subplots
fig, axs = plt.subplots(2, 2, figsize=(10, 6))

# First row - 1 plot that spans both columns
ax1 = fig.add_subplot(2, 2, (1, 2))  # This plot spans two columns
line1, = ax1.plot([], [], label='Training Loss')
line2, = ax1.plot([], [], color='red', label='Validation Loss')
ax1.legend(loc='upper right')
ax1.grid(True)
ax1.set_xlim(0, 10)
ax1.set_ylim(-2, 2)
ax1.set_title('Plot 1 (Row 1, Spanning 2 columns)')

# Second row - 2 plots
ax2 = fig.add_subplot(2, 2, 3)
line3, = ax2.plot([], [], color='blue')
ax2.set_xlim(0, 10)
ax2.set_ylim(-2, 2)
ax2.set_title('Plot 2 (Row 2, Column 1)')
ax2.grid(True)

ax3 = fig.add_subplot(2, 2, 4)
line4, = ax3.plot([], [], color='green')
ax3.set_xlim(0, 10)
ax3.set_ylim(-10, 10)
ax3.set_title('Plot 3 (Row 2, Column 2)')
ax3.grid(True)

# Loop through frames and update the plots
for frame in range(1, len(x)+1):
    # Update each line with new data
    line1.set_data(x[:frame], y1[:frame])
    line2.set_data(x[:frame], y2[:frame])
    line3.set_data(x[:frame], y2[:frame])  # For Plot 2 (Row 2, Column 1)
    line4.set_data(x[:frame], y3[:frame])  # For Plot 3 (Row 2, Column 2)
    
    # Redraw the plot and clear the previous output
    clear_output(wait=True)
    display(fig)
    
    # Pause for a short time to control animation speed
    time.sleep(0.05)

# Display the final plot after the loop finishes
plt.show()
