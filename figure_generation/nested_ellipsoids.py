import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameters for the larger ellipsoid (semi-axis lengths)
a_large = 3  # semi-axis along x
b_large = 5  # semi-axis along y
c_large = 3  # semi-axis along z

# Parameters for the smaller ellipsoid (scaled down by a factor of 4)
a_small = a_large / 2  # semi-axis along x
b_small = b_large / 2  # semi-axis along y
c_small = c_large / 2  # semi-axis along z

# Create a grid of angles for theta (polar) and phi (azimuthal)
theta = np.linspace(0, np.pi, 100)
phi = np.linspace(0, 2 * np.pi, 100)
theta, phi = np.meshgrid(theta, phi)

# Parametric equations for the larger ellipsoid
x_large = a_large * np.sin(theta) * np.cos(phi)
y_large = b_large * np.sin(theta) * np.sin(phi)
z_large = c_large * np.cos(theta)

# Parametric equations for the smaller ellipsoid
x_small = a_small * np.sin(theta) * np.cos(phi)
y_small = b_small * np.sin(theta) * np.sin(phi)
z_small = c_small * np.cos(theta)

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the surface mesh for the larger ellipsoid with transparency (alpha)
ax.plot_surface(x_large, y_large, z_large, cmap='plasma', alpha=0.2)

# Plot the surface mesh for the smaller ellipsoid with transparency (alpha)
ax.plot_surface(x_small, y_small, z_small, cmap='viridis', alpha=0.8)

# Plot a sparse wireframe for the larger ellipsoid
for i in range(0, 100, 20):  # Change the step to adjust the number of lines
    ax.plot(x_large[:, i], y_large[:, i], z_large[:, i], color='black', linewidth=0.1)  # Lines for a larger ellipsoid
    ax.plot(x_small[:, i], y_small[:, i], z_small[:, i], color='black', linewidth=0.1)  # Lines for a smaller ellipsoid
    
# Set the labels for clarity (optional, can be removed)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

# Make the axes have equal scale
max_range = np.array([x_large.max()-x_large.min(), y_large.max()-y_large.min(), z_large.max()-z_large.min()]).max() / 2.0

mean_x = x_large.mean()
mean_y = y_large.mean()
mean_z = z_large.mean()

ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)  # Y-axis range from -4 to 10
ax.set_zlim(-10, 10)

# Add points at the top, bottom, front, back, left, and right limits of the larger ellipsoid
ax.scatter([0], [0], [c_large], color='red',    s=10)       # Top point
ax.scatter([0], [0], [-c_large], color='red',  s=10)  # Bottom point
ax.scatter([0], [b_large], [0], color='green',  s=10)   # Front point
ax.scatter([0], [-b_large], [0], color='green',s=10)  # Back point
ax.scatter([a_large], [0], [0], color='black', s=10)  # Right point
ax.scatter([-a_large], [0], [0], color='black',s=10)  # Left point

ax.scatter([0], [0], [c_small], color='red',   s=10)    # Top point
ax.scatter([0], [0], [-c_small], color='red',  s=10)  # Bottom point
ax.scatter([0], [b_small], [0], color='green', s=10)  # Front point
ax.scatter([0], [-b_small], [0], color='green',s=10)# Back point
ax.scatter([a_small], [0], [0], color='black', s=10) # Right point
ax.scatter([-a_small], [0], [0], color='black',s=10)# Left point



# Plot a line from the top to the bottom of the larger ellipsoid
ax.plot([0, 0], [0, 0], [c_large, -c_large], color='black', linewidth=2, label='Top-Bottom Line')

# Plot a line from the left to the right of the larger ellipsoid
ax.plot([-a_large, a_large], [0, 0], [0, 0], color='black', linewidth=2, label='Left-Right Line')
ax.plot([0, 0], [-b_large, b_large], [0, 0], color='black', linewidth=2, label='Left-Right Line')

# Add a point for the starting and ending positions (on the larger ellipsoid)
ax.scatter([0], [0], [0], color='black', s=50, label='End [0,0,0]')

# Bézier curve for the arc: Start [0, 5, -5], End [0, 0, 0], Control [0, 10, -10] for more curvature
t = np.linspace(0, 1, 100)
p0 = np.array([0, 10, -5])  # Starting point
p1 = np.array([0, 8, -0.05])  # Control point (adjusted for more curvature)
p2 = np.array([0, 0, 0])  # Ending point

# Quadratic Bézier curve formula: B(t) = (1-t)^2 * P0 + 2(1-t)t * P1 + t^2 * P2
arc = (1 - t) ** 2 * p0[:, np.newaxis] + 2 * (1 - t) * t * p1[:, np.newaxis] + t ** 2 * p2[:, np.newaxis]

# Plot the Bézier curve (arc) with a dotted line
ax.plot(arc[0], arc[1], arc[2], color='black', linestyle='dotted', linewidth=2, label='Dotted Arc')
# Add a legend to identify the points
# Create custom legend handles
ellipsoid_1_handle = plt.Line2D([0], [0], color='orange', lw=4, label='max ∆v = 1 km/s')
ellipsoid_2_handle = plt.Line2D([0], [0], color='blue', lw=4, label='max ∆v = 0.5 km/s')
trajectory_handle = plt.Line2D([0], [0], color='black', linestyle='dotted', lw=2, label='Trajectory')
reachable_set_limit_handle = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Reachable Set Limit')

# Add a legend to identify the ellipsoids
ax.legend(handles=[ellipsoid_1_handle, ellipsoid_2_handle, trajectory_handle, reachable_set_limit_handle], loc='upper left')
# Hide the axes and grid
# ax.axis('off')  # Hide axes
ax.grid(False)  # Hide grid

# Show the plot
plt.show()
