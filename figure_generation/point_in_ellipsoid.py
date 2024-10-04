import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameters for the larger ellipsoid (semi-axis lengths)
a = 3  # semi-axis along x
b = 5  # semi-axis along y
c = 3  # semi-axis along z

# Create a grid of angles for theta (polar) and phi (azimuthal)
theta = np.linspace(0, np.pi, 100)
phi = np.linspace(0, 2 * np.pi, 100)
theta, phi = np.meshgrid(theta, phi)

# Parametric equations for the larger ellipsoid
x = a * np.sin(theta) * np.cos(phi)
y = b * np.sin(theta) * np.sin(phi)
z = c * np.cos(theta)

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the surface mesh for the larger ellipsoid with transparency (alpha)
ax.plot_surface(x, y, z, cmap='plasma', alpha=0.2)

# Plot a sparse wireframe for the larger ellipsoid
for i in range(0, 100, 20):  # Change the step to adjust the number of lines
    ax.plot(x[:, i], y[:, i], z[:, i], color='black', linewidth=0.1)  # Lines for a larger ellipsoid
    ax.plot(x[:, i], y[:, i], z[:, i], color='black', linewidth=0.1)  # Lines for a smaller ellipsoid
    
# Set the labels for clarity (optional, can be removed)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

# Make the axes have equal scale
max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0

mean_x = x.mean()
mean_y = y.mean()
mean_z = z.mean()

ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)  # Y-axis range from -4 to 10
ax.set_zlim(-10, 10)

# Add points at the top, bottom, front, back, left, and right limits of the larger ellipsoid
ax.scatter([0], [0], [c], color='red',    s=10)       # Top point
ax.scatter([0], [0], [-c], color='red',  s=10)  # Bottom point
ax.scatter([0], [b], [0], color='green',  s=10)   # Front point
ax.scatter([0], [-b], [0], color='green',s=10)  # Back point
ax.scatter([a], [0], [0], color='black', s=10)  # Right point
ax.scatter([-a], [0], [0], color='black',s=10)  # Left point

# Plot a line from the top to the bottom of the larger ellipsoid
ax.plot([0, 0], [0, 0], [c, -c], color='black', linewidth=1)

# Plot a line from the left to the right of the larger ellipsoid
ax.plot([-a, a], [0, 0], [0, 0], color='black', linewidth=1)
ax.plot([0, 0], [-b, b], [0, 0], color='black', linewidth=1)

# Add a point for the starting and ending positions (on the larger ellipsoid)
ax.scatter([0], [0], [0], color='black', s=30, label='Ellipsoid Center')
ax.scatter([-1], [-3], [1], color='purple', s=30, label='Target Coordinate')
ax.plot([0, -1], [0, -3], [0, 1], color='black', linestyle='dashed', linewidth=1)

# Bézier curve for the arc: Start [0, 5, -5], End [0, 0, 0], Control [0, 10, -10] for more curvature
t = np.linspace(0, 1, 100)
p0 = np.array([0, 10, -5])  # Starting point
p1 = np.array([0, 8, -0.05])  # Control point (adjusted for more curvature)
p2 = np.array([0, 0, 0])  # Ending point

# Add a legend to identify the points
# Create custom legend handles
trajectory_handle = plt.Line2D([0], [0], color='black', linestyle='dotted', lw=2, label='Trajectory')
reachable_set_limit_handle = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Reachable Set Limit')

# Add a legend to identify the ellipsoids
ax.legend()
# Hide the axes and grid
# ax.axis('off')  # Hide axes
ax.grid(False)  # Hide grid

# Show the plot
plt.show()
