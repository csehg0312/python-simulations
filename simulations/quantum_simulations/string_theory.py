import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Set up the figure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Parameters
n_points = 20
n_dimensions = 4  # Extra dimensions beyond 3D
time_steps = 100

# Create points in higher dimensions
points = np.random.randn(n_points, 3 + n_dimensions)

# Function to project higher dimensions to 3D
def project_to_3d(points, time):
    projected = points.copy()
    
    # Add time-dependent rotations in higher dimensions
    for d in range(3, points.shape[1]):
        angle = time * (d-2) * np.pi / 180
        projected[:, 0] += np.sin(angle) * points[:, d]
        projected[:, 1] += np.cos(angle) * points[:, d]
        projected[:, 2] += np.sin(angle + np.pi/2) * points[:, d]
    
    return projected[:, :3]

# Animation update function
def update(frame):
    ax.clear()
    
    # Set axis labels and limits
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    ax.set_zlim([-3, 3])
    ax.set_title('String Theory Network Visualization')
    
    # Project points to 3D
    projected_points = project_to_3d(points, frame)
    
    # Plot points
    ax.scatter(projected_points[:, 0], 
              projected_points[:, 1], 
              projected_points[:, 2], 
              c='blue', alpha=0.6)
    
    # Connect points with strings
    for i in range(len(projected_points)):
        for j in range(i+1, len(projected_points)):
            dist = np.linalg.norm(projected_points[i] - projected_points[j])
            if dist < 2:  # Only connect nearby points
                ax.plot([projected_points[i,0], projected_points[j,0]],
                       [projected_points[i,1], projected_points[j,1]],
                       [projected_points[i,2], projected_points[j,2]],
                       color='green', alpha=0.2)

# Create animation
anim = FuncAnimation(fig, update, frames=360, interval=50, blit=False)
plt.show()