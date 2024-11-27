import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Set up the figure
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111, projection='3d')

# Earth's axial tilt (23.5 degrees)
axial_tilt = 23.5
tilt_rad = np.deg2rad(axial_tilt)

# Create rotation matrix for Earth's tilt
rotation_matrix = np.array([
    [1, 0, 0],
    [0, np.cos(tilt_rad), -np.sin(tilt_rad)],
    [0, np.sin(tilt_rad), np.cos(tilt_rad)]
])

# Create Earth sphere
def create_sphere(radius, resolution=50):
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
    return x, y, z

# Create and rotate Earth
radius = 1
x, y, z = create_sphere(radius)

# Apply tilt to Earth's coordinates
for i in range(x.shape[0]):
    for j in range(x.shape[1]):
        point = np.array([x[i,j], y[i,j], z[i,j]])
        rotated_point = np.dot(rotation_matrix, point)
        x[i,j], y[i,j], z[i,j] = rotated_point

earth = ax.plot_surface(x, y, z, color='lightblue', alpha=0.8)

# Create magnetic field lines
def create_field_line(start_phi, r_factor=1.5):
    t = np.linspace(-np.pi/2, np.pi/2, 100)
    r = r_factor * (1 + np.cos(t)**2)
    x = r * np.cos(t) * np.cos(start_phi)
    y = r * np.cos(t) * np.sin(start_phi)
    z = r * np.sin(t)
    
    # Create points array for rotation
    points = np.array([x, y, z])
    
    # Rotate all points
    rotated_points = np.zeros_like(points)
    for i in range(points.shape[1]):
        rotated_points[:,i] = np.dot(rotation_matrix, points[:,i])
    
    return rotated_points[0], rotated_points[1], rotated_points[2]

# Generate multiple field lines
for phi in np.linspace(0, 2*np.pi, 16):
    x, y, z = create_field_line(phi)
    ax.plot(x, y, z, 'gold', alpha=0.5, linewidth=1.5)

# Plot Earth's rotational axis (tilted)
axis_length = 1.3
rotated_axis_top = np.dot(rotation_matrix, [0, 0, axis_length])
rotated_axis_bottom = np.dot(rotation_matrix, [0, 0, -axis_length])
ax.plot([rotated_axis_bottom[0], rotated_axis_top[0]], 
        [rotated_axis_bottom[1], rotated_axis_top[1]], 
        [rotated_axis_bottom[2], rotated_axis_top[2]], 
        'b--', linewidth=2, label='Earth\'s Axis')

# Add magnetic poles (tilted)
magnetic_north = np.dot(rotation_matrix, [0, 0, 1.1])
magnetic_south = np.dot(rotation_matrix, [0, 0, -1.1])
ax.scatter([magnetic_north[0]], [magnetic_north[1]], [magnetic_north[2]], 
           color='red', s=100, label='Magnetic North Pole')
ax.scatter([magnetic_south[0]], [magnetic_south[1]], [magnetic_south[2]], 
           color='blue', s=100, label='Magnetic South Pole')

# Add labels with rotated positions
ax.text(magnetic_north[0], magnetic_north[1]+0.2, magnetic_north[2], 
        'N. Magnetic Pole', color='red')
ax.text(magnetic_south[0], magnetic_south[1]+0.2, magnetic_south[2], 
        'S. Magnetic Pole', color='blue')

# Add angle annotation for Earth's tilt
angle_radius = 0.5
t = np.linspace(0, tilt_rad, 100)
x_angle = angle_radius * np.sin(t)
z_angle = angle_radius * np.cos(t)
ax.plot(x_angle, np.zeros_like(t), z_angle, 'k-', linewidth=1)
ax.text(angle_radius*0.3, 0.2, angle_radius*0.7, f'{axial_tilt}°', fontsize=10)

# Set plot parameters
ax.set_box_aspect([1,1,1])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title("Earth's Axial Tilt with Aligned Magnetic Field")

# Set viewing angle for better visualization
ax.view_init(elev=20, azim=45)

# Add legend
ax.legend()

# Set axis limits
ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
ax.set_zlim([-2, 2])

plt.show()