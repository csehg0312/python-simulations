import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import LineCollection

# Set up the figure
fig = plt.figure(figsize=(15, 12))
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

# Create magnetic field lines with realistic compression and expansion
def create_field_line(start_phi, r_factor=1.5, compression_factor=0.8, expansion_factor=1.2):
    t = np.linspace(-np.pi/2, np.pi/2, 100)
    r = r_factor * (1 + np.cos(t)**2)
    x = r * np.cos(t) * np.cos(start_phi)
    y = r * np.cos(t) * np.sin(start_phi)
    z = r * np.sin(t)
    
    # Apply tilt rotation to the field line
    x_rot, y_rot, z_rot = np.zeros_like(x), np.zeros_like(y), np.zeros_like(z)
    for i in range(len(x)):
        # Calculate distortion based on position relative to Earth-Sun line
        distance_from_sun = y[i]
        
        # Compression on dayside (sun-facing side, where distance_from_sun is negative)
        if distance_from_sun < 0:
            distortion = 1 - compression_factor * np.exp(distance_from_sun)
        # Expansion on nightside (opposite side)
        else:
            distortion = 1 + expansion_factor * np.exp(-distance_from_sun)
        
        # Apply distortion and tilt rotation
        distorted_point = np.array([x[i] * distortion, y[i] * distortion, z[i] * distortion])
        rotated_point = np.dot(rotation_matrix, distorted_point)
        
        x_rot[i], y_rot[i], z_rot[i] = rotated_point
    
    return x_rot, y_rot, z_rot


def create_dipole_field_line(phi, r_0=1, compression_factor=0.6, expansion_factor=0.6, num_points=100):
    theta = np.linspace(-np.pi / 2, np.pi / 2, num_points)  # Polar angle from South to North
    r = r_0 / (np.cos(theta) ** 2)  # Magnetic dipole distance formula

    # Initialize field line in spherical coordinates, then adjust with compression/expansion
    x = r * np.cos(theta) * np.cos(phi)
    y = r * np.cos(theta) * np.sin(phi)
    z = r * np.sin(theta)
    
    # Apply tilt and calculate distorted field points
    x_distorted, y_distorted, z_distorted = np.zeros_like(x), np.zeros_like(y), np.zeros_like(z)
    for i in range(len(x)):
        distance_from_sun = y[i]  # Positive means nightside, negative means dayside
        # Apply compression on dayside and expansion on nightside
        if distance_from_sun < 0:  # Dayside
            distortion = 1 - compression_factor * np.exp(distance_from_sun)
        else:  # Nightside
            distortion = 1 + expansion_factor * np.exp(-distance_from_sun)

        # Distort the point and apply tilt rotation
        distorted_point = np.array([x[i] * distortion, y[i] * distortion, z[i] * distortion])
        rotated_point = np.dot(rotation_matrix, distorted_point)
        
        x_distorted[i], y_distorted[i], z_distorted[i] = rotated_point

    return x_distorted, y_distorted, z_distorted



# Generate magnetic field lines with realistic compression and expansion
for phi in np.linspace(0, 2 * np.pi, 16):
    x, y, z = create_dipole_field_line(phi, compression_factor=0.6, expansion_factor=0.6)
    ax.plot(x, y, z, 'gold', alpha=0.5, linewidth=1.5)

# Create solar wind particles
def create_solar_wind_particles(num_particles=30):
    start_x = np.random.uniform(-2, 2, num_particles)
    start_y = np.full(num_particles, -3)  # Start from the left side
    start_z = np.random.uniform(-2, 2, num_particles)
    
    paths = []
    for i in range(num_particles):
        # Create particle path
        t = np.linspace(0, 1, 50)
        x = start_x[i] + t * 3  # Move rightward
        y = start_y[i] + t * 6  # Move toward Earth
        z = start_z[i] + np.sin(t * 4) * 0.2  # Add slight waviness
        
        # Deflect particles around magnetosphere
        dist_from_center = np.sqrt((x - 0)**2 + (y - 0)**2 + (z - 0)**2)
        deflection = 1 / (1 + np.exp(-(dist_from_center - 2)))  # Sigmoid function
        
        # Apply deflection
        x = x * deflection
        y = y * deflection
        z = z + (1 - deflection) * np.sign(z)
        
        paths.append(np.column_stack((x, y, z)))
    
    return paths

# Add solar wind particles
solar_wind_paths = create_solar_wind_particles(40)
for path in solar_wind_paths:
    # Create gradient color from yellow to red
    colors = plt.cm.YlOrRd(np.linspace(0, 1, len(path)))
    for i in range(len(path)-1):
        ax.plot(path[i:i+2, 0], path[i:i+2, 1], path[i:i+2, 2], 
                color=colors[i], alpha=0.6, linewidth=1)

# Add Sun indicator
ax.scatter([-2], [-3], [0], color='yellow', s=300, alpha=0.8, label='Sun')
ax.text(-2, -3, 0.5, 'Sun', color='yellow')

# Add magnetic poles (tilted)
magnetic_north = np.dot(rotation_matrix, [0, 0, 1.1])
magnetic_south = np.dot(rotation_matrix, [0, 0, -1.1])
ax.scatter([magnetic_north[0]], [magnetic_north[1]], [magnetic_north[2]], 
           color='red', s=100, label='Magnetic North Pole')
ax.scatter([magnetic_south[0]], [magnetic_south[1]], [magnetic_south[2]], 
           color='blue', s=100, label='Magnetic South Pole')

# Add labels
ax.text(magnetic_north[0], magnetic_north[1]+0.2, magnetic_north[2], 
        'N. Magnetic Pole', color='red')
ax.text(magnetic_south[0], magnetic_south[1]+0.2, magnetic_south[2], 
        'S. Magnetic Pole', color='blue')

# Add solar wind label
ax.text(-1.5, -2.5, 2, 'Solar Wind\nParticles', color='red', fontsize=10)

# Set plot parameters
ax.set_box_aspect([1,1,1])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title("Earth's Magnetic Field and Solar Wind Interaction")

# Set viewing angle for better visualization
ax.view_init(elev=20, azim=45)

# Add legend
ax.legend()

# Set axis limits
ax.set_xlim([-3, 3])
ax.set_ylim([-3, 3])
ax.set_zlim([-3, 3])

plt.show()
