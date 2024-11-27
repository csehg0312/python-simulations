import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Constants
AU = 1.496e11  # Astronomical Unit (in meters)
T_earth = 365.25 * 24 * 3600  # Earth year in seconds

# Orbital parameters (AU and seconds for periods)
planets = {
    "Mercury": {"distance": 0.39 * AU, "period": 88 * 24 * 3600, "color": "gray"},
    "Venus": {"distance": 0.72 * AU, "period": 225 * 24 * 3600, "color": "orange"},
    "Earth": {"distance": 1.00 * AU, "period": 365.25 * 24 * 3600, "color": "blue"},
    "Mars": {"distance": 1.52 * AU, "period": 687 * 24 * 3600, "color": "red"},
}

# Planetary motion calculation
def calculate_planet_position(planet_name, t):
    """Calculate planet's position using circular orbit approximation."""
    d = planets[planet_name]["distance"]
    T = planets[planet_name]["period"]
    omega = 2 * np.pi / T  # Angular velocity
    theta = omega * t  # Angle in radians
    
    x = d * np.cos(theta)
    y = d * np.sin(theta)
    z = 0  # Simplified to xy-plane
    
    return x, y, z

# Sun's galactic motion with more complex trajectory
def sun_motion(t):
    """More complex motion of the Sun through the galaxy."""
    # Combine multiple sinusoidal motions to create a more interesting path
    v_sun = 2e4  # Base velocity 
    x = v_sun * t + 5e4 * np.sin(t / 1e6)  # Linear motion with sinusoidal variation
    y = 3e4 * np.cos(t / 5e5)  # Perpendicular sinusoidal motion
    z = 2e4 * np.sin(t / 3e5)  # Vertical sinusoidal motion
    
    return x, y, z

# Setup the visualization
fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(111, projection='3d')

# Dynamic axis limits that update with sun's motion
ax.set_title('Infinite Solar System Traversal')

# Tracking variables for trails
planet_trail_data = {planet: {'x': [], 'y': [], 'z': []} for planet in planets}
sun_trail_data = {'x': [], 'y': [], 'z': []}

# Create plot objects
planet_plots = {}
planet_trails = {}

# Sun plot objects
sun_plot = ax.plot([], [], [], 'o', color='yellow', markersize=10, label='Sun')[0]
sun_trail = ax.plot([], [], [], '-', color='yellow', alpha=0.3, linewidth=1)[0]

# Initialize plot objects for each planet
for planet, details in planets.items():
    # Planet point
    plot, = ax.plot([], [], [], 'o', color=details['color'], label=planet, markersize=5)
    planet_plots[planet] = plot
    
    # Planet trail
    trail, = ax.plot([], [], [], '-', color=details['color'], alpha=0.2, linewidth=1)
    planet_trails[planet] = trail

# Animation update function
def update(frame):
    # Calculate time with continuous progression
    t = frame * T_earth / 120  # Faster progression
    
    # Get Sun's position
    sun_x, sun_y, sun_z = sun_motion(t)
    
    # Dynamic axis adjustment
    view_range = 3 * AU
    ax.set_xlim(sun_x - view_range, sun_x + view_range)
    ax.set_ylim(sun_y - view_range, sun_y + view_range)
    ax.set_zlim(sun_z - view_range, sun_z + view_range)
    
    # Update Sun position and trail
    sun_plot.set_data([sun_x], [sun_y])
    sun_plot.set_3d_properties([sun_z])
    
    # Store and update Sun trail
    sun_trail_data['x'].append(sun_x)
    sun_trail_data['y'].append(sun_y)
    sun_trail_data['z'].append(sun_z)
    
    # Limit trail length
    max_trail_length = 500
    sun_trail_data['x'] = sun_trail_data['x'][-max_trail_length:]
    sun_trail_data['y'] = sun_trail_data['y'][-max_trail_length:]
    sun_trail_data['z'] = sun_trail_data['z'][-max_trail_length:]
    
    sun_trail.set_data(sun_trail_data['x'], sun_trail_data['y'])
    sun_trail.set_3d_properties(sun_trail_data['z'])
    
    # Update each planet
    for planet in planets:
        # Calculate planet's position relative to Sun
        planet_x, planet_y, planet_z = calculate_planet_position(planet, t)
        
        # Adjust planet position based on Sun's motion
        adjusted_x = planet_x + sun_x
        adjusted_y = planet_y + sun_y
        adjusted_z = planet_z + sun_z
        
        # Store and update planet trail
        planet_trail_data[planet]['x'].append(adjusted_x)
        planet_trail_data[planet]['y'].append(adjusted_y)
        planet_trail_data[planet]['z'].append(adjusted_z)
        
        # Limit trail length
        planet_trail_data[planet]['x'] = planet_trail_data[planet]['x'][-max_trail_length:]
        planet_trail_data[planet]['y'] = planet_trail_data[planet]['y'][-max_trail_length:]
        planet_trail_data[planet]['z'] = planet_trail_data[planet]['z'][-max_trail_length:]
        
        # Update planet position
        planet_plots[planet].set_data([adjusted_x], [adjusted_y])
        planet_plots[planet].set_3d_properties([adjusted_z])
        
        # Update trail
        planet_trails[planet].set_data(
            planet_trail_data[planet]['x'], 
            planet_trail_data[planet]['y']
        )
        planet_trails[planet].set_3d_properties(
            planet_trail_data[planet]['z']
        )
    
    return [sun_plot, sun_trail] + list(planet_plots.values()) + list(planet_trails.values())

# Create animation with a very large number of frames to simulate infinite
anim = FuncAnimation(
    fig, 
    update, 
    frames=10000,  # Large but finite number of frames
    interval=50,   # Update every 50 ms
    blit=True,     # More efficient rendering
    repeat=True    # Loop the animation
)

plt.legend()
plt.show()