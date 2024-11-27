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

# Galactic motion (for simplicity, assume linear motion along x-axis)
def sun_motion(t):
    """Simple linear motion of the Sun through the galaxy."""
    v_sun = 2e4  # Sun's velocity in the galaxy (in meters per second)
    x = v_sun * t
    y = 0  # For now, motion is only along the x-axis
    z = 0
    return x, y, z

# Function to calculate the planet's position relative to the Sun
def calculate_planet_motion(planet_name, t):
    """
    Calculate the position of a planet around the Sun using circular orbit approximation.
    """
    d = planets[planet_name]["distance"]
    T = planets[planet_name]["period"]
    omega = 2 * np.pi / T  # Angular velocity
    theta = omega * t  # Angle in radians
    
    # Position (x, y, z) relative to the Sun
    x = d * np.cos(theta)
    y = d * np.sin(theta)
    z = 0  # Orbits are assumed to be in the xy-plane for simplicity
    
    return x, y, z

# Setup the 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-2 * AU, 2 * AU)
ax.set_ylim(-2 * AU, 2 * AU)
ax.set_zlim(-2 * AU, 2 * AU)
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_zlabel('z (m)')
ax.set_title('Solar System Orbiting the Galactic Plane')

# Plot elements for the Sun and planets
sun_plot, = ax.plot([], [], [], 'o', color='yellow', label='Sun')
planet_plots = {}
planet_trails = {}
planet_positions = {planet: [] for planet in planets}  # Dictionary to store past positions

for planet in planets:
    # Plot the planet
    plot, = ax.plot([], [], [], 'o', color=planets[planet]["color"], label=planet)
    planet_plots[planet] = plot
    
    # Plot the trail
    trail, = ax.plot([], [], [], '-', color=planets[planet]["color"], alpha=0.5)  # Trail with transparency
    planet_trails[planet] = trail

# Legends
ax.legend(loc='upper right')

def init():
    """Initialize the animation."""
    sun_plot.set_data([], [])
    sun_plot.set_3d_properties([])
    
    for planet in planets:
        planet_plots[planet].set_data([], [])
        planet_plots[planet].set_3d_properties([])
        planet_trails[planet].set_data([], [])
        planet_trails[planet].set_3d_properties([])
        
    return [sun_plot] + list(planet_plots.values()) + list(planet_trails.values())

def update(frame):
    """Update the position of the Sun and planets for the current frame."""
    t = frame * T_earth / 360  # Time in seconds (divide one Earth year into 360 frames)
    
    # Update Sun's position based on its motion through the galaxy
    sun_pos = sun_motion(t)
    sun_plot.set_data([sun_pos[0]], [sun_pos[1]])
    sun_plot.set_3d_properties([sun_pos[2]])
    
    for planet in planets:
        # Get the planet's position relative to the Sun
        planet_pos = calculate_planet_motion(planet, t)
        
        # Translate planet's position based on Sun's movement
        pos_x = planet_pos[0] + sun_pos[0]
        pos_y = planet_pos[1] + sun_pos[1]
        pos_z = planet_pos[2] + sun_pos[2]
        
        # Store the planet's past positions for the trail
        planet_positions[planet].append((pos_x, pos_y, pos_z))
        
        # Update planet's current position
        planet_plots[planet].set_data([pos_x], [pos_y])
        planet_plots[planet].set_3d_properties([pos_z])
        
        # Update the trail by plotting all past positions
        trail_x = [p[0] for p in planet_positions[planet]]
        trail_y = [p[1] for p in planet_positions[planet]]
        trail_z = [p[2] for p in planet_positions[planet]]
        planet_trails[planet].set_data(trail_x, trail_y)
        planet_trails[planet].set_3d_properties(trail_z)
    
    return [sun_plot] + list(planet_plots.values())

# Create the animation
anim = FuncAnimation(fig, update, frames=np.arange(0, 720), init_func=init, blit=True, interval=50)

# Display the plot
plt.show()