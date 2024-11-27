import numpy as np
import matplotlib.pyplot as plt
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

# Function to calculate the planet's position
def calculate_planet_motion(planet_name, t):
    """
    Calculate the position of a planet around the Sun using circular orbit approximation.
    """
    d = planets[planet_name]["distance"]
    T = planets[planet_name]["period"]
    omega = 2 * np.pi / T  # Angular velocity
    theta = omega * t  # Angle in radians
    
    # Position (x, y)
    x = d * np.cos(theta)
    y = d * np.sin(theta)
    
    return (x, y)

# Setup the plot
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_xlim(-2 * AU, 2 * AU)
ax.set_ylim(-2 * AU, 2 * AU)
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_title('Rocky Planets Orbiting the Sun')

# Draw the Sun
sun = plt.Circle((0, 0), 0.05 * AU, color='yellow', label='Sun')
ax.add_artist(sun)

# Plot elements for the planets
planet_plots = {}
planet_trails = {}
planet_positions = {planet: [] for planet in planets}  # Dictionary to store past positions

for planet in planets:
    # Plot the planet
    plot, = ax.plot([], [], 'o', color=planets[planet]["color"], label=planet)
    planet_plots[planet] = plot
    
    # Plot the trail
    trail, = ax.plot([], [], '-', color=planets[planet]["color"], alpha=0.5)  # Trail with alpha transparency
    planet_trails[planet] = trail

# Legends
ax.legend(loc='upper right')

def init():
    """Initialize the animation."""
    for planet in planets:
        planet_plots[planet].set_data([], [])
        planet_trails[planet].set_data([], [])
    return list(planet_plots.values()) + list(planet_trails.values())

def update(frame):
    """Update the position and trail of each planet for the current frame."""
    t = frame * T_earth / 360  # Time in seconds (divide one Earth year into 360 frames)
    
    for planet in planets:
        pos = calculate_planet_motion(planet, t)
        planet_positions[planet].append(pos)  # Store the current position in the list
        
        # Update the planet's current position
        planet_plots[planet].set_data([pos[0]], [pos[1]])
        
        # Update the trail by plotting all past positions
        trail_x = [p[0] for p in planet_positions[planet]]
        trail_y = [p[1] for p in planet_positions[planet]]
        planet_trails[planet].set_data(trail_x, trail_y)
    
    return list(planet_plots.values()) + list(planet_trails.values())

# Create the animation
anim = FuncAnimation(fig, update, frames=np.arange(0, 360), init_func=init, blit=True, interval=50)

# Display the plot
plt.show()

