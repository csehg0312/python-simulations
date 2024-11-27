import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Constants
AU = 1.496e11  # Astronomical Unit (in meters)
T_earth = 365.25 * 24 * 3600  # Earth year in seconds

# Orbital parameters
planets = {
    "Mercury": {"distance": 0.39 * AU, "period": 88 * 24 * 3600, "color": "gray"},
    "Venus": {"distance": 0.72 * AU, "period": 225 * 24 * 3600, "color": "orange"},
    "Earth": {"distance": 1.00 * AU, "period": 365.25 * 24 * 3600, "color": "blue"},
    "Mars": {"distance": 1.52 * AU, "period": 687 * 24 * 3600, "color": "red"},
}

def calculate_planet_position(planet_name, t):
    """Calculate planet's position using circular orbit approximation."""
    d = planets[planet_name]["distance"]
    T = planets[planet_name]["period"]
    omega = 2 * np.pi / T
    theta = omega * t
    
    x = d * np.cos(theta)
    y = d * np.sin(theta)
    return x, y

def sun_motion(t):
    """Simplified sun motion through galaxy."""
    v_sun = 2e4
    x = v_sun * t + 5e4 * np.sin(t / 1e6)
    y = 3e4 * np.cos(t / 5e5)
    return x, y

# Prepare figure with explicit configuration
plt.close('all')
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_aspect('equal')
ax.set_title('Solar System Traversal')
ax.set_xlabel('X (meters)')
ax.set_ylabel('Y (meters)')

# Initialize plot objects with empty lists
sun, = ax.plot([], [], 'yo', markersize=10, label='Sun')
planet_plots = {}
trail_plots = {}

for name, details in planets.items():
    planet, = ax.plot([], [], 'o', color=details['color'], label=name, markersize=5)
    trail, = ax.plot([], [], '-', color=details['color'], linewidth=1, alpha=0.5)
    planet_plots[name] = planet
    trail_plots[name] = trail

# Time display
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

# Trail storage
max_trail_length = 200
planet_trails = {name: {'x': [], 'y': []} for name in planets}
sun_trail = {'x': [], 'y': []}

def update(frame):
    # Time calculation
    t = frame * T_earth / 120
    elapsed_years = t / (365.25 * 24 * 3600)
    time_text.set_text(f'Elapsed Time: {elapsed_years:.2f} Earth Years')

    # Sun motion
    sun_x, sun_y = sun_motion(t)
    sun.set_data([sun_x], [sun_y])  # Note the list wrapping

    # Update sun trail
    sun_trail['x'].append(sun_x)
    sun_trail['y'].append(sun_y)
    sun_trail['x'] = sun_trail['x'][-max_trail_length:]
    sun_trail['y'] = sun_trail['y'][-max_trail_length:]

    # Planets
    for name, details in planets.items():
        # Planet position relative to sun
        planet_x, planet_y = calculate_planet_position(name, t)
        adjusted_x = planet_x + sun_x
        adjusted_y = planet_y + sun_y

        # Update planet position with lists
        planet_plots[name].set_data([adjusted_x], [adjusted_y])

        # Update planet trail
        planet_trails[name]['x'].append(adjusted_x)
        planet_trails[name]['y'].append(adjusted_y)
        planet_trails[name]['x'] = planet_trails[name]['x'][-max_trail_length:]
        planet_trails[name]['y'] = planet_trails[name]['y'][-max_trail_length:]

        # Plot trail
        trail_plots[name].set_data(
            planet_trails[name]['x'], 
            planet_trails[name]['y']
        )

    # Dynamically adjust view
    view_range = 3 * AU
    ax.set_xlim(sun_x - view_range, sun_x + view_range)
    ax.set_ylim(sun_y - view_range, sun_y + view_range)

    # Combine all artists to return
    artists = [sun, time_text] + \
              list(planet_plots.values()) + \
              list(trail_plots.values())
    return artists

# Create animation
anim = FuncAnimation(
    fig, 
    update, 
    frames=10000,
    interval=50,
    blit=True
)

plt.legend(loc='upper right')
plt.tight_layout()
plt.show()