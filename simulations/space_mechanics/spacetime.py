import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Constants
AU = 1.496e11  # Astronomical Unit (in meters)
T_earth = 365.25 * 24 * 3600  # Earth year in seconds
G = 6.67430e-11  # Gravitational constant
M_sun = 1.989e30  # Mass of the Sun in kg
c = 299792458

# Orbital parameters (AU and seconds for periods)
planets = {
    "Mercury": {"distance": 0.467 * AU, "period": 88 * 24 * 3600, "color": "gray"},
    "Venus": {"distance": 0.728 * AU, "period": 225 * 24 * 3600, "color": "orange"},
    "Earth": {"distance": 1.017 * AU, "period": 365.25 * 24 * 3600, "color": "blue"},
    "Mars": {"distance": 1.666 * AU, "period": 687 * 24 * 3600, "color": "red"},
}

def sun_motion(t):
    """Simple linear motion of the Sun through the galaxy."""
    v_sun = 2e2  # Reduced Sun's velocity for visualization
    x = v_sun * t
    return x, 0, 0  # Motion only along x-axis

def calculate_planet_motion(planet_name, t):
    """Calculate the position of a planet around the Sun using circular orbit approximation."""
    d = planets[planet_name]["distance"]
    T = planets[planet_name]["period"]
    omega = 2 * np.pi / T
    theta = omega * t
    return d * np.cos(theta), d * np.sin(theta), 0

def spacetime_curvature(x, y, z, sun_pos):
    """Calculate the curvature of spacetime around the Sun."""
    r = np.sqrt((x - sun_pos[0])**2 + (y - sun_pos[1])**2 + (z - sun_pos[2])**2)
    return -G * M_sun / (r * c**2)  # Normalized curvature

# Setup the 3D plot
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-2 * AU, 2 * AU)
ax.set_ylim(-2 * AU, 2 * AU)
ax.set_zlim(-0.5 * AU, 0.5 * AU)
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_zlabel('Spacetime Curvature')
ax.set_title('Solar System and Spacetime Curvature')

# Initialize plot elements
sun_plot, = ax.plot([], [], [], 'o', color='yellow', markersize=20, label='Sun')
planet_plots = {planet: ax.plot([], [], [], 'o', color=data["color"], markersize=5, label=planet)[0] for planet, data in planets.items()}
distance_labels = {planet: ax.text(0, 0, 0, '', fontsize=10) for planet in planets.keys()}

# Initialize spacetime grid
grid_size = 50
x_grid, y_grid = np.meshgrid(np.linspace(-2*AU, 2*AU, grid_size), np.linspace(-2*AU, 2*AU, grid_size))
z_grid = np.zeros((grid_size, grid_size))
spacetime_plot = ax.plot_surface(x_grid, y_grid, z_grid, cmap='viridis', alpha=0.3)

ax.legend(loc='upper right')

def init():
    """Initialize the animation."""
    sun_plot.set_data([], [])
    sun_plot.set_3d_properties([])
    for plot in planet_plots.values():
        plot.set_data([], []) # Initialize planet positions
        plot.set_3d_properties([])
    for label in distance_labels.values():
        label.set_text('')
    return sun_plot, *planet_plots.values(), *distance_labels.values()

def animate(t):
    """Update the animation at each frame."""
    global spacetime_plot  # Declare spacetime_plot as global

    sun_pos = sun_motion(t)
    sun_plot.set_data([sun_pos[0]], [sun_pos[1]])
    sun_plot.set_3d_properties([sun_pos[2]])

    for planet, plot in planet_plots.items():
        x, y, z = calculate_planet_motion(planet, t / 10 )  # Slow down planet motion
        plot.set_data([x], [y])
        plot.set_3d_properties([z])
        distance_labels[planet].set_text(f'{planet}: {np.sqrt((x - sun_pos[0])**2 + (y - sun_pos[1])**2 + (z - sun_pos[2])**2) / AU:.2f} AU')
        distance_labels[planet].set_position((x, y, z))

    x_grid, y_grid = np.meshgrid(np.linspace(-2*AU, 2*AU, grid_size), np.linspace(-2*AU, 2*AU, grid_size))
    z_grid = spacetime_curvature(x_grid, y_grid, 0, sun_pos)

    # Remove the old spacetime plot
    if spacetime_plot is not None:
        spacetime_plot.remove()

    # Create a new spacetime plot
    spacetime_plot = ax.plot_surface(x_grid, y_grid, z_grid, cmap='viridis', alpha=0.3)

    return sun_plot, *planet_plots.values(), *distance_labels.values()

ani = FuncAnimation(fig, animate, frames=np.linspace(0, 10 * T_earth, 1000), interval=20, init_func=init, blit=True)

plt.show()