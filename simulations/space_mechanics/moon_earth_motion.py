import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Constants for the Moon's elliptical orbit
AU = 1.496e11  # Astronomical Unit (not needed for this simulation, but for reference)
initial_semi_major_axis = 384400e3  # Initial semi-major axis of the Moon's orbit (in meters)
eccentricity = 0.0549  # Orbital eccentricity of the Moon's orbit
moon_period = 27.3 * 24 * 3600  # Moon's orbital period (in seconds)
moon_orbit_increase_per_year = 3.8e-2  # 3.8 cm/year increase in orbit (in meters)

# Function to calculate the Moon's elliptical motion relative to Earth
def calculate_moon_elliptical_motion(t, elapsed_years):
    """
    Calculate the position of the Moon around the Earth using elliptical orbit approximation.
    The semi-major axis of the Moon's orbit increases slowly over time.
    """
    # Increase the semi-major axis over time based on elapsed years
    semi_major_axis = initial_semi_major_axis + elapsed_years * moon_orbit_increase_per_year
    
    # Angular velocity of the Moon
    omega_moon = 2 * np.pi / moon_period
    theta_moon = omega_moon * t  # Angle in radians (mean anomaly)
    
    # Use Kepler's equation to find the Moon's true anomaly (θ)
    true_anomaly = theta_moon % (2 * np.pi)
    
    # Calculate distance using the formula for an ellipse: r = a(1 - e^2) / (1 + e * cos(θ))
    distance = semi_major_axis * (1 - eccentricity**2) / (1 + eccentricity * np.cos(true_anomaly))
    
    # Position (x, y, z) relative to the Earth
    x_moon = distance * np.cos(true_anomaly)
    y_moon = distance * np.sin(true_anomaly)
    z_moon = 0  # Assume Moon's orbit is in the xy-plane for simplicity
    
    return x_moon, y_moon, z_moon, distance

# Setup the 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-2 * initial_semi_major_axis, 2 * initial_semi_major_axis)
ax.set_ylim(-2 * initial_semi_major_axis, 2 * initial_semi_major_axis)
ax.set_zlim(-2 * initial_semi_major_axis, 2 * initial_semi_major_axis)
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_zlabel('z (m)')
ax.set_title('Moon Orbiting Earth with Increasing Orbit')

# Plot elements for Earth and Moon
earth_plot, = ax.plot([0], [0], [0], 'o', color='blue', label='Earth')  # Earth at the origin
moon_plot, = ax.plot([], [], [], 'o', color='lightgray', label='Moon')
moon_trail, = ax.plot([], [], [], '-', color='lightgray', alpha=0.5)

# Annotation for displaying real-time distance
distance_text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)

# Legends
ax.legend(loc='upper right')

# Lists to store past positions for trail effect
moon_positions = []

def init():
    """Initialize the animation."""
    moon_plot.set_data([], [])
    moon_plot.set_3d_properties([])
    
    moon_trail.set_data([], [])
    moon_trail.set_3d_properties([])
    
    distance_text.set_text("")  # Clear the initial distance text
    
    return [moon_plot, moon_trail, distance_text]

def update(frame):
    """Update the position of the Moon for the current frame."""
    t = frame * moon_period / 360  # Time in seconds (divide one Moon period into 360 frames)
    
    # Calculate how many years have passed based on the elapsed time (1 year = 365.25 days)
    elapsed_years = (t / (365.25 * 24 * 3600))  # Convert seconds to years
    
    # Calculate Moon's position relative to Earth with an increasing orbit
    moon_x, moon_y, moon_z, distance = calculate_moon_elliptical_motion(t, elapsed_years)
    
    # Store the Moon's past positions for the trail
    moon_positions.append((moon_x, moon_y, moon_z))
    
    # Update Moon's current position
    moon_plot.set_data([moon_x], [moon_y])
    moon_plot.set_3d_properties([moon_z])
    
    # Update the trail by plotting all past positions
    trail_moon_x = [pos[0] for pos in moon_positions]
    trail_moon_y = [pos[1] for pos in moon_positions]
    trail_moon_z = [pos[2] for pos in moon_positions]
    
    moon_trail.set_data(trail_moon_x, trail_moon_y)
    moon_trail.set_3d_properties(trail_moon_z)
    
    # Calculate and display the real-time distance from Earth to the Moon
    distance_text.set_text(f"Moon-Earth Distance: {distance:.2e} m")
    
    return [moon_plot, moon_trail, distance_text]

# Create the animation
anim = FuncAnimation(fig, update, frames=np.arange(0, 720), init_func=init, blit=True, interval=100)

# Display the plot
plt.show()
