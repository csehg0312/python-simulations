import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Constants
G = 6.67430e-11  # Gravitational constant (m^3 kg^-1 s^-2)
M_sun = 1.989e30  # Mass of the Sun (kg)
AU = 1.496e11  # Astronomical Unit (average Earth-Sun distance in meters)
T = 365.25 * 24 * 3600  # One year in seconds (approximate)
omega = 2 * np.pi / T  # Angular velocity (radians per second)

def calculate_earth_motion(t):
    """
    Calculate Earth's position, velocity, and acceleration around the Sun using Newtonian physics.
    
    Parameters:
    t : float
        Time in seconds from a reference point (e.g., start of the year).
    
    Returns:
    (x, y) : tuple
        Position of the Earth in meters (x, y).
    (vx, vy) : tuple
        Velocity of the Earth in meters per second (vx, vy).
    (ax, ay) : tuple
        Acceleration of the Earth in meters per second squared (ax, ay).
    """
    theta = omega * t  # Angle in radians
    
    # Position (x, y)
    x = AU * np.cos(theta)
    y = AU * np.sin(theta)
    
    # Velocity (vx, vy)
    vx = -AU * omega * np.sin(theta)
    vy = AU * omega * np.cos(theta)
    
    # Acceleration (ax, ay)
    ax = -AU * omega**2 * np.cos(theta)
    ay = -AU * omega**2 * np.sin(theta)
    
    return (x, y), (vx, vy), (ax, ay)

# Setup the plot
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_xlim(-1.5 * AU, 1.5 * AU)
ax.set_ylim(-1.5 * AU, 1.5 * AU)
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_title('Earth Orbit around the Sun')

# Draw the Sun
sun = plt.Circle((0, 0), 0.05 * AU, color='yellow', label='Sun')
ax.add_artist(sun)

# Plot elements for Earth, velocity, and acceleration
earth, = ax.plot([], [], 'bo', label='Earth')  # Earth point
velocity_vector = ax.quiver([], [], [], [], color='green', scale=5e7, label='Velocity')  # Velocity vector
acceleration_vector = ax.quiver([], [], [], [], color='red', scale=5e14, label='Acceleration')  # Acceleration vector

# Speed text
speed_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=12, verticalalignment='top')

# Legends
ax.legend(loc='upper right')

def init():
    """Initialize the animation."""
    earth.set_data([], [])
    velocity_vector.set_UVC([], [], [])
    acceleration_vector.set_UVC([], [], [])
    speed_text.set_text('')
    return earth, velocity_vector, acceleration_vector, speed_text

def update(frame):
    """Update the position, velocity, and acceleration of the Earth for the current frame."""
    t = frame * T / 360  # Time in seconds (divide one year into 360 frames)
    pos, vel, acc = calculate_earth_motion(t)
    
    # Calculate speed (magnitude of the velocity vector)
    speed = np.sqrt(vel[0]**2 + vel[1]**2)
    
    # Update Earth's position
    earth.set_data([pos[0]], [pos[1]])  # Ensure x and y are sequences
    
    # Update velocity and acceleration vectors
    velocity_vector.set_offsets([pos])
    velocity_vector.set_UVC([vel[0]], [vel[1]])
    
    acceleration_vector.set_offsets([pos])
    acceleration_vector.set_UVC([acc[0]], [acc[1]])
    
    # Update the speed text
    speed_text.set_text(f'Speed: {speed:.2f} m/s')
    
    return earth, velocity_vector, acceleration_vector, speed_text

# Create the animation
anim = FuncAnimation(fig, update, frames=np.arange(0, 360), init_func=init, blit=True, interval=50)

# Display the plot
plt.show()
