import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Constants
v = 343.0    # Speed of sound in air (m/s)
rho_air = 1.2  # Density of air (kg/m³)
f = 20000.0  # Frequency (Hz)
g = 9.81     # Gravitational acceleration (m/s²)

# Derived constants
lambda_ = v / f  # Wavelength (m)
k = 2 * np.pi / lambda_  # Wave number (m⁻¹)
omega = 2 * np.pi * f  # Angular frequency (rad/s)

# Particle properties
r = 0.001  # Radius of the particle (m)
rho_p = 1000.0  # Density of the particle (kg/m³)
m = (4/3) * np.pi * r**3 * rho_p  # Mass of the particle (kg)

# Pressure wave amplitude based on intensity I = 100 W/m²
Z = rho_air * v  # Impedance of air (kg/(m²s))
I = 100.0  # Intensity (W/m²)
p0 = np.sqrt(2 * Z * I)  # Pressure amplitude (Pa)

# Simulation parameters
dt = 1e-6  # Time step (s)
total_time = 1e-3  # Total simulation time (1 ms)
num_steps = int(total_time / dt)

# Initialize arrays to store data
time = np.zeros(num_steps)
position = np.zeros(num_steps)
velocity = np.zeros(num_steps)

# Initial conditions
x = 0.0  # Initial position (m)
v = 0.0  # Initial velocity (m/s)

# Simulation
for i in range(num_steps):
    # Radiation force
    F_rad = -np.pi * r**2 * 2 * p0 * k * np.cos(k * x) * np.cos(omega * time[i])
    
    # Net force
    F_net = F_rad - m * g
    
    # Acceleration
    a = F_net / m
    
    # Euler's method
    v += a * dt
    x += v * dt
    
    # Store data
    time[i] = i * dt
    position[i] = x
    velocity[i] = v

# Animation setup
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(time.min(), time.max())
ax.set_ylim(position.min() - 0.001, position.max() + 0.001)
ax.set_title('Particle Position over Time')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Position (m)')
ax.grid(True)

# Create a line object
line, = ax.plot([], [], 'r-')

# Initialization function
def init():
    line.set_data([], [])
    return line,

# Animation update function
def update(frame):
    line.set_data(time[:frame], position[:frame])
    return line,

# Create animation
anim = animation.FuncAnimation(fig, update, frames=num_steps, 
                                init_func=init, blit=True, interval=20)

plt.show()