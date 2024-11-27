import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
nx, ny = 100, 100  # Number of grid points
Lx, Ly = 1.0, 1.0  # Domain size
dx, dy = Lx/nx, Ly/ny
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)

g = 9.81  # Gravity
H = 0.1  # Average water depth
dt = 0.0005  # Time step (reduced for stability)

# Initialize variables
h = np.ones((ny, nx)) * H  # Water height
u = np.zeros((ny, nx))  # x-velocity
v = np.zeros((ny, nx))  # y-velocity

# Create an initial disturbance
h[40:60, 40:60] += 0.1 * np.exp(-((X[40:60, 40:60]-0.5)**2 + (Y[40:60, 40:60]-0.5)**2) / 0.01)

def limiter(arr, min_val, max_val):
    return np.clip(arr, min_val, max_val)

def shallow_water_step(h, u, v, dt, dx, dy, g):
    # Add a small constant to h to prevent division by zero
    h_safe = h + 1e-6

    # Compute fluxes
    Fh = h_safe * u
    Fu = h_safe * u**2 + 0.5 * g * h_safe**2
    Fv = h_safe * u * v
    
    Gh = h_safe * v
    Gu = h_safe * u * v
    Gv = h_safe * v**2 + 0.5 * g * h_safe**2
    
    # Update variables
    h[1:-1, 1:-1] -= dt * ((Fh[1:-1, 2:] - Fh[1:-1, :-2]) / (2*dx) + 
                           (Gh[2:, 1:-1] - Gh[:-2, 1:-1]) / (2*dy))
    
    u[1:-1, 1:-1] -= dt * ((Fu[1:-1, 2:] - Fu[1:-1, :-2]) / (2*dx) + 
                           (Gu[2:, 1:-1] - Gu[:-2, 1:-1]) / (2*dy))
    
    v[1:-1, 1:-1] -= dt * ((Fv[1:-1, 2:] - Fv[1:-1, :-2]) / (2*dx) + 
                           (Gv[2:, 1:-1] - Gv[:-2, 1:-1]) / (2*dy))
    
    # Apply boundary conditions (reflective)
    h[0, :] = h[1, :]
    h[-1, :] = h[-2, :]
    h[:, 0] = h[:, 1]
    h[:, -1] = h[:, -2]
    
    u[0, :] = -u[1, :]
    u[-1, :] = -u[-2, :]
    u[:, 0] = -u[:, 1]
    u[:, -1] = -u[:, -2]
    
    v[0, :] = -v[1, :]
    v[-1, :] = -v[-2, :]
    v[:, 0] = -v[:, 1]
    v[:, -1] = -v[:, -2]
    
    # Apply limiter
    h = limiter(h, 0.01, 1.0)  # Ensure water height stays positive and below a maximum
    u = limiter (u, -1.0, 1.0)  # Limit velocities to prevent excessive values
    v = limiter(v, -1.0, 1.0)
    
    return h, u, v

# Set up the figure and axis for animation
fig, ax = plt.subplots(figsize=(8, 8))
im = ax.imshow(h, cmap='coolwarm', extent=(0, Lx, 0, Ly), animated=True)

def update(frame):
    global h, u, v
    h, u, v = shallow_water_step(h, u, v, dt, dx, dy, g)
    im.set_array(h)
    return im,

ani = FuncAnimation(fig, update, frames=1000, blit=True, interval=50)
plt.show()