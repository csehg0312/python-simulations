import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Constants
nx, ny = 100, 100
dx, dy = 1.0, 1.0
g = 9.81
dt = 0.01

# Initialize arrays
h = np.ones((ny, nx)) + 0.1 * np.random.randn(ny, nx)
u = np.zeros((ny, nx))
v = np.zeros((ny, nx))

def limiter(arr, min_val, max_val):
    return np.clip(arr, min_val, max_val)

def shallow_water_step(h, u, v, dt, dx, dy, g):
    # Add a small constant to h to prevent division by zero
    h_safe = h + 1e-6

    Fh = h_safe * u
    Fu = h_safe * u**2 + 0.5 * g * h_safe**2
    Fv = h_safe * u * v

    Gh = h_safe * v
    Gu = h_safe * u * v
    Gv = h_safe * v**2 + 0.5 * g * h_safe**2

    h[1:-1, 1:-1] -= dt * ((Fh[1:-1, 2:] - Fh[1:-1, :-2]) / (2*dx) +
                           (Gh[2:, 1:-1] - Gh[:-2, 1:-1]) / (2*dy))

    u[1:-1, 1:-1] -= dt * ((Fu[1:-1, 2:] - Fu[1:-1, :-2]) / (2*dx) +
                           (Gu[2:, 1:-1] - Gu[:-2, 1:-1]) / (2*dy))

    v[1:-1, 1:-1] -= dt * ((Fv[1:-1, 2:] - Fv[1:-1, :-2]) / (2*dx) +
                           (Gv[2:, 1:-1] - Gv[:-2, 1:-1]) / (2*dy))

    # Apply limiter
    h = limiter(h, 0.1, 10)
    u = limiter(u, -10, 10)
    v = limiter(v, -10, 10)

    return h, u, v

fig, ax = plt.subplots()
im = ax.imshow(h, animated=True, vmin=0, vmax=2)

def update(frame):
    global h, u, v
    for _ in range(10):  # Perform multiple steps per frame for faster simulation
        h, u, v = shallow_water_step(h, u, v, dt, dx, dy, g)
    im.set_array(h)
    return [im]

ani = FuncAnimation(fig, update, frames=200, interval=50, blit=True)
plt.show()
