import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
nx, ny = 41, 41  # Number of grid points
nt = 500  # Number of timesteps
nit = 50  # Number of inner iterations
c = 1  # Speed of sound
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)
X, Y = np.meshgrid(x, y)

rho = 1
nu = 0.1
dt = 0.001

u = np.zeros((ny, nx))
v = np.zeros((ny, nx))
p = np.zeros((ny, nx))
b = np.zeros((ny, nx))

# Set initial condition: Lid-driven cavity
u[-1, :] = 1  # Top wall moving right with u = 1

def build_up_b(b, rho, dt, u, v, dx, dy):
    b[1:-1, 1:-1] = (rho * (1 / dt * ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx) +
                                      (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) -
                            ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx))**2 -
                            2 * ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy) *
                                 (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dx)) -
                            ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy))**2))
    return b

def pressure_poisson(p, dx, dy, b):
    pn = np.empty_like(p)
    pn = p.copy()
    
    for q in range(nit):
        pn = p.copy()
        p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy**2 +
                          (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx**2) /
                         (2 * (dx**2 + dy**2)) -
                         dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * b[1:-1, 1:-1])

        p[:, -1] = p[:, -2]  # dp/dx = 0 at x = 2
        p[0, :] = p[1, :]    # dp/dy = 0 at y = 0
        p[:, 0] = p[:, 1]    # dp/dx = 0 at x = 0
        p[-1, :] = 0         # p = 0 at y = 2
        
    return p

def cavity_flow(nt, u, v, dt, dx, dy, p, rho, nu):
    un = np.empty_like(u)
    vn = np.empty_like(v)
    b = np.zeros((ny, nx))
    
    for n in range(nt):
        un = u.copy()
        vn = v.copy()
        
        b = build_up_b(b, rho, dt, u, v, dx, dy)
        p = pressure_poisson(p, dx, dy , b)
        
        u[1:-1, 1:-1] = (un[1:-1, 1:-1] -
                          un[1:-1, 1:-1] * dt / dx * (un[1:-1, 2:] - un[1:-1, 0:-2]) -
                          vn[1:-1, 1:-1] * dt / dy * (un[2:, 1:-1] - un[0:-2, 1:-1]) -
                          dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, 0:-2]) +
                          nu * dt / dx**2 * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                          nu * dt / dy**2 * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1]))
        
        v[1:-1, 1:-1] = (vn[1:-1, 1:-1] -
                          un[1:-1, 1:-1] * dt / dx * (vn[1:-1, 2:] - vn[1:-1, 0:-2]) -
                          vn[1:-1, 1:-1] * dt / dy * (vn[2:, 1:-1] - vn[0:-2, 1:-1]) -
                          dt / (2 * rho * dy) * (p[2:, 1:-1] - p[0:-2, 1:-1]) +
                          nu * dt / dx**2 * (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                          nu * dt / dy**2 * (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1]))
        
        u[0, :] = 0
        u[:, 0] = 0
        u[:, -1] = 0
        u[-1, :] = 1
        v[0, :] = 0
        v[-1, :] = 0
        v[:, 0] = 0
        v[:, -1] = 0
        
    return u, v, p

fig, ax = plt.subplots(figsize=(11, 7))
quiver = ax.quiver(X[1:-1, 1:-1], Y[1:-1, 1:-1], u[1:-1, 1:-1], v[1:-1, 1:-1], scale=50)
contour = ax.contourf(X, Y, p, alpha=0.5, cmap='viridis')

def init():
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    return quiver,

def update(frame):
    global u, v, p, contour
    u, v, p = cavity_flow(1, u, v, dt, dx, dy, p, rho, nu)
    
    quiver.set_UVC(u[1:-1, 1:-1], v[1:-1, 1:-1])
    
    for c in contour.collections:
        c.remove()
    contour = ax.contourf(X, Y, p, alpha=0.5, cmap='viridis')
    
    return quiver,

ani = FuncAnimation(fig, update, frames=nt, init_func=init, blit=False, repeat=False, interval=50)
plt.show()