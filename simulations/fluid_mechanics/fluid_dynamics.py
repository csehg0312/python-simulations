import numpy as np
import matplotlib.pyplot as plt

# Define grid and parameters
nx, ny = 41, 41  # Number of grid points
nt = 500         # Number of time steps
nit = 50         # Iterations for pressure convergence
dx = 2 / (nx - 1)  # Grid spacing in x
dy = 2 / (ny - 1)  # Grid spacing in y
rho = 1          # Density
nu = 0.1         # Kinematic viscosity
dt = 0.001       # Time step

# Initialize velocity, pressure, and source terms
u = np.zeros((ny, nx))  # x velocity
v = np.zeros((ny, nx))  # y velocity
p = np.zeros((ny, nx))  # Pressure
b = np.zeros((ny, nx))  # Source term for pressure

# Function to compute the source term
def build_up_b(b, rho, dt, u, v, dx, dy):
    b[1:-1, 1:-1] = (rho * (1 / dt *
                   ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx) +
                    (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) -
                   ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx))**2 -
                     2 * ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy) *
                          (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dx)) -
                     ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy))**2))
    return b

# Function to solve for pressure
def pressure_poisson(p, dx, dy, b):
    pn = np.empty_like(p)
    for q in range(nit):
        pn = p.copy()
        p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy**2 +
                         (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx**2) /
                        (2 * (dx**2 + dy**2)) -
                        dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * b[1:-1, 1:-1])

        # Boundary conditions for pressure
        p[:, -1] = p[:, -2]  # dp/dx = 0 at x = 2
        p[0, :] = p[1, :]    # dp/dy = 0 at y = 0
        p[:, 0] = p[:, 1]    # dp/dx = 0 at x = 0
        p[-1, :] = 0         # p = 0 at y = 2

    return p

# Function to solve the Navier-Stokes equations
def cavity_flow(nt, u, v, dt, dx, dy, p, rho, nu):
    un = np.empty_like(u)
    vn = np.empty_like(v)
    b = np.zeros((ny, nx))

    for n in range(nt):
        un = u.copy()
        vn = v.copy()

        b = build_up_b(b, rho, dt, u, v, dx, dy)
        p = pressure_poisson(p, dx, dy, b)

        u[1:-1, 1:-1] = (un[1:-1, 1:-1] -
                         un[1:-1, 1:-1] * dt / dx *
                        (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
                         vn[1:-1, 1:-1] * dt / dy *
                        (un[1:-1, 1:-1] - un[0:-2, 1:-1]) -
                         dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, 0:-2]) +
                         nu * (dt / dx**2 *
                        (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                         dt / dy**2 *
                        (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])))

        v[1:-1, 1:-1] = (vn[1:-1, 1:-1] -
                         un[1:-1, 1:-1] * dt / dx *
                        (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
                         vn[1:-1, 1:-1] * dt / dy *
                        (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) -
                         dt / (2 * rho * dy) * (p[2:, 1:-1] - p[0:-2, 1:-1]) +
                         nu * (dt / dx**2 *
                        (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                         dt / dy**2 *
                        (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])))

        # Boundary conditions for velocity
        u[0, :] = 0
        u[:, 0] = 0
        u[:, -1] = 0
        u[-1, :] = 1  # Lid moving to the right
        v[0, :] = 0
        v[:, 0] = 0
        v[:, -1] = 0
        v[-1, :] = 0

    return u, v, p

# Run the simulation
u, v, p = cavity_flow(nt, u, v, dt, dx, dy, p, rho, nu)

# Plot the results
plt.figure(figsize=(11,7), dpi=100)
plt.contourf(np.linspace(0, 2, nx), np.linspace(0, 2, ny), p, alpha=0.5, cmap='viridis')
plt.colorbar()
plt.contour(np.linspace(0, 2, nx), np.linspace(0, 2, ny), p, cmap='viridis')
plt.streamplot(np.linspace(0, 2, nx), np.linspace(0, 2, ny), u, v)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Fluid Flow Simulation in 2D Cavity')
plt.show()
