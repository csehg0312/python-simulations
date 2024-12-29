import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class NavierStokesSolver:
    def __init__(self, nx=100, ny=100, length=1.0, height=1.0, 
                 nu=0.01, rho=1.0, dt=0.001, total_time=10):
        """
        Initialize the Navier-Stokes solver for 2D incompressible flow
        
        Parameters:
        -----------
        nx, ny : int
            Number of grid points in x and y directions
        length, height : float
            Domain dimensions
        nu : float
            Kinematic viscosity
        rho : float
            Fluid density
        dt : float
            Time step
        total_time : float
            Total simulation time
        """
        # Grid parameters
        self.nx, self.ny = nx, ny
        self.length, self.height = length, height
        
        # Physical parameters
        self.nu = nu  # Kinematic viscosity
        self.rho = rho  # Density
        self.dt = dt  # Time step
        self.total_time = total_time
        
        # Grid spacing
        self.dx = length / (nx - 1)
        self.dy = height / (ny - 1)
        
        # Initialize velocity and pressure fields
        self.u = np.zeros((ny, nx), dtype=np.float64)  # x-velocity
        self.v = np.zeros((ny, nx), dtype=np.float64)  # y-velocity
        self.p = np.zeros((ny, nx), dtype=np.float64)  # pressure
        
        # Create coordinate grids
        self.x = np.linspace(0, length, nx)
        self.y = np.linspace(0, height, ny)
        
        # Initial condition: Lid-driven cavity flow
        # Top boundary moves with a constant velocity
        self.u[-1, :] = 1.0
    
    def compute_divergence(self):
        """
        Compute velocity divergence
        """
        # Central difference for divergence
        div_u = np.zeros_like(self.u, dtype=np.float64)
        div_u[1:-1, 1:-1] = (
            (self.u[1:-1, 2:] - self.u[1:-1, :-2]) / (2 * self.dx) + 
            (self.v[2:, 1:-1] - self.v[:-2, 1:-1]) / (2 * self.dy)
        )
        return div_u
    
    def compute_pressire_poisson(self):
        """
        Solve the pressure Poisson equation using Jacobi iteration
        """
        # Compute divergence
        div_u = self.compute_divergence()
        
        # Pressure Poisson solver with improved stability
        p = self.p.copy()
        for _ in range(100):  # Increased iterations
            p_old = p.copy()
            
            # Improved pressure update with clipping to prevent overflow
            p[1:-1, 1:-1] = 0.25 * (
                np.clip(p_old[1:-1, 2:], -1e10, 1e10) + 
                np.clip(p_old[1:-1, :-2], -1e10, 1e10) + 
                np.clip(p_old[2:, 1:-1], -1e10, 1e10) + 
                np.clip(p_old[:-2, 1:-1], -1e10, 1e10) - 
                div_u[1:-1, 1:-1] * (self.dx * self.dy)
            )
            
            # Boundary conditions (zero Neumann)
            p[0, :] = p[1, :]
            p[-1, :] = p[-2, :]
            p[:, 0] = p[:, 1]
            p[:, -1] = p[:, -2]
        
        self.p = p
    
    def compute_velocities(self):
        """
        Update velocities using Navier-Stokes equation with improved stability
        """
        # Create copies to avoid overwriting during computation
        u, v = self.u.copy(), self.v.copy()
        
        # Compute advection terms with improved numerical stability
        def safe_gradient(field, axis):
            """Compute gradient with clamped values to prevent overflow"""
            grad = np.gradient(np.clip(field, -1e10, 1e10), axis=axis)
            return np.clip(grad, -1e10, 1e10)
        
        # Advection terms with clamping
        u_adv_x = np.clip(u, -1e10, 1e10) * safe_gradient(u, axis=1) / self.dx
        u_adv_y = np.clip(v, -1e10, 1e10) * safe_gradient(u, axis=0) / self.dy
        
        v_adv_x = np.clip(u, -1e10, 1e10) * safe_gradient(v, axis=1) / self.dx
        v_adv_y = np.clip(v, -1e10, 1e10) * safe_gradient(v, axis=0) / self.dy
        
        # Compute diffusion terms
        def safe_laplacian(field):
            """Compute Laplacian with clamped values"""
            grad_x = safe_gradient(safe_gradient(field, axis=1), axis=1) / (self.dx ** 2)
            grad_y = safe_gradient(safe_gradient(field, axis=0), axis=0) / (self.dy ** 2)
            return grad_x + grad_y
        
        u_diff = self.nu * safe_laplacian(u)
        v_diff = self.nu * safe_laplacian(v)
        
        # Pressure gradient terms
        u_press_grad = -safe_gradient(self.p, axis=1) / (self.rho * self.dx)
        v_press_grad = -safe_gradient(self.p, axis=0) / (self.rho * self.dy)
        
        # Update velocities with clipping
        self.u[1:-1, 1:-1] = np.clip(
            u[1:-1, 1:-1] + self.dt * (
                -np.clip(u_adv_x[1:-1, 1:-1], -1e10, 1e10) - 
                np.clip(u_adv_y[1:-1, 1:-1], -1e10, 1e10) + 
                np.clip(u_diff[1:-1, 1:-1], -1e10, 1e10) + 
                np.clip(u_press_grad[1:-1, 1:-1], -1e10, 1e10)
            ),
            -1e10, 1e10
        )
        
        self.v[1:-1, 1:-1] = np.clip(
            v[1:-1, 1:-1] + self.dt * (
                -np.clip(v_adv_x[1:-1, 1:-1], -1e10, 1e10) - 
                np.clip(v_adv_y[1:-1, 1:-1], -1e10, 1e10) + 
                np.clip(v_diff[1:-1, 1:-1], -1e10, 1e10) + 
                np.clip(v_press_grad[1:-1, 1:-1], -1e10, 1e10)
            ),
            -1e10, 1e10
        )
        
        # Enforce boundary conditions
        # Top boundary: lid-driven cavity
        self.u[-1, :] = 1.0
        self.u[0, :] = 0
        self.u[:, 0] = 0
        self.u[:, -1] = 0
        
        self.v[0, :] = 0
        self.v[-1, :] = 0
        self.v[:, 0] = 0
        self.v[:, -1] = 0
    
    def simulate(self):
        """
        Run the full simulation
        """
        # Number of time steps
        nt = int(self.total_time / self.dt)
        print(nt)
        
        # Store velocity magnitude for visualization
        velocity_history = []
        
        for _ in range(nt):
            print("Run")
            # Compute pressure
            self.compute_pressire_poisson()
            
            # Update velocities
            self.compute_velocities()
            
            # Store velocity magnitude
            velocity_magnitude = np.sqrt(np.clip(self.u**2 + self.v**2, 0, 1e10))
            velocity_history.append(velocity_magnitude)
        
        return velocity_history

# Run the simulation
def main():
    # Create solver with reduced parameters for stability
    solver = NavierStokesSolver(
        nx=100, ny=100,  # Grid resolution
        length=1.0, height=1.0,  # Domain size
        nu=0.01,  # Reduced kinematic viscosity
        rho=1.0,  # Density
        dt=0.001,  # Smaller time step
        total_time=3  # Total simulation time
    )
    
    # Run simulation
    velocity_history = solver.simulate()
    
    # Visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create animation
    im = ax.imshow(velocity_history[0], cmap='viridis', animated=True, 
                   extent=[0, 1, 0, 1], origin='lower', vmin=0, vmax=1)
    ax.set_title('Lid-Driven Cavity Flow: Velocity Magnitude')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.colorbar(im, ax=ax, label='Velocity Magnitude')
    
    def update(frame):
        im.set_array(velocity_history[frame])
        return [im]
    
    anim = animation.FuncAnimation(fig, update, frames=len(velocity_history), 
                                   interval=50, blit=True)
    
    # Save animation
    anim.save('navier_stokes_simulation.gif', writer='pillow')
    
    plt.close(fig)
    
    # Create final state plot
    plt.figure(figsize=(10, 8))
    plt.imshow(velocity_history[-1], cmap='viridis',extent=[0.0, 1,0, 0.0, 1.0], origin='lower', vmin=0, vmax=1)
#     extent=[0, 1, 0, 1], origin='lower',
    plt.title('Final State: Velocity Magnitude')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.colorbar(label='Velocity Magnitude')
    plt.savefig('navier_stokes_final_state.png')
    plt.close()

if __name__ == '__main__':
    main()
