import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

class ReentrySimulation:
    def __init__(self):
        # Constants
        self.G = 6.67430e-11  # Universal gravitational constant
        self.M = 5.972e24     # Mass of Earth (kg)
        self.R = 6.371e6      # Radius of Earth (m)
        self.cd = 1.0		   # Drag coefficient (simplified)
        self.rho0 = 1.225     # Sea level air density (kg/m³)
        self.H = 7400         # Scale height (m)

    def air_density(self, height):
        """Calculate air density at given height using exponential model"""
        return self.rho0 * np.exp(-height / self.H)

    def drag_force(self, velocity, height, area, mass):
        """Calculate drag force"""
        rho = self.air_density(height)
        drag = 0.5 * rho * self.cd * area * velocity**2
        return drag

    def gravitational_force(self, height, mass):
        """Calculate gravitational force"""
        r = self.R + height
        return self.G * self.M * mass / r**2

    def equations_of_motion(self, state, t, mass, area, angle_of_attack):
        """Define system of differential equations for reentry"""
        height, velocity, gamma = state
        
        # Calculate forces
        rho = self.air_density(height)
        Fd = self.drag_force(velocity, height, area, mass)  # Drag force
        Fg = self.gravitational_force(height, mass)         # Gravitational force

        Cl = 0.5  # Lift coefficient (adjust as needed)
        Fl = 0.5 * rho * velocity**2 * area * Cl * np.sin(angle_of_attack)

        # Equations of motion
        dheight_dt = -velocity * np.sin(gamma)
        dvelocity_dt = (Fg - Fd * np.cos(angle_of_attack) - Fl * np.sin(gamma)) / mass
        dgamma_dt = (Fd * np.sin(angle_of_attack) - Fl * np.cos(gamma)) / (mass * velocity)

        return [dheight_dt, dvelocity_dt, dgamma_dt]

    def simulate_reentry(self, initial_height, initial_velocity, initial_gamma, mass, area, angle_of_attack, time_span):
        """Run reentry simulation"""
        t = np.linspace(0, time_span, 1000)
        initial_state = [initial_height, initial_velocity, initial_gamma]
        
        solution = odeint(
            self.equations_of_motion,
            initial_state,
            t,
            args=(mass, area, angle_of_attack)  # Pass angle_of_attack here
        )
        
        return t, solution

    def plot_results(self, t, solution, title="Reentry Trajectory"):
        """Plot simulation results"""
        height = solution[:, 0]
        velocity = solution[:, 1]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        # Height vs Time
        ax1.plot(t, height/1000)  # Convert to km
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Height (km)')
        ax1.set_title(f'{title} - Height Profile')
        ax1.grid(True)
        
        # Velocity vs Time
        ax2.plot(t, velocity/1000)  # Convert to km/s
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Velocity (km/s)')
        ax2.set_title('Velocity Profile')
        ax2.grid(True)
        
        plt.tight_layout()
        return fig

# Example usage
sim = ReentrySimulation()

# Simulation parameters
initial_height = 100000  # 100 km
initial_velocity = 7800  # 7.8 km/s (typical orbital velocity)
mass = 1000             # 1000 kg spacecraft
area = 10               # 10 m² cross-sectional area
time_span = 500         # 500 seconds simulation
angle_of_attack = np.radians(15)
initial_gamma = 0.0  # Initial flight path angle in radians

# Run simulation
t, solution = sim.simulate_reentry(initial_height, initial_velocity, initial_gamma, mass, area, angle_of_attack, time_span)
# Plot results
fig = sim.plot_results(t, solution)
plt.show()