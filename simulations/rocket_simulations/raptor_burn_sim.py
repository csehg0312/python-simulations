import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Constants
Isp = 330  # Specific impulse in seconds
g0 = 9.81  # Acceleration due to gravity in m/s^2
mass_flow_rate_fuel = 250  # kg/s (example value)
mass_flow_rate_oxidizer = 500  # kg/s (example value)

# Define fuel consumption rate
def fuel_consumption_rate(fuel, oxidizer):
    return mass_flow_rate_fuel

# Define oxidizer consumption rate
def oxidizer_consumption_rate(fuel, oxidizer):
    return mass_flow_rate_oxidizer

# Define thrust generation
def thrust_generation(fuel, oxidizer):
    return (mass_flow_rate_fuel + mass_flow_rate_oxidizer) * g0 * Isp

# Differential equations
def raptor_engine(t, y):
    fuel, oxidizer, thrust = y
    d_fuel = -fuel_consumption_rate(fuel, oxidizer)
    d_oxidizer = -oxidizer_consumption_rate(fuel, oxidizer)
    d_thrust = thrust_generation(fuel, oxidizer)
    return [d_fuel, d_oxidizer, d_thrust]

# Initial conditions
initial_conditions = [1000, 2000, 0]  # Example values for fuel, oxidizer, thrust

# Time span for simulation
t_span = np.linspace(0, 100, 1000)  # 100 seconds

# Runge-Kutta integration
solution = solve_ivp(raptor_engine, [t_span[0], t_span[-1]], initial_conditions, t_eval=t_span)

# Extract thrust values, fuel, and oxidizer from the solution
thrust_values = solution.y[2]  # Thrust values at each time step
initial_fuel = initial_conditions[0]
initial_oxidizer = initial_conditions[1]

# Calculate total fuel and oxidizer used
fuel_used = initial_fuel - solution.y[0]  # Fuel consumed
oxidizer_used = initial_oxidizer - solution.y[1]  # Oxidizer consumed

# Create subplots for thrust, fuel used, and oxidizer used
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

# Plot thrust
ax1.plot(solution.t, thrust_values, label='Thrust (N)', color='green')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Thrust (N)')
ax1.set_title('Instantaneous Thrust of Raptor Engine Over Time')
ax1.grid()
ax1.legend()

# Plot fuel used
ax2.plot(solution.t, fuel_used, label='Fuel Used (kg)', color='blue')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Fuel Used (kg)')
ax2.set_title('Fuel Used Over Time')
ax2.grid()
ax2.legend()

# Plot oxidizer used
ax3.plot(solution.t, oxidizer_used, label='Oxidizer Used (kg)', color='orange')
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Oxidizer Used (kg)')
ax3.set_title('Oxidizer Used Over Time')
ax3.grid()
ax3.legend()

# Adjust layout
plt.tight_layout()
plt.show()