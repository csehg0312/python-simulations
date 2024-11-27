import numpy as np
import matplotlib.pyplot as plt

# Constants and Parameters
g = 9.81  # m/s^2, gravitational acceleration
rho0 = 1.225  # kg/m^3, air density at sea level
H = 8000.0  # m, scale height for air density

# Rocket parameters
mass = 100.0  # kg
I = 100.0  # kg.m^2, moment of inertia around pitch axis
thrust = 2000.0  # N, constant thrust
Cd = 0.2  # Drag coefficient
A = 0.5  # m^2, cross-sectional area
fin_area = 0.2  # m^2, fin area
max_fin_deflection = np.deg2rad(30.0)  # Maximum fin deflection in radians

# Simulation parameters
dt = 0.01  # s, time step
total_time = 100.0  # s, total simulation time

# Wind profile function (simple model)
def wind_speed(altitude):
    if altitude < 5000.0:
        return 10.0 + (altitude / 1000.0) * 5.0  # m/s, increases with altitude
    else:
        return 30.0  # m/s, constant above 5000 m

# Air density model (barometric formula)
def air_density(altitude):
    return rho0 * np.exp(-altitude / H)

# Initialize arrays to store data
time = np.arange(0.0, total_time, dt)
altitude = np.zeros_like(time)
velocity = np.zeros_like(time)
pitch_angle = np.zeros_like(time)
fin_deflection = np.zeros_like(time)

# Initial conditions
altitude[0] = 0.0  # m
velocity[0] = 0.0  # m/s
pitch_angle[0] = 0.0  # radians
angular_velocity = 0.0  # radians/s

# Control gain for fin deflection
Kp = 0.1  # Proportional gain

# Simulation loop
for i in range(len(time) - 1):
    # Current state
    t = time[i]
    h = altitude[i]
    v = velocity[i]
    phi = pitch_angle[i]
    
    # Wind speed at current altitude
    v_wind = wind_speed(h)
    
    # Relative velocity
    v_rel = v - v_wind * np.sin(phi)
    
    # Dynamic pressure
    q = 0.5 * air_density(h) * v_rel**2
    
    # Drag force
    drag = 0.5 * Cd * A * q
    
    # Net force
    net_force = thrust - mass * g - drag
    
    # Linear acceleration
    acc = net_force / mass
    
    # Lift force on fins (simplified model)
    # Lift = fin_area * q * fin_deflection_angle
    # Torque due to lift
    fin_deflection[i] = np.clip(-Kp * phi, -max_fin_deflection, max_fin_deflection)
    lift = fin_area * q * fin_deflection[i]
    torque = lift * (0.5 * h)  # Assuming fins are at mid-height
    
    # Angular acceleration
    angular_acc = torque / I
    
    # Update angular velocity and pitch angle
    angular_velocity += angular_acc * dt
    pitch_angle[i+1] = phi + angular_velocity * dt
    
    # Update velocity and altitude
    velocity[i+1] = v + acc * dt
    altitude[i+1] = h + v * dt
    
    # Termination condition if rocket lands
    if altitude[i+1] < 0.0:
        altitude[i+1:] = 0.0
        velocity[i+1:] = 0.0
        pitch_angle[i+1:] = 0.0
        fin_deflection[i+1:] = 0.0
        break

# Plotting results
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(time, altitude)
plt.xlabel('Time (s)')
plt.ylabel('Altitude (m)')
plt.title('Altitude vs. Time')

plt.subplot(2, 2, 2)
plt.plot(time, velocity)
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.title('Velocity vs. Time')

plt.subplot(2, 2, 3)
plt.plot(time, np.rad2deg(pitch_angle))
plt.xlabel('Time (s)')
plt.ylabel('Pitch Angle (degrees)')
plt.title('Pitch Angle vs. Time')

plt.subplot(2, 2, 4)
plt.plot(time, np.rad2deg(fin_deflection))
plt.xlabel('Time (s)')
plt.ylabel('Fin Deflection (degrees)')
plt.title('Fin Deflection vs. Time')

plt.tight_layout()
plt.show()