import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Function that defines the differential equation for radioactive decay
def decay(N, t, decay_constant):
    dNdt = -decay_constant * N
    return dNdt

# Initial conditions: Number of atoms at t=0
N0 = {
    "Uranium-238": 100,  # Arbitrary initial number of atoms
    "Thorium-234": 100,
    "Radon-222": 100,
}

# Decay constants (lambda) in inverse time units (1/year)
# These constants represent approximate half-lives of elements
decay_constants = {
    "Uranium-238": 4.468e-18,  # years^-1 (Half-life ~ 4.468 billion years)
    "Thorium-234": 2.34e-2,    # years^-1 (Half-life ~ 24.1 days)
    "Radon-222": 0.181,        # years^-1 (Half-life ~ 3.8 days)
}

# Time points (years)
t_uranium = np.linspace(0, 5e9, 1000)  # for Uranium-238 (long timescale)
t_thorium_radon = np.linspace(0, 100, 1000)  # for Thorium-234 and Radon-222 (short timescale)

# Solve differential equation for each element
results = {}

# Uranium-238 (long timescale)
N_uranium = odeint(decay, N0["Uranium-238"], t_uranium, args=(decay_constants["Uranium-238"],))
results["Uranium-238"] = (t_uranium, N_uranium)

# Thorium-234 and Radon-222 (short timescale)
N_thorium = odeint(decay, N0["Thorium-234"], t_thorium_radon, args=(decay_constants["Thorium-234"],))
N_radon = odeint(decay, N0["Radon-222"], t_thorium_radon, args=(decay_constants["Radon-222"],))
results["Thorium-234"] = (t_thorium_radon, N_thorium)
results["Radon-222"] = (t_thorium_radon, N_radon)

# Plot the results
plt.figure(figsize=(12, 8))

# Plot Uranium-238
plt.subplot(2, 1, 1)
plt.plot(results["Uranium-238"][0], results["Uranium-238"][1], label='Uranium-238 Decay', color='blue')
plt.xlabel('Time (years)')
plt.ylabel('Number of atoms')
plt.title('Radioactive Decay of Uranium-238')
plt.yscale('log')
plt.grid(True)

# Plot Thorium-234 and Radon-222 on a short timescale
plt.subplot(2, 1, 2)
plt.plot(results["Thorium-234"][0], results["Thorium-234"][1], label='Thorium-234 Decay', color='orange')
plt.plot(results["Radon-222"][0], results["Radon-222"][1], label='Radon-222 Decay', color='green')
plt.xlabel('Time (years)')
plt.ylabel('Number of atoms')
plt.title('Radioactive Decay of Thorium-234 and Radon-222')
plt.yscale('log')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
