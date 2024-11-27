
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

class HodgkinHuxley:
    def __init__(self):
        # Membrane capacitance (μF/cm²)
        self.C_m = 1.0
        
        # Maximum conductances (mS/cm²)
        self.g_Na = 120.0
        self.g_K = 36.0
        self.g_L = 0.3
        
        # Reversal potentials (mV)
        self.E_Na = 50.0
        self.E_K = -77.0
        self.E_L = -54.387
        
    def alpha_n(self, V):
        return 0.01 * (V + 55.0) / (1.0 - np.exp(-(V + 55.0) / 10.0))
    
    def beta_n(self, V):
        return 0.125 * np.exp(-(V + 65.0) / 80.0)
    
    def alpha_m(self, V):
        return 0.1 * (V + 40.0) / (1.0 - np.exp(-(V + 40.0) / 10.0))
    
    def beta_m(self, V):
        return 4.0 * np.exp(-(V + 65.0) / 18.0)
    
    def alpha_h(self, V):
        return 0.07 * np.exp(-(V + 65.0) / 20.0)
    
    def beta_h(self, V):
        return 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))
    
    def derivatives(self, X, t, I_ext):
        V, n, m, h = X
        
        # Ion currents
        I_Na = self.g_Na * m**3 * h * (V - self.E_Na)
        I_K = self.g_K * n**4 * (V - self.E_K)
        I_L = self.g_L * (V - self.E_L)
        
        # Membrane potential derivative
        dVdt = (I_ext - I_Na - I_K - I_L) / self.C_m
        
        # Gating variables derivatives
        dndt = self.alpha_n(V) * (1.0 - n) - self.beta_n(V) * n
        dmdt = self.alpha_m(V) * (1.0 - m) - self.beta_m(V) * m
        dhdt = self.alpha_h(V) * (1.0 - h) - self.beta_h(V) * h
        
        return [dVdt, dndt, dmdt, dhdt]

def simulate_HH():
    # Initialize the model
    hh = HodgkinHuxley()
    
    # Time parameters
    T = 50.0  # Total simulation time (ms)
    dt = 0.01  # Time step (ms)
    t = np.arange(0.0, T, dt)
    
    # External current stimulus
    I_ext = np.zeros(len(t))
    I_ext[(t >= 10) & (t <= 40)] = 10  # Apply 10 μA/cm² current between 10-40 ms
    
    # Initial conditions [V, n, m, h]
    X0 = [-65.0, 0.317, 0.05, 0.6]
    
    # Solve ODE system
    solution = []
    X = X0
    for i in range(len(t)):
        solution.append(X)
        X = X + np.array(hh.derivatives(X, t[i], I_ext[i])) * dt
    
    solution = np.array(solution)
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Membrane potential
    plt.subplot(2, 1, 1)
    plt.plot(t, solution[:, 0], 'b', label='Membrane Potential')
    plt.ylabel('Voltage (mV)')
    plt.title('Hodgkin-Huxley Model Simulation')
    plt.grid(True)
    plt.legend()
    
    # Gating variables
    plt.subplot(2, 1, 2)
    plt.plot(t, solution[:, 1], 'r', label='n (K+ activation)')
    plt.plot(t, solution[:, 2], 'g', label='m (Na+ activation)')
    plt.plot(t, solution[:, 3], 'y', label='h (Na+ inactivation)')
    plt.xlabel('Time (ms)')
    plt.ylabel('Gating value')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    simulate_HH()
