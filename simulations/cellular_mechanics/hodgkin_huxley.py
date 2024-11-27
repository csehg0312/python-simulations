import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import networkx as nx
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans

LOKY_MAX_CPU_COUNT = 2

class HodgkinHuxley:
    def __init__(self):
        # Standard Hodgkin-Huxley parameters
        self.C_m = 1.0  # membrane capacitance
        self.g_Na = 120.0  # sodium conductance
        self.g_K = 36.0  # potassium conductance
        self.g_L = 0.3  # leak conductance
        self.E_Na = 55.0  # sodium reversal potential
        self.E_K = -77.0  # potassium reversal potential
        self.E_L = -54.4  # leak reversal potential

    def alpha_n(self, V):
        return 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10)) if abs(V + 55) > 1e-7 else 0.1

    def beta_n(self, V):
        return 0.125 * np.exp(-(V + 65) / 80)

    def alpha_m(self, V):
        return 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10)) if abs(V + 40) > 1e-7 else 1.0

    def beta_m(self, V):
        return 4.0 * np.exp(-(V + 65) / 18)

    def alpha_h(self, V):
        return 0.07 * np.exp(-(V + 65) / 20)

    def beta_h(self, V):
        return 1.0 / (1 + np.exp(-(V + 35) / 10))

    def derivatives(self, X, t, I_ext):
        V, n, m, h = X

        # Calculate ionic currents
        I_Na = self.g_Na * (m**3) * h * (V - self.E_Na)
        I_K = self.g_K * (n**4) * (V - self.E_K)
        I_L = self.g_L * (V - self.E_L)

        # Calculate derivatives
        dVdt = (I_ext - I_Na - I_K - I_L) / self.C_m
        dndt = self.alpha_n(V) * (1 - n) - self.beta_n(V) * n
        dmdt = self.alpha_m(V) * (1 - m) - self.beta_m(V) * m
        dhdt = self.alpha_h(V) * (1 - h) - self.beta_h(V) * h

        return np.array([dVdt, dndt, dmdt, dhdt])

class NeuronNetwork:
    def __init__(self, num_neurons, connectivity_probability=0.3):
        self.num_neurons = num_neurons
        self.neurons = [HodgkinHuxley() for _ in range(num_neurons)]
        self.connectivity = self.create_connectivity(connectivity_probability)
        self.synaptic_strength = 2.0  # Increased synaptic strength
        self.E_syn = 0.0

    def create_connectivity(self, probability):
        connectivity = np.random.random((self.num_neurons, self.num_neurons))
        connectivity = (connectivity < probability).astype(float)
        np.fill_diagonal(connectivity, 0)
        return connectivity * np.random.normal(1.0, 0.1, (self.num_neurons, self.num_neurons))

    def simulate(self, T, dt):
        steps = int(T/dt)
        t = np.linspace(0, T, steps)
        X = np.zeros((steps, self.num_neurons, 4))
        
        # Initial conditions
        X[0] = np.array([[-65.0, 0.317, 0.05, 0.6] for _ in range(self.num_neurons)])
        
        # Create external current input with stronger stimulus
        I_ext = np.zeros((steps, self.num_neurons))
        stim_start = int(0.1 * steps)
        stim_end = int(0.3 * steps)
        I_ext[stim_start:stim_end] = 40.0  # Increased stimulus strength
        
        # Add random fluctuations
        for i in range(self.num_neurons):
            I_ext[:, i] += np.random.normal(0, 2, size=steps)  # Increased noise

        # Simulation loop
        for i in range(1, steps):
            for j in range(self.num_neurons):
                V_pre = X[i-1, :, 0]
                I_syn = self.synaptic_strength * np.sum(self.connectivity[j] * (V_pre - self.E_syn))
                
                derivatives = self.neurons[j].derivatives(X[i-1, j], t[i-1], I_ext[i, j] + I_syn)
                X[i, j] = X[i-1, j] + derivatives * dt
                X[i, j] = np.clip(X[i, j], [-100, 0, 0, 0], [100, 1, 1, 1])

        return t, X

def analyze_network(t, X):
    df = pd.DataFrame()
    
    # Calculate features for each neuron
    for i in range(X.shape[1]):
        voltage = X[:, i, 0]
        df[f'mean_voltage_{i}'] = [np.mean(voltage)]
        df[f'max_voltage_{i}'] = [np.max(voltage)]
        df[f'min_voltage_{i}'] = [np.min(voltage)]
        
        # Count spikes (threshold crossing from below)
        threshold = -20  # Adjusted threshold for spike detection
        voltage_above_threshold = voltage > threshold
        crossings = np.where(np.diff(voltage_above_threshold))[0]
        df[f'spike_count_{i}'] = [len(crossings) // 2]  # Divide by 2 to count complete spikes
    
    # Ensure we have valid data
    if df.empty or df.isnull().values.any():
        raise ValueError("No valid features for clustering")
    
    # Prepare features for clustering
    features = df.values
    n_clusters = min(3, len(features))
    
    # Simple KMeans initialization
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(features)
    
    return df, clusters

if __name__ == '__main__':
    # Create and simulate network
    network = NeuronNetwork(num_neurons=10)
    t, X = network.simulate(T=100.0, dt=0.01)
    
    # Print simulation statistics
    print("\nSimulation Statistics:")
    print(f"Voltage range: {np.min(X[:,:,0]):.2f} to {np.max(X[:,:,0]):.2f} mV")
    print(f"Time steps: {len(t)}")
    print(f"Number of neurons: {X.shape[1]}")
    print(f"Any NaN values: {np.isnan(X).any()}")
    print(f"Any Inf values: {np.isinf(X).any()}")
    
    # Create two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot voltage traces
    for i in range(X.shape[1]):
        ax1.plot(t, X[:, i, 0], label=f'Neuron {i}')
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Membrane Potential (mV)')
    ax1.set_title('Neuron Membrane Potentials')
    ax1.grid(True)
    ax1.legend()
    
    # Plot external current
    I_ext = np.zeros_like(t)
    stim_start = int(0.1 * len(t))
    stim_end = int(0.3 * len(t))
    I_ext[stim_start:stim_end] = 40.0
    ax2.plot(t, I_ext, 'r-', label='External Current')
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Current (µA/cm²)')
    ax2.set_title('External Current Stimulus')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    try:
        df, clusters = analyze_network(t, X)
        print("\nFeature DataFrame:")
        print(df.head())
        print("\nDataFrame Statistics:")
        print(df.describe())
        print("\nCluster assignments:", clusters)
    except Exception as e:
        print(f"\nError in analysis: {e}")
        import traceback
        traceback.print_exc()