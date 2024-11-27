import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import networkx as nx
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from hodgkin_huxley import HodgkinHuxley

class NeuronNetwork:
    def __init__(self, num_neurons, connectivity_probability=0.3):
        self.num_neurons = num_neurons
        self.neurons = [HodgkinHuxley() for _ in range(num_neurons)]
        self.connectivity = self.create_connectivity(connectivity_probability)
        self.synaptic_strength = 0.5
        self.E_syn = 0.0

    def create_connectivity(self, probability):
        connectivity = np.random.random((self.num_neurons, self.num_neurons))
        connectivity = (connectivity < probability).astype(float)
        np.fill_diagonal(connectivity, 0)
        return connectivity * np.random.normal(0.5, 0.1, (self.num_neurons, self.num_neurons))

    def simulate(self, T, dt):
        t = np.arange(0.0, T, dt)
        X = np.zeros((len(t), self.num_neurons, 4))
        X[0] = np.array([[-65.0, 0.317, 0.05, 0.6] for _ in range(self.num_neurons)])

        I_ext = np.zeros((len(t), self.num_neurons))
        I_ext[(t >= 10) & (t <= 40), :] = np.random.normal(20, 5, self.num_neurons)

        for i in range(1, len(t)):
            for j in range(self.num_neurons):
                I_syn = np.sum(self.connectivity[j] * (X[i-1, :, 0] - self.E_syn))
                derivatives = self.neurons[j].derivatives(X[i-1, j], t[i], I_ext[i, j] + I_syn)
                X[i, j] = X[i-1, j] + derivatives * dt
                X[i, j] = np.clip(X[i, j], [-100, 0, 0, 0], [100, 1, 1, 1])

        return t, X

def analyze_network(t, X):
    df = pd.DataFrame()
    
    print("Simulation summary:")
    print(f"Shape of X: {X.shape}")
    print(f"Range of voltage: {np.min(X[:,:,0]):.2f} to {np.max(X[:,:,0]):.2f}")
    print(f"Any NaN values: {np.isnan(X).any()}")
    print(f"Any Inf values: {np.isinf(X).any()}")
    
    for i in range(X.shape[1]):
        df[f'mean_voltage_{i}'] = np.mean(X[:, i, 0])
        df[f'max_voltage_{i}'] = np.max(X[:, i, 0])
        df[f'spike_count_{i}'] = len(np.where(np.diff(X[:, i, 0]) > 20)[0])
    
    print("\nDataFrame summary:")
    print(df.describe())
    
    features = df.values
    
    if len(features) == 0 or np.isnan(features).any() or np.isinf(features).any():
        raise ValueError("No valid features for clustering")
    
    # Add a small amount of noise to prevent singular matrix
    features = features + np.random.normal(0, 1e-6, features.shape)
    
    n_clusters = min(3, len(features))
    kmeans = KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit_predict(features)
    
    return df, clusters

def visualize_results(t, X, clusters, network):
    plt.figure(figsize=(15, 10))
    
    # Plot membrane potentials
    plt.subplot(2, 2, 1)
    for i in range(X.shape[1]):
        plt.plot(t, X[:, i, 0], alpha=0.5, label=f'Neuron {i}')
    plt.ylabel('Voltage (mV)')
    plt.title('Membrane Potentials')
    plt.grid(True)
    
    # Plot correlation matrix
    plt.subplot(2, 2, 2)
    corr_matrix = np.corrcoef(X[:, :, 0].T)
    sns.heatmap(corr_matrix, cmap='coolwarm')
    plt.title('Correlation Matrix')
    
    # Plot network graph
    plt.subplot(2, 2, 3)
    G = nx.from_numpy_array(network.connectivity)
    pos = nx.spring_layout(G)
    nx.draw(G, pos, node_color=clusters, node_size=500, cmap=plt.cm.viridis)
    plt.title('Network Graph')
    
    # Plot clustering results
    plt.subplot(2, 2, 4)
    for cluster in np.unique(clusters):
        mask = clusters == cluster
        plt.plot(t, X[:, mask, 0].mean(axis=1), 
                label=f'Cluster {cluster}', linewidth=2)
    plt.ylabel('Average Voltage (mV)')
    plt.title('Cluster Average Behaviors')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    network = NeuronNetwork(num_neurons=10)
    t, X = network.simulate(T=100.0, dt=0.01)

    print("Simulation summary:")
    print(f"Shape of X: {X.shape}")
    print(f"Range of voltage: {np.min(X[:,:,0]):.2f} to {np.max(X[:,:,0]):.2f}")
    print(f"Any NaN values: {np.isnan(X).any()}")
    print(f"Any Inf values: {np.isinf(X).any()}")

    try:
        df, clusters = analyze_network(t, X)
        visualize_results(t, X, clusters, network)
    except ValueError as e:
        print(f"Error: {e}")
        print("DataFrame summary:")
        print(df.describe())

    # Plot individual neuron voltages
    plt.figure(figsize=(12, 6))
    for i in range(X.shape[1]):
        plt.plot(t, X[:, i, 0], label=f'Neuron {i}')
    plt.xlabel('Time (ms)')
    plt.ylabel('Voltage (mV)')
    plt.title('Individual Neuron Voltages')
    plt.legend()
    plt.grid(True)
    plt.show()