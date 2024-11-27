import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def calabi_yau():
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Generate data
    u = np.linspace(0, 2*np.pi, 100)
    v = np.linspace(0, 2*np.pi, 100)
    u, v = np.meshgrid(u, v)
    
    # Parametric equations for a simplified Calabi-Yau manifold
    x = (2 + np.cos(v)) * np.cos(u)
    y = (2 + np.cos(v)) * np.sin(u)
    z = np.sin(v)
    
    # Plot surface
    surf = ax.plot_surface(x, y, z, cmap='viridis', alpha=0.8)
    
    ax.set_title('Simplified Calabi-Yau Manifold')
    plt.colorbar(surf)
    plt.show()

calabi_yau()