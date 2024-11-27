import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def string_landscape():
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create potential landscape
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    
    # Complex potential function
    Z = np.sin(X) * np.cos(Y) + np.sin(2*X) * np.cos(2*Y)
    
    # Plot landscape
    surf = ax.plot_surface(X, Y, Z, cmap='terrain', alpha=0.8)
    
    # Add some points representing different vacua
    minima = ax.scatter([-2, 0, 2], [-2, 0, 2], [-1, 0, 1], 
                       c='red', s=100, label='Vacuum States')
    
    ax.set_title('String Theory Landscape')
    plt.colorbar(surf)
    plt.show()

string_landscape()