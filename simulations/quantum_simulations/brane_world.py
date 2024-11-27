import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def brane_world():
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create two intersecting branes
    x = np.linspace(-2, 2, 50)
    y = np.linspace(-2, 2, 50)
    X, Y = np.meshgrid(x, y)
    
    # First brane
    Z1 = np.sin(np.sqrt(X**2 + Y**2))
    
    # Second brane
    Z2 = np.cos(np.sqrt(X**2 + Y**2)) + 1
    
    # Plot branes
    ax.plot_surface(X, Y, Z1, alpha=0.5, cmap='coolwarm')
    ax.plot_surface(X, Y, Z2, alpha=0.5, cmap='viridis')
    
    ax.set_title('Brane World Visualization')
    plt.show()

brane_world()