import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def holographic_principle():
    fig = plt.figure(figsize=(15, 5))
    
    # 3D bulk space
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Create a sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    
    # Plot the sphere (bulk)
    ax1.plot_surface(x, y, z, alpha=0.3, cmap='viridis')
    ax1.set_title('Bulk Space')
    
    # 2D boundary
    ax2 = fig.add_subplot(122)
    
    # Create boundary projection
    theta = np.linspace(0, 2*np.pi, 100)
    r = np.linspace(0, 1, 100)
    T, R = np.meshgrid(theta, r)
    Z = np.sin(5*T) * R
    
    # Plot the boundary
    im = ax2.pcolormesh(T, R, Z, cmap='viridis')
    ax2.set_title('Boundary Projection')
    plt.colorbar(im)
    
    plt.tight_layout()
    plt.show()

holographic_principle()
