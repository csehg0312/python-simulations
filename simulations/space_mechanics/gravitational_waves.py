import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

def gravitational_waves():
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    x = np.linspace(-2, 2, 30)
    y = np.linspace(-2, 2, 30)
    X, Y = np.meshgrid(x, y)
    
    def update(frame):
        ax.clear()
        
        # Calculate wave effect
        omega = 2
        k = 1
        Z = np.sin(k*np.sqrt(X**2 + Y**2) - omega*frame/20)
        
        ax.plot_surface(X, Y, Z, cmap='viridis')
        ax.set_zlim(-2, 2)
        ax.set_title('Gravitational Wave Propagation')
    
    anim = FuncAnimation(fig, update, frames=100, interval=50)
    plt.show()

gravitational_waves()