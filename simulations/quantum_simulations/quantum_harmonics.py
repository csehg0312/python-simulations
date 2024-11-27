import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def quantum_oscillator():
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.linspace(-5, 5, 200)
    t = np.linspace(0, 10, 100)
    
    def wave_function(n, x, t):
        # Hermite polynomials for first few states
        if n == 0:
            h = np.ones_like(x)
        elif n == 1:
            h = 2*x
        elif n == 2:
            h = 4*x**2 - 2
        
        return np.exp(-x**2/2) * h * np.exp(-1j*t*(n + 0.5))
    
    def update(frame):
        ax.clear()
        
        # Plot several quantum states
        for n in range(3):
            psi = wave_function(n, x, frame/10)
            ax.plot(x, np.real(psi) + n*2, label=f'n={n}')
        
        ax.set_ylim(-2, 8)
        ax.set_title('Quantum String States')
        ax.legend()
    
    anim = FuncAnimation(fig, update, frames=100, interval=50)
    plt.show()

quantum_oscillator()