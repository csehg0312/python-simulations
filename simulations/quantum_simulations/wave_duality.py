import numpy as np
import matplotlib.pyplot as plt

def double_slit():
    x = np.linspace(-10, 10, 1000)
    y = np.linspace(-10, 10, 1000)
    X, Y = np.meshgrid(x, y)
    
    # Parameters for slits
    slit1_pos = -1
    slit2_pos = 1
    wavelength = 1
    k = 2 * np.pi / wavelength
    
    # Calculate interference pattern
    r1 = np.sqrt(X**2 + (Y - slit1_pos)**2)
    r2 = np.sqrt(X**2 + (Y - slit2_pos)**2)
    wave = np.sin(k * r1) / np.sqrt(r1) + np.sin(k * r2) / np.sqrt(r2)
    intensity = wave**2
    
    plt.figure(figsize=(10, 8))
    plt.imshow(intensity, extent=[-10, 10, -10, 10], cmap='viridis')
    plt.colorbar(label='Intensity')
    plt.title('Double Slit Interference Pattern')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

double_slit()