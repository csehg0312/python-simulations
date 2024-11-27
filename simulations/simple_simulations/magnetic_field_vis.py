import numpy as np
import matplotlib.pyplot as plt

def magnetic_field():
    def B_field(x, y, q1, q2):
        r1 = np.sqrt((x - q1[0])**2 + (y - q1[1])**2)
        r2 = np.sqrt((x - q2[0])**2 + (y - q2[1])**2)
        
        Bx1 = q1[2] * (y - q1[1]) / r1**3
        By1 = -q1[2] * (x - q1[0]) / r1**3
        
        Bx2 = q2[2] * (y - q2[1]) / r2**3
        By2 = -q2[2] * (x - q2[0]) / r2**3
        
        return Bx1 + Bx2, By1 + By2
    
    x = np.linspace(-2, 2, 40)
    y = np.linspace(-2, 2, 40)
    X, Y = np.meshgrid(x, y)
    
    q1 = [-0.5, 0, 1]  # x, y, strength
    q2 = [0.5, 0, -1]
    
    Bx, By = B_field(X, Y, q1, q2)
    
    plt.figure(figsize=(10, 8))
    plt.streamplot(X, Y, Bx, By, density=2, color=np.sqrt(Bx**2 + By**2))
    plt.plot(q1[0], q1[1], 'ro', label='North Pole')
    plt.plot(q2[0], q2[1], 'bo', label='South Pole')
    plt.title('Magnetic Field Lines')
    plt.legend()
    plt.axis('equal')
    plt.show()

magnetic_field()