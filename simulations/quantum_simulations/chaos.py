import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint

def double_pendulum():
    def derivatives(state, t, l1, l2, m1, m2, g):
        theta1, omega1, theta2, omega2 = state
        
        c = np.cos(theta1 - theta2)
        s = np.sin(theta1 - theta2)
        
        theta1_dot = omega1
        theta2_dot = omega2
        
        omega1_dot = (-g*(2*m1 + m2)*np.sin(theta1) - m2*g*np.sin(theta1 - 2*theta2)
                     - 2*s*m2*(omega2**2*l2 + omega1**2*l1*c)) / (l1*(2*m1 + m2 - m2*c**2))
        
        omega2_dot = (2*s*(omega1**2*l1*(m1 + m2) + g*(m1 + m2)*np.cos(theta1)
                     + omega2**2*l2*m2*c)) / (l2*(2*m1 + m2 - m2*c**2))
        
        return theta1_dot, omega1_dot, theta2_dot, omega2_dot
    
    # Parameters
    l1, l2 = 1, 1  # lengths
    m1, m2 = 1, 1  # masses
    g = 9.81
    
    # Initial conditions
    theta1_0 = np.pi/2
    theta2_0 = np.pi /2
    omega1_0 = 0
    omega2_0 = 0
    
    state0 = [theta1_0, omega1_0, theta2_0, omega2_0]
    t = np.linspace(0, 10, 500)
    
    # Solve ODEs
    solution = odeint(derivatives, state0, t, args=(l1, l2, m1, m2, g))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    def update(frame):
        ax.clear()
        
        theta1, omega1, theta2, omega2 = solution[frame]
        
        x1 = l1 * np.sin(theta1)
        y1 = -l1 * np.cos(theta1)
        
        x2 = x1 + l2 * np.sin(theta2)
        y2 = y1 - l2 * np.cos(theta2)
        
        ax.plot([0, x1, x2], [0, y1, y2], 'o-')
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_title('Double Pendulum')
    
    anim = FuncAnimation(fig, update, frames=500, interval=20)
    plt.show()

double_pendulum()