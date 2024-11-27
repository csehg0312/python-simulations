import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, FancyArrowPatch
import mpl_toolkits.mplot3d.art3d as art3d

def create_atmosphere_visualization():
    
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111, projection='3d')
    
    radius = 6371
    theta = np.linspace(0,2*np.pi, 100)
    phi = np.linspace(0,np.pi, 100)
    
    x = radius * np.outer(np.cos(theta), np.sin(phi))
    y = radius * np.outer(np.sin(theta), np.sin(phi))
    z = radius * np.outer(np.ones(np.size(theta)), np.cos(phi))
    
    earth = ax.plot_surface(x,y,z, color='lightblue', alpha=0.6)
    
    layers = {
        'Troposhphere': 12,
        'Stratosphere':50,
        'Mesosphere':80,
        'Thermosphere':700,
        'Exosphere':10000
        }
    
    for layer, scale in layers.items():
        x_layer = scale * x
        y_layer = scale * y
        z_layer = scale * z
        ax.plot_surface(x_layer, y_layer, z_layer, alpha=0.1)
        
    theta = np.linspace(0, 4*np.pi, 100)
    z_jet = 0.3 * np.sin(theta)
    x_jet = 1.2 * np.cos(theta)
    y_jet = 1.2 * np.sin(theta)
    ax.plot(x_jet, y_jet, z_jet, 'r-', label="Jet Stream", linewidth=2)
    
    theta_trade = np.linspace(0,2*np.pi, 100)
    z_trade = 0.1 * np.ones_like(theta_trade)
    x_trade = 1.1 * np.cos(theta_trade)
    y_trade = 1.1 * np.sin(theta_trade)
    ax.plot(x_trade, y_trade, z_trade, 'g-', label="Trade Winds", linewidth=2)
    
    ax.set_title("Earth's atmosphere and Major Air Currents", pad=20)
#     ax.legend()
    
#     ax.set_axis_off()
    
    for layer, scale in layers.items():
        ax.text(scale+0.1, 0,0, layer, fontsize=8)
        
    plt.show()
    
create_atmosphere_visualization()