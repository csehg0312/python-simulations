import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.widgets import Button, Slider

class MagneticFieldVisualizer:
    def __init__(self):
        # Create figure and axes
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_axes([0.1, 0.2, 0.8, 0.7])
        
        # Initialize magnets with position and strength
        self.magnets = [
            {'pos': (-0.5, 0), 'strength': 1.0, 'color': 'red', 'label': 'N'},
            {'pos': (0.5, 0), 'strength': -1.0, 'color': 'blue', 'label': 'S'}
        ]
        
        # Visualization parameters
        self.density = 2.0
        self.grid_points = 40
        self.selected_magnet = None
        self.streamplot = None
        
        # Setup the plot and controls
        self.setup_plot()
        self.setup_controls()
        self.setup_interactions()
        self.update_field()
        
    def setup_plot(self):
        self.ax.set_xlim(-2, 2)
        self.ax.set_ylim(-2, 2)
        self.ax.set_aspect('equal')
        self.ax.set_title('Interactive Magnetic Field Visualization\n'
                         'Click and drag magnets to move them')
        self.ax.grid(True, linestyle='--', alpha=0.3)
        
    def setup_controls(self):
        # Density slider
        density_ax = self.fig.add_axes([0.1, 0.05, 0.3, 0.03])
        self.density_slider = Slider(
            density_ax, 'Field Line Density', 
            0.5, 5.0, valinit=self.density
        )
        self.density_slider.on_changed(self.on_density_change)
        
        # Buttons
        add_n_ax = self.fig.add_axes([0.5, 0.05, 0.1, 0.03])
        add_s_ax = self.fig.add_axes([0.65, 0.05, 0.1, 0.03])
        reset_ax = self.fig.add_axes([0.8, 0.05, 0.1, 0.03])
        
        self.btn_add_n = Button(add_n_ax, 'Add N')
        self.btn_add_s = Button(add_s_ax, 'Add S')
        self.btn_reset = Button(reset_ax, 'Reset')
        
        self.btn_add_n.on_clicked(self.add_north)
        self.btn_add_s.on_clicked(self.add_south)
        self.btn_reset.on_clicked(self.reset)
        
    def setup_interactions(self):
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        
    def calculate_field(self, x, y):
        Bx = np.zeros_like(x)
        By = np.zeros_like(y)
        
        for magnet in self.magnets:
            dx = x - magnet['pos'][0]
            dy = y - magnet['pos'][1]
            r_squared = dx**2 + dy**2
            r_cubed = r_squared * np.sqrt(r_squared)
            r_cubed[r_cubed < 1e-10] = 1e-10  # Avoid division by zero
            
            Bx += magnet['strength'] * dx / r_cubed
            By += magnet['strength'] * dy / r_cubed
            
        return Bx, By
    
    def update_field(self):
        self.ax.clear()
        self.setup_plot()
        
        # Create grid
        x = np.linspace(-2, 2, self.grid_points)
        y = np.linspace(-2, 2, self.grid_points)
        X, Y = np.meshgrid(x, y)
        
        # Calculate field
        Bx, By = self.calculate_field(X, Y)
        B_mag = np.sqrt(Bx**2 + By**2)
        
        # Plot streamlines
        self.streamplot = self.ax.streamplot(
            X, Y, Bx, By,
            density=self.density,
            color=B_mag,
            cmap='viridis',
            linewidth=1,
            arrowsize=1.5
        )
        
        # Plot magnets
        for magnet in self.magnets:
            circle = Circle(
                magnet['pos'], 
                0.1, 
                color=magnet['color'],
                zorder=10
            )
            self.ax.add_patch(circle)
            self.ax.text(
                magnet['pos'][0], 
                magnet['pos'][1], 
                magnet['label'],
                ha='center', 
                va='center', 
                color='white',
                fontweight='bold',
                zorder=11
            )
        
        self.fig.canvas.draw_idle()
    
    def find_nearest_magnet(self, event):
        if event.inaxes != self.ax:
            return None
        
        distances = [np.sqrt((event.xdata - m['pos'][0])**2 + 
                           (event.ydata - m['pos'][1])**2) 
                    for m in self.magnets]
        
        min_dist_idx = np.argmin(distances)
        if distances[min_dist_idx] < 0.2:
            return min_dist_idx
        return None
    
    def on_click(self, event):
        if event.button != 1:  # Left click only
            return
        self.selected_magnet = self.find_nearest_magnet(event)
    
    def on_release(self, event):
        self.selected_magnet = None
    
    def on_motion(self, event):
        if self.selected_magnet is not None and event.inaxes == self.ax:
            self.magnets[self.selected_magnet]['pos'] = (event.xdata, event.ydata)
            self.update_field()
    
    def on_density_change(self, val):
        self.density = val
        self.update_field()
    
    def add_north(self, event):
        self.magnets.append({
            'pos': (0, 0),
            'strength': 1.0,
            'color': 'red',
            'label': 'N'
        })
        self.update_field()
    
    def add_south(self, event):
        self.magnets.append({
            'pos': (0, 0),
            'strength': -1.0,
            'color': 'blue',
            'label': 'S'
        })
        self.update_field()
    
    def reset(self, event):
        self.magnets = [
            {'pos': (-0.5, 0), 'strength': 1.0, 'color': 'red', 'label': 'N'},
            {'pos': (0.5, 0), 'strength': -1.0, 'color': 'blue', 'label': 'S'}
        ]
        self.density = 2.0
        self.density_slider.set_val(2.0)
        self.update_field()

# Create and display the visualization
visualizer = MagneticFieldVisualizer()
plt.show()