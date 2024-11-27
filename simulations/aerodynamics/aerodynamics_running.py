import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class DirectionalStreamlineVisualizer:
    def __init__(self, flow_conditions):
        # Grid setup
        self.x = np.linspace(0, 10, 50)
        self.y = np.linspace(0, 10, 50)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        # Flow parameters
        self.base_velocity = flow_conditions.get('velocity', 5)
        
        # Obstacle parameters
        self.obstacle_x = flow_conditions.get('obstacle_x', 5)
        self.obstacle_y = flow_conditions.get('obstacle_y', 5)
        self.obstacle_radius = flow_conditions.get('obstacle_radius', 1)
        
        # Animation setup
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.time = 0
        
    def generate_unidirectional_velocity_field(self):
        # Consistent horizontal flow from left to right
        u = np.ones_like(self.X) * self.base_velocity
        
        # Vertical component to simulate flow disturbance around obstacle
        v = np.zeros_like(self.Y)
        
        # Create flow disturbance around obstacle
        distance_to_obstacle = np.sqrt(
            (self.X - self.obstacle_x)**2 + 
            (self.Y - self.obstacle_y)**2
        )
        
        # Flow deflection around obstacle
        v += 2 * self.base_velocity * np.exp(
            -((distance_to_obstacle / self.obstacle_radius)**2)
        )
        
        return u, v
    
    def update_streamlines(self, frame):
        # Clear previous plot
        self.ax.clear()
        
        # Generate velocity field
        u, v = self.generate_unidirectional_velocity_field()
        
        # Create streamplot with directional flow
        streamplot = self.ax.streamplot(
            self.X, self.Y,     # Grid coordinates
            u, v,               # Velocity components
            density=1.5,        # Streamline density
            color='blue',       # Consistent color
            linewidth=1,        # Line thickness
            arrowsize=1.2,      # Arrow size
            arrowstyle='->'     # Directional arrows
        )
        
        # Draw obstacle
        obstacle = plt.Circle(
            (self.obstacle_x, self.obstacle_y), 
            self.obstacle_radius, 
            color='red', 
            fill=True
        )
        self.ax.add_artist(obstacle)
        
        # Plot styling
        self.ax.set_title('Directional Flow with Obstacle')
        self.ax.set_xlabel('X Coordinate')
        self.ax.set_ylabel('Y Coordinate')
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(0, 10)
        
        return streamplot
    
    def animate(self, duration=20, fps=15):
        """
        Animate the streamline visualization
        """
        anim = animation.FuncAnimation(
            self.fig, 
            self.update_streamlines, 
            frames=duration * fps,
            interval=1000/fps,
            blit=False
        )
        
        plt.tight_layout()
        plt.show()

# Flow configuration with obstacle
flow_conditions = {
    'velocity': 3,            # Base flow velocity
    'obstacle_x': 6,          # X position of obstacle
    'obstacle_y': 5,          # Y position of obstacle
    'obstacle_radius': 0.8    # Obstacle size
}

# Create and run visualization
visualizer = DirectionalStreamlineVisualizer(flow_conditions)
visualizer.animate()