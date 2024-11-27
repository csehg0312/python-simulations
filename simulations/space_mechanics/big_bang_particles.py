import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Constants
G = 6.67430e-11
PARTICLE_COUNT = 100
PARTICLE_MASS = 1.0
TIMESTEP = 0.05
TOTAL_TIME = 20.0
COLLISION_DISTANCE = 0.5
GRAVITY_BOOST_FACTOR = 1e4
INITIAL_EXPLOSION_SPEED = 10.0  # Reduced speed
EXPLOSION_DELAY = 20

class Particle:
    def __init__(self, x, y, vx, vy, mass):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.mass = mass
        self.active = True
        self.exploded = False

    def merge_with(self, other):
        total_mass = self.mass + other.mass
        self.x = (self.x * self.mass + other.x * other.mass) / total_mass
        self.y = (self.y * self.mass + other.y * other.mass) / total_mass
        self.vx = (self.vx * self.mass + other.vx * other.mass) / total_mass
        self.vy = (self.vy * self.mass + other.vy * other.mass) / total_mass
        self.mass = total_mass
        other.active = False

    def update_position(self):
        self.x += self.vx * TIMESTEP
        self.y += self.vy * TIMESTEP

    def update_velocity(self, ax, ay):
        self.vx += ax * TIMESTEP
        self.vy += ay * TIMESTEP

    def distance_to(self, other):
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

# Initialize particles
particles = []
for _ in range(PARTICLE_COUNT):
    x = np.random.uniform(-0.001, 0.001)
    y = np.random.uniform(-0.001, 0.001)
    vx = 0
    vy = 0
    mass = PARTICLE_MASS * np.random.uniform(0.5, 2.0)
    particles.append(Particle(x, y, vx, vy, mass))

frame_count = 0
explosion_triggered = False
plot_radius = 5.0  # Initial plot radius

def trigger_explosion():
    global explosion_triggered
    if not explosion_triggered:
        for p in particles:
            if p.active and not p.exploded:
                angle = np.arctan2(p.y, p.x)
                # Add some randomness to the explosion direction
                angle += np.random.uniform(-0.5, 0.5)
                speed = INITIAL_EXPLOSION_SPEED * (1 + np.random.uniform(-0.2, 0.2))
                p.vx = speed * np.cos(angle)
                p.vy = speed * np.sin(angle)
                p.exploded = True
        explosion_triggered = True

def calculate_forces():
    for i, p1 in enumerate(particles):
        if not p1.active:
            continue
        ax, ay = 0, 0
        for j, p2 in enumerate(particles):
            if i != j and p2.active:
                r = p1.distance_to(p2)
                if r < COLLISION_DISTANCE:
                    p1.merge_with(p2)
                else:
                    F = GRAVITY_BOOST_FACTOR * G * p1.mass * p2.mass / (r**2 + 1e-10)
                    ax += F * (p2.x - p1.x) / r
                    ay += F * (p2.y - p1.y) / r
        if p1.active:
            p1.update_velocity(ax, ay)
            p1.update_position()

def get_plot_bounds():
    active_particles = [(p.x, p.y) for p in particles if p.active]
    if active_particles:
        x, y = zip(*active_particles)
        max_extent = max(max(abs(np.array(x))), max(abs(np.array(y))))
        return max(5.0, max_extent * 1.2)  # Add some padding
    return 5.0

def animate(frame):
    global frame_count, plot_radius
    frame_count += 1
    
    if frame_count == EXPLOSION_DELAY:
        trigger_explosion()
    
    calculate_forces()
    
    # Update plot bounds based on particle positions
    plot_radius = get_plot_bounds()
    ax.set_xlim(-plot_radius, plot_radius)
    ax.set_ylim(-plot_radius, plot_radius)
    
    active_particles = [(p.x, p.y) for p in particles if p.active]
    if active_particles:
        x, y = zip(*active_particles)
        scatter.set_offsets(np.c_[x, y])
        sizes = [200 * (p.mass/PARTICLE_MASS) for p in particles if p.active]  # Increased particle size
        colors = [p.mass for p in particles if p.active]
        scatter.set_sizes(sizes)
        scatter.set_array(np.array(colors))
    return scatter,

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 10))
scatter = ax.scatter([p.x for p in particles], [p.y for p in particles], 
                    s=[200 * (p.mass/PARTICLE_MASS) for p in particles],
                    c=[p.mass for p in particles],
                    cmap='hot')

# Set initial plot limits
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_facecolor('black')
fig.patch.set_facecolor('black')

# Add title
plt.title('Big Bang Simulation', color='white', size=14)

# Add grid for better perspective
ax.grid(True, color='gray', alpha=0.2)

# Animate
ani = animation.FuncAnimation(fig, animate, 
                            frames=int(TOTAL_TIME/TIMESTEP),
                            interval=20, 
                            blit=True)

plt.show()