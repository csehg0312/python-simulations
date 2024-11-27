import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Constants
G = 6.67430e-11
PARTICLE_COUNT = 50
PARTICLE_MASS = 1.0
TIMESTEP = 0.01
TOTAL_TIME = 10.0
GRAVITY_BOOST_FACTOR = 1000

class Particle:
    def __init__(self, x, y, vx, vy, mass):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.mass = mass

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
    x = np.random.uniform(-10, 10)
    y = np.random.uniform(-10, 10)
    vx = np.random.uniform(-1, 1)
    vy = np.random.uniform(-1, 1)
    particles.append(Particle(x, y, vx, vy, PARTICLE_MASS))

def calculate_forces():
    for i, p1 in enumerate(particles):
        ax, ay = 0, 0
        for j, p2 in enumerate(particles):
            if i != j:
                r = p1.distance_to(p2)
                if r > 0.1:  # Minimum distance to prevent extreme accelerations
                    F = GRAVITY_BOOST_FACTOR * G * p1.mass * p2.mass / (r**2)
                    ax += F * (p2.x - p1.x) / r
                    ay += F * (p2.y - p1.y) / r
        p1.update_velocity(ax, ay)
        p1.update_position()

def animate(i):
    calculate_forces()
    x = [p.x for p in particles]
    y = [p.y for p in particles]
    scatter.set_offsets(np.c_[x, y])
    return scatter,

# Create figure and axis
fig, ax = plt.subplots()
scatter = ax.scatter([p.x for p in particles], [p.y for p in particles], s=50)

# Set plot limits
ax.set_xlim(-15, 15)
ax.set_ylim(-15, 15)

# Animate
ani = animation.FuncAnimation(fig, animate, 
                            frames=int(TOTAL_TIME/TIMESTEP),
                            interval=20, blit=True)

plt.show()