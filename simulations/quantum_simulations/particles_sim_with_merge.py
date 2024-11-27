import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Constants
G = 6.67430e-11
PARTICLE_COUNT = 75
PARTICLE_MASS = 1.0
TIMESTEP = 0.01
TOTAL_TIME = 10.0
GRAVITY_BOOST_FACTOR = 1000
COLLISION_DISTANCE = 0.5
ELASTICITY = 0.5  # 1.0 = perfectly elastic, 0.0 = perfectly inelastic

class Particle:
    def __init__(self, x, y, vx, vy, mass):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.mass = mass
        self.active = True
        self.radius = (mass/PARTICLE_MASS)**(1/3) * COLLISION_DISTANCE

    def merge_with(self, other):
        if not self.active or not other.active:
            return
        total_mass = self.mass + other.mass
        # Conservation of momentum
        self.vx = (self.vx * self.mass + other.vx * other.mass) / total_mass
        self.vy = (self.vy * self.mass + other.vy * other.mass) / total_mass
        # Center of mass position
        self.x = (self.x * self.mass + other.x * other.mass) / total_mass
        self.y = (self.y * self.mass + other.y * other.mass) / total_mass
        self.mass = total_mass
        self.radius = (self.mass/PARTICLE_MASS)**(1/3) * COLLISION_DISTANCE
        other.active = False

    def collide(self, other):
        # Normal vector
        nx = other.x - self.x
        ny = other.y - self.y
        dist = np.sqrt(nx*nx + ny*ny)
        nx /= dist
        ny /= dist

        # Relative velocity
        dvx = self.vx - other.vx
        dvy = self.vy - other.vy
        
        # Normal velocity
        vn = dvx*nx + dvy*ny
        
        # Only collide if objects are approaching
        if vn > 0:
            return

        # Collision impulse
        m1, m2 = self.mass, other.mass
        j = -(1 + ELASTICITY) * vn
        j /= 1/m1 + 1/m2

        # Update velocities
        self.vx += j*nx/m1
        self.vy += j*ny/m1
        other.vx -= j*nx/m2
        other.vy -= j*ny/m2

        # Check for merging (based on relative velocity and mass)
        relative_speed = abs(vn)
        if relative_speed < 0.5:  # Merge if relative speed is low
            self.merge_with(other)

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
        if not p1.active:
            continue
        ax, ay = 0, 0
        for j, p2 in enumerate(particles):
            if i != j and p2.active:
                r = p1.distance_to(p2)
                if r < (p1.radius + p2.radius):
                    p1.collide(p2)
                else:
                    F = GRAVITY_BOOST_FACTOR * G * p1.mass * p2.mass / (r**2 + 1e-10)
                    ax += F * (p2.x - p1.x) / r
                    ay += F * (p2.y - p1.y) / r
        if p1.active:
            p1.update_velocity(ax, ay)
            p1.update_position()

def animate(i):
    calculate_forces()
    active_particles = [(p.x, p.y) for p in particles if p.active]
    if active_particles:
        x, y = zip(*active_particles)
        scatter.set_offsets(np.c_[x, y])
        sizes = [50 * p.mass/PARTICLE_MASS for p in particles if p.active]
        scatter.set_sizes(sizes)
    return scatter,

# Create figure and axis
fig, ax = plt.subplots()
scatter = ax.scatter([p.x for p in particles], [p.y for p in particles], 
                    s=[75 * p.mass/PARTICLE_MASS for p in particles])

# Set plot limits
ax.set_xlim(-15, 15)
ax.set_ylim(-15, 15)

# Animate
ani = animation.FuncAnimation(fig, animate, frames=int(TOTAL_TIME/TIMESTEP), 
                            interval=20, blit=True)

plt.show()