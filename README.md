# Scientific Simulations Repository

## Overview

This repository contains a collection of scientific simulations spanning various domains including:

- Space Mechanics
- Quantum Simulations
- Fluid Dynamics
- Cellular Mechanics
- Earth Simulations
- Rocket Simulations
- Quantum Physics
- Particle Interactions

## Features

- 3D Visualization of Complex Systems
- Numerical Simulations
- Interactive Animations
- Scientific Modeling

## Simulations Catalog

### Space Mechanics
- Solar System Movement
- Moon-Earth Orbital Dynamics
- Planetary Motion Simulation

### Quantum Simulations
- Particle Interactions
- Quantum Harmonic Oscillator
- Holographic Principle Visualization
- String Theory Landscape

### Fluid Dynamics
- 2D Cavity Flow
- Directional Streamline Visualization

### And More...

## Prerequisites

- Python 3.12
- Poetry (Dependency Management)

## Installation

1. Clone the repository
2. Install Poetry (see POETRY_GUIDE.md)
3. Install dependencies:
   ```bash
   poetry install
   ```

## Running simulation

```bash
poetry run python simulations/space_mechanics/sun_movement_3d.py
```



# Poetry Usage Guide

## What is Poetry?

Poetry is a tool for dependency management and packaging in Python. It allows you to declare the libraries your project depends on and it will manage (install/update) them for you.

## Installation

### macOS / Linux / Windows (WSL)
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

### Windown (Powershell)

```bash
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
```

### Basic commands

* Create a new project

```bash
poetry new my_project
cd my_project
```

 * Add dependencies
```bash
# Add a production dependency
poetry add numpy matplotlib

# Add a development dependency
poetry add --group dev pytest
```

* Install project dependencies
```bash
poetry install
```

* Run scripts
```bash
# Run a specific script
poetry run python simulations/space_mechanics/sun_movement_3d.py

# Start a Python shell with project dependencies
poetry shell
```

* Update dependecies
```bash
poetry update
```

* Remove a dependency
```bash
poetry update
```

### Virtual Environment
Poetry automatically creates and manages a virtual environment for your project. You can:

1. Specify Python version
2. Activate/deactivate environment
3. Configure environment location

#### Best Practices
* Always commit pyproject.toml and poetry.lock
* Use `poetry add` instead of `pip install`
* Specify version constraints
* Separate dev and production dependencies
  
#### Troubleshooting
`poetry env list`: List virtual environments
`poetry env remove`: Remove a specific environment
`poetry config virtualenvs.in-project true`: Create virtualenv in project directory

#### More Information
Official Documentation: <https://python-poetry.org/docs/>
