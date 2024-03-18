# Ecosystem Simulation Study

This project simulates an ecosystem using Python, where various entities called "blobs" interact within a grid environment. The simulation model incorporates factors such as movement, predation, reproduction, and adaptation among the blobs.

## Code Overview

The Python code consists of two main classes: `Blob` and `Grid`. Here's a brief description of each:

### Blob Class

The `Blob` class represents individual entities within the ecosystem. Each blob has characteristics such as position, energy level, dietary preferences, speed, vision, and reproductive traits. The class methods handle movement, vital functions (e.g., gaining energy), interactions (e.g., predation), and reproduction.

### Grid Class

The `Grid` class represents the environment in which blobs reside. It manages the spatial arrangement of blobs and provides methods for querying neighboring blobs and updating their positions.

## Simulation Execution

The simulation is executed using Pygame, a Python library for creating interactive applications. Blobs move, interact, and reproduce based on predefined rules and parameters. The simulation loop processes each iteration, updating the state of the ecosystem.

## Features and Functionalities

- **Blob Characteristics:** Each blob possesses unique traits such as diet, speed, vision, and reproductive behavior, influencing its interactions and survival.
- **Predation:** Blobs engage in predation, with outcomes determined by factors like aggressiveness and physical characteristics.
- **Reproduction:** Blobs can reproduce when meeting certain energy thresholds, producing offspring with genetic variations.
- **Interactive Control:** Users can pause the simulation, adjust simulation speed, and interact with individual blobs by selecting them and performing actions such as color coding or elimination.

## Parameters and Tuning

The simulation offers tunable parameters such as metabolism rate, energy thresholds for reproduction, genetic variability, and herbivore gain. These parameters allow for experimentation and observation of their impact on ecosystem dynamics.

## Statistical Analysis and Visualization

The simulation collects statistics on population size, average blob traits, and computational performance. These data are visualized through plots to analyze trends and patterns over time.

## Conclusion

By simulating the interactions within an ecosystem, this project provides insights into emergent behaviors, population dynamics, and the role of individual traits in shaping ecological processes. Through parameter tuning and analysis, users can explore various scenarios and gain a deeper understanding of ecosystem dynamics.
