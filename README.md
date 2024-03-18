# Ecosystem Simulation Study

This project simulates an ecosystem using Python, where various entities called "blobs" interact within a grid environment. The simulation model incorporates factors such as movement, predation, reproduction, and adaptation among the blobs.

## Code Overview

The Python code consists of two main classes: `Blob` and `Grid`. Here's a brief description of each:

### Blob Class

The `Blob` class represents individual entities within the ecosystem. Each blob has characteristics such as position, energy level, dietary preferences, speed, vision, and reproductive traits. The class methods handle movement, vital functions (e.g., gaining energy), interactions (e.g., predation), and reproduction.

#### Blob Parameters

In Blob World, each blob is characterized by a set of parameters that influence its behavior, interactions, and survival. Understanding these parameters is crucial for managing and studying the dynamics of Blob World ecosystems. Here's a breakdown of the key parameters:

#### 1. Position
- **x**: Represents the horizontal position of the blob.
- **y**: Represents the vertical position of the blob.

#### 2. Energy
- **energy**: Indicates the amount of energy stored in the blob, ranging from 0 to 100.

#### 3. Feeding Preferences
- **carno**: Specifies how much energy the blob gains from feeding on prey (0-1 scale).
- **herbo**: Indicates how much energy the blob gains from the surroundings (0-1 scale).

#### 4. Movement
- **speed**: Represents the average speed of the blob as it moves across the environment (0-1 scale).
- **vision**: Determines the blob's visual range, with the actual range calculated as vision divided by 0.15 (0-1 scale).

#### 5. Age
- **age**: Represents the current age of the blob. At a certain maximum age, the blob will die.

#### 6. Combat Abilities
- **offens**: Reflects the likelihood of the blob successfully capturing and consuming other blobs (0-1 scale).
- **defens**: Represents the likelihood of the blob surviving an attack from a predator (0-1 scale).

#### 7. Reproduction
- **energy_for_babies**: Indicates the amount of energy each baby blob inherits from its parent (0-1 scale).
- **number_of_babies**: Specifies the number of babies produced by the blob (0-1 scale).

#### 8. Behavioral Traits
- **curios**: Establishes the probability of the blob moving randomly when there are no specific reasons for movement (0-1 scale).
- **agress**: Represents the likelihood of the blob stalking or pursuing other blobs even with bad combat conditions (0-1 scale).
- **colab**: Determines how likely the blob is to share its energy with other blobs with similar characteristics (0-1 scale).

#### 9. Appearance
- **skin**: Defines the color of the blob and the color inherited by its offspring, specified as a tuple of RGB values (0-255 scale).

#### 10. Preferences
- **fav_meal**: An array of independent values used as parameters for comparing blobs based on their preferred meals.

These parameters collectively shape the behavior, survival, and interactions of blobs. Fine-tuning these parameters can lead to diverse and dynamic ecosystems within the simulation.


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
