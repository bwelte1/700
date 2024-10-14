# MECHENG 700 A/B: Part 4 Project - Benny Welte & Henry Simpson

## Overview

This repository contains the code for our project on optimizing Earth-to-Mars transfer trajectories using a combination of reachability set analysis and reinforcement learning (RL). The goal is to design efficient multiple-impulse transfers to Mars by employing reachability set analysis to define valid states for RL exploration.

The project models the trajectory design as a Markov decision process (MDP), where RL selects optimal position vectors (waypoints) along the transfer path. The Yamanaka-Ankerson State Transition Matrix (STM) is used to generate a reachable set of states based on perturbations at each step of the trajectory. Lambert's arcs are utilized to perform the transfers between states, ensuring constraint satisfaction—addressing a known challenge for RL-based methods.

This project is implemented in Python, creating a framework to enhance the efficiency and reliability of Earth-to-Mars and other interplanetary transfer missions.

## Key Features
- **Reachability Set Analysis**: Defines admissible states for RL exploration.
- **Reinforcement Learning (RL)**: Guides the selection of waypoints along the transfer path.
- **Yamanaka-Ankerson STM**: Models state transitions and defines reachable sets.
- **Lambert's Arc Transfer**: Ensures automatic constraint satisfaction during transfers.
- **Python Implementation**: Offers a reproducible workflow for optimizing interplanetary transfers.

## Requirements

To run the project, you need to install the necessary dependencies. We recommend using **Conda** for environment management.

## Installation Steps

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
2. **Set up Conda environment**:
   ```bash
    conda env create -f environment.yml
    conda activate pykep_env
3. **Install additional dependencies**:
     ```bash
   pip install -r requirements.txt
   ```
   
## Training the Model
To begin training the model with the default settings, run the following command:
```bash
python e2m_main.py --settings settings_def.txt
```
The ```settings_def.txt``` file contains the model parameters and hyperparameters, which can be adjusted according to your preferences.

## Loading and Visualizing Pre-trained Models
Once the training process is complete, the trained model is saved in the ```saved_models``` directory. To load and visualize a specific pre-trained model, use the following command:
```bash
python e2m_load.py --settings settings_def.txt --model_dir saved_models/PPO/Model_27/3000000 --episodes 1
```
The ```--model_dir``` argument points to the directory of the saved model. 
The ```--episodes``` argument specifies the number of episodes to simulate during visualization. Adjust this number to your preference.

## Simulation Data
The ```SimulationData``` branch contains models and datasets used in one of the project reports. These models serve as useful references for comparing your results and conducting further analysis.

## Model Performance
Our model demonstrates strong alignment with results from prior studies, which were used to guide the selection of several default parameters. By tuning mission-specific parameters and RL configurations, the model can generate near-optimal trajectories for Earth-to-Mars transfers. The current framework is capable of designing high-efficiency trajectories for space missions while automatically satisfying complex constraints.

