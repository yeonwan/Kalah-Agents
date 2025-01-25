# Kalah-Agents

A simple repository demonstrating the **Kalah** board game and various reinforcement learning agents, including **Double Deep Q-learning (DQN)** and a **Q-learning** approach with PyTorch.

---

## Table of Contents
1. [Introduction](#introduction)  
2. [File Overview](#file-overview)  
3. [Installation](#installation)  
4. [How to Run](#how-to-run)  
5. [Usage](#usage)  


---

## Introduction
**Kalah** is a well-known Mancala variant in which each player tries to maximize the stones in their own store (Kalah). The rules involve distributing stones from a chosen pit counterclockwise and capturing opponent’s stones under certain conditions.

This project offers:
- A basic Kalah environment (`kalah.py`)  
- Two different reinforcement learning approaches:
  - **Double Deep Q-learning** (in `DQN.py`)  
  - **Q-learning** (in `Qlearning.py`), both implemented with PyTorch  
- Two player scripts demonstrating different playing strategies (`player_random.py` vs. `player_wan.py`)  
- A main runner script (`runner.py`) to execute matches.

---

## File Overview

- **DQN.py**  
  - Contains the neural network (NN) logic for Double Deep Q-learning.  
  - Likely includes the definition of a deep NN using PyTorch, plus methods for updating weights via backpropagation.

- **Qlearning.py**  
  - Implements a standard Q-learning approach (also with PyTorch), possibly using a simpler NN or a different training loop than DQN.  
  - May feature function approximations, a replay buffer, etc.

- **kalah.py**  
  - Core game logic for Kalah:
    - Board representation (pits, stones, stores)  
    - Valid moves and capturing rules  
    - Turn progression

- **player_random.py**  
  - An agent that selects moves at random. Good as a baseline to compare with learning agents.

- **player_wan.py**  
  - Another agent for the Kalah game. This may be a custom (possibly heuristic-based or partially learned) agent.

- **runner.py**  
  - A script that orchestrates one or more matches between two players (agents).  
  - Likely handles command-line or function arguments to specify which agents to pit against each other.

- **README.md**  
  - This file, providing an overview of the project.

---

## Installation

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/yeonwan/Kalah-Agents.git
   cd Kalah-Agents
   ```

2. **Set Up Python Environment**  
   - Ensure you have **Python 3.6+** (or the version recommended by the project; code is ~6 years old, so dependencies may vary).  
   - Create and activate a virtual environment (recommended):
     ```bash
     python -m venv venv
     source venv/bin/activate  # On Linux/Mac
     venv\Scripts\activate     # On Windows
     ```

3. **Install Dependencies**  
   - If there's a `requirements.txt` file, use:
     ```bash
     pip install -r requirements.txt
     ```
   - Otherwise, install necessary packages manually. You might need:  
     - `torch` (PyTorch)  
     - `numpy`  
     - etc.

---

## How to Run

1. **Run a Simple Match**  
   - If `runner.py` supports direct execution, try:
     ```bash
     python runner.py
     ```
   - By default, it may pit two agents (e.g., `player_random.py` vs. `player_wan.py`), or let you choose the agents in the code.

2. **Run Q-learning or DQN**  
   - If you want to train the Q-learning or DQN agent, open `runner.py` (or the relevant file) and ensure it’s set up to use `DQN.py` or `Qlearning.py` as the primary agent.  
   - You may need to adjust hyperparameters (learning rate, discount factor, etc.) directly in `DQN.py` or `Qlearning.py`.

3. **Check for Command-Line Arguments**  
   - If `runner.py` supports CLI options, run `python runner.py --help` or inspect the code to see possible arguments (e.g., specifying which agent to use for Player 1 vs. Player 2).

---

## Usage

- **Training an Agent**  
  - Open `Qlearning.py` or `DQN.py` and review how training is triggered (e.g., a training loop in `runner.py` or a dedicated function call).  
  - Modify the code to control:
    - Number of training episodes  
    - Learning rate and exploration rate  
    - Neural network architecture

- **Evaluating Performance**  
  - After training, run a fixed number of matches against `player_random.py` or `player_wan.py` to gauge the learned policy’s strength.  
  - Look for win rates or final stone counts as metrics.

- **Experimentation**  
  - Tweak neural network size (layers, activation functions) in `DQN.py` or `Qlearning.py` to see how it affects learning.  
  - Try different agents (random, heuristic, Q-learning, DQN) head-to-head.

---
