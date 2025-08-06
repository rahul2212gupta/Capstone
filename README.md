MAPPO-GNN: Adaptive Multi-Agent Traffic Signal Control
Simulation-Based Deep Reinforcement Learning for Urban Traffic Optimization

Overview
This repository implements MAPPO-GNN—an adaptive, multi-agent traffic signal control system combining Multi-Agent Proximal Policy Optimization (MAPPO) with Graph Neural Networks (GNNs). The algorithm is designed to optimize traffic flow in urban networks, reduce congestion, cut travel time, and minimize CO₂ emissions and fuel consumption. All experiments are conducted within the SUMO Traffic Simulator using the TraCI interface, enabling a scalable, controlled, and flexible simulation environment.

Key Features:

Multi-agent, decentralized control: Each intersection is an agent, learning to optimize its own traffic light based on local and neighboring traffic states.

Graph Neural Network Critic: Spatial relations between intersections are modeled as a graph, enhancing coordination and policy learning.

Centralized Training, Decentralized Execution: Agents train jointly with global context but act independently during real-time operation.

Paper
This repository is a companion to:

Adaptive Multi Agent Traffic Signal Control Using MAPPO-GNN: A Simulation-Based Deep Reinforcement Learning Approach
Shaurya Narang, Raghav Rana, Kriti Singh, Rahul Gupta

Table of Contents
Algorithm Overview

Environment Setup

Quickstart

Configuration

Training and Evaluation

Results

References and Credits

Algorithm Overview
MAPPO-GNN is an extension of MAPPO, equipped with a GNN-based critic. This critic leverages the spatial structure of road networks, representing:

Intersections → Nodes

Road connections → Edges

Each agent observes:

Queue lengths,

Number of halted vehicles,

Current signal phase,

Waiting time,

Emissions and fuel consumption metrics.

Action Space: Discrete (hold current phase / switch to next phase)
Reward: Designed to promote decongestion by rewarding reduction in the number of stopped vehicles—encouraging smoother flow, less idling, and reduced emissions.

Architecture:

Actor network: decentralized policy per agent.

Critic network: centralized GNN evaluating joint state-action values using message passing among intersection nodes.

Environment Setup
Requirements
Python 3.8+

SUMO (Simulation of Urban Mobility)

PyTorch (>=1.9)

DGL or PyTorch Geometric for GNNs

numpy

TraCI Python client

matplotlib (for plotting)

(Recommend) gym for simulation wrappers

Installation
Clone Repository:

bash
git clone https://github.com/YOUR-USERNAME/MAPPO-GNN-Traffic-Signal-Control.git
cd MAPPO-GNN-Traffic-Signal-Control
Install Python Dependencies:

bash
pip install -r requirements.txt
Install SUMO:

Download and install per official SUMO docs.

Set SUMO_HOME environment variable.

Quickstart
Configure SUMO Network

Create or modify a SUMO road network (.net.xml), route file (.rou.xml), and scenarios as needed. Sample networks for small 2-intersection grids are provided.

Train MAPPO-GNN Agents

bash
python train_mappo_gnn.py --config configs/sumo_2x1.yaml
Evaluate/ Test

bash
python evaluate.py --model checkpoints/best_model.pth --config configs/sumo_2x1.yaml
Plot Results

bash
python plot_results.py --logs logs/experiment1/
Configuration
All configuration options (network topology, agent count, training parameters) are in the configs/ folder. Example sumo_2x1.yaml:

text
env:
  net_file: "networks/grid_2x1.net.xml"
  route_file: "networks/traffic_2x1.rou.xml"
  step_length: 1.0
  episode_steps: 3600
  intersection_ids: ["junction_1", "junction_2"]
agent:
  obs_dim: 8
  action_dim: 2
  reward_fn: "delta_halted_vehicles"
training:
  gamma: 0.99
  lr: 1e-4
  epochs: 10
  buffer_size: 10000
  gnn_hidden_dim: 64
  update_interval: 10
Training and Evaluation
Training Loop (Pseudocode)
For each episode:

For each agent at every step:

Observe state, select action (hold/switch light phase)

Apply action in SUMO using TraCI

Collect state, reward (decrease in stopped vehicles)

Store in buffer

After interval, update policy and GNN-based critic with PPO loss

Evaluation
Run trained model on test scenarios (different traffic patterns)

Collect metrics: average travel time, fuel usage, CO₂ emissions

Results
Key Findings:

~46% reduction in average travel time

~40% reduction in CO₂ emissions

~18% reduction in fuel consumption

When compared to standard DRL baselines (e.g., MADDPG), MAPPO-GNN achieves:

Higher stability and faster convergence

Better spatial coordination and scalability

Superior ecological performance (lower emissions)

Visualization of results and figures are in results/ folder and can be re-generated with plot_results.py.

References and Credits
Implementation inspired by MAPPO (Yu et al., 2021)

Graph-based MARL literature and prior algorithms (see paper.pdf Reference section)

SUMO and TraCI for simulation environment

Citation:

If you use this codebase or the MAPPO-GNN approach in your own work, please cite:

text
@article{Narang2024MAPPOGNN,
  title={Adaptive Multi Agent Traffic Signal Control Using MAPPO-GNN: A Simulation-Based Deep Reinforcement Learning Approach},
  author={Narang, Shaurya and Rana, Raghav and Singh, Kriti and Gupta, Rahul},
  year={2024},
  journal={Preprint},
}
License
This project is released under the MIT License.

Contact
For questions or collaborations, please open an issue or contact the authors.

Happy simulating smarter, greener cities!
