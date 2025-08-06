# simu.py (Updated for MAPPO-GNN)
import os
import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(r"C:\Users\91896\Desktop\ALL\Capstone\mappo")

from env import TrafficEnv
from mappo import MAPPO_GNN
from utils import get_average_travel_time, get_average_CO2, get_average_fuel, get_average_length, get_total_cars

parser = argparse.ArgumentParser()
parser.add_argument("-R", "--render", action="store_true", help="whether render while training or not")
args = parser.parse_args()

if __name__ == "__main__":
    if "SUMO_HOME" in os.environ:
        tools = os.path.join(os.environ["SUMO_HOME"], "tools")
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")

    state_dim = 10
    action_dim = 2
    n_agents = 2
    n_episode = 1000

    env = TrafficEnv("gui") if args.render else TrafficEnv()
    agent = MAPPO_GNN(n_agents, state_dim, action_dim)

    performance_list, co2_emission, fuel_cons, route_length, cars_list, depart_times = [], [], [], [], [], []

    results_dir = r"C:\Users\91896\Desktop\ALL\Capstone\mappo\result"
    os.makedirs(results_dir, exist_ok=True)

    for episode in range(n_episode):
        state = env.reset()
        done = False

        while not done:
            actions = []
            edge_index = np.array([[i, j] for i in range(n_agents) for j in range(n_agents) if i != j])

            for i in range(n_agents):
                action, log_prob, probs = agent.select_action(state[i, :])
                actions.append(action)
                agent.push(state[i, :], action, log_prob, 0.0, 0.0, edge_index)

            next_state, reward, done = env.step(actions)

            for i in range(n_agents):
                agent.buffer.rewards[-n_agents + i] = reward[i]
                agent.buffer.dones[-n_agents + i] = float(done)

            state = next_state

        agent.train_model()
        env.close()

        avg_travel_time = round(get_average_travel_time(), 2)
        performance_list.append(avg_travel_time)

        avg_length = get_average_length()
        route_length.append(avg_length)

        avg_CO2 = round(get_average_CO2() / avg_length, 2)
        co2_emission.append(avg_CO2)

        avg_fuel = round((get_average_fuel() / avg_length) + 3, 2)
        fuel_cons.append(avg_fuel)

        total_cars = get_total_cars()
        cars_list.append(total_cars)

        depart_times.append(episode * 600)

        print(f"Episode {episode + 1} - Avg Travel Time: {avg_travel_time}s, CO2: {avg_CO2}g/km, Fuel: {avg_fuel}L/100km")

    model_path = os.path.join(results_dir, f"trained_model{n_episode}.th")
    agent.save_model(model_path)

    plt.style.use("seaborn-v0_8-whitegrid")

    plt.figure(figsize=(10, 5))
    plt.plot(depart_times, co2_emission, label='MAPPO-GNN', color='green')
    plt.xlabel("DEPART (SEC)")
    plt.ylabel("CO2 (G/KM)")
    plt.title("Average CO2 Emission Rate for 1-h Traffic Flow")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, "co2_emission.png"))
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(depart_times, fuel_cons, label='MAPPO-GNN', color='green')
    plt.xlabel("DEPART (SEC)")
    plt.ylabel("FUEL (L/100KM)")
    plt.title("Fuel Consumption for 1-h Traffic Flow")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, "fuel_consumption.png"))
    plt.show()
    
    plt.figure(figsize=(10, 5))
    plt.plot(depart_times, performance_list, label='MAPPO-GNN', color='green')
    plt.xlabel("DEPART (SEC)")
    plt.ylabel("AVG TRAVEL TIME (s)")
    plt.title("Average Travel Time per Episode")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, "avg_travel_time.png"))
    plt.show()


np.save(os.path.join(results_dir, "mappo_co2.npy"), np.array(co2_emission))
np.save(os.path.join(results_dir, "mappo_fuel.npy"), np.array(fuel_cons))
np.save(os.path.join(results_dir, "mappo_avg_travel_time.npy"), np.array(performance_list))
