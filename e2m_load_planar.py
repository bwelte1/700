# Example Run:
# python e2m_load_planar.py --settings settings_def.txt --model_dir saved_models/PPO/Model_38/2000 --episodes 1

import os
import argparse
import gymnasium as gym
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from stable_baselines3 import PPO
from e2m_env_planar import Earth2MarsEnvPlanar
import scipy.io
import numpy as np
from numpy.linalg import norm
import shutil

import pykep as pk
from pykep import MU_SUN, DAY2SEC
from pykep.core import lambert_problem



def plot_run(positions, r0, rT):
    positions.insert(0, [r0[0], r0[1], r0[2]])
    x_coords, y_coords, z_coords = zip(*positions)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(x_coords, y_coords, z_coords, c='b', marker='o')
    ax.scatter([0], [0], [0], c='#FFA500', marker='o', s=100)  # Sun at origin
    ax.scatter(r0[0], r0[1], r0[2], c='b', marker='o', s=50)  # Earth
    ax.scatter(rT[0], rT[1], rT[2], c='r', marker='o', s=50)  # Mars

    ax.set_xlabel('X Position (km e8)')
    ax.set_ylabel('Y Position (km e8)')
    ax.set_zlabel('Z Position (km e8)')
    ax.set_title('Spacecraft Position Relative to the Sun')

class EnvLoggingWrapper(gym.Wrapper):
    def __init__(self, env):
        super(EnvLoggingWrapper, self).__init__(env)
        self.info_logs = []
        self.state_logs = []
        self.extra_info_logs = []

    def step(self, action):
        observation, reward, done, truncated, info = self.env.step(action)
        self.info_logs.append(info)
        self.state_logs.append(observation)
        self.extra_info_logs.append(self.env.extra_info.copy())
        return observation, reward, done, truncated, info

    def reset(self, **kwargs):
        self.info_logs = []
        self.state_logs = []
        observation, info = self.env.reset(**kwargs)
        return observation

    def get_info_logs(self):
        return self.info_logs

    def get_state_logs(self):
        return self.state_logs
    
    def get_extra_info_logs(self):
        return self.extra_info_logs

def plot_traj_kepler(plot_data, model_path, ellipsoid_points, TIME_STEP):
    positions = [state[:3] for state in plot_data]
    velocities = [state[3:6] for state in plot_data]

    fig1 = plt.figure()
    ax = fig1.add_subplot(111, projection='3d')
    print("Plot Timestep = " + str(TIME_STEP))

    for ii in range(len(positions)):
        pk.orbit_plots.plot_kepler(
            r0=positions[ii],            # Initial position (3D)
            v0=velocities[ii],           # Initial velocity (3D)
            tof= TIME_STEP * DAY2SEC,    # Time of flight (seconds)
            mu=amu,                      # Gravitational parameter
            color='b',                   # Color of the orbit
            label=None,                  # Optional label
            axes=ax                      # 3D axis for plotting
        )

    x_coords, y_coords, z_coords = zip(*positions)
    ax.scatter(x_coords, y_coords, z_coords, c='b', marker='o')
    ax.scatter([0], [0], [0], c='#FFA500', marker='o', s=100, label="Sun")  # Sun at origin
    ax.scatter(r0[0], r0[1], r0[2], c='b', marker='o', s=50, label="Earth")  # Earth
    ax.scatter(rT[0], rT[1], rT[2], c='r', marker='o', s=50, label="Mars")   # Mars

    ax.set_xlabel('X Position (km e8)')
    ax.set_ylabel('Y Position (km e8)')
    ax.set_zlabel('Z Position (km e8)')
    ax.set_title('Spacecraft Position Relative to the Sun')

    axes_scale = 2e8
    ax.set_xlim([-axes_scale, axes_scale])  # Set X-axis limit
    ax.set_ylim([-axes_scale, axes_scale])  # Set Y-axis limit
    ax.set_zlim([-axes_scale, axes_scale])  # Set Z-axis limit
    ax.set_box_aspect([1,1,1])

    colours = ['red', 'black', 'green', 'orange', 'purple', 'cyan', 'gray']
    for ellipsoid in ellipsoid_points:
        for point in range(7):
            ax.scatter(ellipsoid[0,point], ellipsoid[1,point], ellipsoid[2,point], color=colours[point])

    ax.view_init(elev=90, azim=-90)
    ax.legend()
    
    directory_path = os.path.dirname(args.model_dir)    # each interval zip file
    last_directory = os.path.basename(directory_path)   # model name
    interval_number = os.path.basename(model_path)   # model name
    plot_folder = os.path.join(os.getcwd(), 'Plots', last_directory)    # plot folder for model
    plot_name_png = os.path.join(plot_folder, f'interval_{interval_number}.png')  

    plt.show()

def plot_ellipsoid(ellipsoid, ax):
    colors = ['r', 'g', 'k', 'c', 'm', 'y']

    # Plot the original points as scatter
    for i in range(ellipsoid.shape[0]):
        ax.scatter(ellipsoid[i, 0], ellipsoid[i, 1], ellipsoid[i, 2], color=colors[i])

def load_and_run_model(model_path, env, num_episodes, rI, rT, num_nodes, tof, amu):
    # Ensure the model file has a .zip extension
    model_file_path = f"{model_path}.zip" if not model_path.endswith(".zip") else model_path
    
    if not os.path.exists(model_file_path):
        raise FileNotFoundError(f"Model file not found: {model_file_path}")

    model1 = PPO.load(model_file_path)
    wrapped_env = EnvLoggingWrapper(env)

    for episode in range(num_episodes):
        obs = wrapped_env.reset()
        r_prev = obs[:3]
        v_prev = obs[3:6]
        done = False
        while not done:
            action, _states = model1.predict(obs)
            obs, reward, done, truncated, info = wrapped_env.step(action)
        
        extra_info_logs = wrapped_env.get_extra_info_logs()
        run_log = wrapped_env.get_state_logs()

        ellipsoid_points = [log['semiAxes'] for log in extra_info_logs]
        plotting_data = [log['Plotting'] for log in extra_info_logs]
        TIME_STEP = obs[7] / N_NODES
        plot_traj_kepler(plotting_data, model_path, ellipsoid_points, TIME_STEP)
        

        if num_episodes != 1:
            print(f"Episode {episode + 1} finished.")
        
def display_plots():
    directory_path = os.path.dirname(args.model_dir)    # each interval zip file
    last_directory = os.path.basename(directory_path)   # model name
    plot_folder = os.path.join(os.getcwd(), 'Plots', last_directory)    # plot folder for model
    if os.path.exists(plot_folder):
        shutil.rmtree(plot_folder)
    os.makedirs(plot_folder)
        
    for interval in os.listdir(directory_path):
        path = f'{directory_path}/{interval}'
        load_and_run_model(path, env, args.episodes, r0, rT, tof, amu, N_NODES)

if __name__ == '__main__':
    import argparse

    # Create one argument parser for all command-line arguments
    parser = argparse.ArgumentParser(description="Load settings and run a saved PPO model")
    
    # Add arguments for settings, model directory, and number of episodes
    parser.add_argument('--settings', type=str, default="settings_def.txt", \
        help='Input settings file')
    parser.add_argument('--model_dir', type=str, required=True, help="Path to the saved model directory")
    parser.add_argument('--episodes', type=int, default=1, help="Number of episodes to run")
    args = parser.parse_args()
    settings_file = "./settings_files/" + args.settings
    
    #Read settings and assign environment and model parameters
    with open(settings_file, "r") as input_file: # with open context
        input_file_all = input_file.readlines()
        for line in input_file_all: #read line
            line = line.split()
            if (len(line) > 2):
                globals()[line[0]] = line[1:]
            else:
                globals()[line[0]] = line[1]
            
    input_file.close() #close file
    
    dvMax = float(dvMax)
    N_NODES = int(N_NODES)
    m0 = float(m_initial)
    Isp = float(Isp)                # specific impulse of engine 
    using_reachability = bool(int(using_reachability))
    tof = float(tof)      # predetermined TOF
    
    amu = MU_SUN / 1e9              # km^3/s^2, Gravitational constant of the central body
    rconv = 149600000.              # position, km (sun-earth)
    vconv = np.sqrt(amu/rconv)      # velocity, km/s
    v_ejection = (pk.G0/1000.*Isp)  # propellant ejection velocity

    # Example initial conditions
    r0 = (-140699693.0, -51614428.0, 0)  # Earth position
    rT = (-172682023.0, 176959469.0, 0)  # Mars position

    # Physical constants
    v0 = (9.774596, -28.07828, 0)
    vT = (-16.427384, -14.860506, 0)

    env = Earth2MarsEnvPlanar(
        N_NODES=N_NODES, 
        amu=amu, 
        v0=v0, 
        r0=r0, 
        vT=vT, 
        rT=rT, 
        m0=m0, 
        max_thrust=dvMax,
        v_ejection=v_ejection,   #arbitrary
        mission_time=tof,
        using_reachability=using_reachability
    )

    load_and_run_model(model_path=args.model_dir, env=env, num_episodes=args.episodes, rI=r0, rT=rT, tof=tof, amu=amu, num_nodes=N_NODES)
    display_plots()