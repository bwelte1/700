# Example Run:
# python e2m_load.py --settings settings_def.txt --model_dir saved_models/PPO/Model_27/3000000 --episodes 1

import os
import argparse
import gymnasium as gym
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator
from stable_baselines3 import PPO
from e2m_env import Earth2MarsEnv
import scipy.io
import numpy as np
from numpy.linalg import norm
import shutil

import pykep as pk
from pykep import MU_SUN, DAY2SEC
from pykep.core import lambert_problem

import YA

def plot_run(positions, r0, rT):
    positions.insert(0, [r0[0], r0[1], r0[2]])
    x_coords, y_coords, z_coords = zip(*positions)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(x_coords, y_coords, z_coords, c='b', marker='o')
    ax.scatter([0], [0], [0], c='#FFA500', marker='o', s=100)  # Sun at origin
    ax.scatter(r0[0], r0[1], r0[2], c='b', marker='o', s=50)  # Earth
    ax.scatter(rT[0], rT[1], rT[2], c='r', marker='o', s=50)  # Target Planet

    ax.set_xlabel('X Position (km e8)')
    ax.set_ylabel('Y Position (km e8)')
    ax.set_zlabel('Z Position (km e8)')

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
        self.clear_logs()
        observation, info = self.env.reset(**kwargs)
        return observation

    def get_info_logs(self):
        return self.info_logs

    def get_state_logs(self):
        return self.state_logs
    
    def get_extra_info_logs(self):
        return self.extra_info_logs
    
    def clear_logs(self):
        self.state_logs = []  # Reset state logs
        self.extra_info_logs = []  # Reset any extra info logs, including impulse tracking
        self.info_logs = []

def plot_traj_kepler(plot_data, model_path, ellipsoid_points, dv_data, extra_info_logs):
    positions = [state[:3] for state in plot_data]
    velocities = [state[3:6] for state in plot_data]

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    total_dv = 0

    for ii in range(len(positions)):
        pk.orbit_plots.plot_kepler(
            r0=positions[ii],            # Initial position (3D)
            v0=velocities[ii],           # Initial velocity (3D)
            tof=(tof / N_NODES) * DAY2SEC, # Time of flight (seconds)
            mu=amu,                      # Gravitational parameter
            color='black',                   # Color of the orbit
            label=None,                  # Optional label
            axes=ax1                      # 3D axis for plotting
        )
        if isinstance(dv_data[ii], np.ndarray):
            ax1.quiver(positions[ii][0], positions[ii][1], positions[ii][2], 30000000*dv_data[ii][0], 30000000*dv_data[ii][1], 30000000*dv_data[ii][2], arrow_length_ratio=0.2, color='red')
            state_current = np.concatenate((positions[ii], velocities[ii]))
            dv_rtn = np.transpose(YA.RotMat_RTN2Inertial(state_current)) @ dv_data[ii]
            total_dv += norm(dv_rtn)
            ax2.stem(ii+1-0.2, dv_rtn[0], linefmt='Black', basefmt='White')
            ax2.stem(ii+1, dv_rtn[1], linefmt='Red', basefmt='White')
            ax2.stem(ii+1+0.2, dv_rtn[2], linefmt='Green', basefmt='White')
            ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
        else:
            dv_n_minus_1 = [log.get('dv_n_minus_1', 0) for log in extra_info_logs][-1]
            ax1.quiver(positions[ii][0], positions[ii][1], positions[ii][2], 30000000*dv_n_minus_1[0], 30000000*dv_n_minus_1[1], 30000000*dv_n_minus_1[2], arrow_length_ratio=0.2, color='red')
            state_current = np.concatenate((positions[ii], velocities[ii]))
            dv_rtn = np.transpose(YA.RotMat_RTN2Inertial(state_current)) @ dv_n_minus_1
            total_dv += norm(dv_data[ii])
            ax2.stem(ii+1-0.2, dv_rtn[0], linefmt='Black', basefmt='White')
            ax2.stem(ii+1, dv_rtn[1], linefmt='Red', basefmt='White')
            ax2.stem(ii+1+0.2, dv_rtn[2], linefmt='Green', basefmt='White')
            ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
        
    x_coords, y_coords, z_coords = zip(*positions)
    ax1.scatter([0], [0], [0], c='#FFA500', marker='o', s=100, label="Sun")  # Sun at origin
    ax1.scatter(r0[0], r0[1], r0[2], c='b', marker='o', s=50, label="Earth")  # Earth
    ax1.scatter(rT[0], rT[1], rT[2], c='r', marker='o', s=50, label="Mars")   # Target Planet

    ax1.set_xlabel('X Position (km e8)')
    ax1.set_ylabel('Y Position (km e8)')
    ax1.set_zlabel('Z Position (km e8)')

    ax2.set_xlabel('Impulse')
    ax2.set_ylabel('dv (km/s)')

    axes_scale = 2e8
    ax1.set_xlim([-axes_scale, axes_scale])  # Set X-axis limit
    ax1.set_ylim([-axes_scale, axes_scale])  # Set Y-axis limit
    ax1.set_zlim([-axes_scale, axes_scale])  # Set Z-axis limit
    ax1.set_box_aspect([1,1,1])

    ax1.view_init(elev=90, azim=-90)
    ax1.legend()
    ax2.legend(['Radial', 'Transverse', 'Normal'], loc='lower right')

    directory_path = os.path.dirname(args.model_dir)    # each interval zip file
    last_directory = os.path.basename(directory_path)   # model name
    interval_number = os.path.basename(model_path)   # model name
    plot_folder = os.path.join(os.getcwd(), 'Plots', last_directory)    # plot folder for model
    plot_name_png = os.path.join(plot_folder, f'interval_{interval_number}.png')  
    stem_name_png = os.path.join(plot_folder, f'stem_interval_{interval_number}.png')  
    if not os.path.exists(plot_folder):     # Check if the plot folder exists, and create it if not
        os.makedirs(plot_folder)
    fig1.savefig(plot_name_png)
    fig2.savefig(stem_name_png)

    plt.show()
    plt.close()
    plt.close()
    
    return {
        'total_dv': total_dv
    }
    
    

def plot_ellipsoid(ellipsoid, ax):
    colors = ['r', 'g', 'k', 'c', 'm', 'y']

    for i in range(ellipsoid.shape[0]):
        ax.scatter(ellipsoid[i, 0], ellipsoid[i, 1], ellipsoid[i, 2], color=colors[i])

def load_and_run_model(model_path, env, num_episodes, rI, rT, num_nodes, tof, amu):
    # Ensure the model file has a .zip extension
    model_file_path = f"{model_path}.zip" if not model_path.endswith(".zip") else model_path
    
    if not os.path.exists(model_file_path):
        raise FileNotFoundError(f"Model file not found: {model_file_path}")

    model1 = PPO.load(model_file_path)
    wrapped_env = EnvLoggingWrapper(env)
    
    episode_data_list = []
    total_mass_array = []

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
        
        dv_data = [log['dv'] for log in extra_info_logs]
        
        plotting_data = [log['Plotting'] for log in extra_info_logs]
        mass_data = [log.get('final_mass', 0) for log in extra_info_logs]    
        total_mass_array.append(mass_data[-1])
                
        episode_data = plot_traj_kepler(plotting_data, model_path, ellipsoid_points, dv_data, extra_info_logs)
        episode_data_list.append(episode_data)

    mean_mass = np.mean(total_mass_array)
    mean_dv = np.mean([data['total_dv'] for data in episode_data_list])
    dv_std = np.std([data['total_dv'] for data in episode_data_list])
    mass_std = np.std(total_mass_array)
    print(f'Mean dv across {num_episodes} episodes: {mean_dv}, with std {dv_std}')
    print(f'Mean final mass across {num_episodes} episodes: {mean_mass}, with std {mass_std}')
    boxFig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.boxplot(total_mass_array)
    ax1.set_title("Remaining Mass")
    ax2.boxplot([data['total_dv'] for data in episode_data_list])
    ax2.set_title("Total ∆v")
    plt.show()

        
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
    planet = str(Planet)
    planet = planet.lower()
    
    amu = MU_SUN / 1e9              # km^3/s^2, Gravitational constant of the central body
    rconv = 149600000.              # position, km (sun-earth)
    vconv = np.sqrt(amu/rconv)      # velocity, km/s
    v_ejection = (pk.G0/1000.*Isp)  # propellant ejection velocity

    # Example initial conditions
    r0 = (-140699693.0, -51614428.0, 980.0)
    v0 = (9.774596, -28.07828, 4.337725e-4)

    rT = (-172682023.0, 176959469.0, 7948912.0)
    vT = (-16.427384, -14.860506, 9.21486e-2)

    if planet == 'venus':
        #Same init conds as Zavoli Federici Table 1 (km and km/s)
        rT = (108210000, 0.0, 0.0)  # halfway between aphelion and perihelion in x direction, 0 in y and z. taken from NASA
        vT = (0, 35.02, 0)          # mean velocity in +y direction (perpendicular to distance in +x direction)

    env = Earth2MarsEnv(
        N_NODES=N_NODES, 
        amu=amu, 
        v0=v0, 
        r0=r0, 
        vT=vT, 
        rT=rT, 
        m0=m0, 
        max_thrust=dvMax,
        v_ejection=v_ejection,
        mission_time=tof,
        using_reachability=using_reachability)


    load_and_run_model(model_path=args.model_dir, env=env, num_episodes=args.episodes, rI=r0, rT=rT, tof=tof, amu=amu, num_nodes=N_NODES)