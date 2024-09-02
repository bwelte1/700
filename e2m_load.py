# Example Run:
# python e2m_load.py --model_dir saved_models/PPO/Model_6/300000 --episodes 1

import os
import argparse
import gymnasium as gym
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from e2m_env import Earth2MarsEnv
import scipy.io
import numpy as np
from numpy.linalg import norm
import shutil

import pykep as pk
from pykep import MU_SUN
from pykep.core import lambert_problem

def plot_run(positions, r0, rT):
    positions.insert(0, [r0[0], r0[1], r0[2]])
    x_coords, y_coords, z_coords = zip(*positions)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(x_coords, y_coords, z_coords, c='k', marker='o')
    ax.scatter([0], [0], [0], c='#FFA500', marker='o', s=100)  # Sun at origin
    ax.scatter(r0[0], r0[1], r0[2], c='b', marker='o', s=50)  # Earth
    ax.scatter(rT[0], rT[1], rT[2], c='r', marker='o', s=50)  # Mars

    ax.set_xlabel('X Position (km)')
    ax.set_ylabel('Y Position (km)')
    ax.set_zlabel('Z Position (km)')
    ax.set_title('Spacecraft Position Relative to the Sun')

    # plt.show()

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
    
def upload_matlab(runlog, runlog_extra):
    #semiAxes_values = [info['semiAxes'] for info in runlog_extra if 'semiAxes' in info]
    #print(f"semiAxes values for episode {episode + 1}: {semiAxes_values}")

    directory = 'matlab_exports'
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Convert logs to dictionaries
    data = {
        'extra_info_logs': np.array(runlog_extra),
        'run_log': np.array(runlog)
    }

    # Save data to a MAT file in the specified directory
    scipy.io.savemat(os.path.join(directory, 'data.mat'), data)

def load_and_run_model(model_path, env, num_episodes, rI, rT, tof, amu, num_nodes):
    # Ensure the model file has a .zip extension
    model_file_path = f"{model_path}.zip" if not model_path.endswith(".zip") else model_path
    
    if not os.path.exists(model_file_path):
        raise FileNotFoundError(f"Model file not found: {model_file_path}")

    model1 = PPO.load(model_file_path)
    wrapped_env = EnvLoggingWrapper(env)

    for episode in range(num_episodes):
        fig1 = plt.figure()
        ax = fig1.add_subplot(111, projection='3d')  # Create 3D axes
        obs = wrapped_env.reset()
        done = False
        iteration = 0
        while not done:
            r0 = obs[:3]
            action, _states = model1.predict(obs)
            obs, reward, done, truncated, info = wrapped_env.step(action)
            r1 = obs[:3]
            l = lambert_problem(r0, r1, (tof/num_nodes), pk.MU_SUN)
            v0 = l.get_v1()[0]
            pk.orbit_plots.plot_lambert(l, axes=ax)
            # pk.orbit_plots.plot_kepler(r0 = np.array(r0), v0 = np.array(v0), tof=(tof/num_nodes), mu=amu, axes=ax)
            
            # # Save the figure for each iteration
            # iteration += 1
            # fig1.savefig(f'./Plots/loadrunfig_iteration_{iteration}.png')
            
        
        extra_info_logs = wrapped_env.get_extra_info_logs()
        run_log = wrapped_env.get_state_logs()

        #upload_matlab(run_log,extra_info_logs)

        print(obs[:3])
        if num_episodes != 1:
            print(f"Episode {episode + 1} finished.")
        # positions = [state[:3] for state in run_log]
        # plot_run(positions, rI, rT)
        # positions_alt = [info['state_alt'] for info in extra_info_logs if 'state_alt' in info]
        # sun = np.concatenate((rT, vT))
        # positions_alt.append(sun)
        # plot_run(positions_alt, r0, rT)
        
        ax.set_title("Combined Trajectory of All Episodes")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.scatter([0], [0], [0], c='#FFA500', marker='o', s=100)  # Sun at origin
        ax.scatter(rI[0], rI[1], rI[2], c='b', marker='o', s=50)  # Earth
        ax.scatter(rT[0], rT[1], rT[2], c='r', marker='o', s=50)  # Mars
        # set axis limits to ensure all axes are on the same scale
        axes_scale = 1.8e8
        ax.set_xlim([-axes_scale, axes_scale])  # Set X-axis limit
        ax.set_ylim([-axes_scale, axes_scale])  # Set Y-axis limit
        ax.set_zlim([-axes_scale, axes_scale])  # Set Z-axis limit
        ax.set_box_aspect([1,1,1])
        
        directory_path = os.path.dirname(args.model_dir)    # each interval zip file
        last_directory = os.path.basename(directory_path)   # model name
        interval_number = os.path.basename(model_path)   # model name
        plot_folder = os.path.join(os.getcwd(), 'Plots', last_directory)    # plot folder for model
        plot_name_png = os.path.join(plot_folder, f'interval_{interval_number}.png')  
        plt.show()   
        fig1.savefig(plot_name_png)
        
def display_plots():
    directory_path = os.path.dirname(args.model_dir)    # each interval zip file
    last_directory = os.path.basename(directory_path)   # model name
    plot_folder = os.path.join(os.getcwd(), 'Plots', last_directory)    # plot folder for model
    if os.path.exists(plot_folder):
        shutil.rmtree(plot_folder)
    os.makedirs(plot_folder)
        
    for interval in os.listdir(directory_path):
        path = f'{directory_path}/{interval}'
        load_and_run_model(path, env, args.episodes, r0, rT, tof, amu, num_nodes)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Load and run a saved PPO model")
    parser.add_argument('--model_dir', type=str, required=True, help="Path to the saved model directory")
    parser.add_argument('--episodes', type=int, default=1, help="Number of episodes to run")
    args = parser.parse_args()

    # Example initial conditions
    r0 = (-140699693.0, -51614428.0, 980.0)  # Earth position
    rT = (172682023.0, 176959469.0, 7948912.0)  # Mars position

    # Physical constants
    amu = 132712440018.0  # km^3/s^2, Gravitational constant of the central body
    v0 = (9.774596, -28.07828, 4.337725e-4)
    vT = (-16.427384, -14.860506, 9.21486e-2)
    m0 = 1000.0
    Tmax = 0.5
    tof = 500
    using_reachability = True
    num_nodes = 10

    env = Earth2MarsEnv(
        N_NODES=num_nodes, 
        amu=amu, 
        v0=v0, 
        r0=r0, 
        vT=vT, 
        rT=rT, 
        m0=m0, 
        max_thrust=Tmax,
        v_ejection=15,   #arbitrary
        mission_time=tof,
        using_reachability=using_reachability
    )

    load_and_run_model(args.model_dir, env, args.episodes, r0, rT, tof, amu, num_nodes)
    # display_plots()