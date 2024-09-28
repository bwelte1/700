# Example Run:
# python e2m_load.py --settings settings_def.txt --model_dir saved_models/PPO/Model_27/3000000 --episodes 1

import os
import argparse
import gymnasium as gym
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from stable_baselines3 import PPO
from e2m_env import Earth2MarsEnv
from e2v_env import Earth2VenusEnv
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

def plot_traj_kepler(plot_data, model_path, ellipsoid_points, dv_data):
    positions = [state[:3] for state in plot_data]
    velocities = [state[3:6] for state in plot_data]
    # print(velocities)

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)

    for ii in range(len(positions)):
        #print(positions[ii])
        pk.orbit_plots.plot_kepler(
            r0=positions[ii],            # Initial position (3D)
            v0=velocities[ii],           # Initial velocity (3D)
            tof=(tof / N_NODES) * DAY2SEC, # Time of flight (seconds)
            mu=amu,                      # Gravitational parameter
            color='b',                   # Color of the orbit
            label=None,                  # Optional label
            axes=ax1                      # 3D axis for plotting
        )
        ax1.quiver(positions[ii][0], positions[ii][1], positions[ii][2], 30000000*dv_data[ii][0], 30000000*dv_data[ii][1], 30000000*dv_data[ii][2], arrow_length_ratio=0.2, color='red')
        state_current = np.concatenate((positions[ii], velocities[ii]))
        print(velocities[ii])
        dv_rtn = np.transpose(YA.RotMat_RTN2Inertial(state_current)) @ dv_data[ii]
        # print(dv_rtn)
        ax2.stem(ii-0.2, dv_rtn[0], linefmt='Black', basefmt='White')
        ax2.stem(ii, dv_rtn[1], linefmt='Red', basefmt='White')
        ax2.stem(ii+0.2, dv_rtn[2], linefmt='Green', basefmt='White')
        
    x_coords, y_coords, z_coords = zip(*positions)
    ax1.scatter(x_coords, y_coords, z_coords, c='b', marker='o')
    ax1.scatter([0], [0], [0], c='#FFA500', marker='o', s=100, label="Sun")  # Sun at origin
    ax1.scatter(r0[0], r0[1], r0[2], c='b', marker='o', s=50, label="Earth")  # Earth
    ax1.scatter(rT[0], rT[1], rT[2], c='r', marker='o', s=50, label="Mars")   # Target Planet

    ax1.set_xlabel('X Position (km)')
    ax1.set_ylabel('Y Position (km)')
    ax1.set_zlabel('Z Position (km)')
    ax1.set_title('Spacecraft Position Relative to the Sun')
    ax2.set_xlabel('Impulse')
    ax2.set_ylabel('dv (km/s)')
    ax2.set_title('RTN Components of dv for each Impulse across the Trajectory')

    axes_scale = 2e8
    ax1.set_xlim([-axes_scale, axes_scale])  # Set X-axis limit
    ax1.set_ylim([-axes_scale, axes_scale])  # Set Y-axis limit
    ax1.set_zlim([-axes_scale, axes_scale])  # Set Z-axis limit
    ax1.set_box_aspect([1,1,1])

    colours = ['red', 'black', 'green', 'orange', 'purple', 'cyan', 'gray']
    # for ellipsoid in ellipsoid_points:
    #     for point in range(6):
    #         ax1.scatter(ellipsoid[0,point], ellipsoid[1,point], ellipsoid[2,point], color=colours[point])

    ax1.view_init(elev=90, azim=-90)
    ax1.legend()
    ax2.legend(['r (radial)', 't (transverse)', 'n (normal)'], loc='upper right')

    # colours = ['red', 'blue', 'green', 'orange', 'purple', 'cyan']
    # print(ellipsoid_points)
    # for ellipsoid in ellipsoid_points:
    #     # Plot each of the 6 points in the matrix, with distinct colors
    #     for point in range(6):
    #         ax1.scatter(ellipsoid[point, 0], ellipsoid[point, 1], ellipsoid[point, 2], color=colours[point])
    
    directory_path = os.path.dirname(args.model_dir)    # each interval zip file
    last_directory = os.path.basename(directory_path)   # model name
    interval_number = os.path.basename(model_path)   # model name
    plot_folder = os.path.join(os.getcwd(), 'Plots', last_directory)    # plot folder for model
    plot_name_png = os.path.join(plot_folder, f'interval_{interval_number}.png')  
    # Check if the plot folder exists, and create it if not
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)
    fig1.savefig(plot_name_png)

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
        # fig1 = plt.figure()
        # ax = fig1.add_subplot(111, projection='3d')  # Create 3D axes
        obs = wrapped_env.reset()
        r_prev = obs[:3]
        v_prev = obs[3:6]
        done = False
        while not done:
            # print(f"Analysis of next arc")
            # Plot Lambert
            action, _states = model1.predict(obs)
            obs, reward, done, truncated, info = wrapped_env.step(action)

            # r_new = obs[:3]
            # print(f"v_prev: {v_prev}")
            # l = lambert_problem(r1=r_prev, r2=r_new, tof=((tof/N_NODES)*DAY2SEC), mu=amu, max_revs=0) 
            # r_prev = r_new
            # v_new = l.get_v1()[0]
            # print(f"v_new: {v_new}")
            # dv = np.subtract(v_new, v_prev)
            # print(f"dv: {dv}")
            # print(f"norm: {norm(dv)}")
            # print(obs[6])
            # v_prev = obs[3:6]
            # pk.orbit_plots.plot_lambert(l, axes=ax)
            # ax.scatter(r_new[0], r_new[1], r_new[2], c='k', marker='o', s=10) 
            
            # Plot Kepler
            # v_current = obs[3:6]
            # print(f"velocity at end of previous arc: {v_current}")
            # action, _states = model1.predict(obs)
            # obs, reward, done, truncated, info = wrapped_env.step(action)
            # dv = obs[8:11] # dv now omitted from observation
            # new_v = v_current + dv
            # print(f"dv required to enter current arc from previous arc: {dv}")
            # print(f"velocity at start of current arc: {new_v}")
            # pk.orbit_plots.plot_kepler(r0 = np.array(obs[:3]), v0 = np.array(new_v), tof=(tof/num_nodes)*DAY2SEC, mu=amu, axes=ax)
            
            # # Save the figure for each iteration
            # iteration += 1
            # fig1.savefig(f'./Plots/loadrunfig_iteration_{iteration}.png')
        
        extra_info_logs = wrapped_env.get_extra_info_logs()
        run_log = wrapped_env.get_state_logs()

        ellipsoid_points = [log['semiAxes'] for log in extra_info_logs]
        # print("Ellipsoid Points: " + str(ellipsoid_points))
        plotting_data = [log['Plotting'] for log in extra_info_logs]
        dv_data = [log['dv'] for log in extra_info_logs]
        plot_traj_kepler(plotting_data, model_path, ellipsoid_points, dv_data)
        

        if num_episodes != 1:
            print(f"Episode {episode + 1} finished.")
        # positions = [state[:3] for state in run_log]
        # plot_run(positions, rI, rT)
        # positions_alt = [info['state_alt'] for info in extra_info_logs if 'state_alt' in info]
        # sun = np.concatenate((rT, vT))
        # positions_alt.append(sun)
        # plot_run(positions_alt, r0, rT)
        
        # ax.set_title("Combined Trajectory of All Episodes")
        # ax.set_xlabel("x")
        # ax.set_ylabel("y")
        # ax.set_zlabel("z")
        # ax.scatter([0], [0], [0], c='#FFA500', marker='o', s=100)  # Sun at origin
        # ax.scatter(rI[0], rI[1], rI[2], c='b', marker='o', s=50)  # Earth
        # ax.scatter(rT[0], rT[1], rT[2], c='r', marker='o', s=50)  # Mars
        # # set axis limits to ensure all axes are on the same scale
        # axes_scale = 2e8
        # ax.set_xlim([-axes_scale, axes_scale])  # Set X-axis limit
        # ax.set_ylim([-axes_scale, axes_scale])  # Set Y-axis limit
        # ax.set_zlim([-axes_scale, axes_scale])  # Set Z-axis limit
        # ax.set_box_aspect([1,1,1])
    
        
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
    
    Tmax = float(Tmax)
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

    if planet == 'mars':
        #Same init conds as Zavoli Federici Table 1 (km and km/s)
        rT = (-172682023.0, 176959469.0, 7948912.0)
        vT = (-16.427384, -14.860506, 9.21486e-2)
        env = Earth2MarsEnv(
        N_NODES=N_NODES, 
        amu=amu, 
        v0=v0, 
        r0=r0, 
        vT=vT, 
        rT=rT, 
        m0=m0, 
        max_thrust=Tmax,
        v_ejection=v_ejection,
        mission_time=tof,
        using_reachability=using_reachability
        )
    elif planet == 'venus':
        #Same init conds as Zavoli Federici Table 1 (km and km/s)
        rT = (108210000, 0.0, 0.0)  # halfway between aphelion and perihelion in x direction, 0 in y and z. taken from NASA
        vT = (0, 35.02, 0)          # mean velocity in +y direction (perpendicular to distance in +x direction)
        env = Earth2VenusEnv(
        N_NODES=N_NODES, 
        amu=amu, 
        v0=v0, 
        r0=r0, 
        vT=vT, 
        rT=rT, 
        m0=m0, 
        max_thrust=Tmax,
        v_ejection=v_ejection,
        mission_time=tof,
        using_reachability=using_reachability
        )


    load_and_run_model(model_path=args.model_dir, env=env, num_episodes=args.episodes, rI=r0, rT=rT, tof=tof, amu=amu, num_nodes=N_NODES)
    # display_plots()