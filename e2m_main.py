import argparse
import time
import os
from distutils.util import strtobool
import random

import stable_baselines3
from stable_baselines3.ppo import PPO
import gymnasium as gym

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.modules.activation as F
from torch.optim import Adam

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import use as mpl_use

import pykep as pk
from pykep.planet import _base, jpl_lp
from pykep.core import epoch, lambert_problem
from pykep import MU_SUN

from e2m_env import Earth2MarsEnv
from e2m_load import load_and_run_model

def plotRun(state_logs,r0,rT):

    positions = [state[:3] for state in state_logs]
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

    plt.show()
    
if __name__ == '__main__': 
    env_id = "700Project"
    
    subfolder = "Plots"
    if not os.path.exists(subfolder):
        os.makedirs(subfolder)
    mpl_use('Qt5Agg')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    #Input settings file
    parser = argparse.ArgumentParser()
    parser.add_argument('--settings', type=str, default="settings_def.txt", \
        help='Input settings file')
    # parser.add_argument('--input_model', type=str, default="final_model", \
    #     help='Input model to load')
    args = parser.parse_args()
    settings_file = "./settings_files/" + args.settings
    # input_model = "./settings_files/" + args.input_model
    
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
    
    load_model = bool(int(load_model))
    Tmax = float(Tmax)
    N_NODES = int(N_NODES)
    num_cpu = int(num_cpu)
    init_learning_rate = float(init_learning_rate)
    init_clip_range = float(init_clip_range)
    ent_coef = float(ent_coef)
    nminibatches = int(nminibatches)
    n_steps = int(N_NODES*nminibatches*5)
    gamma = float(gamma)
    gae_lambda = float(gae_lambda)
    n_epochs = int(n_epochs)
    batch_size = int(batch_size)
    f_coef = float(f_coef)
    max_grad_norm = float(max_grad_norm)
    use_sde = bool(int(use_sde))
    normalize_advantage = bool(int(normalize_advantage))
    sde_sample_freq = int(sde_sample_freq)
    stats_window_size = int(stats_window_size)
    verbose = bool(int(verbose))
    _init_setup_model = bool(int(_init_setup_model))
    
    # Physical constants
    amu = MU_SUN / 1e9              # km^3/s^2, Gravitational constant of the central body
    rconv = 149600000.              # position, km
    vconv = np.sqrt(amu/rconv)      # velocity, km/s
    tconv = rconv/vconv             # time, s
    mconv = 1000.                   # mass, kg
    aconv = vconv/tconv             # acceleration, km/s^2
    fconv = mconv*aconv             # force, kN
    Isp = float(Isp)                # specific impulse of engine 
    v_ejection = 100
    #v_ejection = (pk.G0/1000.*Isp)/vconv   # propellant ejection velocity TODO: Confirm if suitable currently 0.658 if Isp = 2000
    # ## INITIAL CONDITIONS ##
    # # planet models
    earth = jpl_lp('earth')
    mars = jpl_lp('mars')
    
    # Timing
    start_date_julian = int(start_date_julian)  # departure date from earth
    departure_date_e = epoch(start_date_julian)
    tof = int(tof)      # predetermined TOF
    arrival_date_e = epoch(tof+start_date_julian)

    #Same init conds as Zavoli Federici Table 1 (km and km/s)
    r0 = (-140699693.0, -51614428.0, 980.0)
    v0 = (9.774596, -28.07828, 4.337725e-4)
    rT = (172682023.0, 176959469.0, 7948912.0)
    vT = (-16.427384, -14.860506, 9.21486e-2)
    m0 = float(m_initial)
    
    tof = int(tof)

    #print([r0, v0, rT, vT, m0])
    # Can do lambert from earth to mars and get v1 and v2
    
    using_reachability = bool(int(using_reachability))
    
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
    
    class envLoggingWrapper(gym.Wrapper):
        def __init__(self, env):
            super(envLoggingWrapper, self).__init__(env)
            self.info_logs = []
            self.state_logs = []

        def step(self, action):
            observation, reward, done, truncated, info = self.env.step(action)
            self.info_logs.append(info)
            self.state_logs.append(observation)
            return observation, reward, done, truncated, info

        def reset(self, **kwargs):
            return self.env.reset(**kwargs)

        def get_info_logs(self):
            return self.info_logs
        
        def get_state_logs(self):
            return self.state_logs
    
    wrapped_env = envLoggingWrapper(env)
        
    policy_kwargs = {
        'share_features_extractor': False
    }

    
    # Function to get the next model number
    def get_next_run_number(base_dir):
        runs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        runs = sorted([int(d.split('_')[-1]) for d in runs if d.split('_')[-1].isdigit()])
        return runs[-1] + 1 if runs else 1


    def create_directories():
        models_dir = "saved_models/PPO"
        log_base_dir = "logs/PPO"

        if not os.path.exists(models_dir):
            os.makedirs(models_dir)

        if not os.path.exists(log_base_dir):
            os.makedirs(log_base_dir)
        
        next_run = get_next_run_number(models_dir)

        # Set up the directories for the current run
        models_dir = f"{models_dir}/Model_{next_run}"
        logdir = f"{log_base_dir}/Model_{next_run}"

        # Create directories for the current run if they don't exist
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)

        if not os.path.exists(logdir):
            os.makedirs(logdir)

        return models_dir, logdir


    models_dir, logdir = create_directories()

    model = PPO(
        policy='MlpPolicy', 
        env=wrapped_env, 
        learning_rate=init_learning_rate, 
        n_steps=n_steps, 
        batch_size=batch_size,
        n_epochs=n_epochs, 
        gamma=gamma, 
        gae_lambda=gae_lambda, 
        clip_range=init_clip_range, 
        clip_range_vf=None, 
        normalize_advantage=normalize_advantage, 
        ent_coef=ent_coef,
        vf_coef=f_coef, 
        max_grad_norm=max_grad_norm, 
        use_sde=use_sde, 
        sde_sample_freq=sde_sample_freq, 
        rollout_buffer_class=None, 
        rollout_buffer_kwargs=None, 
        target_kl=None, 
        stats_window_size=stats_window_size, 
        policy_kwargs=policy_kwargs, 
        verbose=verbose, 
        seed=None, 
        device='auto', 
        _init_setup_model=_init_setup_model,
        tensorboard_log=logdir#Logging disabled for debugging, to enable : logdir
    )
    
    Interval = 5000  # Checkpoint interval
    total_timesteps = 3000000 # One timestep specifies one impulse
    iters = total_timesteps // Interval

    print("Learning Commenced")
    for i in range(iters):
        model.learn(total_timesteps=Interval, reset_num_timesteps=False, tb_log_name="Data")
        model_path = f"{models_dir}/{Interval*(i+1)}"
        model.save(model_path)
        print(f"Model: {model_path}")
        print(f"Model saved at timestep {Interval*(i+1)}")
    
    # Run_log = wrapped_env.get_state_logs()
    # plotRun(Run_log,r0,rT)


    
