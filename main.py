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
    
if __name__ == '__main__': 
    env_id = "700Project"
    
    subfolder = "Plots"
    if not os.path.exists(subfolder):
        os.makedirs(subfolder)
    mpl_use('Agg')
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
    NSTEPS = int(NSTEPS)
    num_cpu = int(num_cpu)
    init_learning_rate = float(init_learning_rate)
    init_clip_range = float(init_clip_range)
    ent_coef = float(ent_coef)
    nminibatches = int(nminibatches)
    n_steps = int(NSTEPS*nminibatches)
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
    amu = MU_SUN                    # km^3/s^2, Gravitational constant of the central body
    rconv = 149600000.              # position, km
    vconv = np.sqrt(amu/rconv)      # velocity, km/s
    tconv = rconv/vconv             # time, s
    mconv = 1000.                   # mass, kg
    aconv = vconv/tconv             # acceleration, km/s^2
    fconv = mconv*aconv             # force, kN
    v_ejection = 50               # propellant ejection velocity #TODO Change
    
    ## INITIAL CONDITIONS ##
    # planet models
    earth = jpl_lp('earth')
    mars = jpl_lp('mars')
    
    # Timing
    start_date_julian = int(start_date_julian)  # departure date from earth
    departure_date_e = epoch(start_date_julian)
    tof = int(tof)      # predetermined TOF
    arrival_date_e = epoch(tof+start_date_julian)
    
    # physical conditions
    r0, v0 = earth.eph(departure_date_e)
    rT, vT = mars.eph(arrival_date_e)
    m0 = float(m_initial)
    # Can do lambert from earth to mars and get v1 and v2
    
    using_reachability = bool(int(using_reachability))
    
    env = Earth2MarsEnv(
        NSTEPS=NSTEPS, 
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

        def step(self, action):
            observation, reward, done, truncated, info = self.env.step(action)
            self.info_logs.append(info)
            return observation, reward, done, truncated, info

        def reset(self, **kwargs):
            return self.env.reset(**kwargs)

        def get_info_logs(self):
            return self.info_logs
    
    wrapped_env = envLoggingWrapper(env)
        
    policy_kwargs = {
        'share_features_extractor': False
    }

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
        tensorboard_log="./logs/"
    )
    
    model.learn(total_timesteps=300000)

    # Access the info logs after training
    info_logs = wrapped_env.get_info_logs()
    # Extract rx, ry, and rz
    rx_out = [log['rx'] for log in info_logs]
    ry_out = [log['ry'] for log in info_logs]
    rz_out = [log['rz'] for log in info_logs]
    rx_out = np.array(rx_out)
    ry_out = np.array(ry_out)
    rz_out = np.array(rz_out)
    
    # Plotting the trajectory in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(rx_out, ry_out, marker='o')

    ax.set_xlabel('rx')
    ax.set_ylabel('ry')
    ax.set_title('Spacecraft Trajectory')

    plot_path = os.path.join("Plots", "test_path1.png")
    plt.savefig(plot_path)
    print(f"Plot saved as {plot_path}")

# TODO: Plot trajectory
