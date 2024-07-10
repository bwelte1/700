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

from e2m_env_no_reachability import Earth2MarsEnv

# import matplotlib
    
if __name__ == '__main__': 
    env_id = "700Project"
    
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
    NITERS = int(NITERS)
    n_steps = int(NSTEPS*nminibatches)
    gamma = float(gamma)
    gae_lambda = float(gae_lambda)
    n_epochs = int(n_epochs)
    batch_size = int(batch_size)
    f_coef = float(f_coef)
    max_grad_norm = floar(max_grad_norm)
    use_sde = bool(int(use_sde))
    normalize_advantage = bool(int(normalize_advantage))
    sde_sample_freq = int(sde_sample_freq)
    stats_window_size = int(stats_window_size)
    verbose = bool(int(verbose))
    _init_setup_model = bool(int(_init_setup_model))
    
    # Physical constants
    amu = 132712440018.             # km^3/s^2, Gravitational constant of the central body
    rconv = 149600000.              # position, km
    vconv = np.sqrt(amu/rconv)      # velocity, km/s
    tconv = rconv/vconv             # time, s
    mconv = 1000.                   # mass, kg
    aconv = vconv/tconv             # acceleration, km/s^2
    fconv = mconv*aconv             # force, kN
    
    # Initial Conditions TODO: CHANGE LATER
    r0 = [0, 0, 0]
    v0 = [0, 0, 0]
    m0 = 1000
    rT = [1, 1, 1]
    vT = [1, 1, 1]
    
    # MISSION TIME
    mission_time = 500
    
    env = Earth2MarsEnv(
        NSTEPS=NSTEPS, 
        NITERS=NITERS, 
        amu=amu, 
        mission_time=mission_time, 
        v0=v0, 
        r0=r0, 
        vT=vT, 
        rT=rT, 
        m0=m0, 
        max_thrust=Tmax
    )
    
    policy_kwargs = {
        'share_features_extractor': False
    }

    model = PPO(
        policy='MlpPolicy', 
        env=env, 
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
        f_coef=f_coef, 
        max_grad_norm=max_grad_norm, 
        use_sde=use_sde, 
        sde_sample_freq=sde_sample_freq, 
        rollout_buffer_class=None, 
        rollout_buffer_kwargs=None, 
        target_kl=None, 
        stats_window_size=stats_window_size, 
        tensorboard_log=None, 
        policy_kwargs=policy_kwargs, 
        verbose=verbose, 
        seed=None, 
        device='auto', 
        _init_setup_model=_init_setup_model
    )
