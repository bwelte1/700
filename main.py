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
    eps_schedule = eps_schedule
    num_cpu = int(num_cpu)
    init_learning_rate = float(init_learning_rate)
    init_clip_range = init_clip_range
    learning_rate = learning_rate
    clip_range = clip_range
    ent_coef = ent_coef
    gamma = gamma
    lam = lam
    noptepochs = noptepochs
    nminibatches = nminibatches
    NITERS = NITERS
    
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
        
    # shared_features_extractor set to False
    policy_kwargs = {
        'share_features_extractor': False
    }

    env = Earth2MarsEnv(NSTEPS=NSTEPS, NITERS=NITERS, amu=amu, mission_time=mission_time, v0=v0, r0=r0, vT=vT, rT=rT, m0=m0, max_thrust=Tmax)

    model = PPO(
        policy='MlpPolicy', 
        env=env, 
        learning_rate=0.0003, 
        n_steps=2048, 
        batch_size=64,
        n_epochs=10, 
        gamma=0.99, 
        gae_lambda=0.95, 
        clip_range=0.2, 
        clip_range_vf=None, 
        normalize_advantage=True, 
        ent_coef=0.0,
        f_coef=0.5, 
        max_grad_norm=0.5, 
        use_sde=False, 
        sde_sample_freq=-1, 
        rollout_buffer_class=None, 
        rollout_buffer_kwargs=None, 
        target_kl=None, 
        stats_window_size=100, 
        tensorboard_log=None, 
        policy_kwargs=None, 
        verbose=0, 
        seed=None, 
        device='auto', 
        _init_setup_model=True
    )
