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

import pykep as pk
from pykep.planet import _base
from pykep.core import epoch, lambert_problem
from pykep import DAY2SEC
from pykep.orbit_plots import plot_lambert
from pykep import MU_SUN

from e2m_env_no_reachability import Earth2MarsEnv

import matplotlib.pyplot as plt
from matplotlib import use as mpl_use

subfolder = "Plots"
if not os.path.exists(subfolder):
    os.makedirs(subfolder)

mpl_use('Agg')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

r1 = [1e8, 0, 0]  # Example: 100,000,000 km in the x-direction
r2 = [0, 1e8, 0]  # Example: 100,000,000 km in the y-direction
tof = 10 * DAY2SEC

try:
    l = lambert_problem(r1=r1, r2=r2, tof=tof, mu=MU_SUN)
    Nmax = l.get_Nmax()
    sol=0
    pk.orbit_plots.plot_lambert(l, sol=sol, axes=ax, legend=True)
    plot_path = os.path.join(subfolder, "test_path.png")
    plt.savefig(plot_path)
    print(f"Plot saved as {plot_path}")
except RuntimeError as e:
    print(f"An error occurred: {e}")
except ValueError as e:
    print(f"An error occurred: {e}")