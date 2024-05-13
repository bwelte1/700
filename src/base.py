import argparse
import os
from distutils.util import strtobool
import time
from torch.utils.tensorboard import SummaryWriter

# import gymnasium as gym
# import matplotlib
# import os
# from stable_baselines3 import PPO
# # import pykep as pk

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"), 
                        help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default="700Project", 
                        help='the id of the gym environment')
    parser.add_argument('--learning-rate', type=float, default=2.5e-4, 
                        help='the learning rate of the experiment')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed of the experiment')
    parser.add_argument('--total-timesteps', type=int, default=25000,
                        help='total timesteps of the experiment')
    parser.add_argument('--torch-deterministic', type=lambda x:bool(strbool(x)), default=True, nargs='?', const=True,
                        help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, cuda will not be enabled by default')
    args = parser.parse_args()
    return args
    
if __name__ == '__main__': 
    args = parse_args()
    print(args)
    run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    for i in range(100):
        writer.add_scalar("test_loss", i*2, global_step=i)