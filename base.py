import argparse
import os
from distutils.util import strtobool
import time
import stable_baselines3
from torch.utils.tensorboard import SummaryWriter
import random
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO
import torch.nn.modules.activation as F
from stable_baselines3.common.torch_layers import FlattenExtractor
from torch.optim import Adam

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
    parser.add_argument('--torch-deterministic', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, cuda will not be enabled by default')
    parser.add_argument('--track', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='if toggled, this experiment will be tracked with Weights and Biases')
    parser.add_argument('--wandb-project-name', type=str, default="cleanRL",
                        help="the wandb's project name")
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help="the entity (team) of wandb's project")
    args = parser.parse_args()
    return args
    
if __name__ == '__main__': 
    env_id = "700Project"
    args = parse_args()
    print(args)
    run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb
        
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    for i in range(100):
        writer.add_scalar("test_loss", i*2, global_step=i)
        
    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    
    # env = gym.make("700Project")
    # env = gym.wrappers.RecordEpisodeStatistics(env)
    # observation = env.reset()
    # for _ in range(200):
    #     action = env.action_space.sample()
    #     observation, reward, done, info = env.step(action)
    #     if done:
    #         observation = env.reset()
    #         print(f"episodic return: {info['episode']['r']}")
    # env.close()
    
    policy = "MlpPolicy"
    
    stable_baselines3.ppo.PPO(policy, 
                              env, 
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
                              _init_setup_model=True)
    
    observation_space = gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(6,), dytpe=float)
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=float)

    stable_baselines3.common.policies.ActorCriticPolicy(observation_space, 
                                                        action_space, 
                                                        learning_rate=0.0003, 
                                                        net_arch=None,
                                                        activation_fn=F.Tanh, 
                                                        ortho_init=True,
                                                        use_sde=False, 
                                                        log_std_init=0.0, 
                                                        full_std=True, 
                                                        use_expln=False,
                                                        squash_output=False,
                                                        features_extractor_class=FlattenExtractor,
                                                        features_extractor_kwargs=None, 
                                                        share_features_extractor=True,
                                                        normalize_images=True, 
                                                        optimizer_class=Adam, 
                                                        optimizer_kwargs=None)
    
    
