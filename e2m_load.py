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

def plot_run(state_logs, r0, rT):
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
        self.extra_info_logs.append(self.env.extra_info)
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
    semiAxes_values = [info['semiAxes'] for info in runlog_extra if 'semiAxes' in info]
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

def load_and_run_model(model_path, env, num_episodes, r0, rT):
    # Ensure the model file has a .zip extension
    model_file_path = f"{model_path}.zip" if not model_path.endswith(".zip") else model_path
    
    if not os.path.exists(model_file_path):
        raise FileNotFoundError(f"Model file not found: {model_file_path}")

    model = PPO.load(model_file_path)
    wrapped_env = EnvLoggingWrapper(env)

    for episode in range(num_episodes):
        obs = wrapped_env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, truncated, info = wrapped_env.step(action)
        
        extra_info_logs = wrapped_env.get_extra_info_logs()
        run_log = wrapped_env.get_state_logs()

        #upload_matlab(run_log,extra_info_logs)

        print(f"Episode {episode + 1} finished.")
        plot_run(run_log, r0, rT)

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
    Tmax = 0.05
    tof = 500
    using_reachability = True

    env = Earth2MarsEnv(
        NSTEPS=10, 
        amu=amu, 
        v0=v0, 
        r0=r0, 
        vT=vT, 
        rT=rT, 
        m0=m0, 
        max_thrust=Tmax,
        v_ejection=150,   #arbitrary
        mission_time=tof,
        using_reachability=using_reachability
    )

    load_and_run_model(args.model_dir, env, args.episodes, r0, rT)