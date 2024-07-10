import gymnasium as gym
from gymnasium import spaces

import numpy as np
from numpy.linalg import norm
from numpy import sqrt, log, exp, cos, sin, arccos, cross, dot, array

import pykep as pk
from pykep.core import propagate_lagrangian, propagate_taylor, ic2par, ic2eq

import stable_baselines3
from stable_baselines3.common.env_checker import check_env


""" RL ENVIRONMENT CLASS """
class Earth2MarsEnv(gym.Env):
    """
    RL Environment for a mass optimal, time fixed, low thrust transfer with fixed initial 
    and final conditions

    Class inputs:
        - NSTEPS:           Number of trajectory segments
        - NITERS:           Number of training iterations
        - amu:              Gravitational constant of central body
        - mission_time:     total mission time, s
        - v0:               initial velocity, km/s, list 
        - r0:               initial position, km, list
        - vT:               target velocity
        - rT:               target position
        - m0:               initial mass, kg
        - max_thrust        maximum thrust force, kN
        - v_ejection        ejection velocity

    Observations:
        Num	Observation                Min     Max
        0	rx                        -max_r   max_r
        1	ry                        -max_r   max_r
        2	rz                        -max_r   max_r
        3	vx                        -max_v   max_v
        4	vy                        -max_v   max_v
        5	vz                        -max_v   max_v
        6	m                           0       1
        7   t                           0       1
        
    Actions:
        Type: Box(3)
        Num	Action
        0	θ - yaw angle                                               -1      1
        1	φ - pitch angle                                             -1      1
        2	r - point distance from centre to edge of ellipsoid         -1      1
    
    Reward:
        [at any time step]                              -mp

    Starting State:
        Start at state: (r0, v0, m0)
    
    Episode Termination:
        - At time tf / after NSTEPS have been completed
    """
    metadata = {'render.modes': ['human']}
    
    """ CLASS CONSTRUCTOR """
    def __init__(self, NSTEPS, NITERS, amu, mission_time, v0, r0, vT, rT, m0, max_thrust, v_ejection):
        super(Earth2MarsEnv, self).__init__()
        # Initialize environment parameters
        self.v0 = array(v0)
        self.r0 = array(r0)
        self.NSTEPS = NSTEPS
        self.NITERS = NITERS
        self.amu = amu
        self.mission_time = mission_time
        self.vT = vT
        self.rT = rT
        self.m0 = m0
        self.max_thrust = max_thrust
        self.v_ejection = v_ejection
        
        self.isDone = False
                
        # Timing
        self.time_step = self.mission_time / self.NSTEPS
        self.training_steps = 0
        
        """ ENVIRONMENT BOUNDARIES """
        coe0 = ic2par(r = self.r0, v = self.v0, mu = self.amu)    # initial classical orbital element of the S/C
        coeT = ic2par(r = self.rTf, v = self.vTf, mu = self.amu)  # classical orbital element of the target
        rpT = coeT[0]*(1 - coeT[1])                               # periapsis radius of the target, km
        raT = coeT[0]*(1 + coeT[1])                               # apoapsis radius of the target, km
        vpT = sqrt(self.amu*(2./rpT - 1/coeT[0]))                 # periapsis velocity of the target, km/s
        self.min_r = 0.1*min(norm(self.r0), rpT)                  # minimum radius, km
        self.max_r = 4.*max(norm(self.r0), raT)                   # maximum radius, km
        self.max_v = 4.*max(norm(self.v0), vpT)                   # maximum velocity, km/s
        
        """ OBSERVATION SPACE """
        # Lower bounds
        o_lb = np.array([-self.max_r, -self.max_r, -self.max_r, \
            -self.max_v, -self.max_v, -self.max_v, \
            0., 0.])
        # Upper bounds
        o_ub = np.array([+self.max_r, +self.max_r, +self.max_r, \
            +self.max_v, +self.max_v, +self.max_v, \
            1., 1.])
        
        self.observation_space = spaces.Box(o_lb, o_ub, dtype=np.float64)
        
        """ ACTION SPACE """
        # Lower bounds
        a_lb = np.array([-1., -1., -1.])
        # Upper bounds
        a_ub = np.array([1., 1., 1.])

        self.action_space = spaces.Box(a_lb, a_ub, dtype=np.float64)

    def step(self, action):
        # Simulate one time step in the environment
        # Update spacecraft state based on action
        # Compute reward based on new state
        # Return observation, reward, and done flag
        
        # Invalid action
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        
        # State at next time step and current control
        r_next, v_next, m_next, u_current = self.propagation_step(action, self.dt)
            
        # Info (state at the beginning of the segment)
        self.sol['rx'] = self.r_current[0]
        self.sol['ry'] = self.r_current[1]
        self.sol['rz'] = self.r_current[2]
        self.sol['vx'] = self.v_current[0]
        self.sol['vy'] = self.v_current[1]
        self.sol['vz'] = self.v_current[2]
        self.sol['ux'] = u_current[0]
        self.sol['uy'] = u_current[1]
        self.sol['uz'] = u_current[2]
        self.sol['m'] = self.m_current
        self.sol['t'] = self.time_passed
        info = self.sol
        
        # Update the spacecraft state
        self.r_current = r_next
        self.v_current = v_next
        self.m_current -= m_next
        self.time_passed += self.time_step
        
        obs = array([self.r_current[0], self.r_current[1], self.r_current[2], \
                self.v_current[0], self.v_current[1], self.v_current[2], \
                self.m_current, self.time_passed]).astype(np.float64)
        
        self.training_steps += 1
        
        reward = self.getReward(action)
        
        return obs, reward, self.isDone, info
    
    def getReward(self, action):
        # minimize fuel consumption
        reward = self.m_current
        
        # Penalty: current action greater than maximum admissible
        reward -= 100.*max(0., norm(action) - 1.)
        
        return reward
        
    def propagation_step(self, action):
        # Position and velocity at the next time step given no dv
        r_next_list, v_next_list = propagate_lagrangian(r0 = self.r_current, v0 = self.v_current, tof = self.time_step, mu = self.amu)
        
        if self.NSTEPS > self.training_steps:
            # TODO: Convert point within ellipsoid to targetable position
            
            # TODO: Use pykep Lambert solver to calculate new velocity
            dv = 0
            pass
        
        # FINAL STEP
        else:  
            r_next = self.rT    # Next position is mars
            # TODO: Calculate required dv with lambert
            self.isDone = True    
        
        # TODO: Add a final final step to equalize velocity with mars
        
        # Spacecraft mass at the next time step
        m_next = self.Tsiolkovsky(dv)
        
        # TODO: Return position and velocity arrays
        r_next = array(r_next_list)
        v_next = array(v_next_list)
        
        return r_next, v_next, m_next, dv
    
        
        
    def reset(self):
        self.r_current = self.r0
        self.v_current = self.v0
        self.m_current = self.m0
        self.time_passed = 0.
        self.isDone = False
        
        # Reset parameters
        self.sol = {'rx': [], 'ry': [], 'rz': [],
                    'vx': [], 'vy': [], 'vz': [],
                    'ux': [], 'uy': [], 'uz': [],
                    'm': [],
                    't': []}
        
        obs = array([self.r_current[0], self.r_current[1], self.r_current[2], \
                self.v_current[0], self.v_current[1], self.v_current[2], \
                self.m_current, self.time_passed]).astype(np.float64)
        
        return obs
    
    def render(self, mode='human', close=False):
        pass
    
    def close(self):
        pass
    
    def Tsiolkovsky(self, dv):
        m_next = self.m_current*exp(-norm(dv)/self.v_ejection)
        return m_next
        


# if __name__ == '__main__':
#     env = Earth2MarsEnv(NSTEPS=10, NITERS=20, amu=5, mission_time=500, v0 = array([0,0,0]), r0=array([0,0,0]), vT=[1,1,1], rT=[1,1,1], m0=1000, max_thrust=0.005)
#     # If the environment don't follow the interface, an error will be thrown
#     check_env(env, warn=True)