import gymnasium as gym
from gymnasium import spaces

import numpy as np
from numpy.linalg import norm
from numpy import sqrt, log, exp, cos, sin, arccos, cross, dot, array

import matplotlib.pyplot as plt
from matplotlib import use as mpl_use

import pykep as pk
from pykep import DAY2SEC
from pykep.core import propagate_lagrangian, ic2par, ic2eq, lambert_problem

import stable_baselines3
from stable_baselines3.common.env_checker import check_env

import YA

import os

""" RL ENVIRONMENT CLASS """
class Earth2MarsEnv(gym.Env):
    """
    RL Environment for a mass optimal, time fixed, low thrust transfer with fixed initial 
    and final conditions

    Class inputs:
        - NSTEPS:           Number of trajectory segments
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
    def __init__(self, NSTEPS, amu, mission_time, v0, r0, vT, rT, m0, max_thrust, v_ejection, using_reachability):
        super(Earth2MarsEnv, self).__init__()
        # Initialize environment parameters
        self.v0 = array(v0)
        self.r0 = array(r0)
        self.NSTEPS = NSTEPS
        self.amu = amu
        self.mission_time = mission_time
        self.vT = vT
        self.rT = rT
        self.m0 = m0
        self.max_thrust = max_thrust
        self.v_ejection = v_ejection
        self.using_reachability = using_reachability
        
        self.isDone = False
        self.extra_info = {}
        # Timing
        self.TIME_STEP = self.mission_time / self.NSTEPS
        self.training_steps = 0
        
        """ ENVIRONMENT BOUNDARIES """
        coe0 = ic2par(r = self.r0, v = self.v0, mu = self.amu)    # initial classical orbital element of the S/C
        coeT = ic2par(r = self.rT, v = self.vT, mu = self.amu)  # classical orbital element of the target
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
        
        """ ACTION SPACE [yaw, pitch, radius]"""
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
        
        #Clips action
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Invalid action
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        
        # State at next time step and current control
        r_next, v_next, m_next, dv = self.propagation_step(action)
            
        # Info (state at the beginning of the segment)
        self.sol['rx'] = self.r_current[0]
        self.sol['ry'] = self.r_current[1]
        self.sol['rz'] = self.r_current[2]
        self.sol['vx'] = self.v_current[0]
        self.sol['vy'] = self.v_current[1]
        self.sol['vz'] = self.v_current[2]
        self.sol['m'] = self.m_current
        self.sol['t'] = self.time_passed
        info = self.sol
        
        # Update the spacecraft state
        # print("Mass used: " + str(self.m_current) + " to " + str(m_next))
        reward = self.getReward(m_next)
        # print("Reward: " + str(reward))


        self.r_current = r_next
        self.v_current = v_next
        self.m_current = m_next
        self.time_passed += self.TIME_STEP
        
        obs = array([self.r_current[0], self.r_current[1], self.r_current[2], \
                self.v_current[0], self.v_current[1], self.v_current[2], \
                self.m_current, self.time_passed]).astype(np.float64)
        
        self.training_steps += 1
        #print("Impulse Number : " + str(self.training_steps))
        
        truncated = False       # necessary return of step, if step is cut off early due to timeout etcc.
        return obs, reward, self.isDone, truncated, info
    
    def getReward(self, mass_next):
        # minimize fuel consumption
        reward = mass_next - self.m_current
        
        return reward
        
    def propagation_step(self, action):
        # Position and velocity at the next time step given no dv, propagate_lagrangian returns tuple containing final position and velocity
        r_centre, v_centre = propagate_lagrangian(r0 = self.r_current, v0 = self.v_current, tof=(self.TIME_STEP*DAY2SEC), mu = self.amu)
        dv = 0
        if (self.NSTEPS-1) > self.training_steps:
            if self.using_reachability == True:   
                state0 = np.concatenate((self.r_current, self.v_current))
                statef = np.concatenate((r_centre, v_centre))


                delta_v_max_RTN = self.max_thrust*np.eye(3)

                #Gets current STM
                STM_Current_Full = YA.YA_STM(state0=state0, tof=(self.TIME_STEP*DAY2SEC), mu=self.amu)
                #Obtains useful STM Quadrant
                STM_Current = STM_Current_Full[0:3, 3:6]
                #print("Useful untransformed STM: " + str(STM_Current))

                #Obtains RTN State Transition Matrix
                STM_RTN = np.dot(YA.DCM_LVLH2RTN(), STM_Current)

                #Constructs Rotation Matrices
                M_RTN2ECI_init = YA.RotMat_RTN2Inertial(state0)
                M_RTN2ECI_init_T = np.transpose(YA.RotMat_RTN2Inertial(state0))
                M_RTN2ECI_f = YA.RotMat_RTN2Inertial(statef)

                #Obtains HCI Frame STM
                STM_HCI = M_RTN2ECI_f @ STM_RTN @ M_RTN2ECI_init_T

                #print(STM_HCI)

                delta_v_max_HCI = M_RTN2ECI_init @ delta_v_max_RTN
                delta_r_max = np.dot(STM_HCI,delta_v_max_HCI)
                semiAxes = delta_r_max + np.transpose(r_centre)
                self.extra_info['semiAxes'] = delta_r_max
                #print("Max position change: " + str(delta_r_max))
                
                Compare = True
                if (Compare == True):
                    without_reach(self = self, action = action,extra_info = self.extra_info)



                #ALTERNATE REACHABILTY FORMULATION 
                # #Creates characteristic ellipsoid matrix and performs eigendecomposition
                # M_Ellipsoid = np.matmul(np.transpose(STM_RTN),STM_RTN)
                # eigvals, eigvecs = np.linalg.eig(M_Ellipsoid)

                #HCI_eigvecs = M_RTN2ECI_f @ eigvecs

                # #Gets body-centric ellipse axes
                # axes = self.getEllipseAxes(self,eigvals,eigvecs,STM_RTN)
                #print("Action: " + str(action))

                #Maps action to points within ellipse to find distance from centre of ellipse
                offset_position = self.action2pos(delta_r_max, action)
                #print("Position Offset: " + str(offset_position))

                #Adds offset to centre position
                r_next = [a + b for a, b in zip(r_centre, offset_position)]
                #print("Next Position: " + str(r_next))

                #Finds velocity at next stage using lambert and produces dv
                final_step_lambert = lambert_problem(r1=self.r_current, r2=r_next, tof=(self.TIME_STEP*DAY2SEC), mu=self.amu)
                v_next = final_step_lambert.get_v2()[0]
                #print(v_next)
                dv = np.subtract(v_next,self.v_current)
                #print("DeltaV: " + str(dv))
            else:
                r_next = r_centre
                v_next = v_centre
                
            m_next = self.Tsiolkovsky(array(dv))
        
        else:  
            # Step to mars (step N-1)
            final_step_lambert = lambert_problem(r1=self.r_current, r2=self.rT, tof=(self.TIME_STEP*DAY2SEC), mu=self.amu)
            lambert_v1 = final_step_lambert.get_v1()[0]
            dv_N_minus_1 = array(lambert_v1) - array(self.v_current)
            m_next = self.Tsiolkovsky(array(dv_N_minus_1))

            # Equalization with mars (step N)
            lambert_v2 = final_step_lambert.get_v2()[0]
            dv_equalization = array(self.vT) - array(lambert_v2) # velocity of mars - velocity at final step 
            m_next = self.Tsiolkovsky(array(dv_equalization))
            
            r_next = self.rT
            v_next = self.vT
            
            self.isDone = True  
        
        return r_next, v_next, m_next, dv
    
        
        
    def reset(self, seed=None, options=None):
        self.r_current = self.r0
        self.v_current = self.v0
        self.m_current = self.m0
        self.time_passed = 0.
        self.isDone = False
        self.training_steps = 0
        
        # Reset parameters
        self.sol = {'rx': [], 'ry': [], 'rz': [],
                    'vx': [], 'vy': [], 'vz': [],
                    'm': [],
                    't': []}
        
        obs = array([self.r_current[0], self.r_current[1], self.r_current[2], \
                self.v_current[0], self.v_current[1], self.v_current[2], \
                self.m_current, self.time_passed]).astype(np.float64)
        
        info = {}
        
        return obs, info
    
    def render(self, mode='human', close=False):
        pass
    
    def close(self):
        pass
    
    def Tsiolkovsky(self, dv):
        m_next = self.m_current*exp(-norm(dv)/self.v_ejection)
        return m_next

    def getEllipseAxes(self,eigenvalues,eigenvectors,STM):
        '''
        Returns the major and both minor axes of ellipsoid.
        x-axis is semimajor axis.
        z-axis is smaller semiminor axis.
        axes = [x,y,z]
        '''
        #Initialising arrays
        axes = np.zeros((3, eigenvectors.shape[1]))
        axes_sorted = np.zeros((3, eigenvectors.shape[1]))
        norms = np.zeros(3, 1)

        for i in range(eigenvectors.shape[1]):
            axes[i] = self.max_thrust*np.dot(STM,eigenvectors[:,i])
            norms[i] = norm(axes[i])

        norm_indices_s = np.argsort(-norms)
        axes_sorted = axes[:, norm_indices_s]
        return axes_sorted

    def action2pos(self, axes, action):
        #print("Action: " + str(action))
        
        #Denormalising angles
        yaw = action[0] * np.pi                 # [-π to π]
        pitch = action[1] * ((np.pi) / 2)       # [-π/2 to π/2]
        r = action[2]                           # [-1 to 1]

        #print("Yaw, pitch, and r" + str([yaw, pitch, r]))
        
        # Spherical to Cartesian
        x = r * np.cos(pitch) * np.cos(yaw)
        y = r * np.cos(pitch) * np.sin(yaw)
        z = r * np.sin(pitch)
        
        # Map the Cartesian coordinates to the ellipsoid axes
        pos = x * axes[:, 0] + y * axes[:, 1] + z * axes[:, 2]
        
        #Eliminating very small values
        pos[np.abs(pos) < 1e-10] = 0

        return pos

    
def without_reach(self, action, extra_info):
    yaw_alt = action[0] * np.pi                 # [-π to π]
    pitch_alt = action[1] * ((np.pi) / 2)       # [-π/2 to π/2]
    r_alt = action[2]                           # [-1 to 1]
    
    # Spherical to Cartesian
    dvx_alt = r_alt * np.cos(pitch_alt) * np.cos(yaw_alt) * self.max_thrust
    dvy_alt = r_alt * np.cos(pitch_alt) * np.sin(yaw_alt) * self.max_thrust
    dvz_alt = r_alt * np.sin(pitch_alt) * self.max_thrust

    v_delta_alt = [dvx_alt, dvy_alt, dvz_alt]
    v_current_alt = [a + b for a, b in zip(self.v_current, v_delta_alt)]
    r_next_alt, v_next_alt = propagate_lagrangian(r0 = self.r_current, v0 = v_current_alt, tof=(self.TIME_STEP*DAY2SEC), mu = self.amu)

    state_alt = np.concatenate((r_next_alt, v_next_alt))

    extra_info['state'] = state_alt
        


# if __name__ == '__main__':
#     env = Earth2MarsEnv(NSTEPS=10, amu=5, mission_time=500, v0 = array([0,0,0]), r0=array([0,0,0]), vT=[1,1,1], rT=[1,1,1], m0=1000, max_thrust=0.005)
#     # If the environment don't follow the interface, an error will be thrown
#     check_env(env, warn=True)

