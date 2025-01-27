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
class Earth2MarsEnvPlanar(gym.Env):
    """
    RL Environment for a mass optimal, time fixed, low thrust transfer with fixed initial 
    and final conditions

    Class inputs:
        - N_NODES:           Number of trajectory segments
        - amu:              Gravitational constant of central body
        - mission_time:     total mission time, days
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
        - At time tf / after N_NODES have been completed
    """
    metadata = {'render.modes': ['human']}
    
    """ CLASS CONSTRUCTOR """
    def __init__(self, N_NODES, amu, mission_time, v0, r0, vT, rT, m0, max_thrust, v_ejection, using_reachability):
        super(Earth2MarsEnvPlanar, self).__init__()
        # Initialize environment parameters
        self.v0 = array(v0)
        self.r0 = array(r0)
        self.N_NODES = N_NODES
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
        self.TIME_STEP = None
        self.training_steps = 0
        
        """ ENVIRONMENT BOUNDARIES """
        coe0 = ic2par(r = self.r0, v = self.v0, mu = self.amu)    # initial classical orbital element of the S/C
        coeT = ic2par(r = self.rT, v = self.vT, mu = self.amu)      # classical orbital element of the target
        rpT = coeT[0]*(1 - coeT[1])                               # periapsis radius of the target, km
        raT = coeT[0]*(1 + coeT[1])                               # apoapsis radius of the target, km
        vpT = sqrt(self.amu*(2./rpT - 1/coeT[0]))                 # periapsis velocity of the target, km/s
        self.min_r = 0.1*min(norm(self.r0), rpT)                  # minimum radius, km
        self.max_r = 4.*max(norm(self.r0), raT)                   # maximum radius, km
        self.max_v = 4.*max(norm(self.v0), vpT)                   # maximum velocity, km/s
        
        """ OBSERVATION SPACE """
        # Lower bounds
        o_lb = np.array([-self.max_r, -self.max_r, 0, \
            -self.max_v, -self.max_v, 0, \
            0., 0.])
        # Upper bounds
        o_ub = np.array([+self.max_r, +self.max_r, 0, \
            +self.max_v, +self.max_v, 0, \
            m0, mission_time*DAY2SEC])
        
        self.observation_space = spaces.Box(o_lb, o_ub, dtype=np.float64)
        
        """ ACTION SPACE [yaw, pitch, radius]"""
        # Lower bounds
        a_lb = np.array([-1., 0, -1., -1.])
        # Upper bounds
        a_ub = np.array([1., 0, 1., 1.])

        self.action_space = spaces.Box(a_lb, a_ub, dtype=np.float64)

    def step(self, action):
        # Simulate one time step in the environment
        # Update spacecraft state based on action
        # Compute reward based on new state
        # Return observation, reward, and done flag

        action[1] = 0

        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        
        if self.TIME_STEP is None:
            # Convert the fourth action component to a TIME_STEP in the range [25, 65]
            self.TIME_STEP = 55 + action[3] * 30
            # Remove the TIME_STEP from the action (we only need the first 3 components for movement)
            action = action[:3]
        
        # State at next time step and current control
        r_next, v_next, m_next, dv = self.propagation_step(action)
            
        # Info (state at the beginning of the segment)
        self.sol['rx'] = self.r_current[0]
        self.sol['ry'] = self.r_current[1]
        self.sol['rz'] = 0
        self.sol['vx'] = self.v_current[0]
        self.sol['vy'] = self.v_current[1]
        self.sol['vz'] = 0
        self.sol['m'] = self.m_current
        self.sol['t'] = self.time_passed
        info = self.sol
        
        # Update the spacecraft state
        # print("Mass used: " + str(self.m_current) + " to " + str(m_next))
        # print("Reward: " + str(reward))
        #print("")

        self.r_current = r_next
        self.v_current = v_next

        reward = self.getReward(m_next, action, dv)

        self.m_current = m_next
        self.time_passed += self.TIME_STEP
        
        obs = array([self.r_current[0], self.r_current[1], self.r_current[2], \
                self.v_current[0], self.v_current[1], self.v_current[2], \
                self.m_current, self.time_passed]).astype(np.float64)
        
        self.training_steps += 1
        
        truncated = False       # necessary return of step, if step is cut off early due to timeout etcc.
        return obs, reward, self.isDone, truncated, info
    
    def getReward(self, mass_next, action, dv):
        reward = 0
        reward -= norm(dv) * 20

        for ii in range(len(action)):
            if abs(abs(action[ii]) > 1):
                reward -= 25
            reward -= 100*max(0, abs(action[ii]) - 1)   # added punishment for actions > 1

        return reward
        
    def propagation_step(self, action):
        # Position and velocity at the next time step given no dv, propagate_lagrangian returns tuple containing final position and velocity
        #print(action)
        r_centre, v_centre = propagate_lagrangian(r0 = self.r_current, v0 = self.v_current, tof=(self.TIME_STEP*DAY2SEC), mu = self.amu)
        dv = [0, 0, 0]
        if (self.N_NODES-1) > self.training_steps:
            if (self.using_reachability == 1):   
                state0 = np.concatenate((self.r_current, self.v_current))
                statef = np.concatenate((r_centre, v_centre))

                #Gets current STM
                STM_Current_Full = YA.YA_STM(state0=state0, tof=(self.TIME_STEP*DAY2SEC), mu=self.amu)
                #Obtains useful STM Quadrant
                STM_Current = STM_Current_Full[0:3, 3:6]

                #Obtains RTN State Transition Matrix
                STM_RTN = YA.DCM_LVLH2RTN() @ STM_Current @ np.transpose(YA.DCM_LVLH2RTN())

                #Constructs Rotation Matrices
                M_RTN2ECI_init = YA.RotMat_RTN2Inertial(state0)
                M_RTN2ECI_init_T = np.transpose(YA.RotMat_RTN2Inertial(state0))
                M_RTN2ECI_f = YA.RotMat_RTN2Inertial(statef)

                #Obtains HCI Frame STM
                STM_HCI = M_RTN2ECI_f @ STM_RTN @ M_RTN2ECI_init_T

                STM_SQUARED = np.transpose(STM_HCI @ STM_HCI)

                eigvals, eigvecs = np.linalg.eig(STM_SQUARED)

                idx = eigvals.argsort()[::-1]
                
                eigvals = eigvals[idx]
                eigvecs = eigvecs[:, idx]

                aligned = STM_HCI @ eigvecs

                radius = aligned[:,0]
                transverse = aligned[:,1]
                normal = aligned[:,2]

                radius_rtn = np.transpose(YA.RotMat_RTN2Inertial(statef)) @ radius
                transverse_rtn = np.transpose(YA.RotMat_RTN2Inertial(statef)) @ transverse
                normal_rtn = np.transpose(YA.RotMat_RTN2Inertial(statef)) @ normal
                
                RS_axes_ordered = np.zeros((3, 3))

                # Ordered TRN
                RS_axes_ordered[:, 0] = radius * np.sign(radius_rtn[1])
                RS_axes_ordered[:, 1] = transverse * np.sign(transverse_rtn[0])
                RS_axes_ordered[:, 2] = normal * np.sign(normal_rtn[2])

                r_next = self.action2pos(RS_axes_ordered, action, r_centre)
                
                p = np.zeros((3,7))
                p[:,0] = r_centre + RS_axes_ordered[:, 0]*self.max_thrust
                p[:,1] = r_centre - RS_axes_ordered[:, 0]*self.max_thrust
                p[:,2] = r_centre + RS_axes_ordered[:, 1]*self.max_thrust
                p[:,3] = r_centre - RS_axes_ordered[:, 1]*self.max_thrust
                p[:,4] = r_centre + RS_axes_ordered[:, 2]*self.max_thrust
                p[:,5] = r_centre - RS_axes_ordered[:, 2]*self.max_thrust
                p[:,6] = r_centre

                self.extra_info['semiAxes'] = p
            
                final_step_lambert = lambert_problem(r1=self.r_current, r2=r_next, tof=(self.TIME_STEP*DAY2SEC), mu=self.amu)
                v_r1 = final_step_lambert.get_v1()[0]
                v_next = final_step_lambert.get_v2()[0]

                dv = np.subtract(v_r1,self.v_current)
                self.plotting = np.concatenate((self.r_current, v_r1))
                self.extra_info['Plotting'] = self.plotting.copy()

            else:
                state_alt, dv = self.without_reach(action)
                r_next = state_alt[:3]
                v_next = state_alt[3:6]
                
            m_next = self.Tsiolkovsky(array(dv))
        
        else:  
            # Step to mars (step N-1)
            final_step_lambert = lambert_problem(r1=self.r_current, r2=self.rT, tof=(self.TIME_STEP*DAY2SEC), mu=self.amu)
            lambert_v1 = final_step_lambert.get_v1()[0]
            dv_N_minus_1 = array(lambert_v1) - array(self.v_current)
            carry_m = self.m_current
            self.m_current = self.Tsiolkovsky(array(dv_N_minus_1))
            self.plotting = np.concatenate((self.r_current, lambert_v1))
            self.extra_info['Plotting'] = self.plotting.copy()

            # Equalization with mars (step N)
            lambert_v2 = final_step_lambert.get_v2()[0]
            dv_equalization = np.subtract(array(self.vT), array(lambert_v2)) # velocity of mars - velocity at final step 
            m_next = self.Tsiolkovsky(array(dv_equalization))

            self.m_current = carry_m
            
            r_next = self.rT
            v_next = self.vT
            
            self.isDone = True  

            #Scalar Dv still works for Tsiolkovsky
            dv = norm(dv_N_minus_1) + norm(dv_equalization)
        
        return r_next, v_next, m_next, dv
    
        
        
    def reset(self, seed=None, options=None):
        self.TIME_STEP = None
        self.r_current = self.r0
        self.v_current = self.v0
        self.m_current = self.m0
        self.time_passed = 0.
        self.isDone = False
        self.training_steps = 0
        dvx = 0
        dvy = 0
        dvz = 0
        
        # Reset parameters
        self.sol = {'rx': [], 'ry': [], 'rz': [],
                    'vx': [], 'vy': [], 'vz': [],
                    'm': [],
                    't': []}
        
        obs = array([self.r_current[0], self.r_current[1], 0, \
                self.v_current[0], self.v_current[1], 0, \
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

    def action2pos(self, axes, action, r_centre):       
        #Denormalising angles
        yaw = action[0] * np.pi                 # [-π to π]
        pitch = action[1] * ((np.pi) / 2)       # [-π/2 to π/2]
        r = action[2]                           # [-1 to 1]
        
        # Spherical to Cartesian and Scale with ellipsoid
        x = r * np.cos(pitch) * np.cos(yaw) * self.max_thrust
        y = r * np.cos(pitch) * np.sin(yaw) * self.max_thrust
        z = r * np.sin(pitch) * self.max_thrust

        points = np.zeros((3,1))
        points[0] = x
        points[1] = y
        points[2] = z

        #Align with ellipsoid
        aligned_point = axes @ points
        aligned_point = np.reshape(aligned_point,(1,3))

        #Translate to centre
        translated_point = aligned_point + r_centre

        translated_point = np.squeeze(translated_point)

        return translated_point
    
    def without_reach(self, action):
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
        self.plotting = np.concatenate((self.r_current, v_current_alt))
        self.extra_info['Plotting'] = self.plotting.copy()
        return state_alt, v_delta_alt
    

