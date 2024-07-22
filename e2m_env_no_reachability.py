import gymnasium as gym
from gymnasium import spaces

import numpy as np
from numpy.linalg import norm
from numpy import sqrt, log, exp, cos, sin, arccos, cross, dot, array

import pykep as pk
from pykep import DAY2SEC
from pykep.core import propagate_lagrangian, ic2par, ic2eq, lambert_problem

import stable_baselines3
from stable_baselines3.common.env_checker import check_env

import YA

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
    def __init__(self, NSTEPS, amu, mission_time, v0, r0, vT, rT, m0, max_thrust, v_ejection):
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
        
        self.isDone = False
                
        # Timing
        self.TIME_STEP = self.mission_time / self.NSTEPS
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
        a_lb = np.array([-180., -180., -1.])
        # Upper bounds
        a_ub = np.array([180., 180., 1.])

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
        self.time_passed += self.TIME_STEP
        
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
        # Position and velocity at the next time step given no dv, propagate_lagrangian returns tuple containing final position and velocity
        r_centre, v_centre = propagate_lagrangian(r0 = self.r_current, v0 = self.v_current, tof = (self.TIME_STEP*DAY2SEC), mu = self.amu)
        
        if (self.NSTEPS-1) > self.training_steps:
            # TODO: Convert point within ellipsoid to targetable position
            state0 = self.r_current + self.v_current
            STM_Current = YA.YA_STM(state0, tof=(self.TIME_STEP*DAY2SEC), mu=self.amu)

            M_Ellipsoid = np.matmul(np.transpose(STM_Current),STM_Current)

            eigvals, eigvecs = np.linalg.eig(M_Ellipsoid)

            #Gets body-centric ellipse axes
            axes = self.getEllipseAxes(self,eigvals,eigvecs,STM_Current)

            offset_position = self.action2pos(self, axes, action)

            r_next = [a + b for a, b in zip(r_centre, offset_position)]

            final_step_lambert = lambert_problem(r1=self.r_current, r2=r_next, tof=(self.TIME_STEP*DAY2SEC), mu=self.amu)
            v_next = final_step_lambert.get_v2()[0]
            dv = v_next - self.v_current
            pass
        
        else:  
            # Step to mars (step N-1)
            final_step_lambert = lambert_problem(r1=self.r_current, r2=self.rT, tof=(self.TIME_STEP*DAY2SEC), mu=self.amu)
            lambert_v1 = final_step_lambert.get_v1()[0]
            dv_N_minus_1 = lambert_v1 - self.v_current
            m_next = self.Tsiolkovsky(dv_N_minus_1)

            # Equalization with mars (step N)
            lambert_v2 = final_step_lambert.get_v2()[0]
            dv_equalization = self.vT - lambert_v2 # velocity of mars - velocity at final step 
            m_next = self.Tsiolkovsky(dv_equalization)
            
            self.isDone = True  
        
        # Spacecraft mass at the next time step
        m_next = self.Tsiolkovsky(dv)
        
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
        #Denormalising
        yaw = action[0] * np.pi                 # [-π to π]
        pitch = action[1] * (np.pi)             # [-π to π]
        r = (action[2] + 1) / 2                 # [0 to 1]
        
        # Spherical to Cartesian
        x = r * np.cos(pitch) * np.cos(yaw)
        y = r * np.cos(pitch) * np.sin(yaw)
        z = r * np.sin(pitch)
        
        # Map the Cartesian coordinates to the ellipsoid axes
        pos = x * axes[:, 0] + y * axes[:, 1] + z * axes[:, 2]
        
        #Eliminating very small values
        pos[np.abs(pos) < 1e-10] = 0

        return pos

    
    # def updateSTM(self):
    #     #This may need to be translated in some way
    #     r_vec = self.r_current
    #     v_vec = self.v_current
    #     r = norm(r_vec)
    #     h_vec = cross(r_vec,v_vec)
    #     h = norm(h_vec)
    #     e_vec = (cross(v_vec,h_vec))/(self.amu) - (r_vec/norm(r_vec))
    #     e = norm(e_vec)
    #     theta = arccos(dot(r_vec,e_vec)/(r*e))
    #     vr = dot(v_vec,(r_vec/norm(r_vec)))
    #     if (vr < 0):
    #         theta = 2*np.pi - theta

    #     rho = 1 + e*np.cos(theta)
    #     s = rho*np.cos(theta)
    #     c = rho*np.cos(theta)
    #     J = (h/rho**2)*(self.time_step)
    #     STM_new = np.array([
    #         [s*(1+1/rho), 0, J*3*(rho**2)],
    #         [0, s/rho, 0],
    #         [c, 0, (2 - 3*e*s*J)]
    #     ])
    #     return STM_new
        


# if __name__ == '__main__':
#     env = Earth2MarsEnv(NSTEPS=10, amu=5, mission_time=500, v0 = array([0,0,0]), r0=array([0,0,0]), vT=[1,1,1], rT=[1,1,1], m0=1000, max_thrust=0.005)
#     # If the environment don't follow the interface, an error will be thrown
#     check_env(env, warn=True)