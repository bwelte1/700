'''
General Process, to be done after action is determined by model.predict():
1. Given current state (without action), propagate s/c to point in space at next timestep given zero action using lagrangian
2. Taking the future point as the centre of the ellipse use max_action to make an ellipsoid
    2.1. Get eigenvalues and eigenvectors of STM
    2.2. 
3. Find which point within ellipse the s/c will reach due to action
4. Solve for velocity, and delta_m using lambert
4. Apply reward function
'''

import numpy as np

amu = 132712440018.             # km^3/s^2, Gravitational constant of the central body
rconv = 149600000.              # position, km
vconv = np.sqrt(amu/rconv)      # velocity, km/s
tconv = rconv/vconv             # time, s
mconv = 1000.                   # mass, kg
aconv = vconv/tconv             # acceleration, km/s^2
fconv = mconv*aconv             # force, kN

tof = 100000

theta = float(theta)     #True anomaly
e = float(e)             #Eccentricity
h = float(h)             #Specific angular momentum
p = float(p)
t = int(t)
t0 = int(t0)

rho = 1 + e*np.cos(theta)
s = rho*np.cos(theta)
c = rho*np.cos(theta)
s_prime = np.cos(theta) + e*np.cos(2*theta)
c_prime = -(np.sin(theta) + e*np.sin(2*theta))
J = (h/p**2)*(t - t0)

STM_ip = np.array([
    [1, -c*(1+1/rho), s*(1+1/rho), J*3*(rho**2)],
    [0, s, c, (2 - 3*e*s*J)],
    [0, 2*s, 2*c-e, 3*(1 - 2*e*s*J)],
    [0, s_prime, c_prime, -3*e*(s_prime*J + s/(rho**2))]
])

STM_oop = STM_ip = (1/rho)*np.array([
    [c, s],
    [-s, c]
])

def step(self, action):
    r_centre,v_centre = propagate_lagrangian(r0 = self.rk, v0 = self.vkm, tof = tof, mu = self.amu)

    r_next = calculate_set(self,r_centre,v_centre,action)

    sln = lambert_problem(self.rk,r_next,tof)

    v2 = sln.get_v2()




def calculate_set(self,r,v):
    eVals_ip, eVecs_ip =  np.linalg.eig(STM_ip)

    eVals_oop, eVecs_oop =  np.linalg.eig(STM_oop)


def Tsiolkowsky(self, DV):
    """
    :param DV: current DV, km/s
    :return mk1: mass at the end of the DV, evaluated through 
        Tsiolkowsky equation, kg

    """
    
    mk1 = self.mk*exp(-norm(DV)/self.ueq)

    return mk1
