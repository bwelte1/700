"""
Script for the computation of the Yamanaka-Ankersen State Transition Matrix (YA-STM)
"""

# Dependencies
import numpy as np
from scipy.optimize import fsolve


"""Conversion function required for the YA_STM computation"""
# Define global tol
global_tol = 1e-13


# Convert eccentric anomaly to mean anomaly
def anomaly_eccentric2mean(E: float,
                           e: float,
                           use_deg: bool = False) -> float:
    """
    Function to convert eccentric anomaly to mean anomaly
    :param E: Eccentric anomaly [rad] or [deg]
    :param e: Eccentricity [--]
    :param use_deg: Flag to use degrees instead of radians for input and output (Default is False)
    :return: Mean anomaly [rad] or [deg]
    """

    # Convert to radians if needed
    if use_deg:
        E = np.deg2rad(E)

    # Compute the mean anomaly
    M = E - e * np.sin(E)

    # Convert to degrees if needed
    if use_deg:
        M = np.rad2deg(M)

    return M


# Convert mean anomaly to eccentric anomaly
def anomaly_mean2eccentric(M: float,
                           e: float,
                           tol: float = global_tol,
                           max_iters: int = 1000,
                           use_deg: bool = False) -> float:
    """
    Function to convert mean anomaly to eccentric anomaly
    :param M: Mean anomaly [rad] or [deg]
    :param e: Eccentricity [--]
    :param tol: Tolerance for convergence (Default is global_tol)
    :param max_iters: Maximum number of iterations (Default is 1000)
    :param use_deg: Flag to use degrees instead of radians for input and output (Default is False)
    :return: Eccentric anomaly [rad] or [deg]
    """

    # Convert to radians if needed
    if use_deg:
        M = np.deg2rad(M)

    # Improved initial guess
    if e < 0.8:
        E = M
    else:
        E = np.pi

    # Newton's method
    for _ in range(max_iters):
        f = E - e * np.sin(E) - M
        f_prime = 1 - e * np.cos(E)
        E_new = E - f / f_prime

        if np.abs(E_new - E) < tol:
            break

        E = E_new

    else:
        E_solution, info, ier, msg = fsolve(
            lambda E: E - e * np.sin(E) - M,
            E,
            full_output=True)
        if ier == 1:
            E = E_solution[0]

        else:
            raise ValueError(f"Newton's method and fsolve did not converge for M={M}, e={e}. fsolve message: {msg}")

    # Convert to degrees if needed
    if use_deg:
        E = np.rad2deg(E)

    return E


# Convert true anomaly to eccentric anomaly
def anomaly_true2eccentric(TA: float,
                           e: float,
                           use_deg: bool = False) -> float:
    """
    Function to convert true anomaly to eccentric anomaly
    :param TA: True anomaly [rad] or [deg]
    :param e: Eccentricity [--]
    :param use_deg: Flag to use degrees instead of radians for input and output (Default is False)
    :return: Eccentric anomaly [rad] or [deg]
    """

    # Convert to radians if needed
    if use_deg:
        TA = np.deg2rad(TA)

    # Compute the eccentric anomaly
    E = 2 * np.arctan(np.sqrt((1 - e) / (1 + e)) * np.tan(TA / 2))

    # Convert to degrees if needed
    if use_deg:
        E = np.rad2deg(E)

    return E


# Convert eccentric anomaly to true anomaly
def anomaly_eccentric2true(E: float,
                           e: float,
                           use_deg: bool = False) -> float:
    """
    Function to convert eccentric anomaly to true anomaly
    :param E: Eccentric anomaly [rad] or [deg]
    :param e: Eccentricity [--]
    :param use_deg: Flag to use degrees instead of radians for input and output (Default is False)
    :return: True anomaly [rad] or [deg]
    """

    # Convert to radians if needed
    if use_deg:
        E = np.deg2rad(E)

    # Compute the true anomaly
    TA = 2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(E / 2))

    # Convert to degrees if needed
    if use_deg:
        TA = np.rad2deg(TA)

    return TA


# Convert true anomaly to mean anomaly
def anomaly_true2mean(TA: float,
                      e: float,
                      tol: float = global_tol,
                      max_iters: int = 1000,
                      use_deg: bool = False) -> float:
    """
    Function to convert true anomaly to mean anomaly
    :param TA: True anomaly [rad] or [deg]
    :param e: Eccentricity [--]
    :param tol: Tolerance for convergence (Default is global_tol)
    :param max_iters: Maximum number of iterations (Default is 1000)
    :param use_deg: Flag to use degrees instead of radians for input and output (Default is False)
    :return: Mean anomaly [rad] or [deg]
    """

    # Convert to radians if needed
    if use_deg:
        TA = np.deg2rad(TA)

    # Compute the eccentric anomaly
    E = 2 * np.arctan(np.sqrt((1 - e) / (1 + e)) * np.tan(TA / 2))

    # Compute the mean anomaly
    M = E - e * np.sin(E)

    # Convert to degrees if needed
    if use_deg:
        M = np.rad2deg(M)

    return M


# Convert mean anomaly to true anomaly
def anomaly_mean2true(M: float,
                      e: float,
                      tol: float = global_tol,
                      max_iters: int = 1000,
                      use_deg: bool = False) -> float:
    """
    Function to convert mean anomaly to true anomaly
    :param M: Mean anomaly [rad] or [deg]
    :param e: Eccentricity [--]
    :param tol: Tolerance for convergence (Default is global_tol)
    :param max_iters: Maximum number of iterations (Default is 1000)
    :param use_deg: Flag to use degrees instead of radians for input and output (Default is False)
    :return: True anomaly [rad] or [deg]
    """

    # Convert to radians if needed
    if use_deg:
        M = np.deg2rad(M)

    # Initial guess
    E = M

    # Newton's method
    for _ in range(max_iters):
        f = E - e * np.sin(E) - M
        f_prime = 1 - e * np.cos(E)
        E_new = E - f / f_prime

        if np.abs(E_new - E) < tol:
            break

        E = E_new

    else:
        raise ValueError("Newton's method did not converge.")

    # Compute the true anomaly
    TA = 2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(E / 2))

    # Convert to degrees if needed
    if use_deg:
        TA = np.rad2deg(TA)

    return TA


# Convert mean anomaly to time of flight
def anomaly_mean2tof(M: float,
                     n: float,
                     use_deg: bool = False) -> float:
    """
    Function to convert mean anomaly to time of flight
    :param M: Mean anomaly [rad] or [deg]
    :param n: Mean motion [rad/s] or [deg/s]
    :param use_deg: Flag to use degrees instead of radians for input and output (Default is False)
    :return: Time of flight [s]
    """

    # Convert to radians if needed
    if use_deg:
        M = np.deg2rad(M)
        n = np.deg2rad(n)

    # Compute the time of flight
    tof = M / n

    return tof


# Convert time of flight to mean anomaly
def anomaly_tof2mean(tof: float,
                     n: float,
                     use_deg: bool = False) -> float:
    """
    Function to convert time of flight to mean anomaly
    :param tof: Time of flight [s]
    :param n: Mean motion [rad/s] or [deg/s]
    :param use_deg: Flag to use degrees instead of radians for input and output (Default is False)
    :return: Mean anomaly [rad] or [deg]
    """

    # Convert to radians if needed
    if use_deg:
        n = np.deg2rad(n)

    # Compute the mean anomaly
    M = n * tof

    # Convert to degrees if needed
    if use_deg:
        M = np.rad2deg(M)

    return M


# Convert eccentric anomaly to time of flight
def anomaly_eccentric2tof(E: float,
                          e: float,
                          n: float,
                          use_deg: bool = False) -> float:
    """
    Function to convert eccentric anomaly to time of flight
    :param E: Eccentric anomaly [rad] or [deg]
    :param e: Eccentricity [--]
    :param n: Mean motion [rad/s] or [deg/s]
    :param use_deg: Flag to use degrees instead of radians for input and output (Default is False)
    :return: Time of flight [s]
    """

    # Convert to radians if needed
    if use_deg:
        E = np.deg2rad(E)
        n = np.deg2rad(n)

    # Compute the mean anomaly
    M = anomaly_eccentric2mean(E, e)

    # Compute the time of flight
    tof = anomaly_mean2tof(M, n)

    return tof


# Convert time of flight to eccentric anomaly
def anomaly_tof2eccentric(tof: float,
                          e: float,
                          n: float,
                          tol: float = global_tol,
                          max_iters: int = 1000,
                          use_deg: bool = False) -> float:
    """
    Function to convert time of flight to eccentric anomaly
    :param tof: Time of flight [s]
    :param e: Eccentricity [--]
    :param n: Mean motion [rad/s] or [deg/s]
    :param tol: Tolerance for convergence (Default is global_tol)
    :param max_iters: Maximum number of iterations (Default is 1000)
    :param use_deg: Flag to use degrees instead of radians for input and output (Default is False)
    :return: Eccentric anomaly [rad] or [deg]
    """

    # Convert to radians if needed
    if use_deg:
        n = np.deg2rad(n)

    # Compute the mean anomaly
    M = anomaly_tof2mean(tof, n)

    # Compute the eccentric anomaly
    E = anomaly_mean2eccentric(M, e, tol, max_iters)

    # Convert to degrees if needed
    if use_deg:
        E = np.rad2deg(E)

    return E


# Convert true anomaly to time of flight
def anomaly_true2tof(TA: float,
                     e: float,
                     n: float,
                     use_deg: bool = False) -> float:
    """
    Function to convert true anomaly to time of flight
    :param TA: True anomaly [rad] or [deg]
    :param e: Eccentricity [--]
    :param n: Mean motion [rad/s] or [deg/s]
    :param use_deg: Flag to use degrees instead of radians for input and output (Default is False)
    :return: Time of flight [s]
    """

    # Convert to radians if needed
    if use_deg:
        TA = np.deg2rad(TA)
        n = np.deg2rad(n)

    # Compute the eccentric anomaly
    E = anomaly_true2eccentric(TA, e)

    # Compute the time of flight
    tof = anomaly_eccentric2tof(E, e, n)

    return tof


# Convert time of flight to true anomaly
def anomaly_tof2true(tof: float,
                     e: float,
                     n: float,
                     tol: float = global_tol,
                     max_iters: int = 1000,
                     use_deg: bool = False) -> float:
    """
    Function to convert time of flight to true anomaly
    :param tof: Time of flight [s]
    :param e: Eccentricity [--]
    :param n: Mean motion [rad/s] or [deg/s]
    :param tol: Tolerance for convergence (Default is global_tol)
    :param max_iters: Maximum number of iterations (Default is 1000)
    :param use_deg: Flag to use degrees instead of radians for input and output (Default is False)
    :return: True anomaly [rad] or [deg]
    """

    # Convert to radians if needed
    if use_deg:
        n = np.deg2rad(n)

    # Compute the mean anomaly
    M = anomaly_tof2mean(tof, n)

    # Compute the eccentric anomaly
    E = anomaly_mean2eccentric(M, e, tol, max_iters)

    # Compute the true anomaly
    TA = anomaly_eccentric2true(E, e)

    # Convert to degrees if needed
    if use_deg:
        TA = np.rad2deg(TA)

    return TA


# Convert R-V state to Classical Orbital Elements (COEs)
def rv2coe(state: np.ndarray,
           mu: float,
           tol: float = global_tol,
           use_deg: bool = False) -> np.ndarray:
    """
    Function to convert the state vector from R-V to COE
    :param state: R-V state of the spacecraft [km, km/s]
    :param mu: Gravitational parameter of the central body [km^3/s^2]
    :param tol: Tolerance for the edge cases (Default is global_tol)
    :param use_deg: Flag to use degrees instead of radians for output (Default is False)
    :return: COE vector [a, e, i, raan, omega, nu]
    """

    # Initialise the R-V state
    r = state[:3]
    v = state[3:6]

    # Magnitudes
    r_norm = np.linalg.norm(r)
    v_norm = np.linalg.norm(v)

    # Angular momentum
    h = np.cross(r, v)
    h_norm = np.linalg.norm(h)
    h_hat = h / h_norm

    # Inclination
    i = np.arctan2(np.sqrt(h_hat[0] ** 2 + h_hat[1] ** 2), h_hat[2])

    # Right Ascension of the Ascending Node (RAAN)
    RAAN = np.arctan2(h_hat[0], -h_hat[1])

    # Semi-latus rectum
    p = h_norm ** 2 / mu

    # Semi-major axis
    a = 1 / (2 / r_norm - v_norm ** 2 / mu)

    # Mean motion
    n = np.sqrt(mu / a ** 3)

    # Numerical stability hack for circular and near-circular orbits
    # Ensures that (1-p/a) is always positive
    if np.isclose(a, p, atol=tol, rtol=tol / 10):
        a = p

    # Eccentricity
    e = np.sqrt(1 - p / a)

    # Eccentric anomaly
    E = np.arctan2(np.dot(r, v) / (n * (a ** 2)), (1 - r_norm / a))

    # True anomaly
    nu = np.arctan2(np.sqrt(1 - e ** 2) * np.sin(E), np.cos(E) - e)

    # Mean longitude
    u = np.arctan2(r[2], (-r[0] * h_hat[1]) + (r[1] * h_hat[0]))

    # Argument of periapsis
    omega = u - nu

    # Correct the angles to run from 0 to 2*pi
    RAAN = (RAAN + 2 * np.pi) % (2 * np.pi)
    omega = (omega + 2 * np.pi) % (2 * np.pi)
    nu = (nu + 2 * np.pi) % (2 * np.pi)

    # Convert to degrees if needed
    if use_deg:
        i = np.rad2deg(i)
        RAAN = np.rad2deg(RAAN)
        omega = np.rad2deg(omega)
        nu = np.rad2deg(nu)

    # COE vector
    COE = np.array([a, e, i, RAAN, omega, nu])

    return COE


"""YA-STM computation function"""


def YA_STM(state0: np.ndarray, tof: float, mu: float) -> np.ndarray:
    """
    Function to compute the state transition matrix using the YA analytical method.
    This function uses the initial R-V state vector and the time of flight to compute the full 6x6 STM.
    The STM is computed in the LVLH frame by default.
    :param state0: Current R-V state vector [km, km/s] (or normalised)
    :param tof: Time of flight [s] (or normalised)
    :param mu: Gravitational parameter of the central body [km^3/s^2] (or normalised)
    :return: YA STM
    """

    # Convert the state vector to the classical orbital elements
    # As the initial state is in [km, km/s], the output will be in [km, rad]
    a, e, i, RAAN, omega, TA0 = rv2coe(state0, mu)

    # Constants
    n = np.sqrt(mu / a ** 3)  # mean motion [1/s]
    p = a * (1 - e ** 2)  # semi-latus rectum [km]
    h = np.sqrt(mu * p)  # angular momentum [km^2/s]
    k = np.sqrt(h) / p  # constant k

    t_offset = anomaly_true2tof(TA0, e, n)
    TA0_prime = anomaly_tof2true(t_offset, e, n)
    TA_prime = anomaly_tof2true(tof + t_offset, e, n)

    # Update the true anomalies according to the t_offset
    TA0 = TA0_prime
    TA = TA_prime

    # At initial true anomaly
    cos_TA0 = np.cos(TA0)
    sin_TA0 = np.sin(TA0)
    rho0 = 1 + e * cos_TA0
    s0 = rho0 * sin_TA0
    c0 = rho0 * cos_TA0

    # First transform to get the pseudo initial conditions for YA at TA0
    ps_Transform = np.block([
        [rho0 * np.eye(3), np.zeros((3, 3))],
        [-e * sin_TA0 * np.eye(3), (1 / ((k ** 2) * rho0)) * np.eye(3)]
    ])

    # Compute the pseudo initial values
    ps_Matrix = (1 / 1 - (e ** 2)) * np.block([
        [1 - e ** 2, 0, 3 * e * s0 * ((1 / rho0) + (1 / (rho0 ** 2))), -e * s0 * (1 + (1 / rho0)), 0,
         -(e * c0) + 2],
        [0, 1 - e ** 2, 0, 0, 0, 0],
        [0, 0, -3 * s0 * ((1 / rho0) + ((e / rho0) ** 2)), s0 * (1 + (1 / rho0)), 0, c0 - (2 * e)],
        [0, 0, -3 * ((c0 / rho0) + e), (c0 * (1 + (1 / rho0))) + e, 0, -s0],
        [0, 0, 0, 0, 1 - e ** 2, 0],
        [0, 0, (3 * rho0) + (e ** 2) - 1, -rho0 ** 2, 0, e * s0]
    ])

    # At the final state
    cos_TA = np.cos(TA)
    sin_TA = np.sin(TA)
    rho = 1 + e * cos_TA
    s = rho * sin_TA
    c = rho * cos_TA
    sPr = cos_TA + e * np.cos(2 * TA)
    cPr = -(sin_TA + e * np.sin(2 * TA))
    J = (k ** 2) * tof

    # At the difference of the final and initial true anomalies
    delta_TA = TA - TA0
    rhoD = 1 + e * np.cos(delta_TA)
    sD = rhoD * np.sin(delta_TA)
    cD = rhoD * np.cos(delta_TA)

    trans_Matrix = np.block([
        [1, 0, -c * (1 + (1 / rho)), s * (1 + (1 / rho)), 0, 3 * (rho ** 2) * J],
        [0, cD / rhoD, 0, 0, sD / rhoD, 0],
        [0, 0, s, c, 0, 2 - (3 * e * s * J)],
        [0, 0, 2 * s, (2 * c) - e, 0, 3 * (1 - (2 * e * s * J))],
        [0, -sD / rhoD, 0, 0, cD / rhoD, 0],
        [0, 0, sPr, cPr, 0, -3 * e * ((sPr * J) + (s / (rho ** 2)))]
    ])

    inv_Ps_Transform = np.block([
        [np.eye(3) / rho, np.zeros((3, 3))],
        [(k ** 2) * e * sin_TA * np.eye(3), (k ** 2) * rho * np.eye(3)]
    ])

    # YA STM
    STM = inv_Ps_Transform @ trans_Matrix @ ps_Matrix @ ps_Transform

    # Eliminate values below the tolerance
    STM[np.abs(STM) < 1e-10] = 0.0

    return STM

def DCM_LVLH2RTN() -> np.ndarray:
    """
    Function to compute the Direction Cosine Matrix (DCM) for LVLH to RTN conversion
    :return: DCM for LVLH to RTN conversion [3x3]
    """

    DCM = np.array([
        [0, 0, -1],
        [1, 0, 0],
        [0, -1, 0]
    ])

    return DCM

def RotMat_RTN2Inertial(state: np.ndarray) -> np.ndarray:
    """
    Function to compute the rotation matrix for RTN to Inertial conversion
    :param state: R-V state of the spacecraft [km, km/s]
    :return: Rotation matrix for RTN to Inertial conversion
    """

    # Extract the position and velocity vectors
    r = state[0:3]
    v = state[3:6]
    n = np.cross(r, v)

    R = r / np.linalg.norm(r)
    N = n / np.linalg.norm(n)
    T = np.cross(N, R)

    R_RTN2Inertial = np.array([
        [R[0], T[0], N[0]],
        [R[1], T[1], N[1]],
        [R[2], T[2], N[2]]
    ])

    return R_RTN2Inertial

