import copy
import numpy as np

from numba import njit

from Constants import Constants
from Utilities import Utilities as UT

from IERS import iers
from SOFA import timediff, invjday, iau_cal2jd, iau_pom00, iau_pnm06a, iau_gst06, iau_rz, iau_sp00, iau_gmst06


def fraction(x):
    return x - np.floor(x)


def gravity_gradient(r, U, AuxParam):
    """
    Computes the gradient of the Earth's harmoic gravity field

    :param r:
    :param U:
    :param AuxParam:
    :return:
    """

    d = 1

    G = np.zeros([3, 3])
    dr = np.zeros([3])

    # Gradient
    for ii in range(3):
        # set offset in i-th component of the position vector
        dr[:] = 0
        dr[ii] = d
        # acceleration difference
        da = acceleration_harmoic(r + dr / 2, U, AuxParam) - acceleration_harmoic(r - dr / 2, U, AuxParam)

        G[:, ii] = da / d

    return G


def acceleration_point_mass(r, s, GM):
    d = r - s
    return -GM * ((d / UT.norm(d) ** 3) + (s / (UT.norm(s) ** 3)))


def chebyshev_3d(t, N, Ta, Tb, Cx, Cy, Cz):
    """

    :param t:
    :param N: number of coefficients
    :param Ta: begin interval
    :param Tb: end interval
    :param Cx: coefficient of chebyshev polynmial x
    :param Cy: coefficient of chebyshev polynmial y
    :param Cz: coefficient of chebyshev polynmial z
    :return:
    """
    # Clenshaw algorithm
    tau = (2 * t - Ta - Tb) / (Tb - Ta)

    f1, f2 = np.zeros([1, 3]), np.zeros([1, 3])

    for ii in range(N, 1, -1):
        old_f1 = copy.copy(f1)
        f1 = 2 * tau * f1 - f2 + np.hstack([Cx[ii], Cy[ii], Cz[ii]])
        f2 = copy.copy(old_f1)

    return tau * f1 - f2 + np.hstack([Cx[0], Cy[0], Cz[0]])


@njit
def legendre_polynomials(n: int, m: int, fi):
    """
    Calculate Legendre polynomials
    :param n:
    :param m:
    :param fi:
    :return:
    """
    pnm = np.zeros((n + 1, m + 1))
    dpnm = np.zeros((n + 1, m + 1))

    pnm[0, 0], dpnm[0, 0] = 1, 0

    pnm[1, 1] = np.sqrt(3) * np.cos(fi)
    dpnm[1, 1] = -np.sqrt(3) * np.sin(fi)

    # diagonal coefficients
    for ii in range(2, n + 1):
        tmp = np.sqrt((2 * ii + 1) / (2 * ii))
        pnm[ii, ii] = tmp * np.cos(fi) * pnm[ii - 1, ii - 1]
        dpnm[ii, ii] = tmp * ((np.cos(fi) * dpnm[ii - 1, ii - 1]) - (np.sin(fi) * pnm[ii - 1, ii - 1]))

    # horizontal first step coefficients
    for ii in range(1, n + 1):
        tmp = np.sqrt(2 * ii + 1)
        pnm[ii, ii - 1] = tmp * np.sin(fi) * pnm[ii - 1, ii - 1]
        dpnm[ii, ii - 1] = tmp * ((np.cos(fi) * pnm[ii - 1, ii - 1]) + (np.sin(fi) * dpnm[ii - 1, ii - 1]))

    j, k = 0, 2

    while True:
        for ii in range(k, n + 1):
            pnm[ii, j] = np.sqrt((2 * ii + 1) / ((ii - j) * (ii + j))) * \
                         ((np.sqrt(2 * ii - 1) * np.sin(fi) * pnm[ii - 1, j]) - (np.sqrt(((ii + j - 1) * (ii - j - 1)) / (2 * ii - 3)) * pnm[ii - 2, j]))

            dpnm[ii, j] = np.sqrt((2 * ii + 1) / ((ii - j) * (ii + j))) * \
                          ((np.sqrt(2 * ii - 1) * np.cos(fi) * pnm[ii - 1, j]) - (np.sqrt(((ii + j - 1) * (ii - j - 1)) / (2 * ii - 3)) * dpnm[ii - 2, j]))

        j += 1
        k += 1
        if j > m:
            break

    return pnm, dpnm


@njit
def get_moments(n_max, m_max, d, pnm, dpnm, lon, Cnm, Snm):
    """

    :param n_max:
    :param m_max:
    :param d:
    :param pnm:
    :param dpnm:
    :param lon:
    :param Cnm:
    :param Snm:
    :return:
    """
    gm = 398600.4415e9
    r_ref = 6378.1366e3

    dUdr, dUdlatgc, dUdlon = 0., 0., 0.
    q1, q2, q3 = 0., 0., 0.

    for nn in range(n_max + 1):
        b1 = (-gm / d ** 2) * (r_ref / d) ** nn * (nn + 1)
        b2 = (gm / d) * (r_ref / d) ** nn
        b3 = (gm / d) * (r_ref / d) ** nn
        for mm in range(m_max + 1):
            mm_lon = mm * lon
            clon = np.cos(mm_lon)
            slon = np.sin(mm_lon)
            q1 += pnm[nn, mm] * (Cnm[nn, mm] * clon + Snm[nn, mm] * slon)
            q2 += dpnm[nn, mm] * (Cnm[nn, mm] * clon + Snm[nn, mm] * slon)
            q3 += mm * pnm[nn, mm] * (Snm[nn, mm] * clon - Cnm[nn, mm] * slon)

        dUdr = dUdr + q1 * b1
        dUdlatgc = dUdlatgc + q2 * b2
        dUdlon = dUdlon + q3 * b3
        q1, q2, q3 = 0., 0., 0.

    return dUdr, dUdlatgc, dUdlon


def acceleration_harmoic(r, E, AuxParam):
    """
    Acceleration harmonics of Earth

    :param r:
    :param E:
    :param AuxParam:
    :return:
    """
    n_max = AuxParam.n_a
    m_max = AuxParam.m_a
    Cnm = AuxParam.Cnm
    Snm = AuxParam.Snm

    # body fixed position
    r_bf = E.dot(r)

    # auxiliary quantities
    d = UT.norm(r_bf)
    latgc = np.arcsin(r_bf[2] / d)
    lon = np.arctan2(r_bf[1], r_bf[0])

    pnm, dpnm = legendre_polynomials(n_max, m_max, latgc)

    dUdr, dUdlatgc, dUdlon = get_moments(n_max, m_max, d, pnm, dpnm, lon, Cnm, Snm)

    # Body fixed acceleration
    r2xy = r_bf[0] ** 2 + r_bf[1] ** 2

    o_d = 1 / d

    d2 = d ** 2
    sr2xy = np.sqrt(r2xy)
    r2dlon = (1 / r2xy * dUdlon)

    ax = (o_d * dUdr - r_bf[2] / (d2 * sr2xy) * dUdlatgc) * r_bf[0] - r2dlon * r_bf[1]
    ay = (o_d * dUdr - r_bf[2] / (d2 * sr2xy) * dUdlatgc) * r_bf[1] - r2dlon * r_bf[0]
    az = o_d * dUdr * r_bf[2] + sr2xy / d2 * dUdlatgc

    a_bf = np.vstack([ax, ay, az])

    return np.dot(E.T, a_bf).squeeze()


def acceleration_iers(t, Y, AuxParam):
    """
    Calculate acceleration due to gravity.
    This will take a state vector, Y, and calculate the rates of change
    It's used as the integration function for an ODE solver
    :param t:
    :param Y:
    :param AuxParam:
    :return:
    """
    SECONDS_PER_DAY = 86400.
    Mjd_UTC = AuxParam.Mjd_UTC + t / SECONDS_PER_DAY

    UT1_UTC, TAI_UTC, x_pole, y_pole, ddpsi, ddeps = iers(AuxParam.eopdata, Mjd_UTC)
    _, _, _, TT_UTC, _ = timediff(UT1_UTC, TAI_UTC)

    year, month, day, hour, minute, second = invjday(Mjd_UTC + 2400000.5)
    DJMJD0, DATE = iau_cal2jd(year, month, day)
    TIME = (60 * (60 * hour + minute) + second) / SECONDS_PER_DAY
    UTC = DATE + TIME
    TT = UTC + TT_UTC / SECONDS_PER_DAY
    TUT = TIME + UT1_UTC / SECONDS_PER_DAY
    UT1 = DATE + TUT

    # polar motion matrix (TIRS->ITRS, IERS 2003)
    Pi = iau_pom00(x_pole, y_pole, (iau_sp00(DJMJD0, TT)))
    # Form bias-precession-nutation matrix
    NPB = iau_pnm06a(DJMJD0, TT)
    # Form earth rotation matrix
    gast = iau_gst06(DJMJD0, UT1, DJMJD0, TT, NPB)
    Theta = iau_rz(gast, np.eye(3))
    # ICRS to ITRS transformation
    E = np.linalg.multi_dot([Pi, Theta, NPB])

    # Acceleration due to harmonic graity field
    a = acceleration_harmoic(Y[:3], E, AuxParam)
    return np.hstack([Y[3:], a])


def variable_equations(yPhi, x, AuxParam):
    Mjd_UTC0 = AuxParam.Mjd_UTC
    eopdata = AuxParam.eopdata

    Mjd_UTC = Mjd_UTC0 + x / Constants.DAYSEC
    UT1_UTC, TAI_UTC, x_pole, y_pole, ddpsi, ddeps = iers(eopdata, Mjd_UTC)
    _, _, _, TT_UTC, _ = timediff(UT1_UTC, TAI_UTC)

    # Transformation matrix
    year, month, day, hour, minute, second = invjday(Mjd_UTC + Constants.DJM0)
    DJMJD0, DATE = iau_cal2jd(year, month, day)
    TIME = (60 * (60 * hour + minute) + second) / Constants.DAYSEC
    UTC = DATE + TIME
    TT = UTC + TT_UTC / Constants.DAYSEC
    TUT = TIME + UT1_UTC / Constants.DAYSEC
    UT1 = DATE + TUT
    U = iau_rz(iau_gmst06(DJMJD0, UT1, DJMJD0, TT), np.eye(3))

    # State vector components
    r = yPhi[:3].squeeze()
    v = yPhi[3:6].squeeze()
    Phi = np.zeros((6, 6))

    # State transition matrix
    try:
        for jj in range(6):
            Phi[:, jj] = yPhi[6*(jj+1):6*(jj+1) + 6].squeeze()
    except Exception as e:
        print(e)
        pass

    a = acceleration_harmoic(r, U, AuxParam)
    G = gravity_gradient(r, U, AuxParam)

    # Time derivative of state transition matrix
    yPhip = np.zeros((42, 1))
    dfdy = np.zeros((6, 6))

    for ii in range(3):
        for jj in range(3):
            dfdy[ii, jj] = 0.0            # dv/dr(i,j)
            dfdy[ii + 3, jj] = G[ii, jj]  # da/dr(i,j)
            if ii == jj:
                dfdy[ii, jj+3] = 1        # dv/dv(i,j)
            else:
                dfdy[ii, jj + 3] = 0.0      # da/dv(i,j)

            dfdy[ii+3, jj+3] = 0.0

    Phip = dfdy.dot(Phi)

    # Derivatie of combined state vector and state transition matrix
    yPhip[:3, 0] = v
    yPhip[3:6, 0] = a

    for ii in range(6):
        for jj in range(6):
            val = 6 * (jj + 1) + (ii + 1)
            yPhip[val - 1] = Phip[ii, jj]

    return yPhip.squeeze()
