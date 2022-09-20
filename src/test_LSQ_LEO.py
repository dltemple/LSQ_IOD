import os
import pickle

import numpy as np
from scipy.integrate import odeint, solve_ivp

from Accelerations import acceleration_iers, variable_equations
from Constants import Constants as Const
from CoordinateChanges import CoordinateChanges, RaDec2AzEl_vect
from SOFA import j2000_eci, ltc_matrix, mjday, iau_cal2jd, iau_gmst06, iau_rz, iers, invjday, timediff
from anglesdr import anglesdr

from aux_params import AuxParam

# from DEInteg import DEInteg
print(os.getcwd())
CC = CoordinateChanges(Const)

with open('../data/DE430Coeff.pkl', 'rb') as f:
	PC = pickle.load(f)

Cnm, Snm = CC.read_harmonics()

eopdata = np.loadtxt('../data/eop19622021.txt')
eopdata = eopdata.T

obs = np.array([2458902.545408345, 58.87856261726468, 14.731759244081474,
                2458902.545436053, 59.27234721283004, 14.210894230605868,
                2458902.545464633, 59.72424896252321, 13.662947113110942,
                2458902.5454930123, 60.166695456812306, 13.07305262024833,
                2458902.545520464, 60.584392136162094, 12.547013704441602,
                2458902.545548699, 61.027979810319685, 11.970820964441293,
                2458902.545576133, 61.436486930172805, 11.430841594485718,
                2458902.5456031663, 61.86396886718889, 10.89408698825507,
                2458902.5456307633, 62.260374230800714, 10.350616114086806,
                2458902.5456603277, 62.67170131717147, 9.789725675970669,
                2458902.5456899777, 63.138615724983836, 9.160130709400882,
                2458902.545718833, 63.55677007170483, 8.59604816484904,
                2458902.545748787, 63.98193991605251, 8.026113799867664,
                2458902.545779709, 64.42103355094028, 7.407506809990097,
                2458902.5458087083, 64.85685320880563, 6.816146644790171,
                2458902.5458365506, 65.2420302503377, 6.2958945528243175,
                2458902.5458641415, 65.6318794698623, 5.759412321982404,
                2458902.5458916947, 66.01330008100292, 5.230946737097588,
                2458902.545919332, 66.39144734459964, 4.682531759949366,
                2458902.5459472155, 66.76052712672337, 4.162533837441324,
                2458902.5459749205, 67.13745202540704, 3.6287794892941734,
                2458902.546002959, 67.53601403362134, 3.069475111040303,
                2458902.546030975, 67.89374005172424, 2.567961917252278,
                2458902.5460589975, 68.28389719509899, 2.0252960240750375,
                2458902.5460861633, 68.62318834503652, 1.5245267244716336,
                2458902.546113516, 68.98378957061303, 1.02006412581391,
                2458902.5461411793, 69.33835802908335, 0.5020223087452188,
                2458902.5461702766, 69.69211981659234, -0.007356974243752226,
                2458902.54619815, 70.05763489801502, -0.5290043566854474,
                2458902.5462263823, 70.4072437159689, -1.035339570740836,
                2458902.546253588, 70.74569660135421, -1.5186516671760322,
                2458902.546282133, 71.08539221130134, -2.0143231977090084,
                2458902.546311317, 70.95111182499451, -3.729669225523737,
                2458902.5463396045, 71.78536745836526, -3.0342245146665845,
                2458902.5463679647, 72.12674361302413, -3.5256291779375597,
                2458902.5463969735, 72.46999251885984, -4.016014818828152,
                2458902.5464251856, 72.7885456380178, -4.48623988131075,
                2458902.546454254, 73.13003284226558, -4.980787467758792,
                2458902.5464833947, 73.47219412967995, -5.464403375876679,
                2458902.54651256, 73.79548747044265, -5.949698980571635,
                2458902.546541102, 74.09930827710612, -6.370159532942733]).reshape([-1, 3])

obs[:, 0] -= Const.DJM0
obs[:, 0] += 0 / Const.DAYSEC
nobs = obs.shape[0]

sigma_az = 10 * 15.6 * 4.84814e-6  # [rad]
sigma_el = 10 * 15.6 * 4.84814e-6  # [rad]

# Convert times to Julian dates
times = obs[:, 0]

sensor_lat = 34.693779
sensor_lon = -86.736969
sensor_alt = 199

lla = [sensor_lat * Const.d2r, sensor_lon * Const.d2r, sensor_alt]

AuxParam.Cnm = Cnm
AuxParam.Snm = Snm
AuxParam.PC = PC
AuxParam.eopdata = eopdata

Rs = CC.llh2ecef(lla[0], lla[1], lla[2])

# number of timesteps, start, mid, end
n2 = int(np.round(nobs / 2))

Mjd1, Mjd2, Mjd3 = times[0], times[n2], times[-1]

E1 = j2000_eci(Mjd1, eopdata)
E2 = j2000_eci(Mjd2, eopdata)
E3 = j2000_eci(Mjd3, eopdata)

Rs1 = E1.T.dot(Rs)
Rs2 = E2.T.dot(Rs)
Rs3 = E3.T.dot(Rs)

# Make sure RA/DEC are converted into Az and El
az, el = RaDec2AzEl_vect(obs[:, 1], obs[:, 2], sensor_lat, sensor_lon, obs[:, 0] + Const.DJM0)

obs[:, 1] = az * Const.d2r
obs[:, 2] = el * Const.d2r

r2, v2 = anglesdr(obs[0, 1], obs[n2, 1], obs[-1, 1],
                  obs[0, 2], obs[n2, 2], obs[-1, 2],
                  Mjd1, Mjd2, Mjd3,
                  Rs1, Rs2, Rs3)

Y0_appr = np.vstack([r2, v2]).squeeze()

state_size = 6

year, month, day, hr, minute, sec = invjday(obs[0, 0] + Const.DJM0)
Mjd0 = mjday(year, month, day, hr, minute, 0.0)

Mjd_UTC = obs[n2, 0]
t = (Mjd_UTC - Mjd0) * Const.DAYSEC
UT1_UTC, TAI_UTC, x_pole, y_pole, ddpsi, ddeps = iers(eopdata, Mjd_UTC)
UT1_TAI, UTC_GPS, UT1_GPS, TT_UTC, GPS_UTC = timediff(UT1_UTC, TAI_UTC)

AuxParam.Mjd_UTC = Mjd_UTC
AuxParam.Mjd_TT = Mjd_UTC + TT_UTC / Const.DAYSEC

t0 = np.array([0, -(obs[n2, 0] - Mjd0) * Const.DAYSEC])
result = solve_ivp(lambda tt, y: acceleration_iers(t, y, AuxParam), t_span=t0, y0=Y0_appr, method='Radau')
Y0 = result.y[:, -1]

npoints = nobs * 2

A = np.zeros([npoints, 6])
b = np.zeros([npoints, 1])
w = np.zeros([npoints, npoints])

# ECEF to ENU
E = ltc_matrix(lla[0], lla[1])

yPhi = np.zeros((42, 1))
Phi = np.zeros((state_size, state_size))

n_iterations = 5
for iteration in range(n_iterations):
	print(f'\nIteration Number. {iteration:4d} \n')
	print('\nResiduals:\n')
	print('    Mjd UTC         Azim(deg)      Elev(deg)\n')

	if iteration == n_iterations - 1:
		y_output = np.zeros((nobs, state_size))
		dAE = np.zeros((nobs, 2))

	for ii in range(nobs):

		Mjd_UTC = times[ii]
		t = (Mjd_UTC - Mjd0) * Const.DAYSEC

		UT1_UTC, TAI_UTC, x_pole, y_pole, ddpsi, ddeps = iers(eopdata, Mjd_UTC)
		UT1_TAI, UTC_GPS, UT1_GPS, TT_UTC, GPS_UTC = timediff(UT1_UTC, TAI_UTC)

		for iii in range(state_size):
			yPhi[iii] = Y0[iii]
			for jj in range(state_size):
				idx = state_size * (jj + 1) + iii
				yPhi[idx] = 1 if iii == jj else 0

		# YPHI INTEGRATION
		yPhi = odeint(variable_equations, yPhi.squeeze(), np.array([0, t]), args=(AuxParam,), rtol=1e-13, atol=1e-6)
		yPhi = np.reshape(yPhi[-1, :], [-1, 1])

		# Extract state transition matrices
		for jj in range(state_size):
			Phi[:, jj] = yPhi[6 * (jj + 1):state_size * (jj + 1) + state_size].squeeze()

		result = solve_ivp(lambda tt, y: acceleration_iers(t, y, AuxParam), t_span=np.array([0, t]), y0=Y0.squeeze(),
		                   method='Radau')
		Y = result.y[:, -1]

		# Topocentric coordinates
		year, month, day, hour, minute, sec = invjday(Mjd_UTC + Const.DJM0)
		DJMJD0, DATE = iau_cal2jd(year, month, day)
		TIME = (60 * (60 * hour + minute) + sec) / Const.DAYSEC
		UTC = DATE + TIME
		TT = UTC + TT_UTC / Const.DAYSEC
		TUT = TIME + UT1_UTC / Const.DAYSEC
		UT1 = DATE + TUT
		U = iau_rz(iau_gmst06(DJMJD0, UT1, DJMJD0, TT), np.eye(3))  # earth rotation
		# r = Y[:3]
		# s = E.dot(U.dot(r) - Rs)
		state_ecef = U.dot(Y[:3])
		ecef_diff = state_ecef - Rs  # difference of state and site in ECEF
		s = E.dot(ecef_diff)  # converted to ENU

		# observations and partial derivs
		Azim, Elev, dAds, dEds = CC.az_el_partials(s)  # these are in ENU
		Dist = np.linalg.norm(s)
		dDds = (s / Dist).T

		# This converts from ECI to ECEF to ENU
		dAdY0 = np.hstack([np.linalg.multi_dot([dAds, E, U])[np.newaxis, :], np.zeros([1, 3])]).dot(Phi)
		dEdY0 = np.hstack([np.linalg.multi_dot([dEds, E, U])[np.newaxis, :], np.zeros([1, 3])]).dot(Phi)

		# Accumulate least-squares system
		A[(2 * ii): 2 * (ii + 1), :] = np.vstack([dAdY0, dEdY0])
		b[(2 * ii): 2 * (ii + 1)] = np.vstack([(obs[ii, 1] - Azim), (obs[ii, 2] - Elev)])

		e3 = np.zeros([2, 2])
		e3[0, 0] = 1 / sigma_az ** 2
		e3[1, 1] = 1 / sigma_el ** 2

		w[2 * ii: 2 * (ii + 1), 2 * ii:2 * (ii + 1)] = e3

		print(
			f'MJD :: {Mjd_UTC:12f} Az :: {(obs[ii, 1] - Azim) * Const.r2d:4.8f} El :: {(obs[ii, 2] - Elev) * Const.r2d:4.8f}')
	# inv(A.T * w * A) * A.T * w * b
	p1 = np.linalg.inv(np.linalg.multi_dot([A.T, w, A])).dot(A.T)
	dY0 = np.linalg.multi_dot([p1, w, b]).squeeze()

	print('Correction:')
	print(' Position [m]')
	print(f'{dY0[0]:6.2f}    {dY0[1]:6.2f}    {dY0[2]:6.2f}')
	print(' Velocity [m/s]')
	print(f'{dY0[3]:6.2f}    {dY0[4]:6.2f}    {dY0[5]:6.2f}')

	Y0 += dY0
