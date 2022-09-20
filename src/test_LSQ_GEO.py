import os
import pickle

import numpy as np
from scipy.integrate import odeint, solve_ivp

from Accelerations import acceleration_iers, variable_equations
from Constants import Constants as Const
from CoordinateChanges import CoordinateChanges
from SOFA import j2000_eci, ltc_matrix, mjday, iau_cal2jd, iau_gmst06, iau_rz, iers, invjday, timediff
from anglesdr import anglesdr
from aux_params import AuxParam

# from DEInteg import DEInteg

print(os.getcwd())

CC = CoordinateChanges(Const)

with open('data/DE430Coeff.pkl', 'rb') as f:
	PC = pickle.load(f)

Cnm, Snm = CC.read_harmonics()

eopdata = np.loadtxt('data/eop19622021.txt')
eopdata = eopdata.T

AuxParam.Cnm = Cnm
AuxParam.Snm = Snm
AuxParam.PC = PC
AuxParam.eopdata = eopdata

times = np.array([49746.1101504629,
                  49746.1102893520,
                  49746.1104398146,
                  49746.1105787037,
                  49746.1107175928,
                  49746.1108564814,
                  49746.1109953704,
                  49746.1111458335,
                  49746.1112847221,
                  49746.1114236112,
                  49746.1115624998,
                  49746.1117013888,
                  49746.1118402779,
                  49746.1119791665,
                  49746.1121180556,
                  49746.1122569446,
                  49746.1123958332,
                  49746.1125347223])

azimuth = np.array([1.0559084894933,
                    1.0846086837131,
                    1.117998577633,
                    1.14996602821253,
                    1.18563706746479,
                    1.22380218221815,
                    1.26564121504696,
                    1.31434811848236,
                    1.36310214580757,
                    1.41580585323004,
                    1.47254825254163,
                    1.53347769672875,
                    1.59864305501047,
                    1.66770398818263,
                    1.74066747756225,
                    1.81668180247436,
                    1.89653585174086,
                    1.97615602688759])

elevation = np.array([0.282624656433946,
                      0.301524826903792,
                      0.323784756183728,
                      0.344285393577653,
                      0.365386424234265,
                      0.386274524722133,
                      0.408216804078206,
                      0.431941064600565,
                      0.453434794338875,
                      0.47454804230025,
                      0.494905562695512,
                      0.514669671145096,
                      0.532616891843354,
                      0.548860671191665,
                      0.562737784074272,
                      0.573848550092468,
                      0.582569960364683,
                      0.586427138011591])

rangem = np.array([2047502,
                   1984677,
                   1918489,
                   1859320,
                   1802186,
                   1747290,
                   1694891,
                   1641201,
                   1594770,
                   1551640,
                   1512085,
                   1476415,
                   1444915,
                   1417880,
                   1395563,
                   1378202,
                   1366010,
                   1359100])

# Observations in JD AZ EL from respective sensor location in LAT LON
obs = np.hstack([times[:, np.newaxis], azimuth[:, np.newaxis], elevation[:, np.newaxis], rangem[:, np.newaxis]])

nobs = rangem.size

# Sensor has range, az, and el uncertainties in measurements
sigma_range = 92.5
sigma_az = 0.0224 * Const.d2r
sigma_el = 0.0139 * Const.d2r

# Sensor position in lat, lon, alt
lat = 21.5748 * Const.d2r
lon = -158.2706 * Const.d2r
alt = 300.20
Rs = CC.llh2ecef(lat, lon, alt)

# number of timesteps, start, mid, end
n2 = int(np.round(nobs / 2)) - 1
Mjd1 = times[0]
Mjd2 = times[n2]
Mjd3 = times[-1]

E1 = j2000_eci(Mjd1, eopdata, 2400000.5)
E2 = j2000_eci(Mjd2, eopdata, 2400000.5)
E3 = j2000_eci(Mjd3, eopdata, 2400000.5)

Rs1 = E1.T.dot(Rs)
Rs2 = E2.T.dot(Rs)
Rs3 = E3.T.dot(Rs)

r2, v2 = anglesdr(obs[0, 1], obs[n2, 1], obs[-1, 1],
                  obs[0, 2], obs[n2, 2], obs[-1, 2],
                  Mjd1, Mjd2, Mjd3,
                  Rs1, Rs2, Rs3)

Y0_appr = np.vstack([r2, v2])

Mjd0 = mjday(1995, 1, 29, 2, 38, 00.0)

Mjd_UTC = obs[n2, 0]
t = (Mjd_UTC - Mjd0) * Const.DAYSEC
UT1_UTC, TAI_UTC, x_pole, y_pole, ddpsi, ddeps = iers(eopdata, Mjd_UTC)
UT1_TAI, UTC_GPS, UT1_GPS, TT_UTC, GPS_UTC = timediff(UT1_UTC, TAI_UTC)

AuxParam.Mjd_UTC = Mjd_UTC

t0 = np.array([0, -(obs[n2, 0] - Mjd0) * 86400])

# result = Radau(lambda t, y: Accel(t, y, Mjd_UTC, eopdata, Cnm, Snm), 0, Y0_appr.squeeze(), -(obs[n2, 0] - Mjd0)*86400, rtol=1e-13, atol=1e-16)

result = solve_ivp(lambda tt, y: acceleration_iers(t, y, AuxParam), t_span=t0, y0=Y0_appr.squeeze(), method='Radau')
Y0 = result.y[:, -1]

A = np.zeros([nobs * 3, 6])
b = np.zeros([nobs * 3, 1])
w = np.zeros([nobs * 3, nobs * 3])

E = ltc_matrix(lat, lon)

yPhi = np.zeros([42, 1])
Phi = np.zeros([6, 6])

for iteration in range(3):
	print('\nIteration Number. {0:4d} \n'.format(iteration))
	print('\nResiduals:\n')
	print('    MjdUTC         Azim(deg)      Elev(deg)      Range(m)\n')

	for ii in range(nobs):

		Mjd_UTC = times[ii]
		t = (Mjd_UTC - Mjd0) * Const.DAYSEC
		UT1_UTC, TAI_UTC, x_pole, y_pole, ddpsi, ddeps = iers(eopdata, Mjd_UTC)
		UT1_TAI, UTC_GPS, UT1_GPS, TT_UTC, GPS_UTC = timediff(UT1_UTC, TAI_UTC)

		for iii in range(6):
			yPhi[iii] = Y0[iii]
			for jj in range(6):
				if iii == jj:
					yPhi[6 * (jj + 1) + iii] = 1
				else:
					yPhi[6 * (jj + 1) + iii] = 0

		# #################################
		# AuxParam.Mjd_UTC = Mjd_UTC
		# AuxParam.Mjd_TT = Mjd_UTC + TT_UTC / Const.DAYSEC
		#
		# yPhi = DEInteg(VarEqn, 0, t, 1e-13, 1e-6, 42, yPhi, AuxParam)
		#
		# # Extract state transition matrices
		# for jj in range(6):
		# 	Phi[:, jj] = yPhi[6 * (jj + 1):6 * (jj + 1) + 6].squeeze()
		#
		# result = solve_ivp(lambda t, y: VarEqn(y, t, AuxParam), t_span=np.array([0, t]), y0=yPhi.squeeze(),
		#                    method='RK45')
		# Y = result.y[:, -1]
		# # y_propagated = propagate(Y0)
		# #################################

		# ODEINT METHOD YPHI INTEGRATION
		yPhi = odeint(variable_equations, yPhi.squeeze(), np.array([0, t]), args=(AuxParam,), rtol=1e-13, atol=1e-6)
		yPhi = np.reshape(yPhi[-1, :], [-1, 1])

		# Extract state transition matrices
		for jj in range(6):
			Phi[:, jj] = yPhi[6 * (jj + 1):6 * (jj + 1) + 6].squeeze()

		result = solve_ivp(lambda tt, y: acceleration_iers(t, y, AuxParam), t_span=np.array([0, t]), y0=Y0.squeeze(),
		                   method='Radau')
		Y = result.y[:, -1]

		# Topocentric coordinates
		year, month, day, hour, minute, sec = invjday(Mjd_UTC + 2400000.5)
		DJMJD0, DATE = iau_cal2jd(year, month, day)
		TIME = (60 * (60 * hour + minute) + sec) / Const.DAYSEC
		UTC = DATE + TIME
		TT = UTC + TT_UTC / Const.DAYSEC
		TUT = TIME + UT1_UTC / Const.DAYSEC
		UT1 = DATE + TUT
		U = iau_rz(iau_gmst06(DJMJD0, UT1, DJMJD0, TT), np.eye(3))  # earth rotation
		r = Y[:3]  # current position
		s = E.dot(U.dot(r) - Rs)  # topocentric position

		# observations and partial derivs
		Azim, Elev, dAds, dEds = CC.az_el_partials(s)
		Dist = np.linalg.norm(s)
		dDds = (s / Dist).T

		dAdY0 = np.hstack([np.linalg.multi_dot([dAds, E, U])[np.newaxis, :], np.zeros([1, 3])]).dot(Phi)
		dEdY0 = np.hstack([np.linalg.multi_dot([dEds, E, U])[np.newaxis, :], np.zeros([1, 3])]).dot(Phi)
		dDdY0 = np.hstack([np.linalg.multi_dot([dDds, E, U])[np.newaxis, :], np.zeros([1, 3])]).dot(Phi)

		# Accumulate least-squares system
		A[(3 * ii): 3 * (ii + 1), :] = np.vstack([dAdY0, dEdY0, dDdY0])
		b[(3 * ii): 3 * (ii + 1)] = np.vstack([(obs[ii, 1] - Azim), (obs[ii, 2] - Elev), (obs[ii, 3] - Dist)])

		e3 = np.zeros([3, 3])
		e3[0, 0] = 1 / sigma_az ** 2
		e3[1, 1] = 1 / sigma_el ** 2
		e3[2, 2] = 1 / sigma_range ** 2

		w[3 * ii: 3 * (ii + 1), 3 * ii:3 * (ii + 1)] = e3

		print(
				f'JD :: {Mjd_UTC:12f} Az Res :: {(obs[ii, 1] - Azim):4.8f} El Res :: {(obs[ii, 2] - Elev):4.8f} Range Res :: {(obs[ii, 3] - Dist):4.8f}')

	# inv(A.T * w * A) * A.T * w * b
	p1 = np.linalg.inv(np.linalg.multi_dot([A.T, w, A]))
	dY0 = np.linalg.multi_dot([p1, A.T, w, b])

	Y0 += dY0.squeeze()
