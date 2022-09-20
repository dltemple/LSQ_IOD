import math
import sys

import numpy as np

from Constants import Constants
from Utilities import Utilities as UT

npna = np.newaxis
npc = np.concatenate


class CoordinateChanges(object):

	def __init__(self, constants: Constants):
		self.constants = constants

	@staticmethod
	def read_harmonics(fname: str = 'data/ITG-Grace03s.txt', n_terms: int = 71):
		cnm = np.zeros([n_terms, n_terms])
		snm = np.zeros([n_terms, n_terms])

		with open(fname, 'r') as fid:
			for nn in range(1, n_terms + 1):
				for mm in range(nn):
					# print('N : ' + str(nn-1) + ' M : ' + str(mm))
					temp = fid.readline()
					ps = temp.split(' ')
					ps2 = [p for p in ps if p != '']
					ps2.pop(-1)
					cnm[nn - 1, mm] = float(ps2[2])
					snm[nn - 1, mm] = float(ps2[3])

		return cnm, snm

	@staticmethod
	def ecef2rae(ecef, lla):
		"""
		########################################################################
		# inputs
		# ecef: a 3xn array of ecef position vectors, measured in (m)
		# lla:  a lat/long/alt reference position, measured in degrees/m
		#
		# outputs
		# rae:  a 3xn array of range/az/el measurements, in meters/degrees
		#       where az is measured from N rotating clockwise is positive
		#       and elevation is measured from horizontal, up positive.
		#########################################################################
		:param ecef:
		:param lla:
		:return:
		"""
		# rad2deg = 180 / np.pi
		if ecef.shape[0] < 3:
			ecef = ecef.T
		ecef_shape = ecef.shape
		rae = np.zeros([3, ecef_shape[1]])
		enu, _ = CoordinateChanges.ecef2enu(lla, ecef[:3, :])
		if enu is None:
			return None

		for ii in range(ecef_shape[1]):
			r = np.sqrt(enu[0, ii] * enu[0, ii]
			            + enu[1, ii] * enu[1, ii]
			            + enu[2, ii] * enu[2, ii])

			az = np.arctan2(enu[0, ii], enu[1, ii])
			el = np.arcsin(enu[2, ii] / r)

			az, el = map(UT.rad2deg, [az, el])

			az = az + 360 if az < 0 else az
			az = az - 360 if az > 360 else az

			rae[:, ii] = [r, az, el]
		return rae

	@staticmethod
	def ecef2enu(lla_ref, pos_ecef=None, vel_ecef=None, ae=Constants.RE, eSQ=Constants.ESQ):
		"""
		########################################################################
		# inputs
		# lla_ref:  the reference geodetic coordinate in lat/long/alt
		#           in degrees & meters not radians & meters
		# posEcef:  position in ECEF coordinates, {3xN}, meters
		# velEcef:  velocity in ECEF coordinates, {3xN}, (velECI required), m/sec
		#
		# outputs
		# posEnu:   an 3xn array of ENU coordinates, measured from the
		#           reference geodetic coordinate, measured in meters.
		# velEnu:   an 3xn array of ENU coordinates, measured from the
		#           reference geodetic coordinate, measured in meters/sec.
		# ti2e:     the transform matrix for ecef to enu for SV vector
		##########################################################################
		:param lla_ref:
		:param pos_ecef:
		:param vel_ecef:
		:param ae:
		:param eSQ:
		:return:
		"""
		# double check that user inputs a 3 component vector for lat/long/alt
		if len(lla_ref.shape) < 2:
			lla_ref = np.expand_dims(lla_ref, 0)

		if lla_ref.shape != (3, 1) and lla_ref.shape != (1, 3):
			print('ERROR (ecef2enu): Vector for lat/long/alt needs to be 3x1 or 1x3')
			return None

		if lla_ref.shape == (1, 3):
			lla_ref = lla_ref.T

		# TRANSFORM MATRIX CREATION

		lat, lon, alt = lla_ref
		# lat,lon input in radians here
		ecef_ref = CoordinateChanges.llh2ecef(lat, lon, alt, ae, eSQ)

		# transformation matrix
		ti2e = np.zeros((3, 3))
		ti2e[0] = [-np.sin(lon), np.cos(lon), 0]
		ti2e[1] = [-np.sin(lat) * np.cos(lon),
		           -np.sin(lat) * np.sin(lon),
		           +np.cos(lat)]
		ti2e[2] = [+np.cos(lat) * np.cos(lon),
		           +np.cos(lat) * np.sin(lon),
		           +np.sin(lat)]

		# if input is only the Lat/Long/Alt, return the transformation matrix
		if pos_ecef is None:
			return ti2e

		if len(pos_ecef.shape) < 2:
			pos_ecef = np.expand_dims(pos_ecef, -1)

		(numRows, numCols) = pos_ecef.shape

		# position array error check
		if numRows != 3:
			print('ERROR (ecef2enu): Input position array must be 3xn array')
			return None

		# initialize output position array
		pos_enu = np.zeros(pos_ecef.shape)
		# loop through all columns of input ECEF_POS and convert to pos_ENU
		for columns in range(numCols):
			calculate = np.dot(ti2e, pos_ecef[:, columns].reshape((3, 1)) - ecef_ref)
			pos_enu[:, columns] = calculate.T
		if vel_ecef is None:
			return pos_enu, ti2e

		if len(vel_ecef.shape) < 2:
			vel_ecef = np.expand_dims(vel_ecef, -1)

		(numRows, numCols) = vel_ecef.shape

		# velocity array error check
		if numRows != 3:
			print('ERROR (ecef2enu): Input velocity array must be 3xn array')
			return None

		# initialize output velocity array
		vel_enu = np.zeros(vel_ecef.shape)

		# loop through each column input of velocities in ECEF
		for columns in range(numCols):
			calculation = np.dot(ti2e, vel_ecef[:, columns])
			vel_enu[:, columns] = calculation
		return pos_enu, vel_enu, ti2e

	@staticmethod
	def llh2ecef(geodetic_latitude, longitude, h=0, ae=Constants.RE, eSQ=Constants.ESQ):
		"""
		########################################################################
		# inputs
		# GeodLat:  Geodetic latitude, {radians}, scalar or {1 x N}
		# Lon:      Longitude, {radians}, scalar or {1 x N}
		# h:        Height above reference ellipsoid, optional with default = 0,
		#           {units of ae, meters}, scalar or {1 x N}
		# ae:       Equatorial earth radius, optional with default defined by
		#           wgs84params routine {distance}, scalar or {1 x N}
		# eSQ:      Eccentricity squared, optional with default defined by
		#           wgs84params routine {unitless}, scalar or {1 x N}
		#
		# outputs
		# ecefPos:    ECEF Position, {distance}, {3 x N}
		########################################################################
		:param geodetic_latitude:
		:param longitude:
		:param h:
		:param ae:
		:param eSQ:
		:return:
		"""
		# print GeodLat, Lon, h, ae, eSQ
		perigee_radius = ae / np.sqrt(1 - eSQ * np.sin(geodetic_latitude) ** 2)
		# if GeodLat.size == 1 and len(h) == 1 and len(Lon) > 1:
		#     h = h + np.zeros(np.shape(Lon))

		return np.array([
			(perigee_radius + h) * np.cos(geodetic_latitude) * np.cos(longitude),
			(perigee_radius + h) * np.cos(geodetic_latitude) * np.sin(longitude),
			(perigee_radius + h - eSQ * perigee_radius) * np.sin(geodetic_latitude)])

	@staticmethod
	def ecef2eci(tepoch_all, ecef_pos, ecef_vel=None, ws=Constants.WE):
		pos_list, vel_list = [], []

		if type(tepoch_all) == int or len(tepoch_all.shape) < 2:
			tepoch_all = np.ones((1, 1)) * tepoch_all

		if len(ecef_pos.shape) < 2:
			ecef_pos = ecef_pos.reshape((1, 3))

		if np.all(ecef_vel) and len(ecef_vel.shape) < 2:
			ecef_vel = ecef_vel.reshape((1, 3))

		nsteps = tepoch_all.shape[0]

		for oo in range(tepoch_all.shape[1]):
			tepoch = tepoch_all[:, oo, 0]
			theta = ws * tepoch
			swdt, cwdt = np.sin(theta), np.cos(theta)

			ecef2eci_mat = np.zeros((nsteps, 3, 3))
			ecef2eci_mat[:, 0, 0] = cwdt
			ecef2eci_mat[:, 0, 1] = -swdt
			ecef2eci_mat[:, 1, 0] = swdt
			ecef2eci_mat[:, 1, 1] = cwdt
			ecef2eci_mat[:, 2, 2] = 1.0

			eci_pos = np.matmul(ecef2eci_mat, ecef_pos[:, oo].reshape((-1, 3, 1))).squeeze(axis=-1)

			pos_list.append(eci_pos)

			if np.all(ecef_vel):
				eci2ecef_mat = UT.trans(ecef2eci_mat)

				inverse_mat = np.linalg.inv(eci2ecef_mat)
				rotation_vector = np.zeros((1, 3))
				rotation_vector[:, -1] = ws
				temp_vector = ecef_vel + np.cross(rotation_vector, ecef_pos[oo, npna, :])
				vel_eci = temp_vector.dot(inverse_mat).squeeze()

				# vel_eci = np.vstack(
				#     [cwdt * ecef_vel[:, 0] - Swdt * ecef_vel[:, 1] - ws * ecef_pos[:, 1],
				#      Swdt * ecef_vel[:, 0] + cwdt * ecef_vel[:, 1] + ws * ecef_pos[:, 0],
				#      ecef_vel[:, 2]])

				vel_list.append(vel_eci)

		pos_out = np.stack(pos_list, axis=0)

		if len(vel_list) > 0:
			vel_out = np.stack(vel_list, axis=0)
		else:
			vel_out = None

		return pos_out, vel_out

	@staticmethod
	def eci2ecef(time, eci_data):
		"""
		########################################################################
		# inputs
		# tepoch:   time (sec) valid input is either scalar
		#           or vector of same length as the number of columns in posECEF
		#           it is the time elapsed since ECI and ECEF were aligned
		# posEci:   position in ECI coordinates, {3xN}, meters
		# velEci:   velocity in ECI coordinates, {3xN}, m/sec
		# accEci:   acceleration in ECI coordinates, {3xN}, m/sec^2
		#
		# outputs
		# posEcef:  position in ECEF coordinates, {3xN}, meters
		# velEcef:  velocity in ECEF coordinates, {3xN}, (velEci required), m/sec
		# accEcef:  acceleration in ECEF coord, {3xN}, (accEci required), m/s^2
		########################################################################
		:param time:
		:param eci_data:
		:return:
		"""

		vel_bool, acc_bool = False, False

		if np.isscalar(time):
			time = np.ones((1, 1)) * time

		if len(eci_data.shape) < 2:
			eci_data = np.expand_dims(eci_data, 0)

		if eci_data.shape[0] > 1 and eci_data.shape[1] < 3:
			eci_data = eci_data.T

		vel_bool = True if eci_data.shape[1] > 3 else vel_bool
		acc_bool = True if eci_data.shape[1] > 6 else acc_bool

		pos = eci_data[:, :3]
		vel = eci_data[:, 3:6] if vel_bool else None
		acc = eci_data[:, 6:9] if acc_bool else None

		cwdt = np.reshape(np.cos(Constants.WE * time), [-1, 1])
		swdt = np.reshape(np.sin(Constants.WE * time), [-1, 1])

		eci2ecef_mat = np.zeros((time.size, 3, 3))
		eci2ecef_mat[:, 0, 0] = cwdt.squeeze()
		eci2ecef_mat[:, 0, 1] = swdt.squeeze()
		eci2ecef_mat[:, 1, 0] = -swdt.squeeze()
		eci2ecef_mat[:, 1, 1] = cwdt.squeeze()
		eci2ecef_mat[:, 2, 2] = 1.0

		ecef_pos = np.matmul(eci2ecef_mat, pos[:, :, npna]).squeeze(axis=-1)
		# ecef_pos = np.hstack([x, y, z])

		if vel_bool:
			ecef_vel = np.matmul(eci2ecef_mat, vel[:, :, npna]).squeeze(axis=-1) - np.cross(
				np.array([0, 0, Constants.WE]),
				ecef_pos)

		# xd = cwdt * vel[:, 0, npna] + swdt * vel[:, 1, npna] + ws * ecef_pos[:, 1, npna]
		# yd = -swdt * vel[:, 0, npna] + cwdt * vel[:, 1, npna] - ws * ecef_pos[:, 0, npna]
		# zd = vel[:, 2, npna]
		#
		# ecef_vel = np.hstack([xd, yd, zd])

		if acc_bool:
			ws2 = pow(Constants.WE, 2)
			ecef_acc = np.hstack(
					[cwdt * acc[:, 0, npna] + swdt * acc[:, 1, npna] + 2 * Constants.WE * (
							-swdt * vel[:, 0, npna] + cwdt * vel[:, 1, npna]) - Constants.WE * ecef_pos[:, 0,
					                                                                           npna] + ws2 * ecef_pos[:,
					                                                                                         0,
					                                                                                         npna],

					 -swdt * acc[:, 0, npna] + cwdt * acc[:, 1, npna] - 2 * Constants.WE * (
							 cwdt * vel[:, 0, npna] + swdt * vel[:, 1, npna]) + Constants.WE * ecef_pos[:, 1,
					                                                                           npna] + ws2 * ecef_pos[:,
					                                                                                         1,
					                                                                                         npna],

					 acc[:, 2, npna]])
		# ecef_acc = acc
		if vel_bool and not acc_bool:
			return np.hstack([ecef_pos, ecef_vel])
		elif acc_bool and vel_bool:
			return np.hstack([ecef_pos, ecef_vel, ecef_acc])
		else:
			return np.hstack([ecef_pos])

	@staticmethod
	def enu2rxy(az, elev, enu_in=None):
		corr_el = UT.deg2rad((90. - elev))
		rot_angle = UT.deg2rad((az - 180.))
		rot_mat_eto_x = np.array([[np.cos(rot_angle), -np.sin(rot_angle), 0.],
		                          [np.sin(rot_angle), np.cos(rot_angle), 0.],
		                          [0., 0., 1.]])

		rot_mat_face_up = np.array([[1., 0., 0.],
		                            [0., np.cos(corr_el), np.sin(corr_el)],
		                            [0., -np.sin(corr_el), np.cos(corr_el)]])

		transmat = rot_mat_face_up @ rot_mat_eto_x
		if enu_in is not None:
			if np.shape(enu_in)[0] != 3:
				if np.shape(enu_in)[1] != 3:
					print("ENU Must be a 3-vector")
					sys.exit(-1)
				else:
					enu_in = enu_in.T
			r_almost = rot_mat_eto_x @ enu_in
			rxy = rot_mat_face_up @ r_almost
			return rxy, transmat
		else:
			return transmat

	@staticmethod
	def uvw2enu_cov(lat, lon):
		slon, clon = np.sin(lon), np.cos(lon)
		slat, clat = np.sin(lat), np.cos(lat)

		uvw2enu = np.zeros((3, 3))
		uvw2enu[0, 0] = np.squeeze(-slon, -1)
		uvw2enu[0, 1] = np.squeeze(clon, -1)
		# uvw2enu[0, 2] = 0
		uvw2enu[1, 0] = np.squeeze(-slat * clon, -1)
		uvw2enu[1, 1] = np.squeeze(-slat * slon, -1)
		uvw2enu[1, 2] = np.squeeze(clat, -1)
		uvw2enu[2, 0] = np.squeeze(clat * clon, -1)
		uvw2enu[2, 1] = np.squeeze(clat * slon, -1)
		uvw2enu[2, 2] = np.squeeze(slat, -1)

		return uvw2enu

	def geodetic_to_lla(self, r: np.array, radius_earth: float = 6378137., f: float = 1.0 / 298.257223563) -> (
			float, float, float):
		"""
		Given ECEF position, tell me the geodetic lat, lon, alt
		:param r:
		:param radius_earth: radius of Earth
		:param f: obliquity
		:return: [deg, deg, m]
		"""
		eps = 2.22044604925031e-16

		required_eps = eps * radius_earth
		e2 = f * (2 - f)

		x, y, z = r.squeeze()
		rho2 = x ** 2 + y ** 2

		dZ = e2 * z

		while True:
			zd_z = z + dZ
			nh = np.sqrt(rho2 + zd_z ** 2)
			sin_phi = zd_z / nh  # sine of geodetic latitude
			N = radius_earth / np.sqrt(1 - e2 * sin_phi ** 2)
			d_z_new = N * e2 * sin_phi
			if np.abs(dZ - d_z_new) < required_eps:
				break

			dZ = d_z_new

		lon = np.arctan2(y, x)
		lat = np.arctan2(zd_z, np.sqrt(rho2))
		alt = (nh - N)

		return lon, lat, alt

	def los_vector(self, az, el):
		return np.array([-np.cos(az) * np.cos(el),
		                 +np.sin(az) * np.cos(el),
		                 +np.sin(el)]).reshape([3, 1])

	def sez_mat(self, lat: float, lon: float):
		"""
		computes SEZ to ECEF rotation matrix
		:param lat:
		:param lon:
		:return:
		"""
		slat, clat = np.sin(lat), np.cos(lat)
		slon, clon = np.sin(lon), np.cos(lon)

		s = np.array([slat * clon, slat * slon, -clat]).reshape([3, 1])
		e = np.array([-slon, clon, 0]).reshape([3, 1])
		z = np.array([clon * clat, slon * clat, slat]).reshape([3, 1])
		return np.hstack([s, e, z])

	def sezl(self, lat, lon, los_vector):
		return self.sez_mat(lat, lon).dot(los_vector)

	def az_el_partials(self, s):
		"""
		Computes partials of azimuth and elevation w.r.t ENU frame
		Building block of Jacobian matrix
		:param s:
		:return: azimuth, elevation, change in az, change in elevation w.r.t time
		"""
		rho = np.sqrt(s[0] ** 2 + s[1] ** 2)

		az = np.arctan2(s[0], s[1])

		if az < 0:
			az += math.pi * 2

		el = np.arctan(s[2] / rho)

		# Partials
		d_ads = np.array([s[1] / rho ** 2, -s[0] / rho ** 2, 0])
		d_eds = np.array([-s[0] * s[2] / rho, -s[1] * s[2] / rho, rho]) / s.dot(s)

		return az, el, d_ads, d_eds

	def radec2rv(self, rr: float, rtasc: float, decl: float, drr: float, drtasc: float, ddecl: float):
		"""
		This function converts the right ascension and declination values to
		position and velocity vectors of a satellite.
		Uses velocity vector to find the solution of singular cases

		:param rr: radius of the satellite
		:param rtasc: right ascension (radians)
		:param decl: declination (radians)
		:param drr: satellite radius rate (er /tu)
		:param drtasc: right ascension rate
		:param ddecl: declination rate
		:return:
		"""
		r, v = np.zeros((3, 1)), np.zeros((3, 1))

		r[0] = rr * np.cos(decl) * np.cos(rtasc)
		r[1] = rr * np.cos(decl) * np.sin(rtasc)
		r[2] = rr * np.sin(decl)

		v[0] = (drr * np.cos(decl) * np.cos(rtasc) -
		        rr * np.sin(decl) * np.cos(rtasc) * ddecl -
		        rr * np.cos(decl) * np.sin(rtasc) * drtasc)

		v[1] = (drr * np.cos(decl) * np.sin(rtasc) -
		        rr * np.sin(decl) * np.sin(rtasc) * ddecl +
		        rr * np.cos(decl) * np.cos(rtasc) * drtasc)

		v[2] = (drr * np.sin(decl) + rr * np.cos(decl) * ddecl)

		return r, v


def ra_dec_to_az_el(ra: float, dec: float, lat: float, lon: float, jd: float) -> (float, float):
	t_ut1 = (jd - 2451545) / 36525
	theta_gmst = (67310.54841 + (876600 * 3600 + 8640184.812866) * t_ut1 +
	              0.093104 * (t_ut1 ** 2) - (6.2 * 10 ** -6) * (t_ut1 ** 3))

	theta_gmst = np.mod((np.mod(theta_gmst, 86400 * (theta_gmst / abs(theta_gmst))) / 240), 360)
	theta_lst = theta_gmst + lon

	d2r = math.pi / 180
	r2d = 180 / math.pi

	LHA = UT.deg2rad(np.mod(theta_lst - ra, 360))

	lat, lon, ra, dec = map(lambda x: d2r * x, [lat, lon, ra, dec])

	el = math.asin(math.sin(lat) * math.sin(dec) + math.cos(lat) * math.cos(dec) * math.cos(LHA))

	az = np.mod(math.atan2(-math.sin(LHA) * math.cos(dec) / math.cos(el),
	                       (math.sin(dec) - math.sin(el) * math.sin(lat)) / (math.cos(el) * math.cos(lat))) * r2d,
	            360)

	return az, el * r2d


def az_el_to_ra_dec(az: float, el: float, lat: float, lon: float, jd: float) -> (float, float):
	"""
	:param az: Local Azimuth Angle (degrees)
	:param el: Local Elevation angle (degrees)
	:param lat: Site latitude in degrees (-90:90)
	:param lon: Site longitude in degrees (-180:180)
	:param jd: Julian Date
	:return:
	"""
	t_ut1 = (jd - 2451545) / 36525
	theta_gmst = (67310.54841 + (876600 * 3600 + 8640184.812866) * t_ut1 +
	              0.093104 * (t_ut1 ** 2) - (6.2 * 10 ** -6) * (t_ut1 ** 3))

	theta_gmst = np.mod((np.mod(theta_gmst, 86400 * (theta_gmst / abs(theta_gmst))) / 240), 360)
	theta_lst = theta_gmst + lon

	d2r = math.pi / 180
	r2d = 180 / math.pi

	lat, lon, az, el = map(lambda x: d2r * x, [lat, lon, az, el])

	dec = np.arcsin(np.sin(el) * np.sin(lat) + np.cos(el) * np.cos(lat) * np.cos(az))
	lha = np.arctan2(-np.sin(az) * np.cos(el) / np.cos(dec),
	                 (np.sin(el) - np.sin(dec) * np.sin(lat)) / (np.cos(dec) * np.cos(lat)))

	ra = np.mod(theta_lst - lha * r2d, 360)

	return ra, dec


AzEl2RaDec_vect = np.vectorize(az_el_to_ra_dec, excluded=['lat', 'lon'])
RaDec2AzEl_vect = np.vectorize(ra_dec_to_az_el, excluded=['lat', 'lon'])

if __name__ == '__main__':
	CoordinateChanges.read_harmonics()

	eci_state = np.array([-1233.5107593334217, 11936.433772555716, 0.0, 0, 0.0, 0])

	ecef_state = CoordinateChanges.eci2ecef(np.array([69020.798671]), eci_state)
	ecef_state
