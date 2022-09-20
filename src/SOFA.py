import copy
import numpy as np

from Constants import Constants as Const
from IERS import iers


def timediff(ut1_utc, tai_utc):
	tt_tai = 32.184
	gps_tai = -19.0
	# TT_GPS = tt_tai - gps_tai
	# TAI_GPS = -gps_tai
	ut1_tai = ut1_utc - tai_utc
	utc_tai = -tai_utc
	utc_gps = utc_tai - gps_tai
	ut1_gps = ut1_tai - gps_tai
	tt_utc = tt_tai - utc_tai
	gps_utc = gps_tai - utc_tai

	return ut1_tai, utc_gps, ut1_gps, tt_utc, gps_utc


def iau_anp(a):
	w = np.mod(a, Const.D2PI)
	if w < 0:
		w += Const.D2PI
	return w


def iau_bpn2xy(rbpn):
	return rbpn[2, 0], rbpn[2, 1]


def iau_cal2jd(year, month, day):
	djm0 = 2400000.5

	b, c = 0, 0
	if month <= 2:
		year -= 1
		month += 12

	if year < 0:
		c = -0.75

	if year > 1582:
		a = np.floor(year / 100)
		b = 2 - a + np.floor(a / 4)
	elif month > 10:
		a = np.floor(year / 100)
		b = 2 - a + np.floor(a / 4)
	elif day > 14:
		a = np.floor(year / 100)
		b = 2 - a + np.floor(a / 4)

	jd = np.floor(365.25 * year + c) + np.floor(30.6001 * (month + 1))
	jd += day + b + 1720994.5
	djm = jd - djm0

	return djm0, djm


def iauCp(p):
	for _ in range(3):
		c[_] = p[_]
	return c


def iauCr(r):
	for _ in range(3):
		c[_, :] = iauCp(r[_, :])
	return c


def iau_eors(rnpb, s):
	x = rnpb[2, 0]
	ax = x / (1 + rnpb[2, 2])
	xs = 1 - ax * x
	ys = -ax * rnpb[2, 1]
	zs = -x
	p = rnpb[0, 0] * xs + rnpb[0, 1] * ys + rnpb[0, 2] * zs
	q = rnpb[1, 0] * xs + rnpb[1, 1] * ys + rnpb[1, 2] * zs

	if (p != 0) or (q != 0):
		return s - np.arctan2(q, p)
	else:
		return s


def iauEra00(dj1, dj2):
	if dj1 < dj2:
		d1, d2 = dj1, dj2
	else:
		d1, d2 = dj2, dj1

	t = d1 + (d2 - Const.DJ00)

	f = np.mod(d1, 1) + np.mod(d2, 1)

	# Earth rotation angle at this UT1
	return iau_anp(Const.D2PI * (f + 0.7790572732640 + 0.00273781191135448 * t))


def iau_fad03(t):
	return np.mod(1072260.703692 +
	              t * (1602961601.2090 +
	                   t * (- 6.3706 +
	                        t * (0.006593 +
	                             t * (- 0.00003169)))), Const.TURNAS) * Const.DAS2R


def iau_fae03(t):
	return np.mod(1.753470314 + 628.3075849991 * t, Const.D2PI)


def iau_faf03(t):
	return np.mod(335779.526232 +
	              t * (1739527262.8478 +
	                   t * (- 12.7512 +
	                        t * (- 0.001037 +
	                             t * 0.00000417))), Const.TURNAS) * Const.DAS2R


def iau_fal03(t):
	return np.mod(485868.249036 +
	              t * (1717915923.2178 +
	                   t * (31.8792 +
	                        t * (0.051635 +
	                             t * (- 0.00024470)))), Const.TURNAS) * Const.DAS2R


def iau_falp03(t):
	return np.mod(1287104.793048 +
	              t * (129596581.0481 +
	                   t * (- 0.5532 +
	                        t * (0.000136 +
	                             t * (- 0.00001149)))), Const.TURNAS) * Const.DAS2R


def iauFaom03(t):
	return np.mod(450160.398036 +
	              t * (- 6962890.5431 +
	                   t * (7.4722 +
	                        t * (0.007702 +
	                             t * (-0.00005939)))), Const.TURNAS) * Const.DAS2R


def iau_fapa03(t):
	return (0.024381750 + 0.00000538691 * t) * t


def iau_fave03(t):
	return np.mod(3.176146697 + 1021.3285546211 * t, Const.D2PI)


def iau_fw2m(gamb, phib, psi, eps):
	rpom = np.eye(3)
	rpom = iau_rz(gamb, rpom)
	rpom = iau_rx(phib, rpom)
	rpom = iau_rz(-psi, rpom)
	rpom = iau_rx(-eps, rpom)

	return rpom


def iau_gmst06(uta, utb, tta, ttb):
	t = ((tta - Const.DJ00) + ttb) / Const.DJC

	return iau_anp(iauEra00(uta, utb) +
	               (0.014506 +
	               (4612.156534 +
	                (1.3915817 +
	                 (-0.00000044 +
	                  (-0.000029956 +
	                   (-0.0000000368)
	                   * t) * t) * t) * t) * t) * Const.DAS2R)


def iau_gst06(uta, utb, tta, ttb, rnpb):
	"""
	Greenwich apparent sidereal time, IAU 2006, given the NPB Matrix
	:param uta:
	:param utb:
	:param tta:
	:param ttb:
	:param rnpb:
	:return:
	"""
	# Extract CIP coordinates
	x, y = iau_bpn2xy(rnpb)

	# The CIO locator, s
	s = iau_s06(tta, ttb, x, y)

	# Greenwich apparent sidereal time
	era = iauEra00(uta, utb)
	eors = iau_eors(rnpb, s)

	return iau_anp(era - eors)


def iau_ir(r=None):
	return np.eye(3)


def iau_nut06a(date1, date2):
	# Interval between fundamental date J2000.0 and given date (JC).
	t = ((date1 - Const.DJ00) + date2) / Const.DJC

	# Factor correcting for secular variation of J2.
	fj2 = -2.7774e-6 * t

	# Obtain IAU 2000A nutation.
	[dp, de] = iau_nut00a(date1, date2)

	# Apply P03 adjustments (Wallace & Capitaine, 2006, Eqs.5).
	dpsi = dp + dp * (0.4697e-6 + fj2)
	deps = de + de * fj2

	return dpsi, deps


def iau_obl06(date1, date2):
	t = ((date1 - Const.DJ00) + date2) / Const.DJC

	# mean obliquity
	eps0 = (84381.406 +
	        (-46.836769 +
	         (-0.0001831 +
	          (0.00200340 +
	           (-0.000000576 +
	            (-0.0000000434) * t) * t) * t) * t) * t) * Const.DAS2R

	return eps0


def iau_pfw06(date1, date2):
	# Interval between fundamental date J2000.0 and given date (JC).
	t = ((date1 - Const.DJ00) + date2) / Const.DJC

	# P03 bias+precession angles
	gamb = (-0.052928 +
	        (10.556378 +
	         (0.4932044 +
	          (-0.00031238 +
	           (-0.000002788 +
	            0.0000000260
	            * t) * t) * t) * t) * t) * Const.DAS2R
	phib = (84381.412819 +
	        (-46.811016 +
	         (0.0511268 +
	          (0.00053289 +
	           (-0.000000440 +
	            (-0.0000000176)
	            * t) * t) * t) * t) * t) * Const.DAS2R
	psib = (-0.041775 +
	        (5038.481484 +
	         (1.5584175 +
	          (-0.00018522 +
	           (-0.000026452 +
	            (-0.0000000148)
	            * t) * t) * t) * t) * t) * Const.DAS2R

	epsa = iau_obl06(date1, date2)

	return gamb, phib, psib, epsa


def iau_pn06(date1, date2, dpsi, deps):
	gamb, phib, psib, eps = iau_pfw06(Const.DJM0, Const.DJM00)

	# B matrix
	r1 = iau_fw2m(gamb, phib, psib, eps)
	rb = iauCr(r1)

	# Bias precession Fukushima-Williams angles of date
	gamb, phib, psib, eps = iau_pfw06(date1, date2)

	# Bias precession matrix
	r2 = iau_fw2m(gamb, phib, psib, eps)
	rbp = iauCr(r2)

	# Solve for precession matrix
	rt = iauTr(r1)
	rp = iau_rxr(r2, rt)

	# Equinox-based bias-precession-nutation matrix
	r1 = iau_fw2m(gamb, phib, psib + dpsi, eps + deps)
	rbpn = iauCr(r1)

	# Solve for nutation matrix
	rt = iauTr(r2)
	rn = iau_rxr(r1, rt)

	# Obliquity, mean of date
	epsa = eps

	return epsa, rb, rp, rbp, rn, rbpn


def iau_pnm06a(date1, date2):
	# Fukushima-Williams angles for frame bias and precession
	gamb, phib, psib, epsa = iau_pfw06(date1, date2)

	# Nutation components
	dp, de = iau_nut06a(date1, date2)

	# Equinox based nutation x precession x bias matrix
	return iau_fw2m(gamb, phib, psib + dp, epsa + de)


def iau_pom00(xp, yp, sp):
	# Construct the matrix.
	rpom = np.eye(3)
	rpom = iau_rz(sp, rpom)
	rpom = iau_ry(-xp, rpom)
	rpom = iau_rx(-yp, rpom)

	return rpom


def iau_rx(phi, r):
	s = np.sin(phi)
	c = np.cos(phi)

	a10 = c * r[1, 0] + s * r[2, 0]
	a11 = c * r[1, 1] + s * r[2, 1]
	a12 = c * r[1, 2] + s * r[2, 2]
	a20 = -s * r[1, 0] + c * r[2, 0]
	a21 = -s * r[1, 1] + c * r[2, 1]
	a22 = -s * r[1, 2] + c * r[2, 2]

	r[2, 0] = a20
	r[2, 1] = a21
	r[2, 2] = a22

	r[1, 0] = a10
	r[1, 1] = a11
	r[1, 2] = a12

	return r


def iau_rxr(a, b):
	wm = np.zeros((3, 3))

	for ii in range(3):
		for jj in range(3):
			w = 0
			for kk in range(3):
				w += a[ii, kk] * b[kk, jj]

			wm[ii, jj] = w

	return iauCr(wm)


def iau_ry(theta, r):
	s = np.sin(theta)
	c = np.cos(theta)

	a00 = c * r[0, 0] - s * r[2, 0]
	a01 = c * r[0, 1] - s * r[2, 1]
	a02 = c * r[0, 2] - s * r[2, 2]
	a20 = s * r[0, 0] + c * r[2, 0]
	a21 = s * r[0, 1] + c * r[2, 1]
	a22 = s * r[0, 2] + c * r[2, 2]

	r[0, 0] = a00
	r[0, 1] = a01
	r[0, 2] = a02
	r[2, 0] = a20
	r[2, 1] = a21
	r[2, 2] = a22

	return r


def iau_rz(psi, r):
	s = np.sin(psi)
	c = np.cos(psi)

	a00 = c * r[0, 0] + s * r[1, 0]
	a01 = c * r[0, 1] + s * r[1, 1]
	a02 = c * r[0, 2] + s * r[1, 2]
	a10 = -s * r[0, 0] + c * r[1, 0]
	a11 = -s * r[0, 1] + c * r[1, 1]
	a12 = -s * r[0, 2] + c * r[1, 2]

	r[0, 0] = a00
	r[0, 1] = a01
	r[0, 2] = a02
	r[1, 0] = a10
	r[1, 1] = a11
	r[1, 2] = a12

	return r


def iau_sp00(date1, date2):
	t = ((date1 - Const.DJ00) + date2) / Const.DJC
	return -47e-6 * t * Const.DAS2R


def iauTr(r):
	# wm = np.zeros([3, 3])
	# for ii in range(3):
	#     for jj in range(3):
	#         wm[ii, jj] = r[jj, ii]
	return iauCr(r.T)


def iau_nut00a(date1, date2):
	u2_r = Const.DAS2R / 1e7

	# Interval between fundamental date J2000.0 and given date (JC).
	t = ((date1 - Const.DJ00) + date2) / Const.DJC

	# -------------------
	# LUNI-SOLAR NUTATION
	# -------------------

	# Fundamental (Delaunay) arguments

	# Mean anomaly of the Moon (IERS 2003).
	el = iau_fal03(t)

	# Mean anomaly of the Sun (MHB2000)
	elp = np.mod(1287104.79305 +
	             t * (129596581.0481 +
	                  t * (-0.5532 +
	                       t * (0.000136 +
	                            t * (-0.00001149)))), Const.TURNAS) * Const.DAS2R

	# Mean longitude of the Moon minus that of the ascending node (IERS 2003).
	f = iau_faf03(t)

	# Mean elongation of the Moon from the Sun (MHB2000).
	d = np.mod(1072260.70369 +
	           t * (1602961601.2090 +
	                t * (-6.3706 +
	                     t * (0.006593 +
	                          t * (-0.00003169)))), Const.TURNAS) * Const.DAS2R

	# Mean longitude of the ascending node of the Moon (IERS 2003).
	om = iauFaom03(t)

	temp_arr = np.array([el, elp, f, d, om]).reshape((1, -1))
	all_args = np.sum(Const.xls_nut00a[:, :5] * temp_arr, axis=-1)
	all_args = np.mod(all_args, Const.D2PI)
	s_args = np.sin(all_args)
	c_args = np.cos(all_args)

	dp = np.sum((Const.xls_nut00a[:, 5] + Const.xls_nut00a[:, 6] * t) * s_args + Const.xls_nut00a[:, 7] * c_args)
	de = np.sum((Const.xls_nut00a[:, 8] + Const.xls_nut00a[:, 9] * t) * s_args + Const.xls_nut00a[:, 10] * c_args)

	# Convert from 0.1 microarcsec units to radians.
	dpsils = dp * u2_r
	depsls = de * u2_r

	# ------------------
	# PLANETARY NUTATION
	# ------------------

	#  n.b.  The MHB2000 code computes the luni-solar and planetary nutation
	#  in different functions, using slightly different Delaunay
	#  arguments in the two cases.  This behaviour is faithfully
	#  reproduced here.  Use of the IERS 2003 expressions for both
	#  cases leads to negligible changes, well below
	#  0.1 microarcsecond.

	# Mean anomaly of the Moon (MHB2000).
	al = np.mod(2.35555598 + 8328.6914269554 * t, Const.D2PI)

	# Mean longitude of the Moon minus that of the ascending node (MHB2000).
	af = np.mod(1.627905234 + 8433.466158131 * t, Const.D2PI)

	# Mean elongation of the Moon from the Sun (MHB2000)
	ad = np.mod(5.198466741 + 7771.3771468121 * t, Const.D2PI)

	# Mean longitude of the ascending node of the Moon (MHB2000)
	aom = np.mod(2.18243920 - 33.757045 * t, Const.D2PI)

	# General accumulated precession in longitude (IERS 2003)
	apa = (0.024381750 + 0.00000538691 * t) * t

	# Mean longitude of Mercury (IERS Conventions 2003)
	alme = np.mod(4.402608842 + 2608.7903141574 * t, Const.D2PI)

	# Mean longitude of Venus (IERS Conventions 2003)
	alve = np.mod(3.176146697 + 1021.3285546211 * t, Const.D2PI)

	# Mean longitude of Earth (IERS Conventions 2003)
	alea = np.mod(1.753470314 + 628.3075849991 * t, Const.D2PI)

	# Mean longitude of Mars (IERS Conventions 2003)
	alma = np.mod(6.203480913 + 334.0612426700 * t, Const.D2PI)

	# Mean longitude of Jupiter (IERS Conventions 2003)
	alju = np.mod(0.599546497 + 52.9690962641 * t, Const.D2PI)

	# Mean longitude of Saturn (IERS Conventions 2003)
	alsa = np.mod(0.874016757 + 21.3299104960 * t, Const.D2PI)

	# Mean longitude of Uranus (IERS Conventions 2003)
	alur = np.mod(5.481293872 + 7.4781598567 * t, Const.D2PI)

	# Neptune longitude (MHB2000).
	alne = np.mod(5.321159000 + 3.8127774000 * t, Const.D2PI)

	temp_arr = np.array([al, af, ad, aom, alme, alve, alea, alma, alju, alsa, alur, alne, apa]).reshape((1, -1))
	all_args = np.sum(Const.xpl_nut00a[:, :13] * temp_arr, axis=-1)
	all_args = np.mod(all_args, Const.D2PI)
	s_args = np.sin(all_args)
	c_args = np.cos(all_args)

	dp = np.sum(Const.xpl_nut00a[:, 13] * s_args + Const.xpl_nut00a[:, 14] * c_args)
	de = np.sum(Const.xpl_nut00a[:, 15] * s_args + Const.xpl_nut00a[:, 16] * c_args)

	# Convert from 0.1 microarcsec units to radians.
	dpsipl = dp * u2_r
	depspl = de * u2_r

	# -------
	# RESULTS
	# -------

	# Add luni-solar and planetary components.
	dpsi = dpsils + dpsipl
	deps = depsls + depspl

	return dpsi, deps


def iau_s06(date1, date2, x, y):
	fa = np.zeros([8])
	sp = np.array([94.00e-6, 3808.65e-6, -122.68e-6, -72574.11e-6, 27.98e-6, 15.62e-6])

	t = ((date1 - Const.DJ00) + date2) / Const.DJC
	# [l,  l', F,  D, Om,LVe, LE, pA      [   sine      [cosine
	s0 = np.array([  # 1-10

		[0, 0, 0, 0, 1, 0, 0, 0, -2640.73e-6, 0.39e-6],
		[0, 0, 0, 0, 2, 0, 0, 0, -63.53e-6, 0.02e-6],
		[0, 0, 2, -2, 3, 0, 0, 0, -11.75e-6, -0.01e-6],
		[0, 0, 2, -2, 1, 0, 0, 0, -11.21e-6, -0.01e-6],
		[0, 0, 2, -2, 2, 0, 0, 0, 4.57e-6, 0.00e-6],
		[0, 0, 2, 0, 3, 0, 0, 0, -2.02e-6, 0.00e-6],
		[0, 0, 2, 0, 1, 0, 0, 0, -1.98e-6, 0.00e-6],
		[0, 0, 0, 0, 3, 0, 0, 0, 1.72e-6, 0.00e-6],
		[0, 1, 0, 0, 1, 0, 0, 0, 1.41e-6, 0.01e-6],
		[0, 1, 0, 0, -1, 0, 0, 0, 1.26e-6, 0.01e-6],
		[1, 0, 0, 0, -1, 0, 0, 0, 0.63e-6, 0.00e-6],  # 11-20
		[1, 0, 0, 0, 1, 0, 0, 0, 0.63e-6, 0.00e-6],
		[0, 1, 2, -2, 3, 0, 0, 0, -0.46e-6, 0.00e-6],
		[0, 1, 2, -2, 1, 0, 0, 0, -0.45e-6, 0.00e-6],
		[0, 0, 4, -4, 4, 0, 0, 0, -0.36e-6, 0.00e-6],
		[0, 0, 1, -1, 1, -8, 12, 0, 0.24e-6, 0.12e-6],
		[0, 0, 2, 0, 0, 0, 0, 0, -0.32e-6, 0.00e-6],
		[0, 0, 2, 0, 2, 0, 0, 0, -0.28e-6, 0.00e-6],
		[1, 0, 2, 0, 3, 0, 0, 0, -0.27e-6, 0.00e-6],
		[1, 0, 2, 0, 1, 0, 0, 0, -0.26e-6, 0.00e-6],
		[0, 0, 2, -2, 0, 0, 0, 0, 0.21e-6, 0.00e-6],  # 21-30
		[0, 1, -2, 2, -3, 0, 0, 0, -0.19e-6, 0.00e-6],
		[0, 1, -2, 2, -1, 0, 0, 0, -0.18e-6, 0.00e-6],
		[0, 0, 0, 0, 0, 8, -13, -1, 0.10e-6, -0.05e-6],
		[0, 0, 0, 2, 0, 0, 0, 0, -0.15e-6, 0.00e-6],
		[2, 0, -2, 0, -1, 0, 0, 0, 0.14e-6, 0.00e-6],
		[0, 1, 2, -2, 2, 0, 0, 0, 0.14e-6, 0.00e-6],
		[1, 0, 0, -2, 1, 0, 0, 0, -0.14e-6, 0.00e-6],
		[1, 0, 0, -2, -1, 0, 0, 0, -0.14e-6, 0.00e-6],
		[0, 0, 4, -2, 4, 0, 0, 0, -0.13e-6, 0.00e-6],
		[0, 0, 2, -2, 4, 0, 0, 0, 0.11e-6, 0.00e-6],  # 31-33
		[1, 0, -2, 0, -3, 0, 0, 0, -0.11e-6, 0.00e-6],
		[1, 0, -2, 0, -1, 0, 0, 0, -0.11e-6, 0.00e-6]])

	# Terms of order t^1
	s1 = np.array([  # 1-3
		[0, 0, 0, 0, 2, 0, 0, 0, -0.07e-6, 3.57e-6],
		[0, 0, 0, 0, 1, 0, 0, 0, 1.73e-6, -0.03e-6],
		[0, 0, 2, -2, 3, 0, 0, 0, 0.00e-6, 0.48e-6]])

	# Terms of order t^2
	s2 = np.array([  # 1-10
		[0, 0, 0, 0, 1, 0, 0, 0, 743.52e-6, -0.17e-6],
		[0, 0, 2, -2, 2, 0, 0, 0, 56.91e-6, 0.06e-6],
		[0, 0, 2, 0, 2, 0, 0, 0, 9.84e-6, -0.01e-6],
		[0, 0, 0, 0, 2, 0, 0, 0, -8.85e-6, 0.01e-6],
		[0, 1, 0, 0, 0, 0, 0, 0, -6.38e-6, -0.05e-6],
		[1, 0, 0, 0, 0, 0, 0, 0, -3.07e-6, 0.00e-6],
		[0, 1, 2, -2, 2, 0, 0, 0, 2.23e-6, 0.00e-6],
		[0, 0, 2, 0, 1, 0, 0, 0, 1.67e-6, 0.00e-6],
		[1, 0, 2, 0, 2, 0, 0, 0, 1.30e-6, 0.00e-6],
		[0, 1, -2, 2, -2, 0, 0, 0, 0.93e-6, 0.00e-6],
		[1, 0, 0, -2, 0, 0, 0, 0, 0.68e-6, 0.00e-6],  # 11-20
		[0, 0, 2, -2, 1, 0, 0, 0, -0.55e-6, 0.00e-6],
		[1, 0, -2, 0, -2, 0, 0, 0, 0.53e-6, 0.00e-6],
		[0, 0, 0, 2, 0, 0, 0, 0, -0.27e-6, 0.00e-6],
		[1, 0, 0, 0, 1, 0, 0, 0, -0.27e-6, 0.00e-6],
		[1, 0, -2, -2, -2, 0, 0, 0, -0.26e-6, 0.00e-6],
		[1, 0, 0, 0, -1, 0, 0, 0, -0.25e-6, 0.00e-6],
		[1, 0, 2, 0, 1, 0, 0, 0, 0.22e-6, 0.00e-6],
		[2, 0, 0, -2, 0, 0, 0, 0, -0.21e-6, 0.00e-6],
		[2, 0, -2, 0, -1, 0, 0, 0, 0.20e-6, 0.00e-6],
		[0, 0, 2, 2, 2, 0, 0, 0, 0.17e-6, 0.00e-6],  # 21-25
		[2, 0, 2, 0, 2, 0, 0, 0, 0.13e-6, 0.00e-6],
		[2, 0, 0, 0, 0, 0, 0, 0, -0.13e-6, 0.00e-6],
		[1, 0, 2, -2, 2, 0, 0, 0, -0.12e-6, 0.00e-6],
		[0, 0, 2, 0, 0, 0, 0, 0, -0.11e-6, 0.00e-6]])

	# Terms of order t^3
	s3 = np.array([  # 1-4
		[0, 0, 0, 0, 1, 0, 0, 0, 0.30e-6, -23.42e-6],
		[0, 0, 2, -2, 2, 0, 0, 0, -0.03e-6, -1.46e-6],
		[0, 0, 2, 0, 2, 0, 0, 0, -0.01e-6, -0.25e-6],
		[0, 0, 0, 0, 2, 0, 0, 0, 0.00e-6, 0.23e-6]])

	# Terms of order t^4
	s4 = np.array([  # 1-1
		[0, 0, 0, 0, 1, 0, 0, 0, -0.26e-6, -0.01e-6]])

	# mean anomaly of the moon
	fa[0] = iau_fal03(t)

	# mean anomaly of the sun
	fa[1] = iau_falp03(t)

	# mean longitude of the moon minus that of the ascending node
	fa[2] = iau_faf03(t)

	# mean elongation of the moon from the sun
	fa[3] = iau_fad03(t)

	# mean longitude of the ascending node of the moon
	fa[4] = iauFaom03(t)

	# mean longitude of venus
	fa[5] = iau_fave03(t)

	# mean longitude of earth
	fa[6] = iau_fae03(t)

	# general precession in longitude
	fa[7] = iau_fapa03(t)

	w0, w1, w2, w3, w4, w5 = sp

	w0 += map_moments(s0, fa)
	w1 += map_moments(s1, fa)
	w2 += map_moments(s2, fa)
	w3 += map_moments(s3, fa)
	w4 += map_moments(s4, fa)

	return (w0 + (w1 + (w2 + (w3 + (w4 + w5 * t) * t) * t) * t) * t) * Const.DAS2R - x * y / 2.0


def map_moments(s, fa, max_dim=8):
	a = np.sum(s[:, :max_dim] * fa[np.newaxis, :], -1, keepdims=True)
	return np.sum(s[:, max_dim, np.newaxis] * np.sin(a) + s[:, max_dim + 1, np.newaxis] * np.cos(a))


def invjday(jd):
	"""
	Given Julian date, tell me the UTC time

	:param jd:
	:return:
	"""
	z = np.floor(jd + 0.5)
	fday = jd + 0.5 - z

	# lt_idx = fday < 0
	# fday[lt_idx] += 1
	# z[lt_idx] -= 1
	if fday < 0:
		fday += 1
		z -= 1

	if z < 2299161:
		a = copy.copy(z)
	else:
		alpha = np.floor((z - 1867216.25) / 36524.25)
		a = z + 1 + alpha - np.floor(alpha / 4)

	b = a + 1524
	c = np.floor((b - 122.1) / 365.25)
	d = np.floor(365.25 * c)
	e = np.floor((b - d) / 30.6001)
	day = b - d - np.floor(30.6001 * e) + fday

	# elt_idx = e < 14
	# month = copy.copy(e)
	# month[elt_idx] = e[elt_idx] - 1
	# month[~elt_idx] = e[~elt_idx] - 13

	if e < 14:
		month = e - 1
	else:
		month = e - 13

	# cgt_idx = month > 2
	# year = copy.copy(c)
	# year[cgt_idx] = c[cgt_idx] - 4716
	# year[~cgt_idx] = c[~cgt_idx] - 4715

	if month > 2:
		year = c - 4716
	else:
		year = c - 4715

	hr = np.abs(day - np.floor(day)) * 24
	minute = np.abs(hr - np.floor(hr)) * 60
	sec = np.abs(minute - np.floor(minute)) * 60

	return year, month, np.floor(day), np.floor(hr), np.floor(minute), sec


def mjday(year, month, day, hour=0, minute=0, sec=0):
	"""
	Given UTC time, tell me the modified Julian date

	:param year:
	:param month:
	:param day:
	:param hour:
	:param minute:
	:param sec:
	:return:
	"""
	y, m, b, c = year, month, 0, 0
	if m <= 2:
		y -= 1
		# Mjday.m:28
		m += 12
	# Mjday.m:29

	if y < 0:
		c = - 0.75
	# Mjday.m:33

	# check for valid calendar date
	if year < 1582:
		# null
		pass
	else:
		if year > 1582:
			a = np.floor(y / 100)
			# Mjday.m:40
			b = 2 - a + np.floor(a / 4)
		# Mjday.m:41
		else:
			if month < 10:
				# null
				pass
			else:
				if month > 10:
					a = np.floor(y / 100)
					# Mjday.m:45
					b = 2 - a + np.floor(a / 4)
				# Mjday.m:46
				else:
					if day <= 4:
						# null
						pass
					else:
						if day > 14:
							a = np.floor(y / 100)
							# Mjday.m:50
							b = 2 - a + np.floor(a / 4)
						# Mjday.m:51
						else:
							print('\n\n  this is an invalid calendar date!!\n')
							return None

	jd = np.floor(np.dot(365.25, y) + c) + np.floor(np.dot(30.6001, (m + 1)))
	# Mjday.m:57
	jd += day + b + 1720994.5
	# Mjday.m:58
	jd += (hour + minute / 60 + sec / 3600) / 24
	# Mjday.m:59
	return jd - Const.DJM0


def jd2gmst(jd):
	# jd0 = np.nan
	jdmin = np.floor(jd) - 0.5
	jdmax = np.floor(jd) + 0.5

	jd0 = jdmin if jd > jdmin else jdmax

	H = (jd - jd0) * 24
	D = jd - 2451545
	D0 = jd0 - 2451545
	T = D / 36525
	return np.mod(6.697374558 + 0.06570982441908 * D0 + 1.00273790935 * H +
	              0.000026 * (T ** 2), 24) * 15


def ltc_matrix(lat, lon):
	m = iau_ry(-lat, np.eye(3)).dot(iau_rz(lon, np.eye(3)))

	for j in range(3):
		aux = copy.copy(m[0, j])
		m[0, j] = copy.copy(m[1, j])
		m[1, j] = copy.copy(m[2, j])
		m[2, j] = copy.copy(aux)

	return m


def hms2deg(ra=None, dec=None):
	ra_deg, dec_deg = None, None
	if dec:
		D, M, S = dec
		dec_deg = D + (M / 60) + (S / 3600)
		# DEC = '{0}'.format(deg * ds)

	if ra:
		H, M, S = ra
		if str(H)[0] == '-':
			rs, H = -1, abs(H)
		ra_deg = (H * 15) + (M / 4) + (S / 240)
		# RA = '{0}'.format(deg * rs)

	if ra and dec:
		return ra_deg, dec_deg
	else:
		return ra_deg or dec_deg


def j2000_eci(jd: float, eopdata: np.array, jd0: float = None):
	"""
	:param jd: julian date of epoch
	:param eopdata:
	:param jd0: reference julian date
	:return: transformation matrix to J2000ECI
	"""

	jd0 = 2400000.5 if not jd0 else jd0

	Mjd_UTC = copy.copy(jd)
	UT1_UTC, TAI_UTC, x_pole, y_pole, ddpsi, ddeps = iers(eopdata, Mjd_UTC)
	UT1_TAI, UTC_GPS, UT1_GPS, TT_UTC, GPS_UTC = timediff(UT1_UTC, TAI_UTC)

	year, month, day, hour, minute, sec = invjday(Mjd_UTC + jd0)
	DJMJD0, DATE = iau_cal2jd(year, month, day)
	TIME = (60 * (60 * hour + minute) + sec) / Const.DAYSEC
	UTC = DATE + TIME
	TT = UTC + TT_UTC / Const.DAYSEC
	TUT = TIME + UT1_UTC / Const.DAYSEC
	UT1 = DATE + TUT

	# Polar motion matrix (TIRS->ITRS, IERS 2003)
	Pi = iau_pom00(x_pole, y_pole, iau_sp00(DJMJD0, TT))
	# Form bias-precession-nutation matrix
	npb = iau_pnm06a(DJMJD0, TT)
	# Form Earth rotation matrix
	gast = iau_gst06(DJMJD0, UT1, DJMJD0, TT, npb)
	theta = iau_rz(gast, np.eye(3))
	# ICRS to ITRS transformation
	# E = Pi * theta * npb
	return np.linalg.multi_dot([Pi, theta, npb])
