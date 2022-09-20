cimport cython

from libc.math cimport log, sin, cos, acos, asin, sinh, cosh, atan, atan2, sqrt, M_PI, ldexp, fabs, fmod, floor
from libc.stdlib cimport strtod

cdef double EPSILON = 1e-10
cdef double EARTH_MU = 398600.5
cdef double PI_OVER_TWO = M_PI / 2.0
cdef double TWO_PI = 2.0 * M_PI
cdef double EARTH_OMEGA = 7.2921159e-5

cdef double SPDOTP = 86400.0 / TWO_PI # sec per day / two pi, converts from rad/s to rev/day
cdef double DTR = M_PI / 180.0
cdef double RTD = 180.0 / M_PI
cdef double D2PI = 2.0 * M_PI

cdef double DAS2R = 4.84813681109536e-06
cdef double TURNAS = 1296000.0

cdef inline double _dtr(double deg):
    return deg * DTR

cdef inline double _rtd(double rad):
    return rad * RTD

# wgs-84 constants
cdef double EARTH_EARTH_ECC_SQR = 0.081819190842622*0.081819190842622
cdef double EARTH_EQUATOR_RADIUS = 6378.1374

#cdef double iauAnp(double a):
#    cdef double w = a % D2PI
#    if w < 0:
#        w = w + D2PI
#    return w
#
#cdef double iauFad03(double t):
#    val = (1072260.703692 +
#               t * (1602961601.2090 +
#                    t * (- 6.3706 +
#                         t * (0.006593 +
#                              t * (- 0.00003169))))
#    val = val % TURNAS
#    val = val * DAS2R
#    return val
#
#cdef double iauFae03(double t):
#    val = 1.753470314 + 628.3075849991 * t
#    val = val % D2PI
#    return val
#
#cdef double iauFaf03(double t):
#    val = (335779.526232 +
#               t * (1739527262.8478 +
#                    t * (- 12.7512 +
#                         t * (- 0.001037 +
#                              t * (0.00000417))))
#    val = val % TURNAS
#    val = val * DAS2R
#    return val
#
#cdef double iauFal03(double t):
#        val = (485868.249036 +
#                   t * (1717915923.2178 +
#                        t * (31.8792 +
#                             t * (0.051635 +
#                                  t * (- 0.00024470)))))
#
#        val = val % TURNAS
#        val = val * DAS2R
#        return val
#
#cdef double iauFalp03(double t):
#    val = 1287104.793048 +
#               t * (129596581.0481 +
#                    t * (- 0.5532 +
#                         t * (0.000136 +
#                              t * (- 0.00001149)))))
#
#    val = val % TURNAS
#    val = val * DAS2R
#    return val
#
#
#cdef double iauFaom03(double t):
#    val = (450160.398036 +
#               t * (- 6962890.5431 +
#                    t * (7.4722 +
#                         t * (0.007702 +
#                              t * (-0.00005939)))))
#    val = val % TURNAS
#    val = val * DAS2R
#    return val
#
#cdef double iauFapa03(double t):
#    return (0.024381750 + 0.00000538691 * t) * t
#
#cdef double iauFave03(double t):
#    val = 3.176146697 + 1021.3285546211 * t
#    val = val % D2PI
#    return val

cpdef object invjday(double jd):

        cdef double day = 0
        cdef double month = 0
        cdef double year = 0
        cdef double hr = 0
        cdef double minute = 0
        cdef double second = 0
        cdef double a = 0

        cdef double z = floor(jd + 0.5)
        cdef double fday = jd + 0.5 - z

        if fday < 0:
            fday = fday + 1
            z = z - 1

        if z < 2299161:
            a = z
        else:
            alpha = floor((z - 1867216.25) / 36524.25)
            a = z + 1 + alpha - floor(alpha / 4)

        b = a + 1524
        c = floor((b - 122.1) / 365.25)
        d = floor(365.25 * c)
        e = floor((b - d) / 30.6001)
        day = b - d - floor(30.6001 * e) + fday

        if e < 14:
            month = e - 1
        else:
            month = e - 13

        if month > 2:
            year = c - 4716
        else:
            year = c - 4715

        hr = fabs(day - floor(day)) * 24
        minute = fabs(hr - floor(hr)) * 60
        sec = fabs(minute - floor(minute)) * 60

        day = floor(day)
        hr = floor(hr)
        minute = floor(minute)

        return (year, month, day, hr, minute, sec)


cpdef object iauCal2jd(double year, double month, double day):
        cdef double djm0 = 2400000.5

        cdef double a = 0
        cdef double b = 0
        cdef double c = 0
        cdef double jd = 0
        cdef double djm = 0

        if month <= 2:
            year -= 1
            month = month + 12

        if year < 0:
            c = -0.75

        if year > 1582:
            a = floor(year / 100)
            b = 2 - a + floor(a / 4)
        elif month > 10:
            a = floor(year / 100)
            b = 2 - a + floor(a / 4)
        elif day > 14:
            a = floor(year / 100)
            b = 2 - a + floor(a / 4)

        jd = floor(365.25 * year + c) + floor(30.6001 * (month + 1))
        jd = jd + day + b + 1720994.5
        djm = jd - djm0

        return (djm0, djm)

cpdef JD2GMST(jd):
    cdef double jdmin = floor(jd) - 0.5
    cdef double jdmax = floor(jd) + 0.5

    jd0 = jdmin if jd > jdmin else jdmax

    H = (jd - jd0) * 24
    D = jd - 2451545
    D0 = jd0 - 2451545
    T = D / 36525
    val = 6.697374558 + 0.06570982441908 * D0 + 1.00273790935 * H + 0.000026 * (T ** 2)

    val = val % 24

    val = val * 15

    #gmst = np.mod(6.697374558 + 0.06570982441908 * D0 + 1.00273790935 * H +
    #              0.000026 * (T ** 2), 24) * 15

    return val