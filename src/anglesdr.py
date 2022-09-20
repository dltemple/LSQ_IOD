import copy
import numpy as np


from Constants import Constants
from CoordinateChanges import CoordinateChanges

CC = CoordinateChanges(Constants)


def doubler(cc1, cc2, magrsite1, magrsite2,
            magr1in, magr2in,
            los1, los2, los3,
            rsite1, rsite2, rsite3,
            t1, t3, direct=1):

    """double R integration method for angles only IOD"""
    mu = 398600.4418e9

    rho1 = (-cc1 + np.sqrt(cc1 ** 2. - 4. * (magrsite1 ** 2. - magr1in ** 2.))) / 2.
    rho2 = (-cc2 + np.sqrt(cc2 ** 2. - 4. * (magrsite2 ** 2. - magr2in ** 2.))) / 2.

    r1 = rho1 * los1 + rsite1.reshape((3, 1))
    r2 = rho2 * los2 + rsite2.reshape((3, 1))

    magr1, magr2 = map(np.linalg.norm, [r1, r2])

    if direct == 1:
        w = np.cross(r1.T, r2.T) / (magr1 * magr2)
    else:
        w = -np.cross(r1.T, r2.T) / (magr1 * magr2)

    rho3 = -np.dot(rsite3.T, w.T) / np.dot(los3.T, w.T)
    r3 = rho3 * los3 + rsite3.reshape((3, 1))
    magr3 = np.linalg.norm(r3)

    # cosdv21 = r2.T.dot(r1) / (magr2 * magr1)
    cosdv21 = np.dot(r2.T, r1) / (magr2 * magr1)
    sindv21 = np.sqrt(1. - np.square(cosdv21))
    dv21 = np.arctan2(sindv21, cosdv21)

    cosdv31 = r3.T.dot(r1) / (magr3 * magr1)
    # cosdv31 = np.dot(r3.T, r1) / (magr3 * magr1)
    sindv31 = np.sqrt(1. - np.square(cosdv31))
    dv31 = np.arctan2(sindv31, cosdv31)

    cosdv32 = r3.T.dot(r2) / (magr3 * magr2)
    # cosdv32 = np.dot(r3.T, r2) / (magr3 * magr2)
    sindv32 = np.sqrt(1. - np.square(cosdv32))

    if dv31[0][0] > np.pi:
        c1 = (magr2 * sindv32) / (magr1 * sindv31)
        c3 = (magr2 * sindv21) / (magr3 * sindv31)
        p = (c1 * magr1 + c3 * magr3 - magr2) / (c1 + c3 - 1.)
    else:
        c1 = (magr1 * sindv31) / (magr2 * sindv32)
        c3 = (magr1 * sindv21) / (magr3 * sindv32)
        p = (c3 * magr3 - c1 * magr2 + magr1) / (-c1 + c3 + 1.)

    ecosv1 = p / magr1 - 1.
    ecosv2 = p / magr2 - 1.
    ecosv3 = p / magr3 - 1.

    delta_pi = np.abs(dv21 - np.pi)

    if delta_pi[0][0] > 1e-6:
        esinv2 = (-cosdv21 * ecosv2 + ecosv1) / sindv21
    else:
        esinv2 = (cosdv32 * ecosv3) / sindv31

    e = np.sqrt(np.square(ecosv2) + np.square(esinv2))
    # e = np.sqrt(ecosv2 ** 2 + esinv2 ** 2)
    e2 = np.square(e)
    a = p / (1. - e2)

    if e2[0][0] < 1:
        n = np.sqrt(mu / a ** 3)
        s = magr2 / p * np.sqrt(1. - e2) * esinv2
        c = magr2 / p * (e2 + ecosv2)

        sinde32 = magr3 / np.sqrt(a * p) * sindv32 - magr3 / p * (1. - cosdv32) * s
        cosde32 = 1. - magr2 * magr3 / (a * p) * (1 - cosdv32)
        deltae32 = np.arctan2(sinde32, cosde32)

        sinde21 = magr1 / np.sqrt(a * p) * sindv21 + magr1 / p * (1. - cosdv21) * s
        cosde21 = 1. - magr2 * magr1 / (a * p) * (1 - cosdv21)
        deltae21 = np.arctan2(sinde21, cosde21)

        deltam32 = deltae32 + 2 * s * np.square((np.sin(deltae32 / 2))) - c * np.sin(deltae32)
        deltam12 = -deltae21 + 2 * s * np.square((np.sin(deltae21 / 2))) + c * np.sin(deltae21)
    else:
        n = np.sqrt(mu / -a ** 3)

        s = magr2 / p * np.sqrt(e2 - 1.) * esinv2
        c = magr2 / p * (e2 + ecosv2)

        sindh32 = magr3 / np.sqrt(-a * p) * sindv32 - magr3 / p * (1. - cosdv32) * s
        sindh21 = magr1 / np.sqrt(-a * p) * sindv21 - magr3 / p * (1. - cosdv21) * s

        deltah32 = np.log(sindh32 + np.sqrt(np.square(sindh32) + 1.))
        deltah21 = np.log(sindh21 + np.sqrt(np.square(sindh21) + 1.))

        deltam32 = -deltah32 + 2. * s * np.square((np.sinh(deltah32 / 2.))) + c * np.sinh(deltah32)
        deltam12 = deltah21 + 2. * s * np.square((np.sinh(deltah21 / 2.))) - c * np.sinh(deltah21)
        deltae32 = deltah32

    f1 = t1 - deltam12 / n
    f2 = t3 - deltam32 / n

    q1 = np.sqrt(np.square(f1) + np.square(f2))

    return r2, r3, f1, f2, q1, magr1, magr2, a, deltae32


def anglesdr(alpha1, alpha2, alpha3,
             delta1, delta2, delta3,
             mjd1, mjd2, mjd3,
             rs1, rs2, rs3):

    magr1in = 2.01 * Constants.RE
    magr2in = 2.11 * Constants.RE

    direct = 1

    tol = 1e-10 * Constants.RE
    percent_change = 5e-6

    t1 = (mjd1 - mjd2) * Constants.DAYSEC
    t3 = (mjd3 - mjd2) * Constants.DAYSEC

    los1 = CC.los_vector(alpha1, delta1)
    los2 = CC.los_vector(alpha2, delta2)
    los3 = CC.los_vector(alpha3, delta3)

    lon1, lat1, _ = CC.geodetic_to_lla(rs1)
    lon2, lat2, _ = CC.geodetic_to_lla(rs2)
    lon3, lat3, _ = CC.geodetic_to_lla(rs3)

    los1 = CC.sezl(lat1, lon1, los1)
    los2 = CC.sezl(lat2, lon2, los2)
    los3 = CC.sezl(lat3, lon3, los3)

    magr1old, magr2old = 99999e3, 99999e3
    magRS1, magRS2 = map(np.linalg.norm, [rs1, rs2])

    cc1 = 2 * los1.T.dot(rs1)
    cc2 = 2 * los2.T.dot(rs2)

    loop_counter = 0
    print('Beginning Double R Iteration for Angles Only')
    while (np.abs(magr1in - magr1old) > tol) and (np.abs(magr2in - magr2old) > tol) and loop_counter <= 10:
        loop_counter += 1
        print(
            f'Iteration : {loop_counter:4d}   dMAGR2 {np.abs(magr2in - magr2old):12.12f}    dMAGR1 {np.abs(magr1in - magr1old):12.12f}')

        r2, r3, f1, f2, q1, magr, magr2, a, deltae32 = doubler(cc1, cc2, magRS1, magRS2,
                                                               magr1in, magr2in, los1, los2, los3,
                                                               rs1, rs2, rs3, t1, t3, direct)

        # f = 1 - a / magr2 * (1 - np.cos(deltae32))
        # g = t3 - np.sqrt(a ** 3 / mu) * (deltae32 - np.sin(deltae32))
        # v2 = (r3 - f * r2) / g

        magr1o = copy.copy(magr1in)
        magr1in *= (1 + percent_change)
        deltar1 = percent_change * magr1in
        r2, r3, f1delr1, f2delr1, q2, magr1, magr2, a, deltae32 = doubler(cc1, cc2, magRS1, magRS2, magr1in, magr2in, los1, los2, los3, rs1, rs2, rs3, t1, t3, direct)

        pf1pr1 = (f1delr1 - f1) / deltar1
        pf2pr1 = (f2delr1 - f2) / deltar1

        magr1in = copy.copy(magr1o)
        # deltar1 = percentChange * magr1in
        magr2o = copy.copy(magr2in)
        magr2in *= (1 + percent_change)
        deltar2 = percent_change * magr2in
        r2, r3, f1delr2, f2delr2, q3, magr1, magr2, a, deltae32 = doubler(cc1, cc2, magRS1, magRS2, magr1in, magr2in, los1, los2, los3, rs1, rs2, rs3, t1, t3, direct)

        pf1pr2 = (f1delr2 - f1) / deltar2
        pf2pr2 = (f2delr2 - f2) / deltar2

        magr2in = copy.copy(magr2o)
        # deltar2 = percentChange * magr2in

        delta = pf1pr1 * pf2pr2 - pf2pr1 * pf1pr2
        delta1 = pf2pr2 * f1 - pf1pr2 * f2
        delta2 = pf1pr1 * f2 - pf2pr1 * f1

        deltar1 = -delta1 / delta
        deltar2 = -delta2 / delta

        magr1old = copy.copy(magr1in)
        magr2old = copy.copy(magr2in)

        magr1in += deltar1.squeeze()
        magr2in += deltar2.squeeze()

    print('COMPLETE : dMAGR2 {0:12.12f}    dMAGR1 {1:12.12f}'.format(np.abs(magr2in - magr2old), np.abs(magr1in - magr1old)))

    r2, r3, f1, f2, q1, magr1, magr2, a, deltae32 = doubler(cc1, cc2, magRS1, magRS2, magr1in, magr2in, los1, los2, los3, rs1, rs2, rs3, t1, t3, direct)

    f = 1 - a / magr2 * (1 - np.cos(deltae32))
    g = t3 - np.sqrt(a ** 3 / Constants.mu) * (deltae32 - np.sin(deltae32))
    v2 = (r3 - f * r2) / g

    return r2, v2


if __name__ == '__main__':
    Alpha1 = 1.0559084894933
    Alpha2 = 1.36310214580757
    Alpha3 = 1.97615602688759
    Delta1 = 0.282624656433946
    Delta2 = 0.453434794338875
    Delta3 = 0.586427138011591
    MJD1 = 49746.1101504629
    MJD2 = 49746.1112847221
    MJD3 = 49746.1125347223

    RS1 = np.array([5854667.68296576, 962016.48606668, 2333503.82563616]).reshape([3, 1])
    RS2 = np.array([5847642.79913641, 1003838.17247155, 2333502.11460873]).reshape([3, 1])
    RS3 = np.array([5839555.14329522, 1049867.92205892, 2333500.06992337]).reshape([3, 1])

    r2, v2 = anglesdr(Alpha1, Alpha2, Alpha3, Delta1, Delta2, Delta3, MJD1, MJD2, MJD3, RS1, RS2, RS3)

    true_r2 = np.array([6147270.33904067, 2498134.81242221, 2872843.19752159]).reshape([3, 1])
    true_v2 = np.array([3764.44555317821, -2217.47263084317, -6141.28151668348]).reshape([3, 1])

    pct_r2 = (np.abs(true_r2 - r2) / np.linalg.norm(true_r2)) * 100
    pct_v2 = (np.abs(true_v2 - v2) / np.linalg.norm(true_v2)) * 100

    print(pct_r2)
    print(pct_v2)
