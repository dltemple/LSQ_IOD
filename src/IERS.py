import numpy as np
from Constants import Constants


def iers(eop0, mjd_utc, interp=True):
    """
    This thing is critical.
    It tells you where the poles are earth are pointed very precisely

    :param eop0:
    :param mjd_utc:
    :param interp:
    :return:
    """
    ut1_utc, tai_utc, x_pole, y_pole, dpsi, deps = [None for _ in range(6)]

    mj = np.floor(mjd_utc)
    if interp:
        # linear interpolation

        pre_idx = np.where(eop0[3, :] == mj)[0][0]
        preeop, nexteop = eop0[:, pre_idx], eop0[:, pre_idx+1]

        mfme = 1440 * (mjd_utc - np.floor(mjd_utc))
        fixf = mfme / 1440

        # Setting of IERS Earth rotation parameters
        x_pole = preeop[4] + (nexteop[4] - preeop[4]) * fixf
        y_pole = preeop[5] + (nexteop[5] - preeop[5]) * fixf
        ut1_utc = preeop[6] + (nexteop[6] - preeop[6]) * fixf
        # LOD = preeop[7] + (nexteop[7] - preeop[7]) * fixf
        dpsi = preeop[8] + (nexteop[8] - preeop[8]) * fixf
        deps = preeop[9] + (nexteop[9] - preeop[9]) * fixf
        dx_pole = preeop[10] + (nexteop[10] - preeop[10]) * fixf
        dy_pole = preeop[11] + (nexteop[11] - preeop[11]) * fixf
        tai_utc = preeop[12]

        x_pole /= Constants.DR2AS
        y_pole /= Constants.DR2AS
        dpsi /= Constants.DR2AS
        deps /= Constants.DR2AS
        dx_pole /= Constants.DR2AS
        dy_pole /= Constants.DR2AS

    elif interp == 0:

        eop_idx = list(eop0[3, :]).index(mj)
        eop = eop0[:, eop_idx]

        x_pole = eop[4] / Constants.DR2AS
        y_pole = eop[5] / Constants.DR2AS
        ut1_utc = eop[6]
        # LOD = eop[7]
        dpsi = eop[8] / Constants.DR2AS
        deps = eop[9] / Constants.DR2AS
        # dx_pole = eop[10] / Constants.DR2AS
        # dy_pole = eop[11] / Constants.DR2AS
        tai_utc = eop[12]

    return ut1_utc, tai_utc, x_pole, y_pole, dpsi, deps
