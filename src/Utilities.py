import os
import math

import numpy as np


class Utilities(object):

    @staticmethod
    def check_folder(log_dir):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return log_dir

    @staticmethod
    def rad2deg(rad):
        return rad * 180 / math.pi

    @staticmethod
    def deg2rad(deg):
        return deg * math.pi / 180

    @staticmethod
    def trans(a):
        if len(a.shape) < 3:
            return a.T
        elif len(a.shape) == 3:
            return np.transpose(a, [0, 2, 1])

    @staticmethod
    def mgtz(a, b):
        if a < 0 and b < 0:
            print('Problem at MGTZ\n')
            return -1
        if a < 0:
            return b
        if b < 0:
            return a
        return np.minimum(a, b)

    @staticmethod
    def unit(vin):
        vout = vin * 0
        n = Utilities.norm(vin)
        if n > 0:
            vout = vin / n

        return vout.squeeze()

    @staticmethod
    def alt(x, earth_radius=6378137.):
        return Utilities.norm(x[:3]) - earth_radius

    @staticmethod
    def norm(x):
        x = x.squeeze()
        return math.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2)

    @staticmethod
    def cross(a, b):
        a = a.reshape((1, 3))
        b = b.reshape((1, 3))
        xc = np.zeros((1, 3))
        xc[0, 0] = a[0, 1] * b[0, 2] - a[0, 2] * b[0, 1]
        xc[0, 1] = a[0, 2] * b[0, 0] - a[0, 0] * b[0, 2]
        xc[0, 2] = a[0, 0] * b[0, 1] - a[0, 1] * b[0, 0]

        return xc

    @staticmethod
    def cross_norm(a, b):
        a = np.reshape(a, [1, 3])
        b = np.reshape(b, [1, 3])

        xc = Utilities.cross(a, b)
        return Utilities.unit(xc)

    @staticmethod
    def dot(a, b):
        a = a.reshape((1, 3))
        b = b.reshape((1, 3))
        return np.sum(a * b, axis=-1, keepdims=True)

    @staticmethod
    def sun_direction(day, hour):
        """
        Find a reasonable sun direction vector
        :param day: 0-365.25
        :param hour: 0-24
        :return:
        """
        theta = Utilities.deg2rad(90 + 23.45 * math.cos((day + 10) / 365.25 * 2 * math.pi))
        phi = Utilities.deg2rad((180 - 36 * hour / 24))

        return np.array([math.cos(phi) * math.sin(theta), math.sin(phi) * math.sin(theta), math.cos(theta)])
