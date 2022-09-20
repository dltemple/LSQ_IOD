import copy
import numpy as np

from scipy.integrate import odeint


def sign_(a: float, b: float):
    """
    Returns the absolute alue of A with sign of B
    :param a:
    :param b:
    :return:
    """
    return np.abs(a) if b >= 0 else -np.abs(a)


def DEInteg(func, t, tout, relerr, abserr, n_eqn, y, AuxParam):
    eps = np.finfo(float).eps
    twou = 2 * eps
    fouru = 4 * eps

    class DE_STATE(object):
        DE_INIT = 1
        DE_DONE = 2
        DE_BADACC = 3
        DE_NUMSTEPS = 4
        DE_STIFF = 5
        DE_INVPARAM = 6

    State_ = DE_STATE.DE_INIT
    PermitTOUT = True
    told = 0

    # powers of two
    two = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0,
                    256.0, 512.0, 1024.0, 2048.0, 4096.0, 8192.0])

    gstr = np.array([1.0, 0.5, 0.0833, 0.0417, 0.0264, 0.0188,
                     0.0143, 0.0114, 0.00936, 0.00789, 0.00679,
                     0.00592, 0.00524, 0.00468])

    yy = np.zeros([n_eqn, 1])
    wt = np.zeros([n_eqn, 1])
    p = np.zeros([n_eqn, 1])
    yp = np.zeros([n_eqn, 1])
    phi = np.zeros([n_eqn, 17])
    g = np.zeros([14, 1])
    sig = np.zeros([14, 1])
    rho = np.zeros([14, 1])
    w = np.zeros([13, 1])
    alpha = np.zeros([13, 1])
    beta = np.zeros([13, 1])
    v = np.zeros([13, 1])
    psi_ = np.zeros([13, 1])

    if t == tout:
        return

    epsilon = np.maximum(relerr, abserr)

    if ((relerr < 0.0) or
            (abserr < 0.0) or
            (epsilon <= 0.0) or
            (State_ > DE_STATE.DE_INVPARAM) or
            ((State_ != DE_STATE.DE_INIT) and (t != told))):
        State_ = DE_STATE.DE_INVPARAM
        return

    # on each call, set interval of integration and counter for
    # number of steps. Adjust input error tolereances to define
    # weight vector for subroutine step

    delta = tout - t
    absdelta = abs(delta)

    tend = t + 100.0 * delta
    if not PermitTOUT:
        tend = tout

    nostep = 0
    kle4 = 0
    stiff = False
    releps = relerr / epsilon
    abseps = abserr / epsilon
    # deltaSgn = 1.0

    if (State_ == DE_STATE.DE_INIT) or (not oldPermit) or (deltaSgn * delta <= 0.0):
        # on start and restart also set the work variables x and yy(*),
        # store the direction of integration and initialize the step size
        start = True
        x = copy.copy(t)
        yy = copy.copy(y)
        deltaSgn = sign_(1.0, delta)
        h = sign_(np.maximum(fouru * np.abs(x), np.abs(tout - x)), tout - x)

    while True:

        # If already past output point, interpolate solution and return
        if abs(x - t) >= absdelta:
            yout = np.zeros([n_eqn, 1])
            ypout = np.zeros([n_eqn, 1])
            g[1] = 1.0
            rho[1] = 1.0
            hi = tout - x
            ki = kold + 1

            # initialize w[*] for computing g[*]
            for ii in range(1, ki + 1):
                w[ii] = 1.0 / ii

            # Compute g[*]
            term = 0.0
            for jj in range(2, ki+1):
                psijm1 = psi_[jj-1]
                gamma = (hi + term) / psijm1
                eta = hi / psijm1
                for ii in range(1, ki + 1 - jj + 1):
                    w[ii] = gamma * w[ii] - eta * w[ii + 1]

                g[jj] = w[1]
                rho[jj] = gamma * rho[jj - 1]
                term = psijm1

            # interpolate for the solution yout and for
            # the derivative of the solution ypout
            for jj in range(1, ki+1):
                ii = ki + 1 - jj
                yout = yout + g[ii] * phi[:, ii, np.newaxis]
                ypout = ypout + rho[ii] * phi[:, ii, np.newaxis]

            yout = y + hi * yout
            y = copy.copy(yout)
            State_ = DE_STATE.DE_DONE
            t = copy.copy(tout)
            told = copy.copy(t)
            OldPermit = PermitTOUT
            return yout

        # If cannot go past output point and sufficiently close,
        # Extrapolate and return
        if not PermitTOUT and (np.abs(tout - x) < fouru * np.abs(x)):
            h = tout - x
            yp = func(yy, x)
            y = yy + h * yp
            State_ = DE_STATE.DE_DONE
            t = copy.copy(tout)
            told = copy.copy(t)
            OldPermit = PermitTOUT
            return y

        # Limit step size, set weight vector and take a step
        h = sign_(np.minimum(abs(h), abs(tend - x)), h)
        wt = releps * np.abs(yy) + abseps
        # for ll in range(n_eqn):
        #     wt[ll] = releps * np.abs(yy[ll]) + abseps

        # Step
        #
        # Begin Block 0
        #
        # Check if step size or error tolerance is too small for machine
        # precision. If first step ,init phi array and estiamte a
        # starting step size. If step size is too small, determine
        # an acceptable one

        if np.abs(h) < fouru * np.abs(x):
            h = sign_(fouru * np.abs(x), h)
            crash = True
            return y

        p5eps = 0.5 * epsilon
        crash = False
        g[1] = 1.0
        g[2] = 0.5
        sig[1] = 1.0

        ifail = 0

        # if error tolerance is too small, increase
        # it to an acceptable value

        roundv = 0.0
        roundv += np.sum((y * y) / (wt * wt))
        # for ll in range(n_eqn):
        #     roundv += (y[ll] * y[ll]) / (wt[ll] * wt[ll])

        roundv = twou * np.sqrt(roundv)

        if p5eps < roundv:
            epsilon = 2.0 * roundv * (10 + fouru)
            crash = True
            return y

        if start:
            # initialize. compute appropriate step size for first step
            yp = func(y, x, AuxParam)
            phi[:, 1] = yp
            phi[:, 2] *= 0.0
            p1 = yp * yp
            p2 = wt * wt

            sumv = np.sum(np.divide(p1.squeeze(), p2.squeeze()))
            # for ll in range(n_eqn):
            #     phi[ll, 1] = copy.copy(yp[ll])
            #     phi[ll, 2] = 0.0
            #     sumv += (yp[ll] * yp[ll]) / (wt[ll] * wt[ll])

            sumv = np.sqrt(sumv)
            absh = np.abs(h)
            if epsilon < 16.0 * sumv * h * h:
                absh = 0.25 * np.sqrt(epsilon / sumv)

            h = sign_(np.maximum(absh, fouru * np.abs(x)), h)
            hold = 0.0
            hnew = 0.0
            k = 1
            kold = 0
            start = False
            phase1 = True
            nornd = True

            if p5eps <= 100.0 * roundv:
                nornd = False
                phi[:, 16] *= 0.0
                # for ll in range(n_eqn):
                #     phi[ll, 16] = 0.0

        #
        # End block 0
        #

        #
        # Repeat blocks 1, 2 (and 3) until step is successful

        while True:
            # print('Beginning Block 1')
            # Begin block 1
            #
            # compute coefficients of formulas for this step.
            # Avoid computing those quantites not changed when
            # step size is not changed
            #

            kp1, kp2, km1, km2 = k+1, k+2, k-1, k-2

            # ns is the number of steps taken with size h
            # including the current one. when k < ns, no coefficients change

            if h != hold:
                ns = 0

            if ns <= kold:
                ns += 1

            nsp1 = ns + 1

            if k >= ns:
                # compute those components of alpha[*]
                # beta[*], psi[*], sig[*], which are changed
                beta[ns] = 1.0
                realns = copy.copy(ns)
                alpha[ns] = 1.0 / realns
                temp1 = h * realns
                sig[nsp1] = 1.0

                if k >= nsp1:
                    for i in range(nsp1, k + 1):
                        im1 = i - 1
                        temp2 = copy.copy(psi_[im1])
                        psi_[im1] = copy.copy(temp1)
                        beta[i] = beta[im1] * psi_[im1] / temp2
                        temp1 = temp2 + h
                        alpha[i] = h / temp1
                        reali = copy.copy(i)
                        sig[i + 1] = reali * alpha[i] * sig[i]

                psi_[k] = copy.copy(temp1)

                # compute coefficients g[*];
                # intiialize v[*] and set w[*]
                if ns > 1:
                    # if order was raised, update diagonal part of v[*]
                    if k > kold:
                        temp4 = k * kp1
                        v[k] = 1.0 / temp4
                        nsm2 = ns - 2
                        for j in range(1, nsm2+1):
                            i = k - j
                            v[i] = v[i] + alpha[j + 1] * v[i + 1]

                    # Update V[*] and set W[*]
                    limit1 = kp1 - ns
                    temp5 = alpha[ns + 1]
                    for iq in range(1, limit1):
                        v[iq] = v[iq] - temp5 * v[iq + 1]
                        w[iq] = v[iq]

                    g[nsp1 + 1] = w[1]
                else:
                    for iq in range(1, k + 1):
                        temp3 = iq * (iq + 1)
                        v[iq] = 1.0 / temp3
                        w[iq] = copy.copy(v[iq])

                # compute the g[*] in the work vector w[*]
                nsp2 = ns + 2
                if kp1 >= nsp2:
                    for i in range(nsp2, kp1 + 1):
                        limit2 = kp2 - i
                        temp6 = alpha[i - 1]
                        for iq in range(1, limit2 + 1):
                            w[iq] = w[iq] - temp6 * w[iq + 1]

                        g[i] = copy.copy(w[1])

            # end if k > ns

            # end block 1

            # begin Block 2
            # print('Beginning Block 2')

            # Predict a solution p[*], evaluate derivatives using predicted
            # Solution, estimate local error at order k and errors at orders
            # k, k-1, k-2, as if constant step size were used

            if k >= nsp1:
                for i in range(nsp1, k + 1):
                    temp1 = beta[i]
                    phi[:, i] *= temp1
                    # for ll in range(n_eqn):
                    #     phi[ll, i] = temp1 * phi[ll, i]

            # Predict solution and differences
            phi[:, kp2] = copy.copy(phi[:, kp1])
            phi[:, kp1] *= 0.0
            p *= 0
            # for ll in range(n_eqn):
            #     phi[ll, kp2] = copy.copy(phi[ll, kp1])
            #     phi[ll, kp1] = 0.0
            #     p[ll] = 0.0

            for j in range(1, k+1):
                i = kp1 - j
                ip1 = i + 1
                temp2 = g[i].squeeze()
                p[:, 0] += temp2 * phi[:, i]
                phi[:, i] += phi[:, ip1]
                # for ll in range(n_eqn):
                #     p[ll] += temp2 * phi[ll, i]
                #     phi[ll, i] = phi[ll, i] + phi[ll, ip1]

            if nornd:
                p = y + h * p
            else:
                tau = h * p - phi[:, 15]
                p = y + tau
                phi[:, 16] = (p - y) * tau
                # for ll in range(n_eqn):
                #     tau = h * p[ll] - phi[ll, 15]
                #     p[ll] = y[ll] + tau
                #     phi[ll, 16] = (p[ll] - y[ll]) - tau

            xold = copy.copy(x)
            x += h
            # print('X = {0:1.4e} h = {1:1.4e}'.format(x[0], h[0]))
            absh = abs(h)
            yp = func(p, x, AuxParam)

            # estimate errors at orders k, k-1, k-2
            erkm2, erkm1, erk = 0.0, 0.0, 0.0

            temp3 = 1.0 / wt
            temp4 = yp[:, np.newaxis] - phi[:, 1, np.newaxis]

            if km2 > 0:
                erkm2 += np.sum(((phi[:, km1, np.newaxis] + temp4) * temp3) * (((phi[:, km1, np.newaxis]) + temp4) * temp3))

            if km2 >= 0:
                erkm1 += np.sum(((phi[:, k, np.newaxis] + temp4) * temp3) * (((phi[:, k, np.newaxis]) + temp4) * temp3))

            erk += np.sum((temp4*temp3)*(temp4*temp3))

            if km2 > 0:
                erkm2 = absh * sig[km1] * gstr[km2] * np.sqrt(erkm2)

            if km2 >= 0:
                erkm1 = absh * sig[k] * gstr[km1] * np.sqrt(erkm1)

            temp5 = absh * np.sqrt(erk)
            err = temp5 * (g[k] - g[kp1])
            erk = temp5 * sig[kp1] * gstr[k]
            knew = copy.copy(k)

            # Test if order should be lowered
            if km2 > 0:
                if np.maximum(erkm1, erkm2) <= erk:
                    knew = copy.copy(km1)

            if km2 == 0:
                if erkm1 <= 0.5 * erk:
                    knew = copy.copy(km1)

            # end block 2

            #
            # if step is successful, continue with block 4, otherwise repeat
            # blocks 1 and 2 after executing block 3

            success = err <= epsilon
            success = success[0]

            if not success:
                # print('Beginning Block 3')
                #
                # begin block 3
                #

                # The step is unsuccessful, restore x, phi[*,*], psi[*].
                # if 3rd consequtive failure, set rder to 1. If step fails,
                # more than 3 times consider an optimal step size. Double
                # error tolerance and return if estimated step size is too small
                # for machine precision
                #

                phase1 = False
                x = copy.copy(xold)
                for i in range(1, k+1):
                    temp1 = 1.0 / beta[i]
                    ip1 = i + 1
                    phi[:, i] = temp1 * (phi[:, i] - phi[:, ip1])

                    # for ll in range(n_eqn):
                    #     phi[ll, i] = temp1 * (phi[ll, i] - phi[ll, ip1])

                if k >= 2:
                    for i in range(2, k+1):
                        psi_[i - 1] = psi_[i] - h

                # On third failure, set order to one
                # Thereafter, use optimal step size

                ifail += 1
                temp2 = 0.5
                if ifail > 3:
                    if p5eps < 0.25 * erk:
                        temp2 = np.sqrt(p5eps / erk)

                if ifail >= 3:
                    knew = 1

                h = temp2 * h
                k = copy.copy(knew)
                if np.abs(h) < fouru * np.abs(x):
                    crash = True
                    h = sign_(fouru * np.abs(x), h)
                    epsilon = epsilon * 2.0
                    return y

                # end block 3
            # end if success

            if success:
                break

        # print('Beginning Block 4')
        #
        # Begin block 4
        #
        # The step is successful. Correct the predicted solution, evaluate
        # the derivatives using the corrected solution and update the
        # difference. determine the best order and step size for next step
        #

        kold = copy.copy(k)
        hold = copy.copy(h)

        # Correct and evaluate
        temp1 = h * g[kp1]
        if nornd:
            y = p + temp1 * (yp[:, np.newaxis] - phi[:, 1, np.newaxis])

            # for ll in range(n_eqn):
            #     y[ll] = p[ll] + temp1 * (yp[ll] - phi[ll, 1])
        else:
            for ll in range(n_eqn):
                rho = temp1 * (yp[ll] - phi[ll, 1]) - phi[ll, 16]
                y[ll] = p[ll] + rho
                phi[ll, 15] = (y[ll] - p[ll]) - rho

        yp = func(y, x, AuxParam)

        # Update Differences for ext step
        phi[:, kp1] = yp - phi[:, 1]
        phi[:, kp2] = phi[:, kp1] - phi[:, kp2]
        # for ll in range(n_eqn):
        #     phi[ll, kp1] = yp[ll] - phi[ll, 1]
        #     phi[ll, kp2] = phi[ll, kp1] - phi[ll, kp2]

        for i in range(1, k+1):
            for ll in range(n_eqn):
                phi[ll, i] = phi[ll, i] + phi[ll, kp1]

        # Estimate error at order k+1 unless
        # - in first phase when always raise order
        # - already deided to lower order
        # - step size not constant so estimate unreliable

        erkp1 = 0.0
        if knew == km1 or k == 12:
            phase1 = False

        if phase1:
            k = copy.copy(kp1)
            erk = copy.copy(erkp1)
        else:
            if knew == km1:
                k = copy.copy(km1)
                erk = copy.copy(erkm1)
            else:
                if kp1 <= ns:
                    erkp1 += np.sum((phi[:, kp2, np.newaxis] / wt) * (phi[:, kp2, np.newaxis] / wt))

                    # for ll in range(n_eqn):
                    #     erkp1 += (phi[ll, kp2] / wt[ll]) * (phi[ll, kp2] / wt[ll])
                    erkp1 = absh * gstr[kp1] * np.sqrt(erkp1)

                    # Using estimated error at order k + 1,
                    # determine appropriate order for next step

                    if k > 1:
                        if erkm1 <= np.minimum(erk, erkp1):
                            # lower order
                            # print('Lowering Order')
                            k = copy.copy(km1)
                            erk = copy.copy(erkm1)
                        else:
                            if erkp1 < erk and k != 12:
                                # raise order
                                k = copy.copy(kp1)
                                erk = copy.copy(erkp1)
                                # print('Raising Order')
                    elif erkp1 < 0.5 * erk:
                        # raise order
                        # here ekp1 < erk < max(erkm1, erkm2) else
                        # order would have been lowered in block 2
                        # thus order is to be raised
                        k = copy.copy(kp1)
                        erk = copy.copy(erkp1)
                    # end
                # end if kp1 <= ns
            # end if knew != km1
        # end if ~phase1

        # with new order determine appropriate step size for next step
        if phase1 or (p5eps >= erk * two[k + 1]):
            hnew = 2.0 * h
        else:
            if p5eps < erk:
                temp2 = k + 1
                r = p5eps / (erk ** (1.0 / temp2))
                hnew = absh * np.maximum(0.5, np.minimum(0.9, r))
                hnew = sign_(np.maximum(hnew, fouru * np.abs(x)), h)
            else:
                hnew = copy.copy(h)

        # print('H -- > {0:1.6e}  HNEW --> {1:1.6e}'.format(h[0], hnew[0]))
        h = copy.copy(hnew)

        #
        # End block 4
        #

        # Test for too small tolerances

        if crash:
            State_ = DE_STATE.DE_BADACC
            relerr = epsilon * releps  # modify relative and absolute
            abserr = epsilon * abseps  # accuracy requirements
            y = copy.copy(yy)
            t = copy.copy(x)
            told = copy.copy(t)
            OldPermit = True
            return y

        nostep += 1

        # Count number of consecutive steps taken with the order of
        # the metod being less or equal to four and test for stiffness

        kle4 += 1
        # print("KLE :: {0:4d}    K :: {1:4d} NS :: {2:4d}".format(kle4, k, ns))

        kle4 = 0 if kold > 4 else kle4
        stiff = True if kle4 >= 50 else stiff


if __name__ == '__main__':

    from Accelerations import acceleration_iers, variable_equations, acceleration_harmoic, gravity_gradient

    # Load all data needed to run tests
    Cnm = np.zeros([71, 71])
    Snm = np.zeros([71, 71])
    with open('/LSQ_IOD/ITG-Grace03s.txt', 'r') as fid:
        for nn in range(1, 72):
            for mm in range(nn):
                # print('N : ' + str(nn-1) + ' M : ' + str(mm))
                temp = fid.readline()
                ps = temp.split(' ')
                ps2 = [p for p in ps if p != '']
                ps2.pop(-1)
                Cnm[nn - 1, mm] = float(ps2[2])
                Snm[nn - 1, mm] = float(ps2[3])

    eopdata = np.loadtxt('C:/TrackFilter-EOIR/LSQ_IOD/eop19622021.txt')
    eopdata = eopdata.T


    class AuxParam(object):
        Cnm = None
        Snm = None
        PC = None
        eopdata = None
        Mjd_UTC = None
        n = 10
        m = 10
        sun = False
        moon = False
        n_a = 10
        m_a = 10
        n_G = 10
        m_G = 10


    AuxParam.Cnm = Cnm
    AuxParam.Snm = Snm
    AuxParam.Mjd_UTC = 58902.0454083448
    AuxParam.eopdata = eopdata

    yPhi = np.zeros([42, 1])
    Phi = np.zeros([6, 6])

    Y0 = np.array([1740735.78436453,
                   6014264.50367203,
                   4427908.81394812,
                   -2965.07461894968,
                   4694.2185049428,
                   -5233.62310402332])

    t = 83.2809805870056

    # Test Accel Harmonics
    r = np.array([1336239.50545621,
                  6587021.79318692,
                  3703412.23573103])
    U = np.array([[-0.980607647160568, 0.195972190026169, 0.00188230352570287],
                  [-0.195971758558225, -0.980609453068139, 0.000412796563495542],
                  [0.00192670127743121, 3.59131348130181e-05, 0.999998143264494]])

    a = acceleration_harmoic(r, U, AuxParam)

    answer1 = np.array([-1.17836450218705,
                        -5.80874917579615,
                        -3.27321451291549])
    # Test Gradient
    G = gravity_gradient(r, U, AuxParam)

    answer_gradient = np.array([[-8.01736485422921e-07, 3.95006533926434e-07, 2.22921469283932e-07],
                                [3.95006534148479e-07, 1.06538705502146e-06, 1.09890571398097e-06],
                                [2.22921469283932e-07, 1.09890571353688e-06, -2.6365057204103e-07]])

    # Test DEIntegration
    Y0 = np.array([1849639.3415658977,
                   6118274.00112913,
                   4511898.170417819,
                   -3292.1565258809496,
                   5051.685437157924,
                   -5794.240908013929])

    for iii in range(6):
        yPhi[iii] = Y0[iii]
        for jj in range(6):
            if iii == jj:
                yPhi[6 * (jj + 1) + iii] = 1
            else:
                yPhi[6 * (jj + 1) + iii] = 0

    # YPHI INTEGRATION
    # yPhi_out = DEInteg(VarEqn, 0, 22.5934621179476, 1e-13, 1e-6, 42, yPhi, AuxParam)

    yPhi = odeint(variable_equations, yPhi.squeeze(), np.array([0, 22.5934621179476]), args=(AuxParam,), rtol=1e-13, atol=1e-6)
    yPhi_out = np.reshape(yPhi[-1, :], [-1, 1])

    for jj in range(6):
        Phi[:, jj] = yPhi_out[6 * (jj + 1):6 * (jj + 1) + 6].squeeze()

    answer_phi = np.array([[0.99982171806219,      0.000117109407990297,      8.53129580384428e-05,          22.6251135981782,      0.000880251471015364,      0.000636143396564503],
                           [0.000117109705242253, 1.00018213359747, 0.000287796259946988, 0.000880252588473535, 22.6278548124391, 0.00216738074139458],
                           [8.53129999848997e-05, 0.000287795671684978, 0.999996193777907, 0.000636143554106563, 0.00216737852930911, 22.6264180645546],
                           [-1.58026641149378e-05, 1.03162451451377e-05, 7.45586189820948e-06, 0.99982070599244, 0.000116302859833916, 8.3380897183893e-05],
                           [1.03163107342369e-05, 1.63247096898688e-05, 2.5401273929134e-05, 0.000116303156683301, 1.00018719340854, 0.000286923412313093],
                           [7.45587110775469e-06, 2.54011439439812e-05, -5.14011529674779e-07, 8.33809388152967e-05, 0.00028692282387452, 0.999992146060742]])
    Phi
