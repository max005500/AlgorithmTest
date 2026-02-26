import numpy as np

def kol_strucfunc(r,lamda, r0=1.0):
    # 6.88*(r/r0)^(5/3)
    return 6.88 * (r / r0) ** (5.0 / 3.0)

def slopecov_kol(nsubx: int,  #<-- sub-aperture number
                 d: float,
                 lam: float,  #<-- lambda 
                 nsamp = 8,   #<-- nsamp
                 scalingFactor = 1,  #<-- scaling factor
                 r0 = 1 ):
    """
    slodar_slopecovKol2(nsubx, nsamp, d) wrapper for Python.
    return cov con shape: (2, nn, nn) where nn = 2*nsubx-1 cov matrix for x-slope and y-slope .

    cov[0, di, dj] -> cov_xx  
    cov[1, di, dj] -> cov_yy 
    """
    scaling = scalingFactor * (206265.0)**2 * 3.0 * (lam / (np.pi * d)) ** 2

    nn = 2 * nsubx - 1
    cov = np.zeros((2, nn, nn), dtype=np.float64)

    # from C: sub-aperture coords in [-0.5,0.5)  offset of 0.5/nsamp
    rxy = (np.arange(nsamp) - (nsamp / 2) + 0.5) / nsamp
    tilt = 2.0 * np.sqrt(3.0) * rxy  # tilt[i] = 2*sqrt(3)*rxy[i]

    # from C (hard-wired lam=500nm and r0=d) now on radians
    n2 = nsamp * nsamp
    n4 = n2 * n2
 
    
    # Integral method for slopes weighting functions

    for i in range(nn):
        for j in range(nn):
            ra_intgrl = np.zeros((nsamp, nsamp), dtype=np.float64)
            rb_intgrl = np.zeros((nsamp, nsamp), dtype=np.float64)
            D_phi = np.zeros((nsamp, nsamp, nsamp, nsamp), dtype=np.float64)

            dbl_intgrl = 0.0

            # 1) build D_phi and acumulation of partial integrals (ra, rb) + total integration 
            for ia in range(nsamp):
                for ja in range(nsamp):
                    for ib in range(nsamp):
                        for jb in range(nsamp):
                            x = (i - nsubx + 1) - rxy[ia] + rxy[ib]
                            y = (j - nsubx + 1) - rxy[ja] + rxy[jb]
                            r = np.sqrt(x * x + y * y)
                            val = kol_strucfunc(r,lam, r0)
                            D_phi[ia, ja, ib, jb] = val

                            ra_intgrl[ib, jb] += val
                            rb_intgrl[ia, ja] += val
                            dbl_intgrl += val

            # 2) build phiphi and acumulation of tilt covariance on x and y
            xtiltcov = 0.0
            ytiltcov = 0.0
            mean_dbl = dbl_intgrl / (nsamp ** 4)

            for ia in range(nsamp):
                for ja in range(nsamp):
                    for ib in range(nsamp):
                        for jb in range(nsamp):

                            # phiphi = 0.5*((ra+rb)/n2) - 0.5*D_phi - 0.5*(dbl/nsamp^4)
                            phiphi = 0.5 * ((ra_intgrl[ib, jb] + rb_intgrl[ia, ja]) / n2)
                            phiphi -= 0.5 * D_phi[ia, ja, ib, jb]
                            phiphi -= 0.5 * mean_dbl

                            xtiltcov += phiphi * tilt[ia] * tilt[ib] * scaling
                            ytiltcov += phiphi * tilt[ja] * tilt[jb] * scaling

            cov[1, i, j] =  xtiltcov / n4   # X-slope cov
            cov[0, i, j] =  ytiltcov / n4   # y-slope cov

    return cov

