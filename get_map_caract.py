"""Compute the EBSD map parameters lambda and sigma"""
import numpy as np
from get_map_weights import get_neig, get_weights
from scipy.optimize import least_squares


def get_lam_val(map):
    """Estimate the EBSDmap lambda value by minimization such that the average map
    weight wii equals 0.375 with SR=1

    :param map: EBSDmap object
    """
    lam_0 = 0.5
    res = least_squares(lam_cost_fun, lam_0, args=(map, 0.375))
    lam = res.x
    return lam[0]


def lam_cost_fun(lam, map, target):
    """Return the difference between wii average with current lam value and the target one

    :param lam: lambda value
    :param map: EBSDmap object
    :param target: wii average target
    """
    map.set_lam(lam)
    weights, wii = get_weights(map, 1)
    return wii - target

def get_sigmas_val(map):
    """Estimate the EBSDmap sigma value

    :param map: EBSDmap object
    """
    sigmas = np.zeros((map.w, map.h))
    n_p = map.patterns[0, 0].shape[0]
    for i in range(0, map.w):
        for j in range(0, map.h):
            d_min = np.nan
            for we in [-1, 0, 1]:
                for ns in [-1, 0, 1]:
                    neig_act, same = get_neig(map, i, j, we, ns)
                    if not(same):
                        d_actu = np.sum((map.patterns[i, j] - neig_act)**2)
                    else:
                        d_actu = np.nan
                    d_min = np.nanmin([d_actu, d_min])
            sigmas[i, j] = np.sqrt(d_min / (2 * n_p))
    return sigmas
