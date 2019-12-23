"""Compute the map weights"""
import numpy as np
import math


def get_weights(map, w_size):
    """Compute the map weights with a given window size

    :param map: EBSDmap object
    :param w_size: window size
    :return weight matrix and average wii over the whole map
    """
    avg_wii = 0.
    weights = np.nan * np.ones((map.w, map.h, w_size * 2 + 1, w_size * 2 + 1))
    neigh_span = list(range(-w_size, w_size+1))
    for i in range(0, map.w):
        for j in range(0, map.h):
            Z = 0
            for we in neigh_span:
                for ns in neigh_span:
                    pat_neigh, same = get_neig(map, i, j, we, ns)
                    if not(type(pat_neigh) == float):
                        sig_act = (map.sigmas[i,j], map.sigmas[i + we, j + ns])
                        dist_act = get_dist(map.patterns[i, j], pat_neigh, sig_act, map.lam)
                        if not np.isnan(dist_act):
                            w_act = math.exp(-(max(dist_act, 0) / map.lam**2))
                            Z += w_act
                        else:
                            w_act = np.nan
                        weights[i, j, we + w_size, ns + w_size] = w_act
            weights[i, j] = weights[i, j] / Z
            if not np.isnan(weights[i, j, w_size, w_size]):
                avg_wii += weights[i, j, w_size, w_size]
    avg_wii /= map.w * map.h
    return weights, avg_wii


def get_dist(ref, neigh, sigmas, lam):
    """Compute the distance between two pattern accounting for noise

    :param ref: reference pattern
    :param neigh: neighbour pattern
    :param sigmas: noise standard deviation (sig_i, sig_j)
    :param lam: lambda distance factor
    :return: distance between two patterns
    """
    n_p = ref.shape[0]
    d = np.sum((ref - neigh) ** 2) - n_p * (sigmas[0] ** 2 + sigmas[1] ** 2)
    if not sigmas[0] ** 2 + sigmas[1] ** 2 == 0:
        d /= math.sqrt(2 * n_p) * (sigmas[0] ** 2 + sigmas[1] ** 2)
    else:
        d = np.nan
    return d


def get_neig(map, i, j, we, ns):
    """Get the actual neighboord of a given pixel. If at the border, returns nan

    :param map: EBSDmap object
    :param i,j: pixel coordinate
    :param we,ns: actual neighbor west-est, north-east relative coordinate
    """
    same = (we == 0) and (ns == 0)
    if i + we < 0 or i + we >= map.w or j + ns < 0 or j + ns >= map.h:
        return np.nan, False
    else:
        return map.patterns[i + we, j + ns], same