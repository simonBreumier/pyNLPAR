"""Denoise kikuchi patterns given the weights"""
from get_map_weights import get_neig
import numpy as np
import matplotlib.pyplot as plt
import h5py


def denoise_map(map, weights):
    """Denoise kikuchi patterns using the NLPAR method

    :param map: EBSDmap object
    :param weights: weights (w, h, w_size, w_size) array
    :return: nothing but write the pattern in the 'denoised' folder
    """
    w_size = int(0.5*(weights[0, 0].shape[0]-1))
    neigh_span = list(range(-w_size, w_size + 1))

    #if map.is_h5:
        #h5_data = h5py.File(map.data_path, 'r+')

    for i in range(0, map.w):
        for j in range(0, map.h):
            new_pat = np.zeros(map.patterns[0, 0].shape)
            for we in neigh_span:
                for ns in neigh_span:
                    pat_neigh, same = get_neig(map, i, j, we, ns)
                    if not(type(pat_neigh) == float):
                        new_pat += weights[i, j, we + w_size, ns + w_size] * pat_neigh
            pat_name = map.pat_names[j * map.w + i]
            pat_resize = new_pat.reshape(map.k_h, map.k_w)

            if map.is_h5:
                map.patterns[map.pat_h5_path][j * map.w + i] = pat_resize
            else:
                plt.imsave('denoised/'+pat_name, pat_resize, cmap='Greys_r')
    #if map.is_h5:
        #h5_data.close()
