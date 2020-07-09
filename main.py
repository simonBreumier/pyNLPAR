"""Implementation of the NLPAR filter
(Brewick et al., 2019, https://doi.org/10.1016/j.ultramic.2019.02.013) in python"""
from EBSDmap import *
from get_map_caract import get_lam_val, get_sigmas_val
from get_map_weights import get_weights
from denoise_map import denoise_map
import matplotlib.pyplot as plt
import h5py

#data_path = "C:/Users/Simon/Desktop/transfer_1767438_files_c64c2032/carto_EBSD_238000_238565/carto_EBSD_238000_238565"
data_path = "MTL_100.h5"
dimensions = (201, 164) #Map dimension
k_dimensions = (1344, 1024) #Pattern dimensions
first_id = 0
pat_h5_path = 'Scan 1/EBSD/Data/Pattern'

print("## Loading data")
map_1 = EBSDmap(data_path, dimensions, k_dimensions, first_id=first_id, pat_h5_path=pat_h5_path)
print("## Compute sigmas")
sigmas = get_sigmas_val(map_1)
map_1.set_sigmas(sigmas)
print("## Compute lambda")
lam = get_lam_val(map_1)
map_1.set_lam(lam)
print("## Compute weights")
weights, wii = get_weights(map_1, 3)

print("## Denoising patterns...")
denoise_map(map_1, weights)

map_1.patterns.close()