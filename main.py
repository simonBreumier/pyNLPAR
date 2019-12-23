"""Implementation of the NLPAR filter
(Brewick et al., 2019, https://doi.org/10.1016/j.ultramic.2019.02.013) in python"""
from EBSDmap import *
from get_map_caract import get_lam_val, get_sigmas_val
from get_map_weights import get_weights
from denoise_map import denoise_map

#data_path = "C:/Users/Simon/Desktop/transfer_1767438_files_c64c2032/carto_EBSD_238000_238565/carto_EBSD_238000_238565"
data_path = "Ni/EDAX-Ni.h5"
dimensions = (186, 151)
k_dimensions = (60, 60)
first_id = 0
pat_h5_path = 'Scan 1/EBSD/Data/Pattern'

map_1 = EBSDmap(data_path, dimensions, k_dimensions, first_id=first_id, pat_h5_path=pat_h5_path)
sigmas = get_sigmas_val(map_1)
map_1.set_sigmas(sigmas)
lam = get_lam_val(map_1)
map_1.set_lam(lam)
weights, wii = get_weights(map_1, 3)

denoise_map(map_1, weights)