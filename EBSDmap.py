"""EBSDmap object containing an EBSD map coordinates and kikuchi patterns"""
import numpy as np
import cv2
import os
import math


class EBSDmap:
    def __init__(self, data_path, dimensions, k_dimensions, separator='_', position=1, first_id=0):
        """Load a set of Kikuchi pattern as an EBSD_map object

        :param data_path: path of the kikuchi patern to be loaded
        :param dimensions: (w,h) doublet indicating the map pixel width and height
        :param k_dimensions: (k_w,k_h) doublet indicating the Kikuchi pattern pixel width and height
        :param separator: pattern file name separator between name and number
        :param position: number position in the file name after split
        :param first_id: first id to be read in the folder
        """
        self.data_path = data_path
        self.w = dimensions[0]
        self.h = dimensions[1]
        self.k_w = k_dimensions[0]
        self.k_h = k_dimensions[1]
        self.lam = 0.
        self.sigmas = np.zeros((self.w, self.h))
        self.patterns, self.pat_names = self.load_pattern(data_path, separator, position, first_id)

    def load_pattern(self, data_path, separator, position, first_id):
        """Load a set of Kikuchi pattern from a given folder into a numpy w*h*k_w*k_h array

        :param data_path: path of the kikuchi patern to be loaded
        :param separator: pattern file name separator between name and number
        :param position: number position in the file name after split
        :param first_id: first id to be read in the folder
        """
        patterns = np.zeros((self.w, self.h, self.k_w * self.k_h))
        pat_names = os.listdir(data_path)
        for pat_act_name in pat_names:
            pat_act = cv2.imread(data_path+'/'+pat_act_name)
            pat_act_gray = cv2.cvtColor(pat_act, cv2.COLOR_BGR2GRAY)
            pat_act_id = int(pat_act_name.split(separator)[position].split('.')[0])
            pat_act_id -= first_id
            x_actu = pat_act_id % self.w
            y_actu = math.floor(pat_act_id / self.w)
            patterns[x_actu, y_actu, :] = pat_act_gray.reshape(self.k_w * self.k_h)
        return patterns, pat_names

    def set_lam(self, lam):
        self.lam = lam

    def set_sigmas(self, sigmas):
        self.sigmas = sigmas
