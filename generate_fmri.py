import numpy as np 
import matplotlib.pyplot as plt
from utils import *
from scipy import ndimage
import scipy.ndimage.filters

class Dataset_fmri:
    def __init__(self, nbSubjects:int=12, n:int=150, grid_length:int=50, k:int=5, sigma:float=0.1, zeta:float=0.1, ksi:float=0.1) -> None:
        """
        Parameters: 
        -nbSubjects: S in the paper, number of subjects under study
        -n         : n in the paper, length of the time series
        -sigma     : sigma in the paper, residual of the regression
        -zeta      : zeta in the paper, independant gaussian noise characterizing subject variability to latent map
        -ksi       : ksi in the paper, scaling factor for the prior map
        """

        self.nbSubjects = nbSubjects
        self.n = n
        self.k = k
        self.grid_length = grid_length
        self.p = self.grid_length ** 2
        self.sigma = sigma
        self.zeta = zeta
        self.ksi  = ksi

        latent_maps = np.array([generate_global_map(grid_length=self.grid_length, max_blobs=3) for _ in range(k)])
        self.V = latent_maps.reshape((grid_length**2, k))

        self.patient_list = []


class Patient:
    def __init__(self, n:int=150, k:int=5, sigma:float=0.1, zeta:float=0.1, ksi:float=0.1) -> None:
        self.n = n
        self.k = k
        self.sigma = sigma
        self.zeta = zeta
        self.ksi  = ksi
