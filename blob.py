import numpy as np
from typing import Tuple, Callable, List, Union
import matplotlib.pyplot as plt
from scipy.stats import binom
from scipy.ndimage import gaussian_filter
import scipy.signal

class Blob:
    def __init__(self, blob_radius=None, center=None, dimensions_ratio=None, angle=None, grid_length=None) -> None:
        """
        Abstract object for blob
        !!! Has to be initiated to be plotted or used !!!
        """
        self.radius = blob_radius
        self.center : tuple = center
        self.dimensions_ratio : tuple = dimensions_ratio
        self.angle = angle

        if grid_length is not None:
            self.grid_length = grid_length
            self.initiate_blob(grid_length)
    
    def initiate_blob(self, grid_length:int):
        """
        Initiates a random blob with suitable parameters on a 2D squared grid of edge grid_length
        """
        if self.radius is None:
            typical_value = grid_length/10
            self.radius = np.random.uniform(0.7*typical_value, 1.1*typical_value)

        if self.dimensions_ratio is None:
            self.dimensions_ratio = np.random.uniform(0.7, 1.2), np.random.uniform(0.7, 1.2)

        if self.center is None: 
            self.center = np.random.randint(self.radius/2, grid_length - self.radius/2, size=2)

        if self.angle is None:
            self.angle = np.random.uniform(0, 2*np.pi) #orientate randomly the blob

        return self
    
    def get_blob_map(self, grid_length:int) -> np.ndarray:
        """
        Returns the map of values with the blob on it
        """
        y, x = np.ogrid[-self.center[0]:grid_length-self.center[0], -self.center[1]:grid_length-self.center[1]]

        spread = 1.3
        a, b = self.dimensions_ratio
        mask = ((x*np.cos(self.angle)+y*np.sin(self.angle))/a)**2 + ((-x*np.sin(self.angle)+y*np.cos(self.angle))/b)**2 <= (spread**2*self.radius)**2 
        

        # Creating the intensity profile
        distance_from_center = np.sqrt((x * np.cos(self.angle) + y * np.sin(self.angle)) ** 2 / a ** 2 +
                                    (-x * np.sin(self.angle) + y * np.cos(self.angle)) ** 2 / b ** 2)
        
        sigma_param = 1.1
        intensity = (1/np.sqrt(sigma_param))*np.exp(-distance_from_center**2 / ((sigma_param * self.radius) ** 2))  # Gaussian-like profile
        
        blob_values = np.zeros((grid_length, grid_length))
        blob_values[mask] = intensity[mask]  # Assign intensity to the blob
        
        return blob_values
    
    def get_copy(self):
        """
        returns a copy of the instantiated blob: useful to get subjects blobs
        """
        return Blob(blob_radius=self.radius, center=self.center, 
                    dimensions_ratio=self.dimensions_ratio, angle=self.angle, grid_length=self.grid_length)
    
######################
######################



def get_random_blobs(grid_length:int=50, max_blobs:int=3, centers_to_avoid:List[np.ndarray]=[], dist_between_blobs=10):
    """Generate a spatial map with blobs and jitter."""
        
    blobs = []
    num_blobs = 0
    while num_blobs == 0: #avoid having blank map
        num_blobs = binom.rvs(n=max_blobs, p=0.5, size=1)[0]  # Binomial distribution for number of blobs
    blob_radii = np.random.randint(grid_length//10, (grid_length)//5, size=num_blobs)  # Random radius for each blob

    for num_blob in range(num_blobs):
        blob_radius = blob_radii[num_blob]

        #avoids overlap with existing blobs
        if len(centers_to_avoid) > 0:
            min_dist = 0
            while min_dist < dist_between_blobs: 
                center_new_blob = np.random.randint(blob_radius/2, grid_length - blob_radius/2, size=2)
                min_dist = np.min([np.sqrt(np.sum(existing_center **2 - center_new_blob**2)) for existing_center in centers_to_avoid])
        else: 
            centers_to_avoid = []
            center_new_blob = np.random.randint(blob_radius/2, grid_length - blob_radius/2, size=2)

        new_blob = Blob(grid_length=grid_length, blob_radius=blob_radius, center=center_new_blob)
        blobs.append(new_blob)
        centers_to_avoid.append(center_new_blob)

    return blobs, centers_to_avoid

def generate_global_map(grid_length, blobs:List[Blob]):
    """
    returns the map of values given by a list of blobs
    """
    map = np.zeros((grid_length, grid_length))
    for blob in blobs:
        map += blob.get_blob_map(grid_length=grid_length)
    return map