import numpy as np 
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.stats import binom
from scipy.ndimage import gaussian_filter
import scipy.signal
from sklearn.decomposition import FastICA

#####################
# Synthetic data generation
#####################

def generate_blob(grid_length:int=50, blob_radius=None, center=None, dimensions_ratio:tuple=None, smoothness:float=0):
    """Generate a single blob with random jitter."""
 
    if blob_radius is None:
        typical_value = grid_length/10
        blob_radius = np.random.uniform(0.8*typical_value, 1.2*typical_value)

    if dimensions_ratio is None:
        a, b = np.random.uniform(0.7, 1.2), np.random.uniform(0.7, 1.2)
    else: 
        a, b = dimensions_ratio

    if center is None: 
        center = np.random.randint(blob_radius/2, grid_length - blob_radius/2, size=2)

    y, x = np.ogrid[-center[0]:grid_length-center[0], -center[1]:grid_length-center[1]]

    spread = 1.3 
    angle = np.random.uniform(0, 2*np.pi) #orientate randomly the blob
    mask = ((x*np.cos(angle)+y*np.sin(angle))/a)**2 + ((-x*np.sin(angle)+y*np.cos(angle))/b)**2 <= (spread**2*blob_radius)**2 
    
    blob = np.zeros((grid_length, grid_length))
    blob[mask] = 1

    # Creating the intensity profile
    distance_from_center = np.sqrt((x * np.cos(angle) + y * np.sin(angle)) ** 2 / a ** 2 +
                                   (-x * np.sin(angle) + y * np.cos(angle)) ** 2 / b ** 2)
    
    sigma_param = 1.1 #control the degrowth of intensity
    intensity = (1/np.sqrt(sigma_param))*np.exp(-distance_from_center**2 / ((sigma_param * blob_radius) ** 2))  # Gaussian-like profile
    
    blob = np.zeros((grid_length, grid_length))
    blob[mask] = intensity[mask]  # Assign intensity to the blob
    
    return blob



def generate_global_map(grid_length:int=50, max_blobs:int=3):
    """Generate a spatial map with blobs and jitter."""
        
    map = np.zeros((grid_length, grid_length))

    num_blobs = 0
    while num_blobs == 0: #avoid having blank map
        num_blobs = binom.rvs(n=max_blobs, p=0.5, size=1)[0]  # Binomial distribution for number of blobs
    blob_radii = np.random.randint(grid_length//10, (grid_length)//5, size=num_blobs)  # Random radius for each blob

    for num_blob in range(num_blobs):
        blob_radius = blob_radii[num_blob]
        center_blob = np.random.randint(blob_radius/2, grid_length - blob_radius/2, size=2)
        blob = generate_blob(grid_length=grid_length, blob_radius=blob_radius, center=center_blob, dimensions_ratio=None)
        map += blob  # Add blob to the map

    return map

def get_random_spatially_correlated_noise(grid_length:int, sigma:float=1, correlation_scale:float=1, sigma_final:float=3):
    """
    Creates a 2D map of size (grid_length, grid_length) with spatially correlated noise
    """
    # Compute filter kernel with radius cor_length 
    cor_length = grid_length * correlation_scale
    x = np.arange(-cor_length, cor_length)
    y = np.arange(-cor_length, cor_length)
    X, Y = np.meshgrid(x, y)
    dist = np.sqrt(X*X + Y*Y)
    sigma_2 = grid_length * correlation_scale
    filter_kernel = np.exp(-dist**2/(2*sigma_2))

    # Generate n-by-n grid of spatially correlated noise
    noise = sigma*np.random.randn(grid_length, grid_length) #random white gaussian noise
    noise = scipy.signal.fftconvolve(noise, filter_kernel, mode='same') #blur the noise with gaussian kernel
    noise = sigma_final * (noise/noise.std()) #rescale the noise so that its std deviation is sigma_final
    return noise

#### deprecated
# def generate_correlated_noise(k, scale=1.0, correlation_length=10):
#     # Generate grid coordinates
#     x, y = np.meshgrid(np.arange(k), np.arange(k))
#     coords = np.stack((x, y), axis=-1)

#     # Create a covariance matrix based on distance
#     distances = np.sqrt(np.sum((coords[:, :, np.newaxis, np.newaxis] - coords[np.newaxis, np.newaxis, :, :]) ** 2, axis=-1))
#     covariance_matrix = np.exp(-distances / correlation_length)

#     # Generate multivariate normal noise
#     noise = np.random.multivariate_normal(mean=np.zeros(k**2), cov=covariance_matrix * scale).reshape(k, k)
#     return noise

#######################

#######################

def laplacian_grid(v_grid):
    """
    returns Lv where L is the laplacian operator
    v is a np.array of size p1*p2
    """
    kernel = np.array([[0,  1, 0],
                        [1, -4, 1],
                        [0,  1, 0]])
    lv = ndimage.convolve(v_grid, kernel, mode='nearest')
    return lv


def omega(v_grid):
    """
    returns omega(v) where v is a np array of size p1*p2
    """
    norm1_v = np.sum(np.abs(v_grid))

    lv = laplacian_grid(v_grid)
    vT_lv = np.sum(v_grid * lv)

    return norm1_v + 1/2 * vT_lv

### deprecated
def laplacian_old(v_grid):
    """
    returns Lv where L is the laplacian operator
    v is a np.array of size p1*p2
    """
    v_grid_padded = np.pad(v_grid, ((1, 1), (1, 1)), mode='edge')
    p1, p2 = v_grid.shape
    Lv = np.zeros((p1+2, p2+2)) #add padding for the extremities
    
    for i in range(p1):
        for j in range(p2):
            i0 = i + 1
            j0 = j + 1
            Lv[i0, j0] = v_grid_padded[i0+1,j0] + v_grid_padded[i0,j0+1] + v_grid_padded[i0-1,j0] + v_grid_padded[i0,j0-1] - 4*v_grid_padded[i0,j0]
    return Lv[1:-1, 1:-1]

def get_init_V(data_fmri: np.ndarray, n_components: int) -> np.ndarray:
    S, n, p1, _ = data_fmri.shape
    X = data_fmri.reshape(S*n, p1, p1)
    X = X.reshape(S*n, p1 * p1) #in 2 steps in case it causes problems
    ica_estimator = FastICA(
                    n_components=n_components, max_iter=400, whiten="arbitrary-variance", tol=15e-5)
    ica_estimator.fit(X)

    components = ica_estimator.components_[:n_components]
    
    return components

def get_corr_matrix(V_maps1: np.ndarray, V_maps2:np.ndarray) -> np.ndarray:
    """
    V_maps1 and V_maps2 are arrays of size (k, p, p) where
    -k is the number of components in the map
    -p is the grid size 
    """
    k , p, _ = V_maps1.shape
    corr_matrix = np.zeros((k,k))
    for i in range(k):
        for j in range(i, k):
            #print(np.corrcoef(V_maps1[i].ravel(),V_maps2[j].ravel())[0, 1])
            corr_matrix[i, j] = np.abs(np.corrcoef(V_maps1[i].ravel(),V_maps2[j].ravel())[0, 1])
            corr_matrix[j, i] = corr_matrix[i, j]
    return corr_matrix


def greedy_matching(set_1, set_2, corr_matrix):
    """
    Finds a matching pairwise between the elements of set_1 and set_2 given the correlation matrix of their elements
    """
    matched_indices = set()
    matched_pairs = []

    # Iterate through set_1
    for i in range(len(set_1)):
        max_corr = -np.inf
        best_pair = None

        # Iterate through set_2 to find the best unmatched pair
        for j in range(len(set_2)):
            if j not in matched_indices:
                if corr_matrix[i, j] > max_corr:
                    max_corr = corr_matrix[i, j]
                    best_pair = (i, j)

        # Mark the best pair as matched
        if best_pair:
            matched_indices.add(best_pair[1])
            matched_pairs.append(best_pair)

    # Reorder set_2 based on the matched pairs
    reordered_set_2 = [set_2[j] for i, j in sorted(matched_pairs)]

    return reordered_set_2

def reorder_maps(real_latent_maps, predicted_maps):
    """
    Inputs: 
    -real_latent_maps: array of size (k, p, p) where k is the number of components and p grid size
    -predicted_maps: array of size (k, p, p) where k is the number of components and p grid size

    Returns: 
    -reordered_predicted_maps: array of size (k, p, p) where k is the number of components and p grid size with maps reordered
    """
    corr_matrix = get_corr_matrix(real_latent_maps, predicted_maps)
    predicted_maps_reordered = greedy_matching(real_latent_maps, predicted_maps, corr_matrix)

    return predicted_maps_reordered