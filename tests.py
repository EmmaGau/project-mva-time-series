import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_blobs = 5
map_size = (100, 100)  # Example: 2D space

# Generate random blob characteristics
blob_positions = np.random.randint(0, map_size[0], size=(num_blobs, 2))  # Random positions
blob_sizes = np.random.normal(loc=10, scale=2, size=num_blobs).astype(int)  # Gaussian distribution for blob sizes

# Create spatial map
spatial_map = np.zeros(map_size)

# Place blobs on spatial map
for i in range(num_blobs):
    blob = np.random.normal(loc=1, scale=0.5, size=(blob_sizes[i], blob_sizes[i]))  # Gaussian blob
    x, y = blob_positions[i]
    spatial_map[x:x+blob.shape[0], y:y+blob.shape[1]] += blob  # Adding blobs to the map

# Display the spatial map
plt.imshow(spatial_map, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.show()
