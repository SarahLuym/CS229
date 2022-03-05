# Question 5 K-means for compression
from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np

# a.
A = imread('./data/mandrill-large.tiff')
plt.imshow(A)
plt.savefig('Q5_a_picture.png')
plt.show()

# b.
B = imread('./data/mandrill-small.tiff')
plt.imshow(B)
plt.savefig('Q5_b_picture.png')
plt.show()


def kmeans(X, n_centroids=16, min_iter=30):
    C = X.reshape(-1, 3)
    centroids = C[np.random.randint(0, C.shape[0], n_centroids)]
    error_history = []
    error = 1
    n_iter = 0

    # Loop until for at least min_iter and while the centroids are not stable
    while (error > 0 or n_iter < min_iter):
        # For each pixel, calculate the distance to all current centroids:
        norms = np.array([np.linalg.norm(C-centroids[k,:],2,1) for k in range(n_centroids)]).T
        # Assign each pixel to the closest centroid:
        assign = np.argmin(norms, 1)
        # For a given centroid, calculate the mean of the pixels assigned to it:
        means = np.array([np.mean(C[np.where(assign == k)], 0) for k in range(n_centroids)])
        # Check for stability of the centroids and keep track of the error history
        error = np.linalg.norm(centroids - means, 1, 1).sum()
        error_history.append(error)
        # Assign the new centroids as the means calculated previously:
        centroids = means
        n_iter += 1

    return centroids, error_history


centroids, error_history = kmeans(B)

plt.plot(error_history)
plt.xlabel("iterations")
plt.ylabel("error")
plt.savefig('Q5_b_error.png')
plt.show()

centroids = centroids.astype(int)
colormap = np.repeat(centroids, 1024, 0).reshape(128, 128, 3)
plt.figure(figsize=(8, 8))
plt.imshow(colormap)
plt.savefig('Q5_b_color.png')
plt.show()


# c.
C = A.reshape(-1, 3)
# For each pixel, calculate the distance to all current centroids:
norms = np.array([np.linalg.norm(C-centroids[k, :], 2, 1) for k in range(centroids.shape[0])]).T
# Assign each pixel to the closest centroid:
assign = np.argmin(norms, 1)

# Create a new image where each pixel is the the centroid closest to the pixel in the original image
compressed_image = np.array([centroids[assign[k]] for k in range(assign.shape[0])])
compressed_image = compressed_image.reshape(512, 512, 3)

# Plot the pictures
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
ax0, ax1 = axes.ravel()

ax0.imshow(A)
ax0.set_title('Original Image')
ax1.imshow(compressed_image)
ax1.set_title('Compressed Image')
plt.savefig('Q5_c.png')
plt.show()


# d.
# The original image is in RGB-256 format. Each pixel requires (log2 256) × 3 = 24 bits of memory to store.
# The compressed image only has 16 colors, therefore each pixel only requires log2 16 = 4 bits of memory. The overall compression factor is 24 / 4 = 6, which is a fairly large amount. As can be seen from the images above, the compression process using k-means only results in minimal loss of detail in this example.