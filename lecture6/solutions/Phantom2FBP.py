import matplotlib.pyplot as plt
import Phantom
import Radon

# Define reconstruction parameters
num_projections = 180  # Number of projections
num_detector_pixels = 256  # Number of detector for each projection

phantom = Phantom.Create()

sinogram = Radon.Transform(phantom,num_detector_pixels,num_projections)

sinogramf = Radon.projFilter(sinogram)

reconstructed_image = Radon.filtered_back_projection(sinogram)

# Display the results
plt.figure(figsize=(8, 8), tight_layout=True)
plt.subplot(221)
plt.title("Phantom")
plt.imshow(phantom, cmap='gray')

plt.subplot(222)
plt.title("Reconstructed Image")
plt.imshow(reconstructed_image, cmap='gray')

plt.subplot(223)
plt.title("Sinogram")
plt.imshow(sinogram, cmap='gray', extent=(0, 180, 0, num_detector_pixels))

plt.subplot(224)
plt.title("Filtered Sinogram")
plt.imshow(sinogramf, cmap='gray', extent=(0, 180, 0, num_detector_pixels))

plt.show()