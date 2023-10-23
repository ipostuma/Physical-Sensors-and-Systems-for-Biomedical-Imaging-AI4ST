import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
from scipy.signal import resample
from scipy.fftpack import fft, fftshift, ifft
import time
from PIL import Image

# Define parameters
num_projections = 180  # Number of projections
num_detector_pixels = 256  # Number of detector pixels
theta = np.linspace(0., 180., num_projections, endpoint=False)  # Projection angles

# Create a simple phantom (you can replace this with your own data)
phantom_size = 1024
c = phantom_size//2
phantom = np.zeros((phantom_size, phantom_size))
Y, X = np.ogrid[:phantom_size,:phantom_size]
body = np.sqrt((X - phantom_size //2)**2 + (Y-phantom_size //2)**2)
radius = phantom_size // 3
phantom[body<=radius] = 3
organ = np.sqrt((X - phantom_size // 5 * 3)**2 + (Y-phantom_size // 40 * 15)**2)
radius = phantom_size // 10
phantom[organ<= radius] = 2
radius = phantom_size // 12
phantom[organ<= radius] = 1
radius = phantom_size // 18
phantom[organ<= radius] = 2
radius = phantom_size // 24
phantom[organ<= radius] = 1
radius = phantom_size // 42
phantom[organ<= radius] = 3
phantom[(c-phantom_size//100):(c+phantom_size//100),:] = 4
radius = phantom_size // 3
phantom[body>radius] = 0

im = Image.fromarray(phantom)
im.save("data/phantom.tif")
im = Image.fromarray(phantom/phantom.max()*256)
im = im.convert('RGB')
im.save("data/phantom.jpeg")

np.save("data/phantom.npy",phantom)

# Create sinogram
sinogram = np.zeros((num_detector_pixels, num_projections))
for i in range(num_projections):
    # Rotate the image
    rot_img = rotate(phantom,theta[i],reshape=False,cval=0)
    # Compute the projection along each row
    projection = np.sum(rot_img,axis=1)
    # Resample the data to fit the shape of the detectors
    detector = resample(projection,num_detector_pixels)
    # Insert the detected array in the sinogram reconstruction map
    sinogram[:,i]=detector

def projFilter(sino):
    """filter projections. Normally a ramp filter multiplied by a window function is used in filtered
    backprojection. The filter function here can be adjusted by a single parameter 'a' to either approximate
    a pure ramp filter (a ~ 0)  or one that is multiplied by a sinc window with increasing cutoff frequency (a ~ 1).
    Credit goes to Wakas Aqram. 
    inputs: sino - [n x m] numpy array where n is the number of projections and m is the number of angles used.
    outputs: filtSino - [n x m] filtered sinogram array"""
    
    a = 0.1;
    projLen, numAngles = sino.shape
    step = 2*np.pi/projLen
    w = np.arange(-np.pi, np.pi, step)
    if len(w) < projLen:
        w = np.concatenate([w, [w[-1]+step]]) #depending on image size, it might be that len(w) =  
                                              #projLen - 1. Another element is added to w in this case
    rn1 = abs(2/a*np.sin(a*w/2));  #approximation of ramp filter abs(w) with a funciton abs(sin(w))
    rn2 = np.sin(a*w/2)/(a*w/2);   #sinc window with 'a' modifying the cutoff freqs
    r = rn1*(rn2)**2;              #modulation of ramp filter with sinc window
    
    filt = fftshift(r)   
    filtSino = np.zeros((projLen, numAngles))
    for i in range(numAngles):
        projfft = fft(sino[:,i])
        filtProj = projfft*filt
        filtSino[:,i] = np.real(ifft(filtProj))

    return filtSino

def filtered_back_projection(sinogram, min_a=0, max_a=180):
    # Projection number
    num_projections = sinogram.shape[1]
    # Detector size
    num_detector_pixels = sinogram.shape[0]
    # maximum value of the synogram pixel
    vmax = sinogram.max()*num_projections/20
    # Initialize the reconstructed image
    reconstructed_image = np.zeros((num_detector_pixels, num_detector_pixels))
    # Calculate the center of the sinogram
    center = num_detector_pixels // 2
    # Define the angles of projection (0 to 180 degrees)
    angles = np.linspace(min_a, max_a, num_projections, endpoint=False)
    angles = np.deg2rad(angles)
    # Coordinate system centered at (0,0)
    x = np.arange(num_detector_pixels)-num_detector_pixels/2 
    y = x.copy()
    X, Y = np.meshgrid(x, y)

    plot=False
    if plot:
        plt.ion()
        fig, ax= plt.subplots(1,1)
        img = ax.imshow(reconstructed_image,vmin=0,vmax=2)
        ax.axis('off')
        plt.show()

    for n in range(num_projections):
        print(n)
        # Determine the rotated coordinates about the origin of the mesh grid
        Xrot = X*np.sin(angles[n])-Y*np.cos(angles[n])

        # Shift to original image coordinates
        XrotCor = np.round(Xrot+num_detector_pixels/2).astype('int')

        # After rotating, you'll inevitably have new coordinates that exceed the size of the original array
        projMatrix = np.zeros((num_detector_pixels, num_detector_pixels))
        m0, m1 = np.where((XrotCor >= 0) & (XrotCor <= (num_detector_pixels-1)))

        # Get projection
        s = sinogram[:,n]

        # Back project 
        projMatrix[m0, m1] = s[XrotCor[m0, m1]]/vmax
        reconstructed_image += projMatrix
        
        if plot:
            img.set_data(reconstructed_image)
            fig.canvas.flush_events()
            plt.savefig("gif/{:03d}.jpg".format(n))
            time.sleep(0.01)
    if plot:
        plt.close()
        plt.ioff()
    
    reconstructed_image = np.flipud(reconstructed_image)
    reconstructed_image = np.fliplr(reconstructed_image)
    return reconstructed_image

sinogramf = projFilter(sinogram)
reconstructed_image = filtered_back_projection(sinogram)

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