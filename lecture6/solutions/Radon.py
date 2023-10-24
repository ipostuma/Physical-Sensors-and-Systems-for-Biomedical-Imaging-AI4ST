import numpy as np
from scipy.ndimage import rotate
from scipy.signal import resample
from scipy.fftpack import fft, fftshift, ifft
import matplotlib.pyplot as plt
import time

def Transform(phantom,num_detector_pixels,num_projections,min_a=0, max_a=180):
    # Create sinogram
    sinogram = np.zeros((num_detector_pixels, num_projections))
    theta = np.linspace(min_a, max_a, num_projections, endpoint=False)  # Projection 
    for i in range(num_projections):
        # Rotate the image
        rot_img = rotate(phantom,theta[i],reshape=False,cval=0)
        # Compute the projection along each row
        projection = np.sum(rot_img,axis=1)
        # Resample the data to fit the shape of the detectors
        detector = resample(projection,num_detector_pixels)
        # Insert the detected array in the sinogram reconstruction map
        sinogram[:,i]=detector
    return sinogram

def projFilter(sino, a=0.1):
    """filter projections. Normally a ramp filter multiplied by a window function is used in filtered backprojection. The filter function here can be adjusted by a single parameter 'a' to either approximate a pure ramp filter (a ~ 0)  or one that is multiplied by a sinc window with increasing cutoff frequency (a ~ 1).
    
    inputs: 
        sino - [n x m] numpy array where n is the number of projections and m is the number of angles used. outputs: filtSino - [n x m] filtered sinogram array.

        a - parameter that modulates the ramp filter window.

    source: https://github.com/csheaff/filt-back-proj  
    """
    
    projLen, numAngles = sino.shape
    step = 2*np.pi/projLen
    w = np.arange(-np.pi, np.pi, step)
    if len(w) < projLen:
        # Depending on image size, it might be that len(w) = projLen - 1.
        # Another element is added to w in this case
        w = np.concatenate([w, [w[-1]+step]])
    # Ppproximation of ramp filter abs(w) with a funciton abs(sin(w)) 
    rn1 = abs(2/a*np.sin(a*w/2))
    # Sinc window with 'a' modifying the cutoff freqs  
    rn2 = np.sin(a*w/2)/(a*w/2)
    # Modulation of ramp filter with sinc window 
    r = rn1*(rn2)**2
    
    # Apply the filter
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