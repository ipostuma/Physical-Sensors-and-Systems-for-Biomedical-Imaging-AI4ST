# Convolution with Python

Image convolution is a fundamental operation in image processing. It involves applying a filter or kernel to an image to perform operations such as blurring, sharpening, edge detection, and more. This process helps enhance or extract features from images.

In image convolution, a kernel (a small matrix) is moved over the image, and for each position, a mathematical operation takes place. The kernel's values determine the operation. Here's a simplified example:

Let's say you have a 3x3 kernel for blurring:

```
1/9 1/9 1/9
1/9 1/9 1/9
1/9 1/9 1/9
```

To blur the image, you multiply the values in the kernel with the pixel values in the image and sum them up. This produces a new pixel value in the output image.

Commonly used kernels include:

* Identity Kernel: [0 0 0, 0 1 0, 0 0 0] - It does nothing, just leaves the pixel as is.
* Gaussian Kernel: Used for blurring.
* Edge Detection Kernels: Such as the Sobel and Prewitt kernels, which highlight edges in images.
* Sharpening Kernel: Enhances edges and fine details.
* Embossing Kernel: Creates a 3D effect in the image.

Now, regarding Python as a platform for image manipulation, it offers several advantages:

* OpenCV and Pillow Libraries: Python has powerful libraries like OpenCV and Pillow for image processing, making it easy to work with images.
* Numpy: Numpy provides efficient array operations, making it ideal for handling pixel values and mathematical operations on images.
* Rich Ecosystem: Python's ecosystem includes a wealth of tools and libraries for data manipulation, visualization, and machine learning. This is especially useful when you're dealing with tasks that require integration with other data processing steps.
* Community and Resources: Python has a large and active community, which means there are plenty of resources, tutorials, and support available for image processing tasks.

To better evaluate your understanding image convolution follor this [Convolution notebook](Notebooks/Lab0_Convolution.ipynb).

## Conda environment definition

To use the notebooks available here, please install the following packages with conda.

```
conda create -n keras
conda activate keras
conda install tensorflow-gpu=2.4.1 cudatoolkit jupyter matplotlib scikit-image numpy
conda install -c conda-forge nibabel
```
