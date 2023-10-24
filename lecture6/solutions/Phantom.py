import numpy as np
from PIL import Image

def Create(phantom_size = 1024):
    # Create a simple phantom (you can replace this with your own data)
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

    return phantom
