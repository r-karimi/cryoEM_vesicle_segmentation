# Helper functions for vesicle picking with segment-anything

import matplotlib.pyplot as plt
import numpy as np
import configparser
from math import prod
from functools import reduce
from vesicle_picker import funcs_mrcio

def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    assert h % nrows == 0, f"{h} rows is not evenly divisible by {nrows}"
    assert w % ncols == 0, f"{w} cols is not evenly divisible by {ncols}"
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))

def unblockshaped(arr, h, w):
    """
    Return an array of shape (h, w) where
    h * w = arr.size

    If arr is of shape (n, nrows, ncols), n sublocks of shape (nrows, ncols),
    then the returned array preserves the "physical" layout of the sublocks.
    """
    n, nrows, ncols = arr.shape
    return (arr.reshape(h//nrows, -1, nrows, ncols)
               .swapaxes(1,2)
               .reshape(h, w))

def show_anns(anns):
    """Adapted from segment-anything"""
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

def sum_masks(masks, key):
    return sum(mask.get(key, 0) for mask in masks)

def multiply_masks(masks, key):
    return prod(mask.get(key, 0) for mask in masks)

def read_config(config_file_path):
    config = configparser.ConfigParser()
    config.read(config_file_path)
    return config

# Taken from: https://stackoverflow.com/questions/6800193/what-is-the-most-efficient-way-of-finding-all-the-factors-of-a-number-in-python/
def factors(n):    
    return np.array(reduce(list.__add__, 
                ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))

def import_mrc(filename):
    """Use funcs_mrcio to open a specified .mrc file"""
    # Read the .mrc file in binary
    micrograph = open(filename,'rb')
    
    # Use funcs_mrcio to extract image array and rescale values to lie between [-1, 1]
    image = funcs_mrcio.irdsec_opened(micrograph,0)
    
    # Use funcs_mrcio to extract header info
    header = funcs_mrcio.irdhdr_opened(micrograph)
    
    # Return the rescaled image and header
    return image, header

def export_mrc(image, filename):
    """Export an mrc image from a numpy array"""
    
    # Generate a new header
    nx, ny = image.shape
    nxyz = np.array([nx, ny, 1], dtype=np.float32)
    dmin = np.min(image)
    dmax = np.max(image)
    dmean = np.sum(image)/(nx*ny)
    
    # Open a new file
    mrc = open(filename, 'wb')
    
    # Write the header to the new file
    funcs_mrcio.iwrhdr_opened(mrc, 
                  nxyz, 
                  dmin, 
                  dmax, 
                  dmean, 
                  mode=2)
    
    # Write the rebinned array to the new file
    funcs_mrcio.iwrsec_opened(image, mrc)

def make_square_bbox(bbox):
    # Unpack the bbox coordinates
    x_min, y_min, x_max, y_max = bbox
    
    # Calculate the center point of the bbox
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    
    # Calculate the width and height of the bbox
    width = x_max - x_min
    height = y_max - y_min
    
    # Determine the size of the square bbox
    side_length = max(width, height)
    
    # Calculate new bbox coordinates to make it square
    new_x_min = center_x - side_length / 2
    new_y_min = center_y - side_length / 2
    new_x_max = center_x + side_length / 2
    new_y_max = center_y + side_length / 2
    
    # Return the new square bbox
    return int(new_x_min), int(new_y_min), int(new_x_max), int(new_y_max)

# Adapted from Jensen, Sigworth et al. 2015
def circular_tukey_window(dims, Rn, alpha):
    """
    Generate a circular Tukey window with updated conditionals.

    :param dims: Tuple of (height, width) for the window dimensions.
    :param Rn: Radius up to which the window value is 1.
    :param alpha: Controls the width of the tapering region.
    """
    # Generate a grid of indices
    rows, cols = np.ogrid[:dims[0], :dims[1]]
    
    # Calculate distances from the center
    c_row, c_col = np.array(dims) / 2  # Center of the image
    r = np.sqrt((rows - c_row)**2 + (cols - c_col)**2)  # Actual radial distances
    
    # Initialize the window based on the conditionals
    window = np.zeros(dims)
    
    # First condition: r <= Rn
    window[r <= Rn] = 1
    
    # Second condition: Rn < r < Rn + 1/alpha
    taper_mask = (r > Rn) & (r < Rn + 1/alpha)
    window[taper_mask] = 0.5 + 0.5 * np.cos(np.pi * alpha * (r[taper_mask] - Rn))
    
    return window