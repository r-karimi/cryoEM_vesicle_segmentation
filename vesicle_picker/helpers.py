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
    
    Taken from: https://stackoverflow.com/questions/16856788/slice-2d-array-into-smaller-2d-arrays
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
    
    Taken from: https://stackoverflow.com/questions/16856788/slice-2d-array-into-smaller-2d-arrays
    """
    n, nrows, ncols = arr.shape
    return (arr.reshape(h//nrows, -1, nrows, ncols)
               .swapaxes(1,2)
               .reshape(h, w))

def show_anns(anns):
    """Adapted from the segment-anything Github repository"""
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

def factors(n):    
    """
    Taken from: https://stackoverflow.com/questions/6800193/what-is-the-most-efficient-way-of-finding-all-the-factors-of-a-number-in-python/
    """
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


