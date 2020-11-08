import numpy as np
from skimage import data, color, io, img_as_float
import cv2
from skimage import measure
from plotly.offline import download_plotlyjs, init_notebook_mode, plot
from plotly.tools import FigureFactory as FF
from pydicom.data import get_testdata_files
import pydicom
import matplotlib.pyplot as plt
from collections import Counter
import matplotlib as mpl
from PIL import Image
from skimage import morphology
import pylab
import copy
import scipy.ndimage
from PIL import Image
from numpy import array
import os
import glob
from plotly.tools import FigureFactory as FF
from skimage import morphology
from plotly.offline import download_plotlyjs, init_notebook_mode, plot
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.ndimage
from skimage import measure
from PIL import ImageFilter
from sklearn.cluster import KMeans

def sample_stack(stack, rows=2, cols=2, start_with=0, show_every=1, display1 = True):
    if (display1):
        new_list = []
        new_list.append(stack)
        new_list.append(stack)
        new_list.append(stack)
        new_list.append(stack)
        sample_stack(new_list, 2, 2, 0, 1, False)
    else:
        fig,ax = plt.subplots(rows,cols,figsize=[12,12])
        for i in range((rows*cols)):
            ind = start_with + i*show_every
            ax[int(i/rows),int(i % rows)].set_title('slice %d' % ind)
            ax[int(i/rows),int(i % rows)].imshow(stack[ind],cmap='gray')
            ax[int(i/rows),int(i % rows)].axis('off')
        plt.show()
def find_seed(imgs_to_process, thresh_min, seed_index):
    test_img = copy.deepcopy(imgs_to_process[seed_index])
    test_img_no_edit = copy.deepcopy(imgs_to_process[seed_index])
    test_img[test_img > 0] = 1
    row_size, col_size = test_img.shape
    sample_stack(test_img)
    labels = measure.label(test_img)  # Different labels are displayed in different colors
    label_vals = np.unique(labels)
    regions = measure.regionprops(labels)
    good_labels = []
    for prop in regions:
        B = prop.bbox
        if B[2] - B[0] < row_size / 10 * 9 and B[3] - B[1] < col_size / 10 * 9 and B[0] > row_size / 5 and B[
            2] < col_size / 5 * 4:
            good_labels.append(prop.label)
    mask = np.ndarray([row_size, col_size], dtype=np.int8)
    mask[:] = 0
    seed_list = []
    for N in good_labels:
        mask = np.where(labels == N, 1, 0)
        one_lung = mask * test_img_no_edit
        x_min = int(np.median((np.where(one_lung > thresh_min))[0]))
        y_min = int(np.median((np.where(one_lung > thresh_min))[1]))
        seed_thresh = thresh_min + 10
        print("Seed Pixel Value: {}".format(test_img_no_edit[x_min][y_min]))
        #If seed lands on a dark spot, re-check using different threshold values until correct
        while (test_img_no_edit[x_min][y_min] < thresh_min):
            x_min = int(np.median((np.where(one_lung > seed_thresh))[0]))
            y_min = int(np.median((np.where(one_lung > seed_thresh))[1]))
            print("Let's kick it up a notch")
            seed_thresh += 10
        seed = [x_min, y_min]
        print("Seed Point: {}".format(seed))
        seed_list.append(seed)
    return seed_list
def connected_threshold(img, img_mask, thresh_min, thresh_max, seed_index = 188):
    def surround_with_ones(L1, center):
        # l1 = list of ones
        x = center[0]
        y = center[1]
        z = center[2]
        L1.append([x + 1, y, z])
        L1.append([x - 1, y, z])
        L1.append([x, y + 1, z])
        L1.append([x, y - 1, z])
        L1.append([x, y, z + 1])
        L1.append([x, y, z - 1])
        return l1

    pt = np.zeros_like(img)
    l1 = []

    seeds = find_seed(img, thresh_min, seed_index)
    left_seed = seeds[0]
    right_seed = seeds[1]
    xseed = seed_index
    yseed = left_seed[0]
    zseed = left_seed[1]
    #pt = array of zeros slowly filled with 255, 0, or 3
    pt[xseed][yseed][zseed] = 255
    center = [xseed, yseed, zseed]
    l1 = surround_with_ones(l1, center)

    xseed = seed_index
    yseed = right_seed[0]
    zseed = right_seed[1]
    # pt = array of zeros slowly filled with 255, 0, or 3
    pt[xseed][yseed][zseed] = 255
    center = [xseed, yseed, zseed]
    l1 = surround_with_ones(l1, center)
    print("Region Growing...")
    while (len(l1) > 0):
        points = l1.pop()
        x = points[0]
        y = points[1]
        z = points[2]
        try:
            if (pt[x][y][z] == 0):
                if ((img[x][y][z] <= thresh_max) & (img[x][y][z] >= thresh_min)):
                    pt[x][y][z] = 255
                    l1 = surround_with_ones(l1, [x, y, z])
                else:
                    pt[x][y][z] = 3
        except:
            pass

    #l1 = list of ones
    return pt
def plot_pixel_count(imgs_after_rg, length, pixel_val= 255):
    print("Plotting Pixel Count...")
    xvals = np.arange(length)
    yvals = np.ones_like(xvals)
    for i in range(length):
        yvals[i] = np.count_nonzero(imgs_after_rg[i] == pixel_val)
    plt.plot(xvals, yvals, 'ro')
    plt.show()

id = 2
out_path = "/Users/paulmccabe/Desktop/Segmentation Project/"
imgs_to_process = np.load(out_path + "nplungs_%d.npy" % (id))
np_mask = np.load(out_path + "justmask_%d.npy" % (id))

seed_index = 188

#sample_stack(imgs_to_process[150])
eroded_to_process = copy.deepcopy(imgs_to_process)
sample_stack(np_mask[seed_index])
sample_stack(imgs_to_process[seed_index])
print("Eroding Masks...")
for i in range(0, np.size(imgs_to_process, 0), 1):
    np_mask_smaller = morphology.erosion(np_mask[i], np.ones([19, 19]))
    eroded_to_process[i] = (np_mask_smaller * imgs_to_process[i]).astype(int)
    #scipy.misc.imsave(out_path + "eroded_tp_%d.jpg" % (i), eroded_to_process[i])
sample_stack(eroded_to_process[seed_index])
thresh_min = 65
thresh_max = 255
imgs_after_rg = connected_threshold(eroded_to_process, np_mask, thresh_min, thresh_max, seed_index)

print("Plotting Pixels...")
plot_pixel_count(imgs_after_rg, len(imgs_after_rg))
sample_stack(imgs_to_process[seed_index])
sample_stack(imgs_after_rg[seed_index])

print("Saving To Numpy Array...")
np.save(out_path + "bronchioles_after_rg_%d.npy" % (id),imgs_after_rg)
