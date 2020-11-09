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
def make_mesh(image, threshold=300, step_size=1):
    print("Transposing surface")
    p = image.transpose(2, 1, 0)

    print("Calculating surface")
    verts, faces, norm, val = measure.marching_cubes(p, threshold, step_size=step_size, allow_degenerate=True)
    return verts, faces
def plotly_3d(verts, faces):
    x, y, z = zip(*verts)

    print("Drawing")

    # Make the colormap single color since the axes are positional not intensity.
    #    colormap=['rgb(255,105,180)','rgb(255,255,51)','rgb(0,191,255)']
    colormap = ['rgb(236, 236, 212)', 'rgb(236, 236, 212)']

    fig = FF.create_trisurf(x=x,
                            y=y,
                            z=z,
                            plot_edges=False,
                            colormap=colormap,
                            simplices=faces,
                            backgroundcolor='rgb(64, 64, 64)',
                            title="Interactive Visualization")
    plot(fig)
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
def grow(img, seed, t):
    """
    img: ndarray, ndim=3
        An image volume.

    seed: tuple, len=3
        Region growing starts from this point.

    t: int
        The image neighborhood radius for the inclusion criteria.
    """
    seg = np.zeros(img.shape, dtype=np.bool)
    checked = np.zeros_like(seg)

    seg[seed] = True
    checked[seed] = True
    needs_check = get_nbhd(seed, checked, img.shape)

    while len(needs_check) > 0:
        pt = needs_check.pop()

        # Its possible that the point was already checked and was
        # put in the needs_check stack multiple times.
        if checked[pt]: continue

        checked[pt] = True

        # Handle borders.
        imin = max(pt[0] - t, 0)
        imax = min(pt[0] + t, img.shape[0] - 1)
        jmin = max(pt[1] - t, 0)
        jmax = min(pt[1] + t, img.shape[1] - 1)
        kmin = max(pt[2] - t, 0)
        kmax = min(pt[2] + t, img.shape[2] - 1)

        if (img[pt] >= 0) or (img[pt] < 60):
            # Include the voxel in the segmentation and
            # add its neighbors to be checked.
            seg[pt] = True
            needs_check += get_nbhd(pt, checked, img.shape)

    return seg
def get_nbhd(pt, checked, dims):
    nbhd = []

    if (pt[0] > 0) and not checked[pt[0]-1, pt[1], pt[2]]:
        nbhd.append((pt[0]-1, pt[1], pt[2]))
    if (pt[1] > 0) and not checked[pt[0], pt[1]-1, pt[2]]:
        nbhd.append((pt[0], pt[1]-1, pt[2]))
    if (pt[2] > 0) and not checked[pt[0], pt[1], pt[2]-1]:
        nbhd.append((pt[0], pt[1], pt[2]-1))

    if (pt[0] < dims[0]-1) and not checked[pt[0]+1, pt[1], pt[2]]:
        nbhd.append((pt[0]+1, pt[1], pt[2]))
    if (pt[1] < dims[1]-1) and not checked[pt[0], pt[1]+1, pt[2]]:
        nbhd.append((pt[0], pt[1]+1, pt[2]))
    if (pt[2] < dims[2]-1) and not checked[pt[0], pt[1], pt[2]+1]:
        nbhd.append((pt[0], pt[1], pt[2]+1))

    return nbhd
def region_grow(vol, mask, start_point, epsilon=5, HU_mid=0, HU_range=0, fill_with=1):
    """
    # `vol` your already segmented 3d-lungs, using one of the other scripts
    # `mask` you can start with all 1s, and after this operation, it'll have 0's where you need to delete
    # `start_point` a tuple of ints with (z, y, x) coordinates
    # `epsilon` the maximum delta of conductivity between two voxels for selection
    # `HU_mid` Hounsfield unit midpoint
    # `HU_range` maximim distance from `HU_mid` that will be accepted for conductivity
    #`fill_with` value to set in `mask` for the appropriate location in vol that needs to be flood filled
    """
    sizez = vol.shape[0] - 1
    sizex = vol.shape[1] - 1
    sizey = vol.shape[2] - 1

    items = []
    visited = []

    def enqueue(item):
        items.insert(0, item)

    def dequeue():
        s = items.pop()
        visited.append(s)
        return s

    enqueue((start_point[0], start_point[1], start_point[2]))

    while not items == []:

        z, x, y = dequeue()

        voxel = vol[z, x, y]
        mask[z, x, y] = fill_with

        if x < sizex:
            tvoxel = vol[z, x + 1, y]
            if abs(tvoxel - voxel) < epsilon and abs(tvoxel - HU_mid) < HU_range:  enqueue((z, x + 1, y))

        if x > 0:
            tvoxel = vol[z, x - 1, y]
            if abs(tvoxel - voxel) < epsilon and abs(tvoxel - HU_mid) < HU_range:  enqueue((z, x - 1, y))

        if y < sizey:
            tvoxel = vol[z, x, y + 1]
            if abs(tvoxel - voxel) < epsilon and abs(tvoxel - HU_mid) < HU_range:  enqueue((z, x, y + 1))

        if y > 0:
            tvoxel = vol[z, x, y - 1]
            if abs(tvoxel - voxel) < epsilon and abs(tvoxel - HU_mid) < HU_range:  enqueue((z, x, y - 1))

        if z < sizez:
            tvoxel = vol[z + 1, x, y]
            if abs(tvoxel - voxel) < epsilon and abs(tvoxel - HU_mid) < HU_range:  enqueue((z + 1, x, y))

        if z > 0:
            tvoxel = vol[z - 1, x, y]
            if abs(tvoxel - voxel) < epsilon and abs(tvoxel - HU_mid) < HU_range:  enqueue((z - 1, x, y))
    return mask
def connected_threshold(img, seed, thresh_min, thresh_max, num3s):
    def Getxyz(i):
        return i[0], i[1], i[2]
    print("Running Connected Threshold...")
    img_height = np.size(img, 0) - 10
    def contains3(voxel, PT):
        total_3s = 0
        for i in voxel:
            x = i[0]
            y = i[1]
            z = i[2]
            #print("pt: {}, img: {}".format(PT[x][y][z], img[x][y][z]))
            if ((x >= 0) & (x < img_height)):
                if ((PT[x][y][z] == 3) or (img[x][y][z] >= thresh_max) or (img[x][y][z] < thresh_min)):
                    total_3s += 1
        #print(total_3s)
        return total_3s
    def surround_with_ones_larger(center, L1):
        x = center[0]
        y = center[1]
        z = center[2]
        voxel = []
        voxel.append([x+1, y, z])
        voxel.append([x+1, y+1, z])
        voxel.append([x+1, y, z+1])
        voxel.append([x+1, y+1, z+1])
        voxel.append([x+1, y-1, z])
        voxel.append([x+1, y, z-1])
        voxel.append([x+1, y-1, z-1])
        voxel.append([x+1, y+1, z-1])
        voxel.append([x+1, y-1, z+1])
        voxel.append([x-1, y, z])
        voxel.append([x-1, y+1, z])
        voxel.append([x-1, y, z+1])
        voxel.append([x-1, y+1, z+1])
        voxel.append([x-1, y-1, z])
        voxel.append([x-1, y, z-1])
        voxel.append([x-1, y-1, z-1])
        voxel.append([x-1, y+1, z-1])
        voxel.append([x-1, y-1, z+1])
        voxel.append([x, y-1, z])
        voxel.append([x, y+1, z])
        voxel.append([x, y, z-1])
        voxel.append([x, y, z+1])
        voxel.append([x, y-1, z+1])
        voxel.append([x, y+1, z-1])
        voxel.append([x, y+1, z+1])
        voxel.append([x, y-1, z-1])
        #voxel.append([x+2, y, z])
        #voxel.append([x-2, y, z])
        #voxel.append([x, y+2, z])
        #voxel.append([x, y-2, z])
        #voxel.append([x, y, z+2])
        #voxel.append([x, y, z-2])
        L1.append(voxel)
        return L1
    xseed = seed[0]
    yseed = seed[1]
    zseed = seed[2]
    pt = np.zeros_like(img)
    pt[xseed][yseed][zseed] = 255
    center = [xseed, yseed, zseed]
    L1 = []
    L1 = surround_with_ones_larger(center, L1)
    while (len(L1) > 0):
        voxel = L1.pop()
        if (contains3(voxel, pt) > num3s):
            #do the normal thing but don't create more voxels
            for pixel in voxel:
                x = pixel[0]
                y = pixel[1]
                z = pixel[2]
                if ((pt[x][y][z] == 0) & (x >= 0) & (x < img_height)):
                    if ((img[x][y][z] <= thresh_max) & (img[x][y][z] >= thresh_min)):
                        pt[x][y][z] = 255
                    else:
                        pt[x][y][z] = 3
        else:
        #do the normal thing
            for pixel in voxel:
                x = pixel[0]
                y = pixel[1]
                z = pixel[2]
                if ((pt[x][y][z] == 0) & (x >= 0) & (x < img_height)):
                    if ((img[x][y][z] <= thresh_max) & (img[x][y][z] >= thresh_min)):
                        pt[x][y][z] = 255
                        L1 = surround_with_ones_larger([x, y, z], L1)
                    else:
                        pt[x][y][z] = 3
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
def find_seed(img):
    x_min = int(np.median((np.where(img < 10))[0]))
    y_min = int(np.median((np.where(img < 10))[1]))
    seed = [10, x_min, y_min]
    print("Seed point: {}".format(seed))
    return seed
def mask_color_img(data, max, min, alpha=0.8):
    def threshold(img2, max, min):
        rows, cols = img2.shape
        color_mask = np.zeros((rows, cols, 3))
        for i in range(rows):
            for j in range(cols):
                if ((img2[i,j] < max) & (img2[i,j] >= min)):
                    color_mask[i, j] = [0, 1, 0]
        return color_mask
    color_mask = threshold(data, max, min)
    img_color = np.dstack((data,data,data))
    img_hsv = color.rgb2hsv(img_color)
    color_mask_hsv = color.rgb2hsv(color_mask)
    img_hsv[..., 0] = color_mask_hsv[..., 0]
    img_hsv[..., 1] = color_mask_hsv[..., 1] * alpha

    img_masked = color.hsv2rgb(img_hsv)

    plt.imshow(img_masked)
    plt.show()
def mask_color_img2(filtered_img, orig_img):
    print("Generating Overlayed Image After Connected Threshold")
    data_filt = filtered_img.copy()
    data_orig = orig_img.copy()
    height, rows, cols = data_filt.shape
    for k in range(height):
        for i in range(rows):
            for j in range(cols):
                if ((data_filt[k][i][j]) == 255):
                    data_orig[k][i][j] = 255
    sample_stack(data_orig, 6, 6, 0, 8, False)

Test = False
id = 0
print("ID: {}".format(id))

img1 = []
counter = 0
datapath = "jpg_images/"
#img0 = Image.open("jpg_images/maskedimage" + str(0) + ".jpg")
for f in glob.glob('/Users/paulmccabe/Desktop/jpg images id0/*.jpg'):
    path = "/Users/paulmccabe/Desktop/jpg images id0/maskedimage" + str(counter) + ".jpg"
    img0 = Image.open(path).convert('L')
    img1.append(array(img0))
    counter += 1
print("Counter: " + str(counter))
imgs_to_process = np.stack([s for s in img1])
print(imgs_to_process.shape)
np.save("/Users/paulmccabe/Desktop" + "/Segmentation Project/nplungs_%d.npy" % (id), imgs_to_process)

#imgs_to_process = cv2.GaussianBlur(imgs_to_process1)


#print("Blurring Image...")
#imgs_to_process = scipy.ndimage.filters.gaussian_filter(imgs_to_process1, .1)
#sample_stack(imgs_to_process[3],2,2,0,1,True)
imgs_to_test = np.stack([imgs_to_process[0], imgs_to_process[1], imgs_to_process[2], imgs_to_process[3], imgs_to_process[4]])
#imgs_to_test_blurred = cv2.GaussianBlur(imgs_to_test,(5,5),0)
#sample_stack(imgs_to_test_blurred[0], 2, 2, 0, 1, False)


#threshold of black should be around 50
#gray area has value of 106

#for i in range(0, 432, 4):
#    print("Data{}: {}".format(i, imgs_to_test[0][i][199]))
print("Region Growing...")
seed = find_seed(imgs_to_process[10])
thresh_min = 0
thresh_max = 50
num_allowed = 2
if(Test):
    imgs_after_rg = connected_threshold(imgs_to_test, seed, thresh_min, thresh_max, num_allowed)
else:
    imgs_after_rg = connected_threshold(imgs_to_process, seed, thresh_min, thresh_max, num_allowed)
#for i in range(0, 432, 4):
#    print("Data{}: {}".format(i, imgs_after_rg[0][i][199]))
#sample_stack(imgs_after_rg[2], 2, 2, 0, 1, True)
#print("Plotting Pixel Data...")
#plot_pixel_count(imgs_after_rg, 4)


#sample_stack(imgs_after_rg[120],2,2,0,1, True)
if(Test):
    pass
else:
    output_path = "/Users/paulmccabe/Desktop"
    np.save(output_path + "/trachea_to_model_%d.npy" % (id), imgs_after_rg)
if(Test):
    plot_pixel_count(imgs_after_rg, 4)
else:
    plot_pixel_count(imgs_after_rg, len(img1))

#mask_color_img2(imgs_after_rg, imgs_to_process)


"""
mask = np.ones_like(imgs_to_test)
print("imgs_to_test: {}".format(mask.ndim))
b = np.array([[250, 200, 200, 255, 250], [250, 0, 0, 1, 250], [250, 0, 1, 20, 250], [250, 0, 1, 20, 250], [250, 250, 250, 250, 250]])
a = np.stack([b, b, b])
seed = np.array([0, 3, 3])
test = connected_threshold(a, seed, 0, 60)
print(test)
#id = 0
#mask_to_process = np.load("/Users/paulmccabe/Desktop/" + "justmask_%d.npy" % (id))
#sample_stack(mask150,2, 2, 0, 1, True)
#for i in range(0, 432, 1):
#    print("Mask: {}".format(mask150[60][60]))
#print(np.where(Test_imgs[0] == np.amin(Test_imgs[0])))
#Min at 189, 191 or 204, 206
#something = grow(imgs_to_test, (0, 189, 191), 1)
#int_type = something.astype(int)
#sample_stack(int_type[0], 2, 2, 0, 1, True)
seed = np.array([0, 189, 191])
something = region_grow(imgs_to_test, mask, seed, 5, 30, 40, 255)
sample_stack(something[0], 2, 2, 0, 1, True)


id = 0
output_path = "/Users/paulmccabe/Desktop"
np.save(output_path + "/trachea_to_model_test_%d.npy" % (id), int_type)

grow_to_process = np.load(output_path + "/trachea_to_model_test_%d.npy" % (id))
for i in range(432):
    print(grow_to_process[0][i][100])
#v, f = make_mesh(grow_to_process, 255, 2)
#plotly_3d(v, f)

#sample_stack(something[150], 2, 2, 0, 1, True)

#for i in range(432):
#    print("Data Value{}: {}".format(i, something[150][i][100]))
"""
