for i in range(w):
    for j in range(h):
        if (data[i][j] < 100):
            data[i][j] = 0;
        elif (data[i][j] > 240):
            data[i][j] = 250
        else:
            data[i][j] = 180;
import numpy as np
import png, os, pydicom
PNG = True
# Specify the .dcm folder path
source_folder = "/Users/paulmccabe/Desktop/LungCT-Diagnosis/R_014/04-11-1998-Diagnostic Pre-Surgery Contrast Enhanced CT-02847/2- NONE -62871"
# Specify the output jpg/png folder path
output_folder = "/Users/paulmccabe/Desktop/LungCT-Diagnosis/R_014/04-11-1998-Diagnostic Pre-Surgery Contrast Enhanced CT-02847/2- NONE -62871"
images_path = os.listdir(source_folder)

list_of_files = os.listdir(source_folder)
for file in list_of_files:
    try:
        ds = pydicom.dcmread(os.path.join(source_folder,file))
        shape = ds.pixel_array.shape
            # Convert to float to avoid overflow or underflow losses.
        image_2d = ds.pixel_array.astype(float)

            # Rescaling grey scale between 0-255
        image_2d_scaled = (np.maximum(image_2d,0) / image_2d.max()) * 255.0

            # Convert to uint
        image_2d_scaled = np.uint8(image_2d_scaled)

            # Write the PNG file
        with open(os.path.join(output_folder,file)+'.png' , 'wb') as png_file:
            w = png.Writer(shape[1], shape[0], greyscale=True)
            w.write(png_file, image_2d_scaled)
    except:
        print('Could not convert: ', file)




"/Users/paulmccabe/Desktop/LungCT-Diagnosis/R_004/06-30-1997-Diagnostic Pre-Surgery Contrast Enhanced CT-71813/3- NONE -29295"
"/Users/paulmccabe/Desktop/LungCT-Diagnosis/R_006/09-24-1998-Diagnostic Pre-Surgery Contrast Enhanced CT-41946/2- NONE -71225"
"/Users/paulmccabe/Desktop/LungCT-Diagnosis/R_014/04-11-1998-Diagnostic Pre-Surgery Contrast Enhanced CT-02847/2- NONE -62871"
good ct scan "/Users/paulmccabe/Desktop/LungCT-Diagnosis/R_006/09-24-1998-Diagnostic Pre-Surgery Contrast Enhanced CT-41946"

for f in range(0, length, 1):
    i = place_holder_list[f]
    filename = get_testdata_files("CTScans/*.dcm")[i]
    dataset = pydicom.dcmread(filename)
    print("Slice location...:", dataset.get('SliceLocation', "(missing)"))

    plt.imshow(dataset.pixel_array, cmap=plt.cm.bone)
    plt.show()
    i = i + 1

[2, 6, 5, 9, 12, 16, 20, 19, 24, 29, 26, 33, 37, 42, 54, 45, 44, 59, 47, 48, 49, 50, 51, 56, 55, 63, 58, 60, 66, 65, 70, 74, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75]

img1[img1 == -2000] = 0

    # Convert to Hounsfield units (HU)
    intercept = imgs_to_process[0].RescaleIntercept
    print("Intercept: {}".format(intercept))
    slope = imgs_to_process[0].RescaleSlope
    print("Slope: {}".format(slope))

    if slope != 1:
        img1 = slope * img1.astype(np.float64)
        img1 = img1.astype(np.int16)

    img1 += np.int16(intercept)
    # for i in range(0, 432, 1):
    test_images.append(img1)
    test_images.append(img2)
    test_images.append(img3)
    test_images.append(img4)

import numpy as np
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
    print "Transposing surface"
    p = image.transpose(2, 1, 0)

    print "Calculating surface"
    verts, faces, norm, val = measure.marching_cubes(p, threshold, step_size=step_size, allow_degenerate=True)
    return verts, faces
def plotly_3d(verts, faces):
    x, y, z = zip(*verts)

    print "Drawing"

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
def sample_stack(stack, rows=2, cols=2, start_with=127, show_every=3, display1 = False):
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
def connected_threshold(img, seed, thresh_min, thresh_max):

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
    xseed = seed[0]
    yseed = seed[1]
    zseed = seed[2]
    #pt = array of zeros slowly filled with 255, 0, or 3
    pt[xseed][yseed][zseed] = 255
    center = [xseed, yseed, zseed]
    l1 = []
    l1 = surround_with_ones(l1, center)
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
    xvals = np.arange(length)
    yvals = np.ones_like(xvals)
    for i in range(length):
        yvals[i] = np.count_nonzero(imgs_after_rg[i] == pixel_val)

    plt.plot(xvals, yvals, 'ro')
    plt.show()

img1 = []
counter = 0
datapath = "jpg_images/"
img0 = Image.open("jpg_images/maskedimage" + str(0) + ".jpg")

for f in glob.glob('/Users/paulmccabe/Desktop/jpg images/*.jpg'):
    path = "jpg_images/maskedimage" + str(counter) + ".jpg"
    img0 = Image.open(path).convert('L')
    img1.append(array(img0))
    counter += 1
print("Counter: " + str(counter))
imgs_to_process1 = np.stack([s for s in img1])
#imgs_to_process = cv2.GaussianBlur(imgs_to_process1)
print("Blurring Image...")
imgs_to_process = scipy.ndimage.filters.gaussian_filter(imgs_to_process1, .1)
sample_stack(imgs_to_process, 6, 6, 40, 3, False)
#sample_stack(imgs_to_process[3],2,2,0,1,True)
imgs_to_test = np.stack([imgs_to_process[0], imgs_to_process[1], imgs_to_process[2], imgs_to_process[3], imgs_to_process[4]])
#imgs_to_test_blurred = cv2.GaussianBlur(imgs_to_test,(5,5),0)
#sample_stack(imgs_to_test_blurred[0], 2, 2, 0, 1, False)


#threshold of black should be around 50
#gray area has value of 106
print("Region Growing...")
seed = np.array([0, 189, 199])
imgs_after_rg = connected_threshold(imgs_to_process, seed, 0, 30)
#sample_stack(imgs_after_rg[3],2,2,0,1,True)
#sample_stack(imgs_after_rg[2], 2, 2, 0, 1, True)
print("Plotting Pixel Data...")
plot_pixel_count(imgs_after_rg, len(img1))

id = 0
output_path = "/Users/paulmccabe/Desktop"
np.save(output_path + "/trachea_to_model_%d.npy" % (id), imgs_after_rg)



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

import numpy as np
from mayavi import mlab

id = 0
imgs_after_rg = np.load("/Users/paulmccabe/Desktop/" + "trachea_to_model_%d.npy" % (id))
h, r, c = imgs_after_rg.shape





mlab.figure(bgcolor=(0,0,0), size = (400, 400))
src = mlab.pipeline.scalar_field(imgs_after_rg)
#src.spacing = [1, 1, 1]
#src.update_image_data = True

blur = mlab.pipeline.user_defined(src, filter='ImageGaussianSmooth')
voi = mlab.pipeline.extract_grid(blur)
voi.trait_set(x_min=0, x_max=300, y_min=0, y_max=300, z_min=0, z_max=300)

mlab.pipeline.iso_surface(voi)

mlab.view(90,90, 400)
mlab.roll(-90)

mlab.show()

#Bronchial Grower
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
def find_seed(imgs_to_process, img_mask, seed_index = 188):
    test_img = copy.deepcopy(imgs_to_process[seed_index])
    test_img_no_edit = copy.deepcopy(imgs_to_process[seed_index])
    test_img[test_img > 0] = 1
    row_size, col_size = test_img.shape

    labels = measure.label(img_mask[seed_index])  # Different labels are displayed in different colors
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
        sample_stack(one_lung)
        x_min = int(np.median((np.where(one_lung > 70))[0]))
        y_min = int(np.median((np.where(one_lung > 70))[1]))
        seed_thresh = 80
        #If seed lands on a dark spot, re-check using different threshold values until correct
        while (test_img_no_edit[x_min][y_min] < 30):
            x_min = int(np.median((np.where(one_lung > seed_thresh))[0]))
            y_min = int(np.median((np.where(one_lung > seed_thresh))[1]))
            print("Let's kick it up a notch")
            seed_thresh += 10
        seed = [x_min, y_min]
        print("Seed Point: {}".format(seed))
        seed_list.append(seed)
    return seed_list
def connected_threshold(img, img_mask, thresh_min, thresh_max):
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

    left_seed, right_seed = find_seed(img, img_mask, seed_index=188)
    xseed = 150
    yseed = left_seed[0]
    zseed = left_seed[1]
    #pt = array of zeros slowly filled with 255, 0, or 3
    pt[xseed][yseed][zseed] = 255
    print("Left Seed Pixel Value: {}".format(img[xseed][yseed][zseed]))
    center = [xseed, yseed, zseed]
    l1 = surround_with_ones(l1, center)

    xseed = 150
    yseed = right_seed[0]
    zseed = right_seed[1]
    # pt = array of zeros slowly filled with 255, 0, or 3
    pt[xseed][yseed][zseed] = 255
    print("Right Seed Pixel Value: {}".format(img[xseed][yseed][zseed]))
    center = [xseed, yseed, zseed]
    l1 = surround_with_ones(l1, center)
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

id = 3
out_path = "/Users/paulmccabe/Desktop/Segmentation Project/"
imgs_to_process = np.load(out_path + "nplungs_%d.npy" % (id))
np_mask = np.load(out_path + "justmask_%d.npy" % (id))

#sample_stack(imgs_to_process[150])
eroded_to_process = copy.deepcopy(imgs_to_process)
sample_stack(np_mask[188])
sample_stack(imgs_to_process[188])
print("Eroding Masks...")
for i in range(0, np.size(imgs_to_process, 0), 1):
    np_mask_smaller = morphology.erosion(np_mask[i], np.ones([19, 19]))
    eroded_to_process[i] = np_mask_smaller * imgs_to_process[i]
    #scipy.misc.imsave(out_path + "eroded_tp_%d.jpg" % (i), eroded_to_process[i])

sample_stack(eroded_to_process[188])
print("Region Growing...")
thresh_min = 75
thresh_max = 255
imgs_after_rg = connected_threshold(eroded_to_process, np_mask, thresh_min, thresh_max)

print("Plotting Pixels...")
plot_pixel_count(imgs_after_rg, len(imgs_after_rg))
sample_stack(imgs_after_rg[150])

print("Saving To Numpy Array...")
np.save(out_path + "bronchioles_after_rg_%d.npy" % (id),imgs_after_rg)
