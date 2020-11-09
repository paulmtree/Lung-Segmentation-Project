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
import vtk

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

        if img[pt] >= img[imin:imax + 1, jmin:jmax + 1, kmin:kmax + 1].mean():
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
def find_seed(img, thresh_min, thresh_max):
    """
    fill array with zeros
    set the seed value to 255 and pixels around it to 1
    if there are ones, check that value.
        if <threshold set to 255 and pixels around it equal to 1 if equal to zero
        if >threshold set to 3
    loop back to if statement

    create center
    create list of ones locations around center
    for each array of points in list of ones, try setting pt to that value at that point
    try
        if pt[][][] = 0 and if images[][][] is less than threshold, pass as new center and set pt = 255
            else set pt[][][] = 3
        else pt[][][] == 1, 3, or 255 do nothing
        pop point
        if the number of 255's is greater than a certain value
    except:
        pop point
        pass
    """
    def surround_with_ones(L1, center):
        # l1 = list of ones
        x = center[0]
        y = center[1]
        L1.append([x + 1, y])
        L1.append([x - 1, y])
        L1.append([x, y + 1])
        L1.append([x, y - 1])
        return l1
    pt = np.zeros_like(img)
    temp_Arr = np.where(img == np.amin(img))
    xseed = temp_Arr[0]
    yseed = temp_Arr[1]
    print(xseed)
    print(yseed)
    #pt = array of zeros slowly filled with 255, 0, or 3
    pt[xseed][yseed] = 255
    center = [xseed, yseed]
    l1 = []
    l1 = surround_with_ones(l1, center)
    while (len(l1) > 0):
        points = l1.pop()
        x = points[0]
        y = points[1]
        try:
            if (pt[x][y] == 0):
                if ((img[x][y] <= thresh_max) & (img[x][y] >= thresh_min)):
                    pt[x][y] = 255
                    l1 = surround_with_ones(l1, [x, y])
                else:
                    pt[x][y] = 3
        except:
            pass

    #l1 = list of ones
    return pt
def make_mesh(image, threshold=300, step_size=.5):
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
                            backgroundcolor='rgb(0, 0, 0)',
                            title="Interactive Visualization")
    plot(fig)
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
    sample_stack(data_orig, 2, 2, 0, 50, False)
id = 1
imgs_after_rg1 = np.load("/Users/paulmccabe/Desktop/" + "trachea_to_model_1.npy")
imgs_after_rg = np.flip(imgs_after_rg1, 0)
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
imgs_to_process_orig = np.stack([s for s in img1])


#or i in range(len(imgs_to_process)):
#test_1 = connected_threshold(imgs_to_process[0], 0, 70)

#sample_stack(Test,2,2,0,1,True)
#imgs_to_process[0][196][205] = 0
#sample_stack(imgs_to_process,6,6,60,3)
v, f = make_mesh(imgs_after_rg, 254, 1)
plotly_3d(v, f)

mask_color_img2(imgs_after_rg1, imgs_to_process_orig)


"""
nx = np.size(imgs_to_process, 0)
ny = np.size(imgs_to_process, 1)
nz = np.size(imgs_to_process, 2)
tx = np.linspace(-3,3,nx)
ty = np.linspace(-3,3,ny)
tz = np.linspace(-3,3,nz)
x,y,z = np.meshgrid(tx,ty,tz)
w =
"""



"""
a = np.array([100, 250, 250, 250, 250])
b = np.array([250, 2, 3, 4, 250])
c = np.array([160, 2, 3, 4, 250])
d = np.array([250, 2, 3, 4, 250])
e = np.array([250, 250, 250, 250, 250])
f = np.array([a, b, c, d, e])
print(f)
test = np.stack([f, f, f, f, f])
seed = np.array([0, 2, 2])
print(test[0][2][2])
pt = connected_threshold(test, seed, 0, 60)
print(pt)


186624
a = np.array([100, 250, 250, 250, 250])
b = np.array([250, 2, 3, 4, 250])
c = np.array([160, 2, 3, 4, 250])
d = np.array([250, 2, 3, 4, 250])
e = np.array([250, 250, 250, 250, 250])
test = np.stack([a, b, c, d, e])
print(test)
list = []
list.append(a)
list.append(b)
list.append(c)
list.append(d)

print(list.pop()[4])

id = 0
imgs_to_process = np.load("/Users/paulmccabe/Desktop/" + "trachea_to_model_%d.npy" % (id))

print(np.amin(imgs_to_process[150]))
print(np.amax(imgs_to_process[150]))
img150 = imgs_to_process[150]
#print("Before Smoothing: {}".format(img150[200][80]))

print(np.size(img150))



#img3 = img2.filter(ImageFilter.)

#imgCoord = copy.deepcopy(img150)
# [y,x] 200, 80, range -300, -50
#print("After Smoothing: {}".format(img150[200][80]))
"""

