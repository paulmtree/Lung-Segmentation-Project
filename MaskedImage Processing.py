import numpy as np
from skimage import measure
from plotly.offline import download_plotlyjs, init_notebook_mode, plot
from plotly.tools import FigureFactory as FF
from pydicom.data import get_testdata_files
import pydicom
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image
from skimage import morphology
import SimpleITK as sitk
import pylab

def sitk_show(img, title=None, margin=0.05, dpi=40 ):
    nda = sitk.GetArrayFromImage(img)
    spacing = img.GetSpacing()
    figsize = (1 + margin) * nda.shape[0] / dpi, (1 + margin) * nda.shape[1] / dpi
    extent = (0, nda.shape[1]*spacing[1], nda.shape[0]*spacing[0], 0)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])

    plt.set_cmap("gray")
    ax.imshow(nda, extent=extent, cmap = 'gray')

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
def show_4_images(imgs_to_process):
    test_images = []
    img1 = (imgs_to_process[130])
    img2 = (imgs_to_process[133])
    img3 = (imgs_to_process[136])
    img4 = (imgs_to_process[139])
    threshold = -290
    for x in range(0, 432, 1):
            print("Before: {}".format(img1[x][150]))
            for y in range(0, 432, 1):
                img1[x][y] += 1000
                if img1[x][y] < threshold + 1000:
                    img1[x][y] = 1
                else:
                    img1[x][y] = 0
            print("After: {}".format(img1[x][150]))
    eroded = morphology.erosion(img1, np.ones([3, 3]))
    img2 = eroded
    dilation = morphology.dilation(eroded, np.ones([4, 4]))
    img3 = dilation

    #for i in range(0, 432, 1):
    #    img1[i][174] = -80
    #    print("{}. {}".format(i, img1[i][175]))
    #    img1[150][i] = 80
    # for i in range(0, 432, 1):
    test_images.append(img1)
    test_images.append(img2)
    test_images.append(img3)
    test_images.append(img4)

    sample_stack(test_images, 2, 2, 0, 1)
def make_mesh(image, threshold = 300, step_size = 2):
    print "Transposing surface"
    p = image.transpose(2, 1, 0)
    verts, faces, norm, val = measure.marching_cubes(p, threshold, step_size=step_size, allow_degenerate=True)

    return verts, faces
def plot_3D(verts, faces):
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
def get_pixels_hu(scans):
    image = np.stack([s.pixel_array for s in scans])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 1
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope

    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)

    image += np.int16(intercept)

    return np.array(image, dtype=np.int16)
def get_trachea(scans):
    label_Airway = 1
    label_Else = 2
    threshold = -270
    threshold_scans = []
    trachea_scans = []
    #threshold image
    if (False):
        print("Thresholding...")
        for x in range(0, 432, 1):
            for y in range(0, 432, 1):
                img[x][y] += 1000
                if img[x][y] < threshold + 1000:
                    img[x][y] = 1
                else:
                    img[x][y] = 0
    if (True):
        nda = scans[150]
        
        #sample_stack(nda4, 2, 2, 0, 1, True)
        #Image.fromarray(np.hstack((np.array(nda), np.array(nda2)))).show()
        #plt.imshow(nda)
        #for i in range(0, 400, 4):
            #print("Data Value: {}".format(nda[i][200]))

    #Various Methods
        #which clusters have nothing inside them? maybe measure mean values
        #choose the smallest of the 3 kmean clusters
        #Use AI to fill in volume
    if (False):
        fig, ax = plt.subplots(3, 2, figsize=[12, 12])
        ax[0, 0].set_title("Original", cmap = 'grey')
        ax[0, 0].imshow()
        ax[0, 0].axis('off')
        ax[0, 1].set_title("Threshold", cmap = 'grey')
        ax[0, 1].imshow()
        ax[0, 1].axis('off')
        ax[1, 0].set_title("")
    return img_smooth

id = 0
imgs_to_process = np.load("/Users/paulmccabe/Desktop/" + "maskedimages_forsure_%d.npy" % (id))
number_of_images = len(imgs_to_process)
voxels = 432

#for i in range(0, 400, 1):
#    print("Data Value: {}".format(imgs_to_process[130][i][200]))

#for i in range(0, 400, 1):
#    print("Data Value in H units: {}".format(imgs_to_process[130][i][200]))

#Image.fromarray(np.hstack()).show()

#float64 type
#verts, faces = make_mesh(imgs_to_process)
#plot_3D(verts, faces)

for img in imgs_to_process:
    img *= 1000.0/img.max()
#threshold around -270
trachea_3D_array = get_trachea(imgs_to_process)
#show_4_images(imgs_to_process)



