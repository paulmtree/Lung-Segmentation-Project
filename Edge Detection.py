from PIL import Image, ImageFilter
import numpy as np
import glob
from numpy import array
import matplotlib.pyplot as plt
from skimage import morphology
import scipy.ndimage

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
"""
datapath = "jpg_images/"
img0 = Image.open("jpg_images/maskedimage" + str(0) + ".jpg")
counter = 0
img1 = []
for f in glob.glob('/Users/paulmccabe/Desktop/jpg images/*.jpg'):
    path = "jpg_images/maskedimage" + str(counter) + ".jpg"
    img0 = Image.open(path).convert('L')
    img1.append(array(img0))
    counter += 1
print("Counter: " + str(counter))
imgs_to_process_orig = np.stack([s for s in img1])
"""
id = 2

imgs = np.load("/Users/paulmccabe/Desktop/Segmentation Project/" + "justmask_%d.npy" % (id))
counter = 0
print("Saving as jpg Images...")
for img in imgs:
    scipy.misc.imsave('/Users/paulmccabe/Desktop/Segmentation Project' + '/jpg mask images/justmask{}.jpg'.format(counter), img)
    counter += 1
counter = 0
#print("Re-Importing jpg Images...")
#for f in glob.glob('/Users/paulmccabe/Desktop/Segmentation Project/jpg mask images/*.jpg'):
#    path = "jpg_images/maskedimage" + str(counter) + ".jpg"
#    img0 = Image.open(path).convert('L')
#    img1.append(array(img0))
#    counter += 1
imgs[imgs == 1] = 255
list = []
for img in imgs:
    PIL_img = Image.fromarray(img.astype('uint8'))
    PIL_edge = PIL_img.filter(ImageFilter.FIND_EDGES)
    np_img = array(PIL_edge)
    dilation = morphology.dilation(np_img, np.ones([4,4]))
    list.append(dilation)

imgs_after_processing = np.stack([s for s in list])

np.save("/Users/paulmccabe/Desktop/Segmentation Project" + "/justedge_%d.npy" % (id), imgs_after_processing[:284])

#sample_stack(np_img)