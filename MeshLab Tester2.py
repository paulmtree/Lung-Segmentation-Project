import numpy as np
from mayavi import mlab
import matplotlib.pyplot as plt
from PIL import Image
from numpy import array
import os, sys
def plot_img(img, title):
    plt.title(title)
    plt.imshow(img, cmap='gray')
    plt.show()

id = 0




tumor_img_rg = np.load("/Users/paulmccabe/Desktop/Segmentation Project/" + "Thresh_tumor_%d.npy" % (id))
tumor_img = np.load("/Users/paulmccabe/Desktop/Segmentation Project/" + "Thresh_roi_%d.npy" % (id))
print(tumor_img_rg.shape)

mlab.figure(bgcolor=(0,0,0), size = (400, 400))
src = mlab.pipeline.scalar_field(tumor_img_rg)
src2 = mlab.pipeline.scalar_field(tumor_img)

#src.spacing = [1, 1, 1]
#src.update_image_data = True

blur = mlab.pipeline.user_defined(src, filter='ImageGaussianSmooth')
blur2 = mlab.pipeline.user_defined(src2, filter='ImageGaussianSmooth')
voi = mlab.pipeline.extract_grid(blur)
voi2 = mlab.pipeline.extract_grid(blur2)

#mlab.pipeline.iso_surface(voi, color=(1.0, 0.1, 0.1))
mlab.pipeline.iso_surface(voi2, color=(0.0, 0.5, 0.7), opacity = .7)
mlab.view(90,90, 400)
mlab.roll(-90)

mlab.show()