import numpy as np
from mayavi import mlab
import matplotlib.pyplot as plt
from PIL import Image
from numpy import array
import os, sys
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

id = 0




imgs_after_rg = np.load("/Users/paulmccabe/Desktop/" + "trachea_to_model_%d.npy" % (id))
np_edge = np.load("/Users/paulmccabe/Desktop/Segmentation Project/" + "justedge_%d.npy" % (id))
bronchioles_after_rg = np.load("/Users/paulmccabe/Desktop/Segmentation Project/" + "bronchioles_after_rg_%d.npy" % (id))
tumor_after_rg = np.load("/Users/paulmccabe/Desktop/Segmentation Project/" + "Tumor_in_lungs_%d.npy" % (id))
print(np_edge.shape)
print(imgs_after_rg.shape)
print(bronchioles_after_rg.shape)
print(tumor_after_rg.shape)

PIL_edge = Image.fromarray(np_edge[0])
np_edge_new = array(PIL_edge)
#sample_stack(np_edge_new)
#sample_stack(imgs_after_rg[0])
#sample_stack(bronchioles_after_rg[45])

mlab.figure(bgcolor=(0,0,0), size = (400, 400))
src = mlab.pipeline.scalar_field(imgs_after_rg)
src_outer = mlab.pipeline.scalar_field(np_edge)
src_bronchioles = mlab.pipeline.scalar_field(bronchioles_after_rg)
src_tumor =  mlab.pipeline.scalar_field(tumor_after_rg)
#src.spacing = [1, 1, 1]
#src.update_image_data = True

blur = mlab.pipeline.user_defined(src, filter='ImageGaussianSmooth')
voi = mlab.pipeline.extract_grid(blur)
blur2 = mlab.pipeline.user_defined(src_outer, filter = 'ImageGaussianSmooth')
voi_outer = mlab.pipeline.extract_grid(blur2)
blur3 = mlab.pipeline.user_defined(src_bronchioles, filter='ImageGaussianSmooth')
voi_bronchioles = mlab.pipeline.extract_grid(blur3)
blur4 = mlab.pipeline.user_defined(src_tumor, filter='ImageGaussianSmooth')
voi_tumor = mlab.pipeline.extract_grid(blur4)

mlab.pipeline.iso_surface(voi)
mlab.pipeline.iso_surface(voi_outer, opacity = .05, colormap = 'gray')
mlab.pipeline.iso_surface(voi_bronchioles, colormap='winter')
mlab.pipeline.iso_surface(voi_tumor, color=(1.0, 0.1, 0.1))

mlab.view(90,90, 400)
mlab.roll(-90)

mlab.show()