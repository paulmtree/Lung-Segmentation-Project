import pydicom
import numpy as np
import matplotlib.pyplot as plt
import sys
import glob
from PIL import Image
from numpy import array

id = 0

img1 = []
counter = 0
datapath = "jpg_images/"
#img0 = Image.open("jpg_images/maskedimage" + str(0) + ".jpg")
for f in glob.glob('/Users/paulmccabe/Desktop/asdf/*.jpg'):
    path = "/Users/paulmccabe/Desktop/asdf/maskedimage" + str(counter) + ".jpg"
    img0 = Image.open(path).convert('L')
    img1.append(array(img0))
    counter += 1
print("Counter: " + str(counter))
imgs_to_process = np.stack([s for s in img1])
print(imgs_to_process.shape)
plt.title("Convert to Numpy Test Image")
plt.imshow(imgs_to_process[150], cmap = 'gray')
plt.show()
np.save("/Users/paulmccabe/Desktop" + "/Segmentation Project/nplungs_%d.npy" % (id), imgs_to_process)