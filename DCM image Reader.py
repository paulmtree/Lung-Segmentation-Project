import pydicom
import matplotlib.pyplot as plt
from pydicom.data import get_testdata_files
import glob
from PIL import Image

import numpy as np
import os

filename_array = get_testdata_files("CTScans/*.dcm")
dataset = pydicom.dcmread(filename_array[1])
print(dataset.get('SliceLocation', "(missing)"))
pixels = dataset.pixel_array
plt.imshow(dataset.pixel_array, cmap=plt.cm.bone)
plt.show()

#img1 = Image.fromarray(pixels)
#img1.show()

for f in glob.glob('CTScans/*.dcm'):
    img0 = Image.open(f)
    img0.show()
