from PIL import Image
from numpy import array
import os
import glob
import numpy as np
from sklearn.cluster import KMeans

from matplotlib import pyplot as plt
from PIL import ImageFilter

def threshold_filter(data, w, h):
    for i in range(w):
        for j in range(h):
            if (data[i][j] < 130):
                data[i][j] = 0
            elif (data[i][j] > 240):
                data[i][j] = 250
            else:
                data[i][j] = 180
    return data
def pointwise_transform(data, w, h, T1, T2):
    slope = 255 / (T2 - T1)
    output = data
    for i in range(h):
        for j in range(w):
            if (data[i][j] < T1):
                output[i][j] = 0
            elif data[i][j] > T2:
                output[i][j] = 240
            else:
                output[i][j] = (data[i][j]*slope) - T1
    return output

#Set preconditions
file_type = 'jpg'
def kmeans_filter(img1):
    kmeans = KMeans(n_clusters=2).fit(img1)
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)
    img2 = np.where(img1 < threshold, 240, 30)
    return img2

for f in glob.glob('ImageScans/*.{}'.format(file_type)):
    if f.endswith('.jpg') == True:
        img0 = Image.open(f).convert('L')
        img1 = img0
        fn, fext = os.path.splitext(f)
        #img1 = img1.filter(ImageFilter.EDGE_ENHANCE)
        # image.show()
        #img1 = img1.filter(ImageFilter.MedianFilter(3))
        data = array(img1)
        w, h = img1.size

#data analytics

        #data = threshold_filter(data, w, h)
        data = pointwise_transform(data, w, h, 80, 220)
        # save image
        #img1 = Image.fromarray(data)

#Filter
        img1 = Image.fromarray(kmeans_filter(data))
        #img3 = img2.filter(ImageFilter.FIND_EDGES)
        #img3 = img2.filter(ImageFilter.MedianFilter)
        #histogram = img2.histogram(mask=None, extrema=None)
        #x = range(0, 256)
        #plt.plot(x, histogram)
        #plt.show()
        #ID = fn.replace('ImageScans/', '')
        #print(ID)
        Image.fromarray(np.hstack((np.array(img0), np.array(img1)))).show()
        #Image.fromarray(np.hstack((np.array(img0), np.array(img1), np.array(img2)))).save("ImageScanResults/{}a.jpg".format(ID))
        img1.save("ImageScanResults/receipt_DFA.jpg")