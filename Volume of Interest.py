import numpy as np
from numpy import array
from skimage import morphology
import matplotlib.pyplot as plt
import copy
from sklearn.cluster import KMeans
from skimage import measure
from PIL import Image, ImageDraw
def seperate_roi(imgs_to_process, bottom_left, top_right, id, out_path, lung_mask):
    print("Seperating ROI...")
    zsize = imgs_to_process.shape[0]
    ysize = imgs_to_process.shape[1]
    xsize = imgs_to_process.shape[2]
    reduced_roi = np.zeros((zsize, bottom_left[0] - top_right[0]+1,top_right[1]-bottom_left[1]+1))
    imgs_to_process.shape[0]

    for z in range(0, zsize, 1):
        np_mask_smaller = morphology.erosion(lung_mask[z], np.ones([17, 17]))
        slice = (np_mask_smaller * imgs_to_process[z]).astype(int)

        small_slice = reduced_roi[z]
        for y in range(0, ysize, 1):
            for x in range(0, xsize, 1):
                if ((x >= bottom_left[1]) & (x <= top_right[1]) & (y >= top_right[0]) & (y <= bottom_left[0])):
                    small_slice[y-top_right[0]][x-bottom_left[1]] = slice[y][x]
                else:
                    slice[y][x] = 0
        reduced_roi[z] = small_slice
        imgs_to_process[z] = slice
    np.save(out_path + "Tumor_roi_%d.npy" % (id), imgs_to_process)
    np.save(out_path + "Reduced Tumor_roi_%d.npy" % (id), reduced_roi)
    plt.title("Isolated Tumor")
    plt.imshow(imgs_to_process[229], cmap='gray')
    plt.show()
    return imgs_to_process, reduced_roi
def plot_img(img, title):
    print("Plotting Image...")
    plt.title(title)
    plt.imshow(img, cmap='gray')
    plt.show()
def threshold_kmeans(middle):
    kmeans = KMeans(n_clusters=2).fit(np.reshape(middle, [np.prod(middle.shape), 1]))
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)
    thresh_img = np.where(middle < threshold, 0, 255)
    return thresh_img
def threshold_reg(img, min_intensity):
    thresh_img = np.where(img < min_intensity, 0, 255)
    return thresh_img
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
def show_roi(RGB_img, bottom_left, top_right):
    print("Displaying ROI...")
    # for the 4 lines of the box, make them red
    # left vertical
    for i in range(top_right[0], bottom_left[0], 1):
        RGB_img[i][bottom_left[1]][0] = 255
        RGB_img[i][bottom_left[1]][1] = 0
        RGB_img[i][bottom_left[1]][2] = 0
    # right vertical
    for i in range(top_right[0], bottom_left[0], 1):
        RGB_img[i][top_right[1]][0] = 255
        RGB_img[i][top_right[1]][1] = 0
        RGB_img[i][top_right[1]][2] = 0
    # top horizontal
    for i in range(bottom_left[1], top_right[1], 1):
        RGB_img[top_right[0]][i][0] = 255
        RGB_img[top_right[0]][i][1] = 0
        RGB_img[top_right[0]][i][2] = 0

    # bottom horizontal
    for j in range(bottom_left[1], top_right[1], 1):
        RGB_img[340][j][0] = 255
        RGB_img[340][j][1] = 0
        RGB_img[340][j][2] = 0

    RGB_img[340][290][0] = 200
    RGB_img[340][290][1] = 100
    RGB_img[340][290][2] = 50

    plt.title('Region of Interest')
    plt.imshow(RGB_img)
    plt.show()
def apply_lung_mask(imgs, lungs):
    z = imgs.shape[0]
    for i in range(0, z, 1):
        imgs[i] = imgs[i]*lungs[i]
    return imgs
def connected_threshold(img, seed, thresh_min):

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
    print("Region Growing...")
    pt = np.zeros_like(img)
    xseed = int(np.median((np.where(img[seed] > thresh_min))[0]))
    yseed = int(np.median((np.where(img[seed] > thresh_min))[1]))
    zseed = seed
    #pt = array of zeros slowly filled with 255, 0, or 3
    pt[zseed][xseed][yseed] = 255
    center = [zseed, xseed, yseed]
    l1 = []
    l1 = surround_with_ones(l1, center)
    while (len(l1) > 0):
        points = l1.pop()
        x = points[0]
        y = points[1]
        z = points[2]

        try:
            if (pt[x][y][z] == 0):
                if (img[x][y][z] >= thresh_min):
                    pt[x][y][z] = 255
                    l1 = surround_with_ones(l1, [x, y, z])
                else:
                    pt[x][y][z] = 3
        except:
            pass

    #l1 = list of ones
    return pt
def place_into_lungs(id, tumor, bottom_left, top_right):
    imgs = np.load(out_path + "nplungs_%d.npy" % (id))

    entire_lungs = np.zeros_like(imgs)

    print("Re-placing ROI...")
    zsize = tumor.shape[0]
    ysize = tumor.shape[1]
    xsize = tumor.shape[2]

    for z in range(0, zsize, 1):
        slice = tumor[z]
        entire_lungs_slice = entire_lungs[z]
        for y in range(0, ysize, 1):
            for x in range(0, xsize, 1):
                entire_lungs_slice[y+bottom_left[1]][x+top_right[0]] = slice[y][x]
        entire_lungs[z] = entire_lungs_slice
    np.save(out_path + "Tumor_in_lungs_%d.npy" % (id), entire_lungs)
    plt.title("Tumor in Lungs")
    plt.imshow(entire_lungs[229], cmap='gray')
    plt.show()
    return entire_lungs
def connected_threshold_larger(img, seed, thresh_min, thresh_max, num3s):
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
                    #print(img[x][y][z])
        #print("Total 3s: "+ str(total_3s))
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
    seed = [seed, int(np.median((np.where(img[seed] > thresh_min))[0])), int(np.median((np.where(img[seed] > thresh_min))[1]))]
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

id = 0
seed_index = 229
min_intensity = 45
bottom_left = [340,275]
top_right = [300,310]
out_path = "/Users/paulmccabe/Desktop/Segmentation Project/"

"""
imgs_to_process = np.load(out_path + "nplungs_%d.npy" % (id))
np_mask = np.load(out_path + "justmask_%d.npy" % (id))
#id0 tumor at 229
tum_slice = copy.deepcopy(imgs_to_process[seed_index])

plt.title('Grayscale Image')
plt.imshow(tum_slice, cmap = 'gray')
plt.show()
RGB_img = np.dstack((tum_slice, tum_slice, tum_slice))

#show_roi(RGB_img, bottom_left, top_right)
#seperate out roi into array

sep_roi, reduced_roi = seperate_roi(imgs_to_process, bottom_left, top_right, id, out_path, np_mask)
#sep_roi = np.load(out_path + "Tumor_roi_%d.npy" % (id))
#reduced_roi = np.load(out_path + "Reduced Tumor_roi_%d.npy" % (id))
plt.title('Reduced ROI')
plt.imshow(reduced_roi[seed_index], cmap = 'gray')
plt.show()
#thresh_img = threshold_kmeans(reduced_roi)
#thresh_img = threshold_reg(reduced_roi, min_intensity)
#plot_img(thresh_img[seed_index], "Thresholded Image")

reduced_roi = np.load(out_path + "Thresh_roi_%d.npy" % (id))
tumor_after_rg = connected_threshold_larger(reduced_roi, seed_index, min_intensity, 300, 14)
#num3s works well at 14 for id0
np.save(out_path + "Thresh_tumor_%d.npy" % (id), tumor_after_rg)
plot_img(tumor_after_rg[seed_index], "Tumor After RG")
"""
tumor_after_rg = np.load(out_path + "Thresh_tumor_%d.npy" % (id))
tumor_in_lungs = place_into_lungs(id, tumor_after_rg, bottom_left, top_right)
#place back into lungs

#tumor_in_lungs =

#thresh_roi = threshold(sep_roi)

#multiply by lung mask to remove outer lung

"""
PIL_img = Image.fromarray(RGB_img)
PIL_img_background = Image.fromarray(RGB_img)

new_image = Image.new("RGB", PIL_img.size, "WHITE")
draw = ImageDraw.Draw(new_image)
draw.rectangle([(305, 275), (340,310)], fill=None, outline= "Red")
del draw
np_new_image = array(new_image)
plt.title('New Image/Box')
plt.imshow(np_new_image)
#plt.show()
PIL_img_background = PIL_img_background.convert("RGBA")
new_image = new_image.convert("RGBA")
final_img = Image.blend(PIL_img_background, new_image, 1)
final_img.show()
"""