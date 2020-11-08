import numpy as np
import matplotlib.pyplot as plt
import cv2
import copy
def draw_circle(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(img, center = (x, y), radius = 5,
                       color = (87, 184, 237), thickness = -1)
    elif event == cv2.EVENT_RBUTTONDOWN:
            cv2.circle(img, center = (x, y), radius = 10,
                       color = (87, 184, 237), thickness = 1)

id = 0

out_path = "/Users/paulmccabe/Desktop/Segmentation Project/"
imgs_to_process = np.load(out_path + "nplungs_%d.npy" % (id))

#convert to RGB from greyscale
tum_slice = copy.deepcopy(imgs_to_process[150])
img_copy = np.dstack((tum_slice, tum_slice, tum_slice))


cv2.rectangle(img_copy, pt1 = (300, 350), pt2 = (260, 290),
              color = (255, 0, 0), thickness = 5)
cv2.circle(img_copy, center = (150, 300), radius = 50,
           color = (0, 0, 255), thickness = 5)
cv2.putText(img_copy, text = "Sample Text",
            org = (50, 250),
            fontFace = cv2.FONT_HERSHEY_DUPLEX,
            fontScale = 1,
            color = (0, 255, 0),
            thickness = 1,
            lineType = cv2.LINE_AA)
img = img_copy
cv2.namedWindow(winname = 'my_drawing')
cv2.setMouseCallback('my_drawing', draw_circle)
while True:
    cv2.imshow('my_drawing',img)
    if cv2.waitKey(10) & 0xFF == 27:
        break
cv2.destroyAllWindows()


plt.imshow(img_copy)
#plt.show()

#plt.title("Test Import Image")
#plt.imshow(imgs_to_process[150], cmap = 'gray')
#plt.show()