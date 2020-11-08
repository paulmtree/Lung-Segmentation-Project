import pydicom
import matplotlib.pyplot as plt
from pydicom.data import get_testdata_files

filename_array = get_testdata_files("CTScans/*.dcm")
length = len(filename_array)
i = 0
slice_values_list = [0]*length
variable = 0
temp_int = 1
place_holder_list = [0]*length
#put slice location values into list
for f in range(1, length, 1):
    dataset = pydicom.dcmread(filename_array[f])
    arrayyyy = dataset.pixel_array
    slice_values_list[f] = dataset.get('SliceLocation', "(missing)")
    place_holder_list[f] = f
print(slice_values_list)
#print(place_holder_list)

#sort list of slice locations algorithm from least to greatest
for a in range(0, length, 1):

    for b in range(a, length-1, 1):
        if slice_values_list[a] < slice_values_list[b+1]:
            temp_int = slice_values_list[a]
            slice_values_list[a] = slice_values_list[b+1]
            slice_values_list[b+1] = temp_int
            temp_int = place_holder_list[a]
            place_holder_list[a] = place_holder_list[b+1]
            place_holder_list[b+1] = temp_int

#read data in each file
#print(slice_values_list)
#print(place_holder_list)
for f in range(0, length, 1):
    dataset = pydicom.dcmread(filename_array[place_holder_list[f]])

    plt.imshow(dataset.pixel_array, cmap=plt.cm.bone)
    plt.show()
    #print("Slice location...:", dataset.get('SliceLocation', "(missing)"))

