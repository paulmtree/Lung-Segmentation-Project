# Lung-Segmentation-Project (3-5 minute read)
Use CT scans of the lungs to generate 3D models of the airway, bronchioles, outer lung structure, and cancerous growths. Mathematical descriptions of these objects can be used for AI research, such as predicting benign vs malignant tumors to prevent unnecessary and invasive cancer treatments, early recognition of tumors, and modeling the growth rate of tumors.
### License 
See the [LICENSE](https://github.com/paulmtree/Lung-Segmentation-Project/blob/main/LICENSE.md) file for license rights and limitations (MIT). If this code was useful for you, please let me know via email at pkm29@case.edu so I can brag about it. 
# Intro
This project was part of my research at the Center for Computational Imaging and Personalized Diagnostics (CCIPD) lab under Dr. Mehdi Alilou at Case Western Reserve University in 2019. This was also my first experience with Python and took approximately 2 months. Each slide corresponds to a python file used to generate the slides content. You can find these slides in this [Powerpoint Presentation](https://github.com/paulmtree/Lung-Segmentation-Project/blob/main/Powerpoint%20Images/Image%20Segmentation_%20Paul%20McCabe%20(3).pptx). (MeshlabVisualizer.py is used on a couple different slides)
## [DICOM Processing.py](https://github.com/paulmtree/Lung-Segmentation-Project/blob/main/DICOM%20Processing.py)
### Import lung slices and create lung mask
1. Import DICOM images and convert to numpy arrays.
2. Use Kmeans thresholding to display empty space as white and non-empty space as black. 
3. Erode away the finer details and dilate to return to former size.
4. Label regions using python's skimage measure tools.
5. Choose appropriate regions based on length and width, then dilate out mask to make sure it doesn't clip the lungs.
6. Apply mask to each lung slice.

<img align="center" width="960"  src="https://github.com/paulmtree/Lung-Segmentation-Project/raw/main/Powerpoint%20Images/PP1.png">


### Check lung mask and save files


<img align="center" width="960"  src="https://github.com/paulmtree/Lung-Segmentation-Project/raw/main/Powerpoint%20Images/Powerpoint1.gif">

## [RegionGrowerSmall.py](https://github.com/paulmtree/Lung-Segmentation-Project/blob/main/RegionGrowerSmall.py)


<img align="center" width="960"  src="https://github.com/paulmtree/Lung-Segmentation-Project/raw/main/Powerpoint%20Images/Powerpoint2.gif">


##
<img align="center" width="960"  src="https://github.com/paulmtree/Lung-Segmentation-Project/raw/main/Powerpoint%20Images/Powerpoint3.gif">

##
With a small hole in our airway, our region growing algorithm spills into the lungs. However with a larger region growing voxel (3D pixel), we won't have a leak. Below is the paper that describes this theory in region growing algorithms.

<img align="center" width="960"  src="https://github.com/paulmtree/Lung-Segmentation-Project/raw/main/Powerpoint%20Images/PP3.png">


## [RegionGrowerLarge.py](https://github.com/paulmtree/Lung-Segmentation-Project/blob/main/RegionGrowerLarge.py)

<img align="center" width="960"  src="https://github.com/paulmtree/Lung-Segmentation-Project/raw/main/Powerpoint%20Images/Powerpoint4.gif">


##
<img align="center" width="960"  src="https://github.com/paulmtree/Lung-Segmentation-Project/raw/main/Powerpoint%20Images/Powerpoint5.gif">


##
<img align="center" width="960"  src="https://github.com/paulmtree/Lung-Segmentation-Project/raw/main/Powerpoint%20Images/PP4.png">


## [MeshlabVisualizer.py](https://github.com/paulmtree/Lung-Segmentation-Project/blob/main/MeshLabVisualizer.py)

<img align="center" width="960"  src="https://github.com/paulmtree/Lung-Segmentation-Project/raw/main/Powerpoint%20Images/PP5.png">


## [Bronchial Grower.py](https://github.com/paulmtree/Lung-Segmentation-Project/blob/main/Bronchial%20Grower.py)

<img align="center" width="960"  src="https://github.com/paulmtree/Lung-Segmentation-Project/raw/main/Powerpoint%20Images/PP6.png">


##
<img align="center" width="960"  src="https://github.com/paulmtree/Lung-Segmentation-Project/raw/main/Powerpoint%20Images/PP7.png">


## [Volume of Interest.py](https://github.com/paulmtree/Lung-Segmentation-Project/blob/main/Volume%20of%20Interest.py)

<img align="center" width="960"  src="https://github.com/paulmtree/Lung-Segmentation-Project/raw/main/Powerpoint%20Images/PP8.png">


## [Meshlab ROI.py](https://github.com/paulmtree/Lung-Segmentation-Project/blob/main/Meshlab%20ROI.py)


<img align="center" width="960"  src="https://github.com/paulmtree/Lung-Segmentation-Project/raw/main/Powerpoint%20Images/PP9.png">


##
<img align="center" width="960"  src="https://github.com/paulmtree/Lung-Segmentation-Project/raw/main/Powerpoint%20Images/PP10.png">


##
<img align="center" width="960"  src="https://github.com/paulmtree/Lung-Segmentation-Project/raw/main/Powerpoint%20Images/PP11.png">


