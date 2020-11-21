# Lung-Segmentation-Project (≈ 5-10 minute read)
This project uses a process known as segmentation to extract individual lung components from CT scans such as the airway, bronchioles, outer lung structure, and cancerous growths. Mathematical descriptions of these objects can be used for AI research, such as predicting benign vs malignant tumors to prevent unnecessary and invasive cancer treatments, early recognition of tumors, and modeling the growth rate of tumors. The final result of this segmentation project is shown below:

<img align="center" width="300"  src="https://github.com/paulmtree/Lung-Segmentation-Project/blob/main/Powerpoint%20Images/Lung.gif">

### Please Note
Anyone looking to use this code, I highly recommend using Howard Chen's article: https://www.raddq.com/dicom-processing-segmentation-visualization-in-python/ for the first few steps of the process. This code may be outdated/depreciated by the time you read it however the approaches I use should still work. 

### License 
See the [LICENSE](https://github.com/paulmtree/Lung-Segmentation-Project/blob/main/LICENSE.md) file for license rights and limitations (MIT). If this code was useful for you, please let me know via email at pkm29@case.edu so I can brag about it. 
# Introduction
This project was part of my research at the Center for Computational Imaging and Personalized Diagnostics (CCIPD) lab under Dr. Mehdi Alilou at Case Western Reserve University in 2019. This was also my first experience with Python and took approximately 2 months. Each slide corresponds to a python file used to generate the slides content. You can find these slides in this [Powerpoint Presentation](https://github.com/paulmtree/Lung-Segmentation-Project/blob/main/Powerpoint%20Images/Image%20Segmentation_%20Paul%20McCabe%20(3).pptx). (MeshlabVisualizer.py is used on a couple different slides)
## Import lung slices and filter out the lung and airway region for each slice
### [DICOM Processing.py](https://github.com/paulmtree/Lung-Segmentation-Project/blob/main/DICOM%20Processing.py)
First we will use the DICOM Processing.py script to import our CT scans, which are DICOM files containg slice numbers and other metrics, and process them for segmentation. Below is a description of the script's processes and outputs. 
1. Import DICOM images and convert to numpy arrays.
2. Use Kmeans thresholding to display empty space as white and non-empty space as black. 
3. Erode away the finer details and dilate to return to former size.
4. Label regions using python's skimage measure tools.
5. Choose appropriate regions based on length and width, then dilate out mask to make sure it doesn't clip the lungs.
6. Apply mask to each lung slice.

<img align="center" width="960"  src="https://github.com/paulmtree/Lung-Segmentation-Project/raw/main/Powerpoint%20Images/PP1.png">


## Check lung mask and save files

<img align="center" width="960"  src="https://github.com/paulmtree/Lung-Segmentation-Project/raw/main/Powerpoint%20Images/Powerpoint1.gif">

## Use a small region growing algorithm to extract just the airway (doesn't work)

### [RegionGrowerSmall.py](https://github.com/paulmtree/Lung-Segmentation-Project/blob/main/RegionGrowerSmall.py)

To fill the area known as the trachea (or airway), we need to use a region growing algorithm. With a regular threshold, we would mistakenly include the air in the lungs since it is the same material and brightness. A region growing algorithm will instead fill a closed container/volume and nothing outside of it. It begins with a seed point and spreads from that seed with a voxel as shown below to adjacent pixels, determining suitable material with a set brightness threshold. If the algorith comes into contact with a wall as determined by our brightness threshold, it stops growing that branch and moves onto other areas that still need to be checked and grown. Eventually, the algorithm grows until the volume is completely filled.


<img align="center" width="960"  src="https://github.com/paulmtree/Lung-Segmentation-Project/raw/main/Powerpoint%20Images/Powerpoint2.gif">

##
However with a small hole in our airway, our region growing algorithm spills into the lungs as indicated by the pixel density per slice graph. Since the airway only extends halfway down our whole CT scan, we would expect to see it drop off sharply to zero once it reaches the bottom of the airway. Instead, we see a high number of captured pixels throughout the entire CT scan.

<img align="center" width="960"  src="https://github.com/paulmtree/Lung-Segmentation-Project/raw/main/Powerpoint%20Images/Powerpoint3.gif">

##
While some papers recommend using Gaussian filters to smooth over leaks from the trachea to the lungs, I found that increasing the size of the growing voxel fixes this issue nicely. The paper below was my source of inspiration and it nicely illustrates how a larger growing size voxel is not able to get through small holes in the wall of our airway. 

<img align="center" width="960"  src="https://github.com/paulmtree/Lung-Segmentation-Project/raw/main/Powerpoint%20Images/PP3.png">

## Use a large region growing algorithm to extract just the airway (works!)
### [RegionGrowerLarge.py](https://github.com/paulmtree/Lung-Segmentation-Project/blob/main/RegionGrowerLarge.py)
Now with the large voxel growing size.

<img align="center" width="960"  src="https://github.com/paulmtree/Lung-Segmentation-Project/raw/main/Powerpoint%20Images/Powerpoint4.gif">

##
And voila, no leaks! 

<img align="center" width="960"  src="https://github.com/paulmtree/Lung-Segmentation-Project/raw/main/Powerpoint%20Images/Powerpoint5.gif">


## Check quality of region growing algorithm
Further inspection of the airway illustrates the revised algorithm is of sufficient quality. 

<img align="center" width="960"  src="https://github.com/paulmtree/Lung-Segmentation-Project/raw/main/Powerpoint%20Images/PP4.png">

## Model the airway in 3D with the surrounding lung structure with 3 different sets of CT scans
### [MeshlabVisualizer.py](https://github.com/paulmtree/Lung-Segmentation-Project/blob/main/MeshLabVisualizer.py)

<img align="center" width="960"  src="https://github.com/paulmtree/Lung-Segmentation-Project/raw/main/Powerpoint%20Images/PP5.png">

## Use the smaller region growing algorithm to extract the bronchials from the lungs
### [Bronchial Grower.py](https://github.com/paulmtree/Lung-Segmentation-Project/blob/main/Bronchial%20Grower.py)
To extract our bronchials, the branches that fill up our lungs, we can use the smaller growing voxel algorithm (RegionGrowerSmall.py). However since the 2 bronchial groups are not connected to each other, it is necessary to have 2 seed points. Using scimage measuring tools as we did when creating the mask and labeling regions in step 4 of DICOM Processing.py, we can approach the left and right lung separately to determine a seed point automatically. We also do not want our algorithm to grow to the walls of the lungs as they are similiarly colored, so we erode away the walls of the lungs. 

Unlike the airway however, we cannot simply use the median/center position of our particular material, since the bronchials are less uniform and contain regions of dark space where if the seed point was placed there, it would not grow at all. However the advantage of using the center position approach is that the seed lands in the center of our mass and not on some isolated but also bright pixel created by noise. 

Fortunately, I found a solution that utilizes both techniques. A check is set so that if the seed point landed in a dark area, change the selected brightness of the material that the algorithm finds the center of. This changes the shape of our bronchials slightly and should yield a slightly different seed point until a satisfactory one is found. Here is an example of this recursive process:
1. Select a middle slice of the scan where we expect the bronchials to be prominant and remove the edges of our lung through erosion. (This doesn't need to be done each time, just an approximate middle slice like number 160 out of 300 slices)
2. Threshold the image for a brightness value greater than 210. (255 being pure white)
3. Find the median or center position of this thresholded image and check if that location is in a light or a dark area. 
4. If the median position lands in a dark area, change our thresholded image to be for a brightness greater than 215. 
5. With this slightly altered image, find the median area again and repeat the process of using different brightness thresholds until the seed lands in a bright area. 
With the check in place, our seed point should change until a suitable seed is found. 
##

<img align="center" width="960"  src="https://github.com/paulmtree/Lung-Segmentation-Project/raw/main/Powerpoint%20Images/PP6.png">


##
The bronchial growing algorithm works for all 3 patients seamlessly and is shown in 3D below.

<img align="center" width="960"  src="https://github.com/paulmtree/Lung-Segmentation-Project/raw/main/Powerpoint%20Images/PP7.png">

## Define a volume of interest such as a tumor and extract it from the lungs
### [Volume of Interest.py](https://github.com/paulmtree/Lung-Segmentation-Project/blob/main/Volume%20of%20Interest.py)
To model a cancerous growth separately, our simplistic approach requires the user to input a slice number and 2 x,y coordinates to form a region of interest around the tumor. Advanced ML algorithms, specifically the field of machine vision, are capable of finding these growths automatically as well. 

Interestingly, we find that when we dilated our CT scans to create the walls of our lung regions at the beginning, our program included the tumor as part of the lung wall as indicated by the unusually uniform indent. Where does the tumor end and the wall of the lungs begin? This phenomenon remains a current problem for tumor recognition and modeling and will be part of future work as mentioned at the end of this presentation.

<img align="center" width="960"  src="https://github.com/paulmtree/Lung-Segmentation-Project/raw/main/Powerpoint%20Images/PP8.png">


## Extract the tumor and place it in the overall lung structure as a separate entity
### [Meshlab ROI.py](https://github.com/paulmtree/Lung-Segmentation-Project/blob/main/Meshlab%20ROI.py)
For our last region growing application, we use the large growing voxel (RegionGrowerLarge.py) so that we do not spread from the tumor to a nearby bronchial. We can then place it back into our lung structure and create a 3D model with all of our segmented components!

<img align="center" width="960"  src="https://github.com/paulmtree/Lung-Segmentation-Project/raw/main/Powerpoint%20Images/PP9.png">


### Summary of future Work

<img align="center" width="960"  src="https://github.com/paulmtree/Lung-Segmentation-Project/raw/main/Powerpoint%20Images/PP10.png">


### Sources

<img align="center" width="960"  src="https://github.com/paulmtree/Lung-Segmentation-Project/raw/main/Powerpoint%20Images/PP11.png">


