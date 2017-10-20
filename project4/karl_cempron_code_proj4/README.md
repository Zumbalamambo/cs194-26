# CS194-26: Face Morphing
## Overview
The goal of this project is to produce a visual morph between two images. In this particular project, I define a correspondence between two faces, mapping facial features such as the eyes, mouth, nose, and hair shape. Along with the morphing of two faces, I find the mean face of a given population to make comparisons with my own facial features.
## Accessing the code
To run this project, you must open the main.py file and uncomment the respective section in the main method that you want to run. Currently, all the code in the main method is commented out such that running `py main.py` will have no effect. Furthermore, uncommenting a given section will run that section of the project with a preset input of images. If you want to change the name of the image that is to be altered, simply change the `filename` and the respective `stored_points` if you want to avoid selecting new correspondence points. To run.
## Code Breakdown
The primary methods in this file are:
1. morph(imA, imB imA_points, imB_points)
    1. This method runs the steps of finding correspondence points, finding the Mid-Way Face, and then morphing all 45 frames.
2. findMeanFace(directory)
    1. This method parses the Danes dataset and then generates the mean face as well as the mean face shape.
3. warpMeanFace(im_names, im_points, average_points, saved_name, t = 1)
    1. This method takes in a list of image names, list of lists of corresponding points, a list of average_points, a destination path, and a predefined scalar of 1. This method is used primarily in generating warped faces for components in the Mean Face section as well as getting caricatures.
