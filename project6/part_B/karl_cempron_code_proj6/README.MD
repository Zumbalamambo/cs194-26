# CS194-26: Image Warping and Mosaicing
## Overview
The purpose of this project is to use image warping to achieve wide panoramas from multiple images of the same source content. This part will contain the latter half of the project, discussing Harris corners, ANMS, feature matching, and RANSAC.
## Accessing the Code
By running the main.py file, you will run the panoramic warping and stitching of a 3 part image. To make different images, simply change the filenames of the images you want. Also as an aside, if only doing a mosaic of 2 images, you can leave Image C as is and only blend the first two images. Currently this file is set to run for the Thames images and through it's process will display the images with plotted points (for each corresponding automatic process). Take a screenshot of the image and close out to proceed to the next one.
## Code Breakdown
The file has 4 methods particular to this part:
- ANMS
- find_descriptors
- feature_match
- RANSAC

called sequentially in that order.
