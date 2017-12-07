# CS194-26: Image Warping and Mosaicing
## Overview
The purpose of this project is to use image warping to achieve wide panoramas from multiple images of the same source content. This part will contain the first half of the project, discussing homography, rectifying an image, and finally, creating a mosaic.
## Accessing the Code
By running the main.py file, you will run the panoramic warping and stitching of a 3 part image. To make different images, simply change the filenames of the images you want. Also as an aside, if only doing a mosaic of 2 images, you can leave Image C as is and only blend the first two images.
## Code Breakdown
This project does the following:
- Finds Correspondences
- Calculates Homography Matrices
- Warps images
- Stitches images together
