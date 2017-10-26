# CS194-26: Light Field Camera
## Overview
Capturing multiple images over a plane orthogonal to the optical axis enables achieving complex effects using very simple operations like shifting and averaging. The goal of this project is to reproduce some of these effects using real lightfield data. Using rectified images from the Stanford Light Field Archive, I reproduced both the refocusing and aperture adjustment effects that can be made with a light field camera.
## Accessing the Code
By running the main.py file, you will run the whole project. This will produce sets of images for both the chess and rock image source(5 focus images of varying scaler values and 9 aperture images of varying radius values).
## Code Breakdown
The primary function in thie project is `getAvgShiftedImage(directory, scale, radius, name)` which takes in the name of the directory you want to read from, the scale used for focusing, the radius for aperture size, and the name for the resulting images.
The rest of the functions serve as utility for the primary function to run.
