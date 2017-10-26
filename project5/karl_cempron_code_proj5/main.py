from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
import math
import os

CENTERX = CENTERY = 8

def shift(image, shifts):
    assert len(shifts) == len(image.shape), 'Dimensions must match'
    new = np.zeros(image.shape)
    og_selector, target_selector = [], []
    for shift in shifts:
        if shift == 0:
            og_s, target_s = slice(None), slice(None)
        elif shift > 0:
            og_s, target_s = slice(shift, None), slice(None, -shift)
        else:
            og_s, target_s = slice(None, shift), slice(-shift, None)
        og_selector.append(og_s)
        target_selector.append(target_s)
    new[og_selector] = image[target_selector]
    return new

def calculateShift(imageName, scale):
	text = imageName.split("-")
	prefix = text[0].split("_")
	x = int(prefix[2])
	y = int(prefix[1])

	shiftx = x - CENTERX
	shifty = CENTERY - y
	return (x, y), (scale * shifty, scale * shiftx, 0)

def getAvgShiftedImage(directory, scale, radius, name):
	cwd = os.getcwd()
	im_directory = os.path.join(cwd, directory)
	count = 1
	average_image = None

	for filename in os.listdir(im_directory):
		orig, coord = calculateShift(filename, scale)
		if "out_08_08" in filename:
			filepath = os.path.join(im_directory, filename)
			average_image =  misc.imread(filepath)/255.
			break

	assert average_image.all() != None, "image_name not found"
	for filename in os.listdir(im_directory):
		orig, coord = calculateShift(filename, scale)
		if "out_08_08" in filename:
			continue
		if calculateDistance(CENTERX, CENTERY, orig[0], orig[1]) <= radius:
			filepath = os.path.join(im_directory, filename)
			curr_image = misc.imread(filepath)/255.
			curr_image = shift(curr_image, coord)
			average_image += curr_image
			count += 1

	average_image = average_image/float(count)
	misc.imsave("{}_average.png".format(name), average_image)

def calculateDistance(x1,y1,x2,y2):  
     dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)  
     return dist

def main(directory):
	## Part 1
	### Calculate the average image with shifting
	for i in range(0, 5):
		getAvgShiftedImage(directory, i, 25, "./{}/shifted_scale{}".format("processed_" + directory, i))

	## Part 2
	### Calculate aperture
	for i in range (0, 9):
		getAvgShiftedImage(directory, 1, i, "./{}/aperature_{}".format("processed_" + directory, i))

main("chess")
main("rock")
