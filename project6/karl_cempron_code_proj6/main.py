import matplotlib.pyplot as plt
from skimage.transform import warp
from scipy import misc
import numpy as np
import sys

POINT_COUNT = 4

def findPoints(im):
	plt.imshow(im)
	i = 0
	im_points = []
	while i < POINT_COUNT:
		x = plt.ginput(1,timeout = 0)
		im_points.append([x[0][0], x[0][1]])
		plt.scatter(x[0][0], x[0][1])
		plt.draw()
		i += 1
	plt.close()

	return im_points

def findCorrespondences(imA, imB, imC, filename):
	imA_points = findPoints(imA)
	imB_points = findPoints(imB)
	imC_points = findPoints(imC)
	f = open("./stored_{}_points.py".format(filename), "w")
	f.write("imA_points = " + str(imA_points) + "\n")
	f.write("imB_points = " + str(imB_points) + "\n")
	f.write("imC_points = " + str(imC_points) + "\n")
	f.close()
	return imA_points, imB_points, imC_points

def computeHomographyMatrix(imA_points, imB_points):
	A = computeAMatrix(imA_points, imB_points)
	b = computeBVector(imB_points)
	
	result = np.linalg.lstsq(A, np.transpose(b))[0]
	return formatHMatrix(result)

def computeAMatrix(imA_points, imB_points):
	matrix_string = ""
	for i in range(POINT_COUNT):
		x, y = imA_points[i][0], imA_points[i][1]
		x_1, y_1 = imB_points[i][0], imB_points[i][1] 
		if i + 1 == POINT_COUNT:
			value = "{} {} 1 0 0 0 {} {};".format(x, y, -1 * x * x_1, -1 * y * x_1) +"0 0 0 {} {} 1 {} {}".format(x, y, -1 * x * y_1, -1 * y * y_1)
			matrix_string += value
		else:
			value = "{} {} 1 0 0 0 {} {};".format(x, y, -1 * x * x_1, -1 * y * x_1) +"0 0 0 {} {} 1 {} {};".format(x, y, -1 *x * y_1, -1 * y * y_1)
			matrix_string += value

	return np.matrix(matrix_string)

def computeBVector(imB_points):
	vector_string = ""
	for i in range(POINT_COUNT):
		x, y = imB_points[i][0], imB_points[i][1]
		vector_string += " {} {} ".format(x, y)

	return np.matrix(vector_string)

def formatHMatrix(result):
	H = np.matrix("{} {} {};".format(result[0], result[1], result[2])
				 +"{} {} {};".format(result[3], result[4], result[5])
				 +"{} {} 1".format(result[6], result[7]))
	return H

def linearBlend(imA, imB, weight):
	height, width = imA.shape[0], imA.shape[1]
	blendedIm = np.zeros((imA.shape))
	for y in range(height):
		for x in range(width):
			if np.sum(imA[y, x, :]) != 0.0 and np.sum(imB[y, x, :]) != 0.0:
				blendedIm[y, x, :] = imA[y, x, :] * (1 - weight) + imB[y, x, :] * weight
			else:
				blendedIm[y, x, :] = imA[y, x, :] + imB[y, x, :]

	return np.clip(blendedIm, 0, 1)

def main():
	im_left = plt.imread("./images/a.jpg")
	im_center = plt.imread("./images/b.jpg")
	im_right = plt.imread("./images/c.jpg")

	try:
		from stored_test_points import imA_points
		from stored_test_points import imB_points
		from stored_test_points import imC_points

	except:
		imA_points, imB_points, imC_points = findCorrespondences(im_left, im_center, im_right, "road")
	
	H_LC = computeHomographyMatrix(imA_points, imB_points)
	H_CC = computeHomographyMatrix(imB_points, imB_points)
	H_RC = computeHomographyMatrix(imC_points, imB_points)

	warpedImL = warp(im_left, np.linalg.inv(H_LC), output_shape = (im_left.shape[0] * 1.5, im_left.shape[1] * 3))
	warpedImC = warp(im_center, np.linalg.inv(H_CC), output_shape = (im_left.shape[0] * 1.5, im_left.shape[1] * 3))
	warpedImR = warp(im_right, np.linalg.inv(H_RC), output_shape = (im_left.shape[0] * 1.5, im_left.shape[1] * 3))

	result = linearBlend(linearBlend(warpedImL, warpedImC, weight = .5), warpedImR, weight = .5)

	misc.imsave("warpedL.png", warpedImL)
	misc.imsave("warpedC.png", warpedImC)
	misc.imsave("warpedR.png", warpedImR)
	misc.imsave("warped_result.png", result)

main()