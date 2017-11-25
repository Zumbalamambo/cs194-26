import matplotlib.pyplot as plt
import skimage.io as skio
from skimage.transform import warp
from scipy import misc
import numpy as np
import sys
import harris
import math
import random


def computeHomographyMatrix(imA_points, imB_points):
	A = computeAMatrix(imA_points, imB_points)
	b = computeBVector(imB_points)
	
	result = np.linalg.lstsq(A, np.transpose(b))[0]
	return formatHMatrix(result)

def computeAMatrix(imA_points, imB_points):
	matrix_string = ""
	for i in range(len(imA_points)):
		x, y = imA_points[i][0], imA_points[i][1]
		x_1, y_1 = imB_points[i][0], imB_points[i][1] 
		if i + 1 == len(imA_points):
			value = "{} {} 1 0 0 0 {} {};".format(x, y, -1 * x * x_1, -1 * y * x_1) +"0 0 0 {} {} 1 {} {}".format(x, y, -1 * x * y_1, -1 * y * y_1)
			matrix_string += value
		else:
			value = "{} {} 1 0 0 0 {} {};".format(x, y, -1 * x * x_1, -1 * y * x_1) +"0 0 0 {} {} 1 {} {};".format(x, y, -1 *x * y_1, -1 * y * y_1)
			matrix_string += value

	return np.matrix(matrix_string)

def computeBVector(imB_points):
	vector_string = ""
	for i in range(len(imB_points)):
		x, y = imB_points[i][0], imB_points[i][1]
		vector_string += " {} {} ".format(x, y)

	return np.matrix(vector_string)

def formatHMatrix(result):
	H = np.matrix("{} {} {};".format(result[0], result[1], result[2])
				 +"{} {} {};".format(result[3], result[4], result[5])
				 +"{} {} 1".format(result[6], result[7]))
	return H

def ANMS(points, eps, H):
	radius_values = {}
	for center in points:
		H_i = H[center[0], center[1]]
		interest_points = []
		for point in points:
			H_j = H[point[0], point[1]]
			if H_i < (eps * H_j):
				interest_points.append(point)
		if len(interest_points) > 0:
			radius_values[center] = np.amin(harris.dist2(np.array([center]), np.array(interest_points)))
	radius_values = sorted((value, key) for (key, value) in radius_values.items())[::-1]
	
	top_500 = [[], []]
	for i in range(500):
		top_500[0].append(radius_values[i][1][0])
		top_500[1].append(radius_values[i][1][1])

	return top_500

def find_descriptors(im, points):
	results = {}
	patch_size = 40
	for point in points:
		corner_left_x = point[0] - 20
		corner_left_y = point[1] - 20
		sample_patch = np.zeros((40, 40))
		for i in range(patch_size):
			for j in range(patch_size):
				pixel = im[corner_left_x + i][corner_left_y + j]
				sample_patch[i][j] = pixel

		subsample_patch = misc.imresize(sample_patch, (8, 8))

		mean = np.mean(subsample_patch)
		std = np.std(subsample_patch)
		normalized_patch = (subsample_patch - mean)/std

		results[point] = np.reshape(normalized_patch, (1, 64))
	return results

def feature_match(desc_imA, desc_imB):
	results = {}
	for point_A, vector_A in desc_imA.items():
		dists = {}
		for point_B, vector_B in desc_imB.items():
			dists[point_B] = harris.dist2(vector_A, vector_B)[0][0]
		dists = sorted((value, key) for (key, value) in dists.items())

		if dists[0][0]/dists[1][0] < .3:
			results[point_A] = dists[0][1]
	return results

def RANSAC(matched_points):
	points_A = list(matched_points.keys())
	points_B = list(matched_points.values())
	results = {}
	sub_points = random.sample(range(1, len(points_A)), 4)
	subpoints_A = np.array([points_A[sub_points[0]], points_A[sub_points[1]], points_A[sub_points[2]], points_A[sub_points[3]]])
	subpoints_B = np.array([points_B[sub_points[0]], points_B[sub_points[1]], points_B[sub_points[2]], points_B[sub_points[3]]])

	H = computeHomographyMatrix(subpoints_A, subpoints_B)
	b = np.array(points_B)

	error = np.dot(H, np.transpose(np.hstack((points_A, np.ones((len(points_A), 1))))))	
	A = np.zeros_like(error)
	for i in range(3):
		A[i, :] = error[i, :] /error[2, :]

	A = np.transpose(A)[:,:2]
	val_1 = (A[:,0] - b[:,0])**2
	val_2 = (A[:,1] - b[:,1])**2
	sqrd_err = np.sqrt(val_1 + val_2)

	for i in range(len(sqrd_err)):
		if sqrd_err[i] < 0.5:
			results[points_A[i]] = points_B[i]
	return results

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
	im_left = misc.imread("./images/thames1.jpg")/255.
	im_left_bw = misc.imread("./images/thames1.jpg", flatten = True)/255.
	im_right = misc.imread("./images/thames2.jpg")/255.
	im_right_bw = misc.imread("./images/thames2.jpg", flatten = True)/255.

	H_left, coors_left = harris.get_harris_corners(im_left_bw)
	H_right, coors_right = harris.get_harris_corners(im_right_bw)

	plt.imshow(im_left)
	plt.scatter(coors_left[1], coors_left[0], s = 40)
	plt.show()

	plt.imshow(im_right)
	plt.scatter(coors_right[1], coors_right[0], s = 40)
	plt.show()

	points_left = []
	ANMS_points_left = []
	for i in range(len(coors_left[0])):
		points_left.append((coors_left[0][i], coors_left[1][i]))
	ANMS_coors_left = ANMS(points_left, .9, H_left)

	points_right = []
	ANMS_points_right = []
	for i in range(len(coors_right[0])):
		points_right.append((coors_right[0][i], coors_right[1][i]))
	ANMS_coors_right = ANMS(points_right, .9, H_right)

	plt.imshow(im_left)
	plt.scatter(ANMS_coors_left[1], ANMS_coors_left[0], s = 40)
	plt.show()

	plt.imshow(im_right)
	plt.scatter(ANMS_coors_right[1], ANMS_coors_right[0], s = 40)
	plt.show()

	for i in range(len(ANMS_coors_left[0])):
		ANMS_points_left.append((ANMS_coors_left[0][i], ANMS_coors_left[1][i]))
	descriptors_left = find_descriptors(im_left_bw, ANMS_points_left)

	for i in range(len(ANMS_coors_right[0])):
		ANMS_points_right.append((ANMS_coors_right[0][i], ANMS_coors_right[1][i]))
	descriptors_right = find_descriptors(im_right_bw, ANMS_points_right)

	matched_features = feature_match(descriptors_left, descriptors_right)

	matched_left_points = list(matched_features.keys())
	plt.imshow(im_left)
	plt.scatter([x for (y, x) in matched_left_points], [y for (y, x) in matched_left_points], s = 40)
	plt.show()

	matched_right_points = list(matched_features.values())
	plt.imshow(im_right)
	plt.scatter([x for (y, x) in matched_right_points], [y for (y, x) in matched_right_points], s = 40)
	plt.show()

	RANSAC_points = {}
	for i in range(500):
		points = RANSAC(matched_features)
		if len(points) > len(RANSAC_points):
			RANSAC_points = points

	H_L = computeHomographyMatrix(list(RANSAC_points.keys()), list(RANSAC_points.values()))
	H_R = computeHomographyMatrix(list(RANSAC_points.keys()), list(RANSAC_points.values()))

	warpedImL = warp(im_left, np.linalg.inv(H_L), output_shape = (im_left.shape[0] * 1.5, im_left.shape[1] * 3))
	warpedImR = warp(im_right, np.linalg.inv(H_R), output_shape = (im_left.shape[0] * 1.5, im_left.shape[1] * 3))

	result = linearBlend(warpedImL, warpedImR, weight = .5)

	misc.imsave("warpedL.png", warpedImL)
	misc.imsave("warpedR.png", warpedImR)
	misc.imsave("warped_result.png", result)

main()
	