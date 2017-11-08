import matplotlib.pyplot as plt
from scipy import misc
import numpy as np

POINT_COUNT = 10

def findPoints(im):
	plt.imshow(im)
	i = 0
	im_points = []
	while i < POINT_COUNT:
		x = plt.ginput(1, timeout = 0)
		im_points.append([x[0][0], x[0][1]])
		plt.scatter(x[0][0], x[0][1])
		plt.draw()
		i += 1
	plt.close()

	return im_points

def findCorrespondences(imA, imB, filename):
	imA_points = findPoints(imA)
	imB_points = findPoints(imB)

	f = open("./stored_{}_points.py".format(filename), "w")
	f.write("imA_points = " + str(imA_points) + "\n")
	f.write("imB_points = " + str(imB_points) + "\n")
	f.close()
	return imA_points, imB_points

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
			matrix_string += "{} {} 1 0 0 0 {} {};".format(x, y, x * x_1, y * x_1)
				            +"0 0 0 {} {} 1 {} {}".format(x, y, x * y_1, y * y_1)
		else:
			matrix_string += "{} {} 1 0 0 0 {} {};".format(x, y, x * x_1, y * x_1)
				            +"0 0 0 {} {} 1 {} {};".format(x, y, x * y_1, y * y_1)

	return np.matrix(matrix_string)

def computeBVector(imB_points):
	vector_string = ""
	for i in range(POINT_COUNT):
		x, y = imB_points[i][0], imB_points[i][1]
		vector_string += "{} {}".format(x, y)

	return np.matrix(vector_string)

def formatHMatrix(result):
	H = np.matrix("{} {} {};".format(result[0], result[1], result[2])
				 +"{} {} {};".format(result[3], result[4], result[5])
				 +"{} {} 1".format(result[6], result[7]))

	return H

def main():
	im_left = plt.imread("./images/1.jpg")
	im_center = plt.imread("./images/2.jpg")
	im_right = plt.imread("./images/3.jpg")

	imA_points, imB_points = findCorrespondences(im_left, im_center, "test_LC")
	H_LC = computeHomographyMatrix(imA_points, imB_points)

	imA_points, imB_points = findCorrespondences(im_center, im_right, "test_CR")
	H_CR = computeHomographyMatrix(imA_points, imB_points)

	print(H_LC)
	print(H_CR)
	
main()