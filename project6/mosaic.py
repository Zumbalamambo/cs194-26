import matplotlib.pyplot as plt
from scipy import misc
import numpy as np

POINT_COUNT = 10

def findPoints(imA):
	plt.imshow(imA)
	i = 0
	imA_points = []
	while i < POINT_COUNT:
		x = plt.ginput(1, timeout = 0)
		imA_points.append([x[0][0], x[0][1]])
		plt.scatter(x[0][0], x[0][1])
		plt.draw()
		i += 1
	plt.close()

	return imA_points

def findCorrespondences(imA, imB, filename):
	imA_points = findPoints(imA)
	imB_points = findPoints(imB)

	f = open("./stored_{}_points.py".format(filename), "w")
	f.write("imA_points = " + str(imA_points) + "\n")
	f.write("imB_points = " + str(imB_points) + "\n")
	f.close()
	return imA_points, imB_points


def main():
	imA = 
	imB = 
	imA_points, imB_points = findCorrespondences(imA, imB, "test")


main()