import matplotlib.pyplot as plt
from scipy import ndimage
from scipy import sparse
from scipy import misc
import numpy as np
from numpy import linalg as LA
import skimage as sk
import skimage.io as skio
from sklearn import preprocessing
import copy

def toy_reconstruct(im):
	height = len(im)
	width =len(im[0])

	# create im2var
	temp_im = np.zeros((height, width))
	
	k = 0
	for i in range(height):
		for j in range(width):
			temp_im[i][j] = k
			k = k + 1			
	
	A = sparse.lil_matrix(((height * (width - 1)) + (width * (height - 1)) + 1, height * width), dtype = np.float32)
	b = np.zeros(((height * (width - 1)) + (width * (height - 1)) + 1, 1))

	e = 0
	# Get the x-gradient
	for y in range(height):
		for x in range(width - 1):
			A[e,temp_im[y][x + 1]] = 1
			A[e,temp_im[y][x]] = -1
			b[e] = im[y][x + 1] - im[y][x]
			e = e + 1

	# Get the y-gradient
	for y in range(height - 1):
		for x in range(width):
			A[e, temp_im[y + 1][x]] = 1
			A[e, temp_im[y][x]] = -1
			b[e] = im[y + 1][x] - im[y][x]
			e = e + 1

	A[e, temp_im[0][0]] = 1
	b[e] = im[0][0]

	A = sparse.csr_matrix(A)
	return np.reshape(sparse.linalg.lsqr(A, b)[0], (height, width))

def main():
	im_toy = plt.imread('samples/toy_problem.png')/255.
	im_toy_reconst = toy_reconstruct(im_toy)
	misc.imsave("./2_1/toy_problem_reconst.png", im_toy_reconst)

main()
