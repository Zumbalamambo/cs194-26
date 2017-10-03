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

def mask_check(y, x, mask):
	return mask[y][x] == 1.

def calculate_vars(height, width, mask):
	i = 0;
	for y in range(height):
		for x in range(width):
			if mask_check(y, x, mask):
				i = i + 1
	return i

def calc_adjacent(x, y, height, width, mask, A , b, e, im2var, source, source_i, target):
	if mask_check(y, x, mask):
		A[e, im2var[y][x]] = -1
		b[e] = source_i - source[y][x]
	else:
		b[e] = target[y][x]

def calc_all_sides(x, y, height, width, mask, A , b, e, im2var, source, target):
	#calc up
	calc_adjacent(x, y + 1, height, width, mask, A, b, e, im2var, source, source[y][x], target)
	A[e, im2var[y][x]] = 1
	e = e + 1
	#calc down
	calc_adjacent(x, y - 1, height, width, mask, A, b, e, im2var, source, source[y][x], target)
	A[e, im2var[y][x]] = 1
	e = e + 1
	#calc_right
	calc_adjacent(x + 1, y, height, width, mask, A, b, e, im2var, source, source[y][x], target)
	A[e, im2var[y][x]] = 1
	e = e + 1
	#calc_left
	val = calc_adjacent(x - 1, y, height, width, mask, A, b, e, im2var, source, source[y][x], target)
	A[e, im2var[y][x]] = 1
	e = e + 1

	return e

def poisson_blend(source, target, mask):
	height = len(source)
	width = len(source[0])
	num_vars = calculate_vars(height, width, mask)
	im2var = np.zeros((height, width))

	k = 0
	for y in range(height):
		for x in range(width):
			if mask_check(y, x, mask):
				im2var[y][x] = k	
				k = k + 1
			else:
				im2var[y][x] = -1

	A = sparse.lil_matrix(((num_vars * 4), num_vars), dtype = np.float32)
	b = np.zeros(((num_vars * 4), 1))
	e = 0

	for y in range(height):
		for x in range(width):
			if mask_check(y, x, mask):
				e = calc_all_sides(x, y, height, width, mask, A , b, e, im2var, source, target)		
			
	A = sparse.csr_matrix(A)

	result = sparse.linalg.lsqr(A, b)[0]
	# if result.min() < 0:
	# 	result -= result.min()
	# result /= result.max()
	return np.clip(sparse.linalg.lsqr(A, b)[0], 0, 1)

def pixel_replace(result_vec, source, target, mask):
	k = 0
	for y in range(len(source)):
		for x in range(len(source[0])):
			if mask_check(y, x, mask):
				print(str(target[y][x]) + "original")
				print(result_vec[k])
				target[y][x] = result_vec[k]
				k = k + 1
	return target

def gray_blend(source, target, mask):
	result_vec = poisson_blend(source, target, mask)
	target = pixel_replace(result_vec, source, target, mask)

	return target

def split_RGB(im):
	red = im[:,:,2]
	green = im[:,:,1]
	blue = im[:,:,0]
	return red, green, blue

def color_blend(source, target, mask):
	red_source, green_source, blue_source = split_RGB(source)
	red_target, green_target, blue_target = split_RGB(target)

	result_vec_red = poisson_blend(red_source, red_target, mask)
	result_vec_green = poisson_blend(green_source, green_target, mask)
	result_vec_blue = poisson_blend(blue_source, blue_target, mask)

	pixel_replace(result_vec_red, red_source, red_target, mask)
	pixel_replace(result_vec_green, green_source, green_target, mask)
	pixel_replace(result_vec_blue, blue_source, blue_target, mask)

	return np.dstack([blue_target, green_target, red_target])

def main():
	source = misc.imread('./2_2/source_rap.png', flatten = True)/255.
	# misc.imsave("source_image_gray.png", source)
	target = misc.imread('./2_2/target_rap.jpg', flatten = True)/255.
	# misc.imsave("target_image_gray.png", target)
	mask = misc.imread('./2_2/mask_rap.png', flatten = True)/255.

	### grayscale images only
	target = gray_blend(source, target, mask)

	### color images only
	# target = color_blend(source, target, mask)

	misc.imsave("./2_2/blended_rap.png", target)

main()
