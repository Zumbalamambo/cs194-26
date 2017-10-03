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

from align_image_code import align_images

### 1.1 sharpening
def sharpen(im, k):
	smoothed = ndimage.gaussian_filter(im, k)
	misc.imsave("1_1/karl_smoothed100.png", smoothed)
	detail = np.clip(np.subtract(im, smoothed), 0, 1)
	misc.imsave("1_1/karl_detail100.png", detail)
	return np.clip(np.add(im, detail), 0, 1)

### 1.1 file read
# fname = "./1_1/karl_headshot.png"
# im = misc.imread(fname, flatten = False)/255.
# sharpened_im = sharpen(im, k = 100)
# misc.imsave("1_1/karl_sharpened100.png", sharpened_im)

def low_pass(im, sigma):
	return ndimage.gaussian_filter(im, sigma)

def high_pass(im, sigma):
	low_pass_im = low_pass(im, sigma)
	return np.clip(np.subtract(im,low_pass_im), 0, 1)

def fourier_save(name, im):
	plt.imsave(name, np.log(np.abs(np.fft.fftshift(np.fft.fft2(im)))))

def hybrid_image(im1, im2, sigma1, sigma2):
	# Next align images (this code is provided, but may be improved)
	im1_aligned, im2_aligned = align_images(im1, im2)

	low_pass_im = low_pass(im1_aligned, sigma1)
	skio.imsave("./1_2/karl_filtered.png", low_pass_im)
	gray_low_pass = misc.imread("./1_2/karl_filtered.png", flatten = True)/255.
	fourier_save("./1_2/karl_fft.png", gray_low_pass)
	high_pass_im = high_pass(im2_aligned, sigma2)
	skio.imsave("./1_2/akita_filtered.png", high_pass_im)
	gray_high_pass = misc.imread("./1_2/akita_filtered.png", flatten = True)/255.
	fourier_save("./1_2/akita_fft.png", gray_high_pass)
	return np.clip(low_pass_im + high_pass_im, 0, 1)

def hybrid_save(name, image):
	skio.imsave(name, hybrid)

## 1.2 File reads
# f1 = './1_2/karl.jpg'
# f2 = './1_2/akita.jpg'
# # high sf
# im1 = misc.imread(f1, flatten = True)/255.
# fourier_save("./1_2/karl_orig_fft.png", im1)
# # low sf
# im2 = misc.imread(f2, flatten = True)/255
# fourier_save("./1_2/akita_orig_fft.png", im2)
# ### 1.2: Hybrid images
# sigma1 = 3
# sigma2 = 2

# # im_1 is from far, im_2 is from close
# # for mark/matt: s1 = 2,s2 = 20
# # for joseph/karl: s1 = 4, s2 = 20
# # for karl/akita: s1 = 3, s2 = 2
# hybrid = hybrid_image(im1, im2, sigma1, sigma2)
# hybrid_save("./1_2/dogman.png", hybrid)
hybrid = misc.imread("./1_2/dogman.png", flatten = True)/255.
fourier_save("./1_2/dogman_fft.png", hybrid)

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def stacks(hybrid, N):
	gaus_stack = []
	prev_gaus = copy.copy(hybrid)
	lap_stack = []
	lap_image = copy.copy(hybrid)
	for i in range(0, N):
		gaus_image = low_pass(prev_gaus, 2 ** i)
		lap_image = np.clip(np.subtract(prev_gaus,gaus_image), 0, 1)
		if gaus_image.min() < 0:
			gaus_image -= gaus_image.min()
		gaus_image/= gaus_image.max()
		gaus_stack.append(gaus_image)
		prev_gaus = gaus_image
		if lap_image.min() < 0:
			lap_image -= lap_image.min()
		lap_image/= lap_image.max()
		lap_stack.append(lap_image)
	return lap_stack, gaus_stack

def multi_save(arr, prefix):
	fbasename = 'out_stack.jpg'
	for i in range(0, len(arr)):
		fname = prefix + str(i) + fbasename
		skio.imsave(fname, np.clip(arr[i], -1, 1))


### 1.3: Compute and display Gaussian and Laplacian Pyramids
## suggested number of pyramid levels (your choice)
# N = 5 
# dali = plt.imread("./1_3/dali.jpg")/255.
# lap_stack, gaus_stack = stacks(dali, N)

# multi_save(lap_stack, "./1_3/laps/dali_lap")
# multi_save(gaus_stack, "./1_3/gauss/dali_gaus")

def multi_res_blend(im1, im2):
	# im1_aligned, im2_aligned = align_images(im1, im2);
	im1_aligned, im2_aligned = im1, im2
	N = 5

	im1_lap_stack, im1_gaus_stack = stacks(im1_aligned, N)
	im2_lap_stack, im2_gaus_stack = stacks(im2_aligned, N)

	multi_save(im1_lap_stack, "./1_4/laps/kendrick")
	multi_save(im2_lap_stack, "./1_4/laps/jcole")

	height = len(im1_aligned)
	width = len(im1_aligned[0])
	depth = len(im1_aligned[0][0])

	r_mask = np.zeros((height, width, depth))
	for i in range(int(width/2)):
		r_mask[:, i] = np.ones((height, 1))
	r_mask = ndimage.gaussian_filter(r_mask, 30)

	# r_mask_lap, r_mask_gaus = stacks(r_mask, N)

	im1_arr = []
	im2_arr = []
	blended_stack = []

	for l in range(N):
		im_blend = np.zeros((height, width, depth))
		im1_stack = im1_lap_stack[l]
		im2_stack = im2_lap_stack[l]
		for j in range(width):
			GR = r_mask[:,j][0]
			im1_stack[:,j] = im1_stack[:,j] * GR
			im2_stack[:,j] = im2_stack[:,j] * (1 - GR)
		
		im_blend = np.add(im1_stack,im2_stack)
		blended_stack.append(im_blend)
		im1_arr.append(im1_stack)
		im2_arr.append(im2_stack)

	multi_save(im1_arr, "./1_4/mask_1/kendrick")
	multi_save(im2_arr, "./1_4/mask_2/jcole")
	multi_save(blended_stack, "./1_4/end/layer_fire")

	end_result = np.zeros((height, width, depth))
	for i in range(N):
		end_result += blended_stack[i]

	# if end_result.min() < 0:
	# 	end_result -= end_result.min()
	# end_result /= end_result.max()

	multi_save([end_result], "./1_4/end/blended_fire")

#### 1.4 multires
# im1 = plt.imread("./1_4/kendrick.jpg")/255.
# im2 = plt.imread("./1_4/jcole.jpg")/255.

# multi_res_blend(im1, im2)

def main():
	## 1.1 file read
	fname = input("choose image to sharpen")
	im = misc.imread(fname, flatten = False)/255.
	sharpened_im = sharpen(im, k = 100)
	misc.imsave("1_1/karl_sharpened100.png", sharpened_im)

	## 1.2 File reads
	f1 = input("choose your high pass image")
	f2 = input("choose your low pass image")
	# high sf
	im1 = misc.imread(f1, flatten = True)/255.
	fourier_save("./1_2/karl_orig_fft.png", im1)
	# low sf
	im2 = misc.imread(f2, flatten = True)/255
	fourier_save("./1_2/akita_orig_fft.png", im2)
	### 1.2: Hybrid images
	sigma1 = 3
	sigma2 = 2

	# im_1 is from far, im_2 is from close
	# for mark/matt: s1 = 2,s2 = 20
	# for joseph/karl: s1 = 4, s2 = 20
	# for karl/akita: s1 = 3, s2 = 2
	hybrid = hybrid_image(im1, im2, sigma1, sigma2)
	hybrid_save("./1_2/dogman.png", hybrid)

	## 1.3: Compute and display Gaussian and Laplacian Pyramids
	# suggested number of pyramid levels (your choice)
	N = 5 
	dali = plt.imread("./1_3/dali.jpg")/255.
	lap_stack, gaus_stack = stacks(dali, N)

	multi_save(lap_stack, "./1_3/laps/dali_lap")
	multi_save(gaus_stack, "./1_3/gauss/dali_gaus")

	### 1.4 multires
	f1 = input("choose your first image")
	f2 = input("choose your second image")
	im1 = plt.imread(f1)/255.
	im2 = plt.imread(f2)/255.

	multi_res_blend(im1, im2)

main()
