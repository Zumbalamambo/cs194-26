from scipy import ndimage
from scipy import misc
import numpy as np
from matplotlib import pyplot as plt
from skimage.filters import gaussian as gaussian_filter
import skimage.io as skio

def create_horizontal_mask(im, fname):
	plt.imshow(im)
	y = int(plt.ginput(1, timeout = 0)[0][1])
	
	top = np.reshape(np.linspace(1, 0, y), (y, 1))
	bottom = np.reshape(np.linspace(0, 1, im.shape[0] - y), (im.shape[0] - y, 1))

	blended_top = np.tile(top, (1, im.shape[1]))
	blended_bottom = np.tile(bottom, (1, im.shape[1]))
	mask = np.vstack((blended_top, blended_bottom))
	skio.imsave("mask_{}.jpg".format(fname), mask)

	return mask

def prepare_stack(im, factor):
	g_stack = [im]
	for i in range(factor):
		im = gaussian_filter(im, 1, multichannel=True)
		g_stack.append(im)
	return g_stack

def tilt_image(im, mask):
	im_blend = np.array(im)
	g_stack = prepare_stack(im, int(10**2))
	for j in range(len(im)):
		GR = mask[j][0]
		i = int((GR * 10)**2)
		im_blend[j] = g_stack[i][j]
	return im_blend

def main():
	fname = "nyc"
	im = misc.imread(fname + ".jpg", flatten = False)/255.
	mask = create_horizontal_mask(im, fname)
	im_tilt = tilt_image(im, mask)
	skio.imsave("tilt_{}_final.jpg".format(fname), im_tilt)
	
main()
