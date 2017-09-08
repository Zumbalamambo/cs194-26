import numpy as np
import skimage as sk
import skimage.io as skio
from sklearn.preprocessing import StandardScaler
from skimage.filters import roberts, sobel, scharr, prewitt
from skimage.feature import canny

# Shifts image by a given x,y displacement
def shift(image, displacement):
    return np.roll(np.roll(image, displacement[0], axis = 0), displacement[1], axis = 1)

# aligns two images together using SSD
def align(A, B):
    displacement = [0,0]
    ssd = np.sum(np.sum(np.square(A - B)))
    
    for i in range(-15, 15):
        modx_A = np.roll(A, i, axis = 0)
        for j in range(-15, 15):
            modxy_A = np.roll(modx_A, j, axis = 1)
            mod_ssd = np.sum(np.sum(np.square(modxy_A - B)))
            if (mod_ssd < ssd):
                ssd = mod_ssd
                displacement = [i, j]
    
    return displacement

# pyramid imaging to align two high resolution images.
def pyramid_align(A, B, level):
    if A.shape[0] <= 400 or B.shape[0] <= 400 or level == 5:
        return align(A,B)
    else:
        displacement = pyramid_align(sk.transform.rescale(A, .5), sk.transform.rescale(B, .5), level + 1)
        scaled_displacement = [displacement[0] * 2, displacement[1] * 2]
        rel_A = shift(A, scaled_displacement)
        
        x_reduction = rel_A.shape[0]//2 - 100
        y_reduction = rel_A.shape[1]//2 - 100
        displacement = align(rel_A[x_reduction: - x_reduction, y_reduction: - y_reduction], B[x_reduction: - x_reduction, y_reduction: - y_reduction])
        result = [scaled_displacement[0] + displacement[0], scaled_displacement[1] + displacement[1]]
        return result

# method initializer for pyramid aligning
def pyramid_method(A,B):
    cropped_A = A[800:-800, 800:-800]
    cropped_B = B[800:-800, 800:-800]
    
    edge_A = roberts(cropped_A)
    edge_B = roberts(cropped_B)
    
    return pyramid_align(edge_A, edge_B, 1)

def colorize_jpeg(filename):
    # name of the input file
    imname = filename + ".jpg"

    # read in the image
    im = skio.imread(imname)

    # convert to double (might want to do this later on to save memory)    
    im = sk.img_as_float(im)

    # compute the height of each part (just 1/3 of total)
    height = int(np.floor(im.shape[0] / 3.0))
    
    # separate color channels
    b = im[:height]
    g = im[height: 2*height]
    r = im[2*height: 3*height]
    
    cropped_g = g[20:-20, 20:-20]
    cropped_r = r[20:-20, 20:-20]
    cropped_b = b[20:-20, 20:-20]
    
    edge_g = roberts(cropped_g)
    edge_r = roberts(cropped_r)
    edge_b = roberts(cropped_b)
    
    ag_displacement = align(edge_g, edge_b)
    ar_displacement = align(edge_r, edge_b)

    ag = shift(g, ag_displacement)
    ar = shift(r, ar_displacement)
    
    print(filename)
    print("green: " + str(ag_displacement[0]) + ", " + str(ag_displacement[1]))
    print("red: " + str(ar_displacement[0]) + ", " + str(ar_displacement[1]))

    # create a color image
    im_out = np.dstack([ar, ag, b])

    # save the image
    fname = 'out_edge_' + filename + '.jpeg'
    skio.imsave(fname, im_out)
    
def colorize_tif(filename):
    # name of the input file
    imname = filename + ".tif"

    # read in the image
    im = skio.imread(imname)

    # convert to double (might want to do this later on to save memory)    
    im = sk.img_as_float(im)

    # compute the height of each part (just 1/3 of total)
    height = int(np.floor(im.shape[0] / 3.0))
    
    # separate color channels
    b = im[:height]
    g = im[height: 2*height]
    r = im[2*height: 3*height]
    
    ag_displacement = pyramid_method(g,b)
    ar_displacement = pyramid_method(r,b)

    ag = shift(g, ag_displacement)
    ar = shift(r, ar_displacement)
    
    print(filename)
    print("green: " + str(ag_displacement[0]) + ", " + str(ag_displacement[1]))
    print("red: " + str(ar_displacement[0]) + ", " + str(ar_displacement[1]))

    # create a color image
    im_out = np.dstack([ar, ag, b])

    # save the image
    fname = 'out_edge_' + filename + '.jpeg'
    skio.imsave(fname, im_out)
    
def main():
    jpg_image_names = ['cathedral', 'monastery', 'nativity', 'settlers']
    tif_image_names = ['beans', 'squad', 'hut', 'emir', 'harvesters', 'icon', 'lady', 'self_portrait', 'three_generations', 'train', 'turkmen', 'village']
    
    for filename in jpg_image_names:
        colorize_jpeg(filename)
        
    for filename in tif_image_names:
        colorize_tif(filename)
    
main()
