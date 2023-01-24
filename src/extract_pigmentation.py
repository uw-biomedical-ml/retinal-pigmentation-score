import matplotlib.image as mpimage
from PIL import Image
from scipy.ndimage import binary_dilation, center_of_mass, generate_binary_structure 
from skimage.color import rgb2gray, rgb2lab
from skimage.morphology import flood
import numpy as np
import pandas as pd
from glob import glob


def remove_dc_island(im):
    """ removes the pixels that are not connected to the disc cup segmentations using flood fill"""

    # finds the segmentation in the blue channel (correlates to the cup)
    dc = np.zeros(im.shape[:2])
    dc[np.where(im>0)[:2]] = 1
    com = center_of_mass(im[:,:,2] == 1)
    if not np.isnan(com).any(): #TODO could be bug if it doesn't creat the correct min and max
        seed = tuple(int(x) for x in com)
        flooded = flood(dc, seed)
        return flooded
    else: 
        return np.sum(im, axis=2) #TODO deal with when there is no disc segmentation

def get_masks(vessel_path, disc_path):
    """code to generate masks with the paths

    Args:
        vessel_path (str): path to vessels
        disc_path (str): path to disc

    Returns:
        _type_: _description_
    """

    vessel = mpimage.imread(vessel_path)
    artery = vessel[:,:,0] #artery in red channel 
    vein = vessel[:,:,2]  # vein in blue channel
    uncertain = vessel[:,:,1] # uncertain in green channel
    dc = mpimage.imread(disc_path)
    dc = remove_dc_island(dc) 

    masks = [artery, vein, uncertain, dc]

    return masks

def get_inverted_masks(masks, raw_im):
    """creates the inverted image from an array of the masks and the raw image

    Args:
        masks (array): list of the mask arrays (artery, vein disc)
        raw_im (array): array of the raw image 

    Returns:
        inv_mask (array): array of the backgorund image inverted
    """

    mask = np.sum(masks, axis=0)
    mask[mask>0] = 1

    # find the background
    gray_im = rgb2gray(raw_im)
    th = np.percentile(gray_im, 0.5)
    mask[gray_im <= th] = 1 #TODO think of a better thresholding

    # dilate the mask it will be scaled by every 600 pixels in the image to roughly have the same dilation irrespective of size
    struct_size = [int(x/600) for x in raw_im.shape]
    mask = binary_dilation(mask, structure = generate_binary_structure(2,2,),\
                           iterations=4*struct_size[0]) 

    # invert values to create boolean array where searchable areas are equal to 1
    inv_mask = np.array(~(mask).astype('bool'))

    return inv_mask

def crop_img(c_w, c_h, r, img):
    """Code to crop an image based on the passed crops in the good quality csv with a center and radius

    Args:
        c_w (int): center of img to crop along width dimension
        c_h (int): center of img to crop along height dimension
        radius (int): "radius for the crop
        img (np.array): PIL Image object of image to crop

    Returns:
        crop_im: image that has been cropped
    """
    
    w_min, w_max = int(c_w-r), int(c_w+r) 
    h_min, h_max = int(c_h-r), int(c_h+r)
    
    crop_img = img.crop((h_min, w_min, h_max, w_max))

    return crop_img

def get_pigmentation(config):
    """extracts the median color from the retinal background in the Lab space and
    stores it as a csv
    """

    crop_csv = pd.read_csv(config.results_dir+ "M1/Good_quality/image_list.csv")
    vp = config.results_dir + "M2/binary_vessel/raw_binary/"
    dp = config.results_dir + "M2/optic_disc_cup/raw/"
    out_csv = config.results_dir  + 'retinal_background_lab_values.csv'

    f_list = []
    L = []
    a = []
    b = []

    for _,row in crop_csv.iterrows():

        im_pth = row.Name
        f = im_pth.split('/')[-1]

        im = mpimage.imread(im_pth)
        im = crop_img(row.centre_w, row.centre_h, row.radius, im)
        masks = get_masks(vp+f, dp+f)
        inv_mask = get_inverted_masks(masks, im)
    
        vals = rgb2lab(im[inv_mask])
        med  = np.median(vals, axis=0)
        f_list.append(im_pth)
        L.append(med[0])
        a.append(med[1])
        b.append(med[2])

    data = {'Name': f_list, 'L': L, 'a': a, 'b':b}
    df = pd.DataFrame.from_dict(data)
    print('Lab color values are stored at {}'.format(out_csv))
    df.to_csv(out_csv, index=False)
