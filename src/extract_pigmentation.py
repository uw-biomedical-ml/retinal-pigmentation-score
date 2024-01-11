import matplotlib.image as mpimage
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import binary_dilation, center_of_mass, generate_binary_structure
from skimage.color import rgb2gray, rgb2lab
from skimage.morphology import flood
import numpy as np
import pandas as pd
import os
import logging

log = logging.getLogger("extract_pigmentation")

def convert_to_png(fname):
    return fname.replace('.jpg', '.png')

def remove_dc_island(im):
    """removes the pixels that are not connected to the disc cup segmentations using flood fill"""

    # finds the segmentation in the blue channel (correlates to the cup)
    dc = np.zeros(im.shape[:2])
    dc[np.where(im > 0)[:2]] = 1
    com = center_of_mass(im[:, :, 2] == 1)
    if not np.isnan(
        com
    ).any():  # TODO could be bug if it doesn't creat the correct min and max
        seed = tuple(int(x) for x in com)
        flooded = flood(dc, seed)
        return flooded
    else:
        return np.sum(im, axis=2)  # TODO deal with when there is no disc segmentation


def get_masks(vessel_path, disc_path):
    """code to generate masks with the paths

    Args:
        vessel_path (str): path to vessels
        disc_path (str): path to disc

    Returns:
        masks: masks
    """

    vessel = mpimage.imread(vessel_path)
    artery = vessel[:, :, 0]  # artery in red channel
    vein = vessel[:, :, 2]  # vein in blue channel
    uncertain = vessel[:, :, 1]  # uncertain in green channel
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
    mask[mask > 0] = 1

    # find the background
    gray_im = rgb2gray(raw_im)
    th = np.percentile(gray_im, 0.5)
    mask[gray_im <= th] = 1  # TODO think of a better thresholding

    # dilate the mask it will be scaled by every 600 pixels in the image to roughly have the same dilation irrespective of size
    struct_size = [int(x / 600) for x in raw_im.shape]
    mask = binary_dilation(
        mask,
        structure=generate_binary_structure(
            2,
            2,
        ),
        iterations=4 * struct_size[0],
    )

    # invert values to create boolean array where searchable areas are equal to 1
    inv_mask = np.array(~(mask).astype("bool"))

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

    w_min, w_max = int(c_w - r), int(c_w + r)
    h_min, h_max = int(c_h - r), int(c_h + r)

    crop_img = img.crop((h_min, w_min, h_max, w_max))

    return crop_img


def adjust_to_median(im, mask, med):
    # pass mask and the median values and adjust them to the median
    mask_vals = rgb2lab(np.array(im)[mask])
    mask_med = np.median(mask_vals, axis=0)
    return med / mask_med


def get_pigmentation(config):
    """takes the segmentation masks
    inverts and dilates the masks
    finds the median pixel value of retinal background
    stores it in L,a,b colorspace

    Args:
        config (dict): config file with parameters

    Returns:
        df (pandas.dataframe): dataframe with the median L,a,b values for each image
    """

    crop_csv = pd.read_csv(config.results_dir + "M1/Good_quality/image_list.csv")
    vp = config.results_dir + "M2/binary_vessel/raw_binary/"
    dp = config.results_dir + "M2/optic_disc_cup/raw/"

    data = {"Name": [], "L": [], "a": [], "b": []}

    for _, row in crop_csv.iterrows():
        im_pth = row.Name
        f = im_pth.split("/")[-1].split('.')[0]

        # convert to .png if ends in jpg
        f = convert_to_png(f)

        try:
            im = Image.open(im_pth)
            im = crop_img(row.centre_w, row.centre_h, row.radius, im)
            masks = get_masks(vp + f + '.png', dp + f +'.png')
            inv_mask = get_inverted_masks(masks, np.array(im))
        except IOError:
            log.warning(
                "{} was not processsed, image size {}, masks size {}".format(im_pth, im.size, [x.sizee for x in masks])
            )
            #TODO add extra debugging here regarding what actually caused the error
            continue

        vals = rgb2lab(np.array(im)[inv_mask])
        med = np.median(vals, axis=0)
        data["Name"].append(im_pth)
        data["L"].append(med[0])
        data["a"].append(med[1])
        data["b"].append(med[2])

        if config.debug == True: #TODO this causes use memory usage if you run this on a loop of patients
            if not os.path.exists(config.results_dir + "debug/"):
                os.makedirs(config.results_dir + "debug/")

            fig = plt.figure(figsize=(10, 10))
            plt.imshow(np.array(im))
            plt.imshow(inv_mask, alpha=0.14)
            plt.title("file:{}".format(f))
            plt.axis("off")
            plt.savefig(config.results_dir + "debug/{}_debug.png".format(f))

    df = pd.DataFrame.from_dict(data)
    return df
