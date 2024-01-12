from glob import glob
import pandas as pd
import os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from .fundus_prep import process_without_gb, imread, imwrite
from random import sample
from pathlib import Path
import logging
from .csv_tests import validate_csv

log = logging.getLogger("pre-processing")


def create_resolution_information(image_dir):
    # create resolution of 1 for all images if information not known.

    images = glob("{}*.png".format(image_dir))
    print("{} images found with glob".format(len(images)))

    res_csv_pth = Path(__file__).parent / "../resolution_information.csv"
    with open(res_csv_pth, "w") as f:
        f.write("fundus,res\n")
        f.writelines("{},1\n".format(x.split("/")[-1]) for x in images)


def process(image_list, save_path, cfg):
    """Crops each image in the image list to create the smallest square that fits all retinal colored information and
    removes background retina pixels
    """

    radius_list = []
    centre_list_w = []
    centre_list_h = []
    name_list = []
    list_resolution = []
    scale_resolution = []

    resolution_csv_path = Path(__file__).parent / "../resolution_information.csv"
    if not os.path.exists(resolution_csv_path):
        create_resolution_information(cfg.image_dir)
    resolution_list = pd.read_csv(resolution_csv_path)

    for image_path in image_list:
        
        fname = image_path.split('/')[-1]
        # check to see if image already exists, therefore don't do this step
        if os.path.exists("{}M0/images/".format(save_path) + image_path.split("/")[-1]):
            print("continue...")
            continue
        try:
            if (
                len(
                    resolution_list["res"][
                        resolution_list["fundus"] == image_path
                    ].values
                )
                == 0
            ):
                resolution_ = 1
            else:
                resolution_ = resolution_list["res"][
                    resolution_list["fundus"] == image_path
                ].values[0]
            list_resolution.append(resolution_)

            img = imread(image_path)
            
            (   r_img,
                borders,
                mask,
                label,
                radius_list,
                centre_list_w,
                centre_list_h,
            ) = process_without_gb(img, img, radius_list, centre_list_w, centre_list_h)

            if not cfg.sparse:
                imwrite(save_path + fname.split(".")[0] + ".png", r_img)

            name_list.append(image_path)

        except IndexError:
            print(
                "\nThe file {} has not been added to the resolution_information.csv found at {}\n\
                   Please update this file with the script found at /lee_lab_scripts/create_resolution.py and re-run the code".format(
                    image_path, resolution_csv_path
                )
            )
            exit(1)
        except:
            log.error("exception occurred", exc_info=True)
            log.warning("error with {}".format(image_path))

    scale_list = [a * 2 / 912 for a in radius_list]

    scale_resolution = [a * b * 1000 for a, b in zip(list_resolution, scale_list)]

    Data4stage2 = pd.DataFrame(
        {
            "Name": name_list,
            "centre_w": centre_list_w,
            "centre_h": centre_list_h,
            "radius": radius_list,
            "Scale": scale_list,
            "Scale_resolution": scale_resolution
        }
    )
    Data4stage2.to_csv(
        "{}M0/crop_info.csv".format(cfg.results_dir), index=None, encoding="utf8"
    )


def EyeQ_process(cfg):

    #If csv is set in the config file
    if cfg.csv_path:
        validate_csv(cfg.csv_path)
        image_list = pd.read_csv(cfg.csv_path)["Path"].values
    elif cfg.sample_num: # if you want a random sample of images from the image directory
        print("Sampling {} images from {}".format(cfg.sample_num, cfg.image_dir))
        image_list = sample(sorted(os.listdir(cfg.image_dir)), cfg.sample_num)
    else: # if you don't want random sample and all the images are in the same directory
        image_list = sorted(os.listdir(cfg.image_dir))

    save_path = cfg.results_dir + "M0/images/"

    if not os.path.exists("{}".format(save_path)):
        os.makedirs("{}".format(save_path))

    log.info("Number of images to be processed:{}".format(len(image_list)))
    process(image_list, save_path, cfg)
