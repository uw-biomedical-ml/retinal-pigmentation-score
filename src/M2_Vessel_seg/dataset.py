from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
from torchvision import transforms, utils
import random
from torchvision.utils import save_image
from scipy.ndimage import rotate
from PIL import Image, ImageEnhance
import pandas as pd



log = logging.getLogger('M2_vesselseg.dataset')

class SEDataset_out(Dataset):

    def __init__(self, imgs_dir, label_dir, mask_dir, img_size, dataset_name, pthrehold, uniform, crop_csv, train_or=True):
        self.imgs_dir = imgs_dir
        self.label_dir = label_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.dataset_name = dataset_name
        self.pthrehold = pthrehold
        self.uniform = uniform
        self.train_or = train_or
        self.crop_csv = crop_csv

        fps = pd.read_csv(crop_csv, usecols=['Name']).values.ravel()
        self.file_paths = fps
        
        log.info(f'Creating dataset with {len(self.file_paths)} examples')

    def __len__(self):
        return len(self.file_paths)

    @classmethod
    def pad_imgs(self, imgs, img_size):
        img_h,img_w=imgs.shape[0], imgs.shape[1]
        target_h,target_w=img_size[0],img_size[1] 
        if len(imgs.shape)==3:
            d=imgs.shape[2]
            padded=np.zeros((target_h, target_w,d))
        elif len(imgs.shape)==2:
            padded=np.zeros((target_h, target_w))
        padded[(target_h-img_h)//2:(target_h-img_h)//2+img_h,(target_w-img_w)//2:(target_w-img_w)//2+img_w,...]=imgs
        return padded

    @classmethod
    def random_perturbation(self,imgs):
        for i in range(imgs.shape[0]):
            im=Image.fromarray(imgs[i,...].astype(np.uint8))
            en=ImageEnhance.Color(im)
            im=en.enhance(random.uniform(0.8,1.2))
            imgs[i,...]= np.asarray(im).astype(np.float32)
        return imgs 

    @classmethod
    def crop_img(self, crop_csv, f_path, pil_img):
        """ Code to crop the input image based on the crops stored in a csv. This is done to save space and having to store intermediate cropped
        files.
        Params:
        crop_csv - csv containing the name with filepath stored at gv.image_dir, and crop info
        f_path - str containing the filepath to the image
        pil_img - PIL Image of the above f_path
        Returns:
        pil_img - PIL Image cropped by the data in the csv
        """ 

        
        df = pd.read_csv(crop_csv)
        row = df[df['Name'] == f_path]
        
        c_w = row['centre_w']
        c_h = row['centre_h']
        r = row['radius']
        w_min, w_max = int(c_w-r), int(c_w+r) 
        h_min, h_max = int(c_h-r), int(c_h+r)
        
        pil_img = pil_img.crop((h_min, w_min, h_max, w_max))

        return pil_img
 

    @classmethod
    def preprocess(self, img, dataset_name, img_size, train_or, pthrehold):

        img_array = np.array(img)

        if np.sum(img_array[...,2])==0:
            img_array = np.concatenate((img_array[...,1][...,np.newaxis],img_array[...,1][...,np.newaxis],img_array[...,1][...,np.newaxis]),axis=2)
            mean_value=np.mean(img_array[img_array[...,0] > pthrehold],axis=0)
            std_value=np.std(img_array[img_array[...,0] > pthrehold],axis=0)
            img_array=(img_array-mean_value)/std_value        
            
        else:
            mean_value=np.mean(img_array[img_array[...,0] > pthrehold],axis=0)
            std_value=np.std(img_array[img_array[...,0] > pthrehold],axis=0)
            img_array=(img_array-mean_value)/std_value

        if len(img_array.shape) == 2:
            img_array = np.expand_dims(img_array, axis=2)
            
        img_array = img_array.transpose((2, 0, 1))

        return img_array


    def __getitem__(self, i):

        f_path = self.file_paths[i]

        # credit here:  https://github.com/pytorch/pytorch/issues/1137
        try:

            img = Image.open(f_path)
            img = self.crop_img(self.crop_csv, f_path, img)
            ori_width, ori_height = img.size
            img = img.resize(self.img_size)
            img = self.preprocess(img, self.dataset_name, self.img_size, self.train_or, self.pthrehold)

        except:
            # insert log statement
            log.error("Exception occurred", exc_info=True)
            log.warning("file: {} has failed V2 vessel segmentation pre-processing".format(f_path))
            return None

        return {
            'name': f_path.split('/')[-1].split('.')[0],
            'width': ori_width,
            'height': ori_height,
            'image': torch.from_numpy(img).type(torch.FloatTensor)
        }

