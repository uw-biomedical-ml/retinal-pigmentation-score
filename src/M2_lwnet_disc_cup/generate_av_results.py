import os, json, sys
import os.path as osp
import argparse
import warnings
from tqdm import tqdm
import cv2
import numpy as np
from skimage.io import imsave
from skimage.util import img_as_ubyte
from skimage.transform import resize
from skimage.color import label2rgb
import shutil
from PIL import Image
from torchvision.utils import save_image
import torch
import torch.nn.functional as F
import torchvision
from .models.get_model import get_arch
from .utils.get_loaders import get_test_dataset
from .utils.model_saving_loading import load_model
from skimage import filters, measure
import skimage
import pandas as pd
from skimage.morphology import skeletonize, remove_small_objects
from pathlib import Path


def intersection(mask, vessel_, it_x, it_y):
    """
    Remove the intersection in case the whole vessel is too long
    """
    x_less = max(0, it_x - 1)
    y_less = max(0, it_y - 1)
    x_more = min(vessel_.shape[0] - 1, it_x + 1)
    y_more = min(vessel_.shape[1] - 1, it_y + 1)

    active_neighbours = (
        (vessel_[x_less, y_less] > 0).astype("float")
        + (vessel_[x_less, it_y] > 0).astype("float")
        + (vessel_[x_less, y_more] > 0).astype("float")
        + (vessel_[it_x, y_less] > 0).astype("float")
        + (vessel_[it_x, y_more] > 0).astype("float")
        + (vessel_[x_more, y_less] > 0).astype("float")
        + (vessel_[x_more, it_y] > 0).astype("float")
        + (vessel_[x_more, y_more] > 0).astype("float")
    )

    if active_neighbours > 2:
        cv2.circle(mask, (it_y, it_x), radius=1, color=(0, 0, 0), thickness=-1)

    return mask, active_neighbours


def misc_measures(true_vessel_arr, pred_vessel_arr):
    cm = confusion_matrix(true_vessel_arr, pred_vessel_arr)
    mse = mean_squared_error(true_vessel_arr, pred_vessel_arr)

    try:
        acc = 1.0 * (cm[0, 0] + cm[1, 1]) / np.sum(cm)
        sensitivity = 1.0 * cm[1, 1] / (cm[1, 0] + cm[1, 1])
        specificity = 1.0 * cm[0, 0] / (cm[0, 1] + cm[0, 0])
        precision = 1.0 * cm[1, 1] / (cm[1, 1] + cm[0, 1])
        G = np.sqrt(sensitivity * specificity)
        F1_score = 2 * precision * sensitivity / (precision + sensitivity)
        iou = 1.0 * cm[1, 1] / (cm[1, 0] + cm[0, 1] + cm[1, 1])
        return acc, sensitivity, specificity, precision, G, F1_score, mse, iou
    except:
        return 0, 0, 0, 0, 0, 0, 0, 0


def evaluate_disc(results_path, label_path):
    if os.path.exists(results_path + ".ipynb_checkpoints"):
        shutil.rmtree(results_path + ".ipynb_checkpoints")
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    seg_list = os.listdir(results_path)

    tot = []
    sent = []
    spet = []
    pret = []
    G_t = []
    F1t = []
    mset = []
    iout = []

    n_val = len(seg_list)

    for i in seg_list:
        label_name = i.split(".")[0] + "_OD.png"
        label_ = cv2.imread(label_path + label_name) / 255
        label_ = label_[..., 0]
        seg_ = cv2.imread(results_path + i)
        seg_ = (seg_ < 255).astype("float")[..., 0]

        acc, sensitivity, specificity, precision, G, F1_score, mse, iou = misc_measures(
            label_.flatten(), seg_.flatten()
        )

        tot.append(acc)
        sent.append(sensitivity)
        spet.append(specificity)
        pret.append(precision)
        G_t.append(G)
        F1t.append(F1_score)
        mset.append(mse)
        iout.append(iou)

    Data4stage2 = pd.DataFrame(
        {
            "ACC": tot,
            "Sensitivity": sent,
            "Specificity": spet,
            "Precision": pret,
            "G_value": G_t,
            "F1-score": F1t,
            "MSE": mset,
            "IOU": iout,
        }
    )
    Data4stage2.to_csv(
        "./results/IDRID_optic/performance.csv", index=None, encoding="utf8"
    )

    # return tot / n_val, sent / n_val, spet / n_val, pret / n_val, G_t / n_val, F1t / n_val, auc_roct / n_val, auc_prt / n_val, iout/n_val, mset/n_val


def prediction_eval(
    model_1,
    model_2,
    model_3,
    model_4,
    model_5,
    model_6,
    model_7,
    model_8,
    test_loader,
    device,
    cfg,
):
    n_val = len(test_loader)
    seg_results_raw_path = "{}M2/optic_disc_cup/raw/".format(cfg.results_dir)

    if not os.path.isdir(seg_results_raw_path):
        os.makedirs(seg_results_raw_path)

    with tqdm(total=n_val, desc="Validation round", unit="batch", leave=False) as pbar:
        for batch in test_loader:
            imgs = batch["image"]
            img_name = batch["name"]
            ori_width = batch["original_sz"][0]
            ori_height = batch["original_sz"][1]
            mask_pred_tensor_small_all = 0

            imgs = imgs.to(device=device, dtype=torch.float32)

            with torch.no_grad():
                _, mask_pred = model_1(imgs)
                mask_pred_tensor_small = mask_pred.clone().detach()
                mask_pred_tensor_small_1 = F.softmax(mask_pred_tensor_small, dim=1)
                mask_pred_tensor_small_all += mask_pred_tensor_small_1.type(
                    torch.FloatTensor
                )

                _, mask_pred = model_2(imgs)
                mask_pred_tensor_small = mask_pred.clone().detach()
                mask_pred_tensor_small_2 = F.softmax(mask_pred_tensor_small, dim=1)
                mask_pred_tensor_small_all += mask_pred_tensor_small_2.type(
                    torch.FloatTensor
                )

                _, mask_pred = model_3(imgs)
                mask_pred_tensor_small = mask_pred.clone().detach()
                mask_pred_tensor_small_3 = F.softmax(mask_pred_tensor_small, dim=1)
                mask_pred_tensor_small_all += mask_pred_tensor_small_3.type(
                    torch.FloatTensor
                )

                _, mask_pred = model_4(imgs)
                mask_pred_tensor_small = mask_pred.clone().detach()
                mask_pred_tensor_small_4 = F.softmax(mask_pred_tensor_small, dim=1)
                mask_pred_tensor_small_all += mask_pred_tensor_small_4.type(
                    torch.FloatTensor
                )

                _, mask_pred = model_5(imgs)
                mask_pred_tensor_small = mask_pred.clone().detach()
                mask_pred_tensor_small_5 = F.softmax(mask_pred_tensor_small, dim=1)
                mask_pred_tensor_small_all += mask_pred_tensor_small_5.type(
                    torch.FloatTensor
                )

                _, mask_pred = model_6(imgs)
                mask_pred_tensor_small = mask_pred.clone().detach()
                mask_pred_tensor_small_6 = F.softmax(mask_pred_tensor_small, dim=1)
                mask_pred_tensor_small_all += mask_pred_tensor_small_6.type(
                    torch.FloatTensor
                )

                _, mask_pred = model_7(imgs)
                mask_pred_tensor_small = mask_pred.clone().detach()
                mask_pred_tensor_small_7 = F.softmax(mask_pred_tensor_small, dim=1)
                mask_pred_tensor_small_all += mask_pred_tensor_small_7.type(
                    torch.FloatTensor
                )

                _, mask_pred = model_8(imgs)
                mask_pred_tensor_small = mask_pred.clone().detach()
                mask_pred_tensor_small_8 = F.softmax(mask_pred_tensor_small, dim=1)
                mask_pred_tensor_small_all += mask_pred_tensor_small_8.type(
                    torch.FloatTensor
                )

                mask_pred_tensor_small_all = (mask_pred_tensor_small_all / 8).to(
                    device=device
                )

                # print(mask_pred_tensor_small_all.is_cuda)
                # print(mask_pred_tensor_small_1.is_cuda)

                uncertainty_map = torch.sqrt(
                    (
                        torch.square(
                            mask_pred_tensor_small_all - mask_pred_tensor_small_1
                        )
                        + torch.square(
                            mask_pred_tensor_small_all - mask_pred_tensor_small_2
                        )
                        + torch.square(
                            mask_pred_tensor_small_all - mask_pred_tensor_small_3
                        )
                        + torch.square(
                            mask_pred_tensor_small_all - mask_pred_tensor_small_4
                        )
                        + torch.square(
                            mask_pred_tensor_small_all - mask_pred_tensor_small_5
                        )
                        + torch.square(
                            mask_pred_tensor_small_all - mask_pred_tensor_small_6
                        )
                        + torch.square(
                            mask_pred_tensor_small_all - mask_pred_tensor_small_7
                        )
                        + torch.square(
                            mask_pred_tensor_small_all - mask_pred_tensor_small_8
                        )
                    )
                    / 8
                )

                _, prediction_decode = torch.max(mask_pred_tensor_small_all, 1)
                prediction_decode = prediction_decode.type(torch.FloatTensor)

                n_img = prediction_decode.shape[0]

                if len(prediction_decode.size()) == 3:
                    torch.unsqueeze(prediction_decode, 0)
                if len(uncertainty_map.size()) == 3:
                    torch.unsqueeze(uncertainty_map, 0)

                for i in range(n_img):
                    img_r = np.zeros(
                        (
                            prediction_decode[i, ...].shape[0],
                            prediction_decode[i, ...].shape[1],
                        )
                    )
                    img_g = np.zeros(
                        (
                            prediction_decode[i, ...].shape[0],
                            prediction_decode[i, ...].shape[1],
                        )
                    )
                    img_b = np.zeros(
                        (
                            prediction_decode[i, ...].shape[0],
                            prediction_decode[i, ...].shape[1],
                        )
                    )

                    img_r[prediction_decode[i, ...] == 1] = 255
                    img_b[prediction_decode[i, ...] == 2] = 255
                    # img_g[prediction_decode[i,...]==3]=255

                    img_b = remove_small_objects(img_b > 0, 50)
                    img_r = remove_small_objects(img_r > 0, 100)

                    img_ = np.concatenate(
                        (
                            img_b[..., np.newaxis],
                            img_g[..., np.newaxis],
                            img_r[..., np.newaxis],
                        ),
                        axis=2,
                    )

                    img_ww = cv2.resize(
                        np.float32(img_) * 255,
                        (int(ori_width[i]), int(ori_height[i])),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    cv2.imwrite(seg_results_raw_path + img_name[i] + ".png", img_ww)

                pbar.update(imgs.shape[0])


class M2_DC_args:
    def __init__(self, cfg):
        self.experiment_path = ""
        self.csv_train = False
        self.seed = 30
        self.model_name = "wnet"
        self.batch_size = cfg.batch_size
        self.grad_acc_steps: 0
        self.min_lr = 1e-08
        self.max_lr = 0.01
        self.cycle_lens = "20/50"
        self.metric = "auc"
        self.im_size = "512"
        self.in_c = 3
        self.do_not_save = False
        self.save_path = "wnet_ALL_three_1024_disc_cup"
        self.csv_test = False
        self.path_test_preds = False
        self.check_point_folder = False
        self.num_workers = cfg.worker
        self.device = cfg.device
        self.experiment_path = "experiments/wnet_All_three_1024_disc_cup/30"
        self.results_path = cfg.results_dir


def M2_disc_cup(cfg):
    args = M2_DC_args(cfg)
    results_path = args.results_path

    device = torch.device(args.device)
    model_name = args.model_name

    experiment_path = args.experiment_path  # this should exist in a config file
    if experiment_path is None:
        raise Exception("must specify path to experiment")

    im_size = tuple([int(item) for item in args.im_size.split(",")])
    if isinstance(im_size, tuple) and len(im_size) == 1:
        tg_size = (im_size[0], im_size[0])
    elif isinstance(im_size, tuple) and len(im_size) == 2:
        tg_size = (im_size[0], im_size[1])
    else:
        sys.exit("im_size should be a number or a tuple of two numbers")

    data_path = "{}M1/Good_quality/".format(results_path)
    crop_csv = "{}M1/Good_quality/image_list.csv".format(results_path)

    test_loader = get_test_dataset(
        data_path,
        crop_csv,
        tg_size=tg_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model_1 = get_arch(model_name, n_classes=3).to(device)
    model_2 = get_arch(model_name, n_classes=3).to(device)
    model_3 = get_arch(model_name, n_classes=3).to(device)
    model_4 = get_arch(model_name, n_classes=3).to(device)
    model_5 = get_arch(model_name, n_classes=3).to(device)
    model_6 = get_arch(model_name, n_classes=3).to(device)
    model_7 = get_arch(model_name, n_classes=3).to(device)
    model_8 = get_arch(model_name, n_classes=3).to(device)

    experiment_path_1 = (
        Path(__file__).parent / "./experiments/wnet_All_three_1024_disc_cup/28/"
    )
    experiment_path_2 = (
        Path(__file__).parent / "./experiments/wnet_All_three_1024_disc_cup/30/"
    )
    experiment_path_3 = (
        Path(__file__).parent / "./experiments/wnet_All_three_1024_disc_cup/32/"
    )
    experiment_path_4 = (
        Path(__file__).parent / "./experiments/wnet_All_three_1024_disc_cup/34/"
    )
    experiment_path_5 = (
        Path(__file__).parent / "./experiments/wnet_All_three_1024_disc_cup/36/"
    )
    experiment_path_6 = (
        Path(__file__).parent / "./experiments/wnet_All_three_1024_disc_cup/38/"
    )
    experiment_path_7 = (
        Path(__file__).parent / "./experiments/wnet_All_three_1024_disc_cup/40/"
    )
    experiment_path_8 = (
        Path(__file__).parent / "./experiments/wnet_All_three_1024_disc_cup/42/"
    )

    model_1, stats = load_model(model_1, experiment_path_1, device)
    model_1.eval()

    model_2, stats = load_model(model_2, experiment_path_2, device)
    model_2.eval()

    model_3, stats = load_model(model_3, experiment_path_3, device)
    model_3.eval()

    model_4, stats = load_model(model_4, experiment_path_4, device)
    model_4.eval()

    model_5, stats = load_model(model_5, experiment_path_5, device)
    model_5.eval()

    model_6, stats = load_model(model_6, experiment_path_6, device)
    model_6.eval()

    model_7, stats = load_model(model_7, experiment_path_7, device)
    model_7.eval()

    model_8, stats = load_model(model_8, experiment_path_8, device)
    model_8.eval()

    prediction_eval(
        model_1,
        model_2,
        model_3,
        model_4,
        model_5,
        model_6,
        model_7,
        model_8,
        test_loader,
        device,
        cfg,
    )
