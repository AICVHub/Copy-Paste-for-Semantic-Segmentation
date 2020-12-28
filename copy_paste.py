"""
Unofficial implementation of Copy-Paste for semantic segmentation
"""

from PIL import Image
import cv2
import argparse
import os
import numpy as np


def random_flip_horizontal(mask, img, p=0.5):
    if np.random.random() < p:
        _, w_img, _ = img.shape
        img = img[:, ::-1, :]
        _, w_mask, _ = mask.shape
        mask = mask[:, ::-1, :]
    return mask, img


def img_add(img_src, img_main, mask_src):
    _, _, c = mask_src.shape
    for i in range(c):
        sub_img01 = cv2.add(img_src, np.zeros(np.shape(img_src), dtype=np.uint8), mask=mask_src[:, :, i])
        sub_img02 = cv2.add(img_main, np.zeros(np.shape(img_main), dtype=np.uint8),
                            mask=cv2.resize(mask_src[:, :, i], (img_main.shape[1], img_main.shape[0]),
                                            interpolation=cv2.INTER_NEAREST))
        img_main = img_main - sub_img02 + cv2.resize(sub_img01, (img_main.shape[1], img_main.shape[0]),
                                                     interpolation=cv2.INTER_NEAREST)
    return img_main


def rescale_src(mask_src, img_src, h, w):
    h_src, w_src, _ = mask_src.shape
    max_reshape_ratio = min(h / h_src, w / w_src)
    rescale_ratio = np.random.uniform(0.2, max_reshape_ratio)

    # reshape src img and mask
    rescale_h, rescale_w = int(h_src * rescale_ratio), int(w_src * rescale_ratio)
    mask_src = cv2.resize(mask_src, (rescale_w, rescale_h),
                          interpolation=cv2.INTER_NEAREST)
    img_src = cv2.resize(img_src, (rescale_w, rescale_h),
                         interpolation=cv2.INTER_LINEAR)

    # set paste coord
    py = int(np.random.random() * (h - rescale_h))
    px = int(np.random.random() * (w - rescale_w))

    # paste src img and mask to a zeros background
    img_pad = np.zeros((h, w, 3), dtype=np.uint8)
    mask_pad = np.zeros((h, w, 3), dtype=np.uint8)
    img_pad[py:int(py + h_src * rescale_ratio), px:int(px + w_src * rescale_ratio), :] = img_src
    mask_pad[py:int(py + h_src * rescale_ratio), px:int(px + w_src * rescale_ratio), :] = mask_src

    return mask_pad, img_pad


def main(args):
    # input path
    segclass = os.path.join(args.input_dir, 'SegmentationClass')
    JPEGs = os.path.join(args.input_dir, 'JPEGImages')

    # create output path
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'SegmentationClass'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'JPEGImages'), exist_ok=True)

    masks_path = os.listdir(segclass)
    for mask_path in masks_path:
        # get source mask and img
        mask_src = cv2.imread(os.path.join(segclass, mask_path))
        img_src = cv2.imread(os.path.join(JPEGs, mask_path.replace('.png', '.jpg')))
        mask_src, img_src = random_flip_horizontal(mask_src, img_src)

        # random choice main mask/img
        mask_main_path = np.random.choice(masks_path)
        mask_main = cv2.imread(os.path.join(segclass, mask_main_path))
        img_main = cv2.imread(os.path.join(JPEGs, mask_main_path.replace('.png', '.jpg')))
        mask_main, img_main = random_flip_horizontal(mask_main, img_main)

        # rescale mask_src/img_src to less than mask_main/img_main's size
        h, w, _ = img_main.shape
        mask_src, img_src = rescale_src(mask_src, img_src, h, w)

        img = img_add(img_src, img_main, mask_src)
        mask = img_add(mask_src, mask_main, mask_src)

        mask_filename = "copy_paste_" + mask_path
        img_filename = mask_filename.replace('.png', '.jpg')
        cv2.imwrite(os.path.join(args.output_dir, 'SegmentationClass', mask_filename), mask)
        cv2.imwrite(os.path.join(args.output_dir, 'JPEGImages', img_filename), img)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="VOC_root/VOCdevkit2012/VOC2012", type=str,
                        help="input annotated directory")
    parser.add_argument("--output_dir", default="VOC_root/VOCdevkit2012/VOC2012_copy_paste", type=str,
                        help="output dataset directory")

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    # test_03(args)
    main(args)
