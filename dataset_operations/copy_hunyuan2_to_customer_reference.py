import os
import random
import re
import shutil
import cv2
import numpy as np
from tqdm import tqdm

from config.dataset_paths_selector import dataset_images_path_selector
from utils.utils import file_reader


def main():
    random.seed(42)

    dataset_paths = dataset_images_path_selector("hunyuan2")
    unsplitted = dataset_paths.get("unsplitted")
    customer = dataset_paths.get("customer")
    reference = dataset_paths.get("reference")

    src_imgs = unsplitted.get("images")
    src_labels = unsplitted.get("segmentation_labels")
    src_masks = unsplitted.get("mask_images")

    dst_imgs_customer = customer.get("customer_images")
    dst_masks_customer = customer.get("customer_mask_images")
    dst_labels_customer = customer.get("customer_segmentation_labels")

    dst_imgs_reference = reference.get("reference_images")
    dst_masks_reference = reference.get("reference_mask_images")
    dst_labels_reference = reference.get("reference_segmentation_labels")

    src_img_files = file_reader(src_imgs, "jpg")

    pill_files = {}

    # Group files by pill name
    for f in src_img_files:
        match = re.search(r"(?<=object_)(\d+_\d+)", f)
        if match:
            pill_name = match.group(1)
            if pill_name not in pill_files:
                pill_files[pill_name] = []
            pill_files[pill_name].append(f)

    for pill_name, files in tqdm(pill_files.items(), desc="Processing pills"):
        file_white_pixels = []
        for file in files:
            image_filename = os.path.basename(file)
            mask_filename = image_filename.replace('.jpg', '.png')
            mask_path = os.path.join(src_masks, mask_filename)
            # Check if mask exists
            if not os.path.exists(mask_path):
                print(f"Mask file {mask_path} does not exist. Skipping {image_filename}.")
                continue
            # Read mask and count white pixels
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"Failed to read mask {mask_path}. Skipping {image_filename}.")
                continue
            white_pixels = cv2.countNonZero(mask)
            file_white_pixels.append((file, white_pixels))

        if not file_white_pixels:
            print(f"No valid mask files found for pill {pill_name}. Skipping.")
            continue

        # Sort files by descending white pixel count
        sorted_files = sorted(file_white_pixels, key=lambda x: x[1], reverse=True)
        sorted_filenames = [x[0] for x in sorted_files]

        num_files = len(sorted_filenames)
        if num_files <= 2:
            print(f"Less than 2 files found for pill {pill_name}. Skipping.")
            continue
        else:
            reference_collected_files = sorted_filenames[:2]
            customer_collected_files = sorted_filenames[2:]

        # Copy reference files (largest masks)
        for file in reference_collected_files:
            image_filename = os.path.basename(file)
            mask_filename = image_filename.replace('.jpg', '.png')
            label_filename = image_filename.replace('.jpg', '.txt')

            # Check if label exists
            label_src = os.path.join(src_labels, label_filename)
            if not os.path.exists(label_src):
                print(f"Label file {label_filename} does not exist. Skipping.")
                continue

            # Copy image
            img_dst = os.path.join(dst_imgs_reference, image_filename)
            shutil.copy(file, img_dst)
            # Copy mask
            mask_src = os.path.join(src_masks, mask_filename)
            mask_dst = os.path.join(dst_masks_reference, mask_filename)
            shutil.copy(mask_src, mask_dst)
            # Copy label
            label_dst = os.path.join(dst_labels_reference, label_filename)
            shutil.copy(label_src, label_dst)

        # Copy customer files (smallest masks)
        for file in customer_collected_files:
            image_filename = os.path.basename(file)
            mask_filename = image_filename.replace('.jpg', '.png')
            label_filename = image_filename.replace('.jpg', '.txt')

            label_src = os.path.join(src_labels, label_filename)
            if not os.path.exists(label_src):
                print(f"Label file {label_filename} does not exist. Skipping.")
                continue

            # Copy image
            img_dst = os.path.join(dst_imgs_customer, image_filename)
            shutil.copy(file, img_dst)
            # Copy mask
            mask_src = os.path.join(src_masks, mask_filename)
            mask_dst = os.path.join(dst_masks_customer, mask_filename)
            shutil.copy(mask_src, mask_dst)
            # Copy label
            label_dst = os.path.join(dst_labels_customer, label_filename)
            shutil.copy(label_src, label_dst)


if __name__ == "__main__":
    main()