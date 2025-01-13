import os
import random
import re
import shutil

from jinja2.compiler import operators
from tqdm import tqdm

from config.dataset_paths_selector import dataset_images_path_selector
from dataset_operations.augmentation_utils import rotate_image_segmentation
from utils.utils import file_reader


def main():
    random.seed(42)

    src_imgs = dataset_images_path_selector("synthetic").get("unsplitted").get("images")
    src_labels = dataset_images_path_selector("synthetic").get("unsplitted").get("segmentation_labels")


    dst_imgs_customer = dataset_images_path_selector("synthetic").get("customer").get("customer_images")
    dst_masks_customer = dataset_images_path_selector("synthetic").get("customer").get("customer_mask_images")
    dst_labels_customer = dataset_images_path_selector("synthetic").get("customer").get("customer_segmentation_labels")

    dst_imgs_reference = dataset_images_path_selector("synthetic").get("reference").get("reference_images")
    dst_masks_reference = dataset_images_path_selector("synthetic").get("reference").get("reference_mask_images")
    dst_labels_reference = dataset_images_path_selector("synthetic").get("reference").get("reference_segmentation_labels")

    src_img_files = file_reader(src_imgs, "jpg")


    for file in tqdm(src_img_files):
        # if the name starts with c_ it's a customer image, if it starts with r_ it's a reference image
        image_filenames = os.path.basename(file)
        if image_filenames.startswith("c_"):
            dst_imgs = dst_imgs_customer
            des_masks = dst_masks_customer
            des_labels = dst_labels_customer
        elif image_filenames.startswith("r_"):
            dst_imgs = dst_imgs_reference
            des_masks = dst_masks_reference
            des_labels = dst_labels_reference
        else:
            # error
            print("Error: ", file)
            continue


        image_filenames = os.path.basename(file)
        mask_files = file.replace("images", "gt_masks")
        label_filenames = image_filenames.replace(".jpg", ".txt")

        shutil.copy(file, str(os.path.join(dst_imgs, image_filenames)))
        #shutil.copy(mask_files, str(os.path.join(des_masks, image_filenames)))
        shutil.copy(os.path.join(src_labels, label_filenames), os.path.join(des_labels, label_filenames))


if __name__ == "__main__":
    main()