import os
import random
import re
import shutil

from jinja2.compiler import operators
from tqdm import tqdm

from config.dataset_paths_selector import dataset_images_path_selector
from utils.utils import file_reader


def main():
    random.seed(42)

    src_imgs = dataset_images_path_selector("cure").get("unsplitted").get("images")
    src_labels = dataset_images_path_selector("cure").get("unsplitted").get("segmentation_labels")


    dst_imgs_customer = dataset_images_path_selector("cure").get("customer").get("customer_images")
    dst_masks_customer = dataset_images_path_selector("cure").get("customer").get("customer_mask_images")
    dst_labels_customer = dataset_images_path_selector("cure").get("customer").get("customer_segmentation_labels")

    dst_imgs_reference = dataset_images_path_selector("cure").get("reference").get("reference_images")
    dst_masks_reference = dataset_images_path_selector("cure").get("reference").get("reference_mask_images")
    dst_labels_reference = dataset_images_path_selector("cure").get("reference").get("reference_segmentation_labels")

    src_img_files = file_reader(src_imgs, "jpg")

    pill_files = {}

    for f in src_img_files:
        # only get the filename without the id wich is behind the second "_"
        # 0_bottom_22.jpg -> 0_bottom
        # 0_top_22.jpg -> 0_top
        # 2_bottom_111.jpg -> 2_bottom
        filename = os.path.basename(f).split("_")[0] + "_" + os.path.basename(f).split("_")[1]
        if filename not in pill_files:
            pill_files[filename] = []
        pill_files[filename].append(f)


    for pill_name, files in pill_files.items():
        #random.shuffle(files)

        reference_collected_files = files[2:]
        customer_collected_files = files[:2]
        for file in tqdm(reference_collected_files, total=len(reference_collected_files)):
            image_filenames = os.path.basename(file)
            mask_files = file.replace("images", "gt_masks")
            label_filenames = image_filenames.replace(".jpg", ".txt")

            shutil.copy(file, str(os.path.join(dst_imgs_customer, image_filenames)))
            # shutil.copy(mask_files, str(os.path.join(dst_mask_customer, image_filenames)))
            shutil.copy(os.path.join(src_labels, label_filenames), os.path.join(dst_labels_customer, label_filenames))

        for file in tqdm(customer_collected_files, total=len(customer_collected_files)):
            image_filenames = os.path.basename(file)
            mask_files = file.replace("images", "gt_masks")
            label_filenames = image_filenames.replace(".jpg", ".txt")

            shutil.copy(file, str(os.path.join(dst_imgs_reference, image_filenames)))
            #shutil.copy(mask_files, str(os.path.join(dst_masks_reference, image_filenames)))
            shutil.copy(os.path.join(src_labels, label_filenames), os.path.join(dst_labels_reference, label_filenames))


if __name__ == "__main__":
    main()