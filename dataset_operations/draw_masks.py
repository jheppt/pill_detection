"""
File: draw_mask.py
Author: Richárd Rádli
E-mail: radli.richard@mik.uni-pannon.hu
Date: Apr 12, 2023

Description: The program based on YOLO format segmentation annotations, creates binary masks of the corresponding
images.
"""

import concurrent.futures
import cv2
import logging
import numpy as np
import os

from numpy import ndarray, dtype
from typing import Tuple, List, Dict, Any

from tqdm.auto import tqdm

from config.dataset_paths_selector import dataset_images_path_selector
from config.json_config import json_config_selector
from utils.utils import file_reader, setup_logger, load_config_json

cfg = (
    load_config_json(
        json_schema_filename=json_config_selector("stream_images").get("schema"),
        json_filename=json_config_selector("stream_images").get("config")
    )
)


# ----------------------------------------------------------------------------------------------------------------------
# -------------------------------------------- P A T H   S E L E C T O R -----------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def path_selector(operation: str) -> dict:
    """
    Selects the correct directory paths based on the given operation string.

    Args:
        operation: A string indicating the operation mode (train or test).

    Returns:
        A dictionary containing directory paths for images, masks, and other related files.

    Raises ValueError:
        If the operation string is not "train", "valid", "test" or "whole".
    """

    dataset_type = cfg.get("dataset_type")

    if operation.lower() == "customer":
        path_to_images = {
            "images": dataset_images_path_selector(dataset_type).get(operation).get("customer_images"),
            "labels": dataset_images_path_selector(dataset_type).get(operation).get("customer_segmentation_labels"),
            "masks": dataset_images_path_selector(dataset_type).get(operation).get("customer_mask_images")
        }
    elif operation.lower() == "reference":
        path_to_images = {
            "images": dataset_images_path_selector(dataset_type).get(operation).get("reference_images"),
            "labels": dataset_images_path_selector(dataset_type).get(operation).get("reference_segmentation_labels"),
            "masks": dataset_images_path_selector(dataset_type).get(operation).get("reference_mask_images")
        }
    else:
        raise ValueError("Wrong operation!")

    return path_to_images


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------- L O A D   F I L E S --------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def load_files(images_dir: str, labels_dir: str) -> List[str]:
    """
    This function loads the image and label files from two directories: train_dir and labels_dir.
    Args:
        images_dir: it is the path to the directory containing the image files.
        labels_dir: it is the path to the directory containing the corresponding label files for each image

    Returns:
         two lists of file paths: image_files and text_files.
    """

    if not os.path.isdir(images_dir):
        raise ValueError(f"Invalid path: {images_dir} is not a directory")

    if not os.path.isdir(labels_dir):
        raise ValueError(f"Invalid path: {labels_dir} is not a directory")

    image_files = file_reader(images_dir, "jpg")

    if not image_files:
        raise ValueError(f"No image files found in {images_dir}")

    for img in image_files:
        # check if image has corresponding label file in labels_dir
        if not os.path.exists(os.path.join(labels_dir, os.path.basename(img).replace(".jpg", ".txt"))):
            image_files.remove(img)



    return image_files

def convert_yolo_label_to_mask(original_image_file_path, label_file_path):
    original_image = cv2.imread(original_image_file_path, cv2.IMREAD_GRAYSCALE)
    h, w = original_image.shape
    mask = np.zeros((h, w), np.uint8)

    with open(label_file_path, 'r') as f:
        for line in map(lambda x: x.rsplit(), f.readlines()):
            x_points = list(map(lambda x: int(float(x) * w), line[1::2]))
            y_points = list(map(lambda y: int(float(y) * h), line[2::2]))
            pts = np.array(list(zip(x_points, y_points)),np.int32).reshape((-1, 1, 2))
            # cv2.polylines(mask, [pts], True, 255, 1) # Not filling
            cv2.fillPoly(img=mask, pts=[pts], color=[255]) # Filling
    return mask

# ----------------------------------------------------------------------------------------------------------------------
# --------------------------------------------- P R O C E S S   D A T A ------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def process_data(img_file: str, txt_path: str)-> tuple[None, None] | tuple[ndarray[tuple[int, ...], dtype[Any]], str]:
    """
    Given the file paths to an image file and a corresponding text file with object coordinates in YOLO format,
    loads the image, extracts the object coordinates, and creates a binary mask indicating where the object is.

    Args:
        img_file: A string specifying the path to an image file.
        txt_path: A string specifying the path to a text file with YOLO object coordinates.

    Returns:
        A tuple of (img, mask), where img is a PIL Image object representing the loaded image,
        and mask is a numpy array representing a binary mask indicating the object location.
        Returns (None, None) if either file path is invalid.
    """
    original_image = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    h, w = original_image.shape
    mask = np.zeros((h, w), np.uint8)

    label_path = os.path.join(txt_path, os.path.basename(img_file).replace(".jpg", ".txt"))
    if not os.path.exists(label_path):
        return None, None

    with open(label_path, 'r') as f:
        for line in map(lambda x: x.rsplit(), f.readlines()):
            x_points = list(map(lambda x: int(float(x) * w), line[1::2]))
            y_points = list(map(lambda y: int(float(y) * h), line[2::2]))
            pts = np.array(list(zip(x_points, y_points)), np.int32).reshape((-1, 1, 2))
            # cv2.polylines(mask, [pts], True, 255, 1) # Not filling
            mask = cv2.fillPoly(img=mask, pts=[pts], color=(255, 0, 0))

    return mask , img_file


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------- S A V E   M A S K S --------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def save_masks(mask: np.ndarray, img_file: str, path_to_files: Dict[str, str]) -> None:
    """
    This function saves the mask to a given path.

    Args:
        mask: Mask image.
        img_file: path of the image file.
        path_to_files: path to the files.

    Returns:
        None
    """

    name = os.path.basename(img_file)
    save_path = (os.path.join(path_to_files.get("masks"), name))
    save_path = save_path.replace(".jpg", ".png")
    cv2.imwrite(save_path, mask)


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------- M A I N --------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def main(operation: str = "train", batch_size: int = 10) -> None:
    """
    Runs the main processing pipeline.

    Returns:
        None
    """

    setup_logger()
    path_to_files = path_selector(operation)

    img_files = load_files(images_dir=path_to_files.get("images"), labels_dir=path_to_files.get("labels"))
    label_dir = path_to_files.get("labels")

    total_files = len(img_files)
    num_batches = (total_files + batch_size - 1) // batch_size

    with concurrent.futures.ProcessPoolExecutor(max_workers=cfg.get("max_workers")) as executor:
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, total_files)
            batch_img_files = img_files[start_idx:end_idx]

            futures = []
            for img_file in batch_img_files:
                futures.append(executor.submit(process_data, img_file, label_dir))

            for future in tqdm(futures,total=len(batch_img_files),
                                      desc=f"Processing batch {i+1}/{num_batches}"):

                try:
                    mask, image_path = future.result()
                    if mask is not None:
                        save_masks(mask=mask, img_file=image_path, path_to_files=path_to_files)
                except Exception as e:
                    logging.error(f"Error processing {image_path}: {e}")


# ----------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------- __M A I N__ ------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        operations = [ "customer","reference"] #"reference",
        for op in operations:
            main(operation=op, batch_size=500)
    except KeyboardInterrupt as kie:
        logging.error(f"The following error has occurred: {kie}")
