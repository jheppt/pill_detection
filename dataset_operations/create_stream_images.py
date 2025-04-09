"""
File: create_stream_images.py
Author: Richárd Rádli
E-mail: radli.richard@mik.uni-pannon.hu
Date: Apr 12, 2023

Description: The program creates the different images (contour, lbp, rgb, texture) for the substreams.
"""
from math import ceil

import cv2
import logging
import numpy as np
import re
import os
import shutil

from concurrent.futures import ThreadPoolExecutor, wait
from skimage.feature import local_binary_pattern
from tqdm import tqdm
from typing import Tuple

from config.dataset_paths_selector import dataset_images_path_selector as dips
from config.json_config import json_config_selector
from utils.utils import file_reader, measure_execution_time, setup_logger, load_config_json


class CreateStreamImages:
    def __init__(self, operation:str):
        setup_logger()

        self.cfg = (
            load_config_json(
                json_schema_filename=json_config_selector("stream_images").get("schema"),
                json_filename=json_config_selector("stream_images").get("config")
            )
        )

        self.dataset_type = self.cfg.get('dataset_type')
        self.operation = self.cfg.get("operation") if operation is None else operation
        self.max_worker = self.cfg.get("max_worker")
        self.img_size = self.cfg.get("image_size")

        # Output paths
        self.contour_images_path = (
            dips(self.dataset_type).get("src_stream_images").get(self.operation).get("stream_images_contour")
        )
        self.lbp_images_path = (
            dips(self.dataset_type).get("src_stream_images").get(self.operation).get("stream_images_lbp")
        )
        self.rgb_images_path = (
            dips(self.dataset_type).get("src_stream_images").get(self.operation).get("stream_images_rgb")
        )
        self.texture_images_path = (
            dips(self.dataset_type).get("src_stream_images").get(self.operation).get("stream_images_texture")
        )

    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------- U S E  M A S K  & C R O P  I M A G E-------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def apply_mask_and_crop_image(self, in_img: np.ndarray, seg_map: np.ndarray, output_path: str) -> None:
        """
        Applies the Maks to cut out the background and crop the image to the object.

        Args:
            in_img: input testing image
            seg_map: output of the unet for the input testing image
            output_path: where the file should be saved

        Returns:
            None
        """
        # Ensure the segmentation map is binary
        seg_map_binary = (seg_map > 0.5).astype(np.uint8)  # Assuming seg_map is a probability map

        # Ensure the mask has the same size as the input image
        if seg_map_binary.shape != in_img.shape[:2]:
            logging.error(f"The segmentation map and input image must have the same dimensions. Mapshape:{seg_map_binary.shape} vs image_Shape:{in_img.shape[:2]} {output_path}")
            return
        # Apply the mask to the input image
        masked_img = cv2.bitwise_and(in_img, in_img, mask=seg_map_binary)

        # Find contours to get bounding box around the object
        contours, _ = cv2.findContours(seg_map_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            logging.error(f"No contours found for image: {output_path}")
            return

        # Get the obb of the largest contour
        largest_contour = max(contours, key=cv2.contourArea)


        if self.cfg.get("rotate_image"):
            cropped_img = self.rotate_and_crop_image(largest_contour, masked_img)
            if cropped_img is None:
                return
        else:
            # Get the bounding box of the largest contour
            x, y, w, h = cv2.boundingRect(largest_contour)
            cropped_img = masked_img[y:y + h, x:x + w]

        # *** Resize the cropped image to self.img_size with black background padding ***
        cropped_h, cropped_w = cropped_img.shape[:2]

        # Calculate scaling factor to fit into self.img_size without distortion
        scaling_factor = min(self.img_size / cropped_h, self.img_size / cropped_w)

        # Resize the cropped image while maintaining aspect ratio
        new_w = int(cropped_w * scaling_factor)
        new_h = int(cropped_h * scaling_factor)
        resized_img = cv2.resize(cropped_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Create a black canvas of self.img_size x self.img_size
        final_img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

        # Center the resized image on the black canvas
        x_offset = (self.img_size - new_w) // 2
        y_offset = (self.img_size - new_h) // 2
        final_img[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_img

        # Save the cropped image
        cv2.imwrite(output_path, final_img)

    def rotate_and_crop_image(self, largest_contour, masked_img):
        # get obb
        rect = cv2.minAreaRect(largest_contour)
        # Berechnung des Rotationswinkels
        angle = rect[2]
        # Bildmitte
        center = rect[0]
        # Rotationsmatrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
        # Get the rotated image

        rotated_img = cv2.warpAffine(masked_img, rotation_matrix, (masked_img.shape[1], masked_img.shape[0]))

        # Crop the image
        width = int(rect[1][0])
        height = int(rect[1][1])
        cropped_img = cv2.getRectSubPix(rotated_img, (width, height), center)
        if cropped_img is None:
            print("Das Bild wurde nicht richtig geladen oder ist leer!")
            return None
        # Rotate the image back to the original orientation
        if cropped_img.shape[0] > cropped_img.shape[1]:
            cropped_img = cv2.rotate(cropped_img, cv2.ROTATE_90_CLOCKWISE)

        return cropped_img

    # ------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------- P R O C E S S   I M A G E -------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def process_image(self, image_paths: str) -> None:
        """
        Processes a single image by reading in the color and mask images, drawing the bounding box, and saving the
        result.
            image_paths: tuple of paths to the color and mask images
        Returns: None
        """

        color_path= image_paths
        mask_path = color_path.replace("images", "mask_images").replace("jpg", "png")
        output_name = os.path.basename(color_path)
        output_file = (os.path.join(self.rgb_images_path, output_name))
        color_images = cv2.imread(str(color_path), 1)
        mask_images = cv2.imread(str(mask_path), 0)
        if color_images is None or mask_images is None:
            logging.error(f"Error reading image: {color_path} or {mask_path}")
            return
        # check if image is empty
        if np.all(mask_images == 0):
            logging.error(f"Empty mask image: {mask_path}")
            return
        self.apply_mask_and_crop_image(color_images, mask_images, output_file)

    # ------------------------------------------------------------------------------------------------------------------
    # ----------------------------------- S A V E   B O U N D I N G   B O X   I M G S ----------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def save_rgb_images(self) -> None:
        """
        Reads in the images, draws the bounding box, and saves the images in parallel.
        Returns: None
        """

        images = (
            "customer_images" if self.operation == "customer"
            else ("reference_images" if self.operation == "reference"
                  else None)
        )

        masks = (
            "customer_mask_images" if self.operation == "customer"
            else ("reference_mask_images" if self.operation == "reference"
                  else None)
        )

        color_images_dir = dips(self.dataset_type).get(self.operation).get(images)
        mask_images_dir = dips(self.dataset_type).get(self.operation).get(masks)

        color_images = file_reader(color_images_dir, "jpg")

        with ThreadPoolExecutor(max_workers=self.max_worker) as executor:
            list(tqdm(executor.map(self.process_image, color_images), total=len(color_images),
                      desc="RGB images"))

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------ C R E A T E   C O N T O U R   I M A G E S -----------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def create_contour_images(args) -> None:
        """
        Applies edge detection on an input image and saves the resulting edge map.

            args: Tuple containing:
            - cropped_image (numpy.ndarray): The input image to apply edge detection on.
            - output_path (str): The path to save the output edge map.
            - kernel_size (int): The size of the kernel used in median blur.
            - canny_low_thr (int): The lower threshold for Canny edge detection.
            - canny_high_thr (int): The higher threshold for Canny edge detection.
        Returns: None
        """

        cropped_image, output_path, kernel_size, canny_low_thr, canny_high_thr = args
        blured_images = cv2.GaussianBlur(cropped_image, (kernel_size, kernel_size), 0)
        edges = cv2.Canny(blured_images, canny_low_thr, canny_high_thr)
        cv2.imwrite(output_path, edges)

    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------- S A V E   C O N T O U R   I M G S ---------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def save_contour_images(self) -> None:
        """
        Loads RGB images from the designated directory and applies Canny edge detection algorithm to extract contours
        for each image. The resulting images are saved in a new directory designated for contour images.

        Returns: None
        """

        rgb_images = file_reader(self.rgb_images_path, "jpg")
        args_list = []

        for img_path in tqdm(rgb_images, desc="Contour images"):
            output_name = "contour_" + os.path.basename(img_path)
            output_file = os.path.join(self.contour_images_path, output_name)
            bbox_images = cv2.imread(img_path, 0)
            args_list.append(
                (
                    bbox_images,
                    output_file,
                    self.cfg.get("kernel_median_contour"),
                    self.cfg.get("canny_low_thr"),
                    self.cfg.get("canny_high_thr")
                )
            )

        with ThreadPoolExecutor(max_workers=self.max_worker) as executor:
            futures = [executor.submit(self.create_contour_images, args) for args in args_list]
            wait(futures)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------ C R E A T E   T E X T U R E   I M A G E S -----------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def create_texture_images(args) -> None:
        """
        Given a cropped input image, this method applies a Gaussian blur, subtracts the blurred image from the original,
        normalizes the resulting image intensities between 0 and 255, and saves the resulting texture image to a
        given path.

        Args:
            A tuple containing the following elements:
            - cropped_image: A numpy array representing the cropped input image
            - output_path: A string representing the path to save the output texture image
            - kernel_size: An integer representing the size of the Gaussian blur kernel

        Returns:
            None
        """

        cropped_image, output_path, kernel_size = args
        blured_image = cv2.GaussianBlur(cropped_image, kernel_size, 0)
        blured_image = cv2.GaussianBlur(blured_image, (15, 15), 0)
        sub_img = cv2.subtract(cropped_image, blured_image)
        cv2.imwrite(output_path, sub_img*15)

    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------- S A V E   T E X T U R E   I M G S ---------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def save_texture_images(self) -> None:
        """
        Save texture images using Gaussian Blur filter.

        Returns:
            None
        """

        rgb_images = file_reader(self.rgb_images_path, "jpg")
        args_list = []

        for img_path in tqdm(rgb_images, desc="Texture images"):
            output_name = "texture_" + os.path.basename(img_path)
            output_file = os.path.join(self.texture_images_path, output_name)
            bbox_images = cv2.imread(img_path, 0)
            args_list.append((bbox_images, output_file,
                              (self.cfg.get("kernel_gaussian_texture"), self.cfg.get("kernel_gaussian_texture"))))

        with ThreadPoolExecutor(max_workers=self.max_worker) as executor:
            futures = [executor.submit(self.create_texture_images, args) for args in args_list]
            wait(futures)

    # ------------------------------------------------------------------------------------------------------------------
    # --------------------------------------- P R O C E S S   L B P   I M A G E S --------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def process_lbp_image(img_gray: np.ndarray, dst_image_path: str) -> None:
        """
        Process the LBP image for a given image file and save it to the destination directory.

            img_gray: The BGR image as a numpy array.
            dst_image_path: The destination path to save the LBP image.
        Returns: None.
        """

        lbp_image = local_binary_pattern(image=img_gray, P=8, R=2, method="default")
        lbp_image = np.clip(lbp_image, 0, 255)
        cv2.imwrite(dst_image_path, lbp_image)

    # ------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------- S A V E   L B P   I M A G E S ---------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def save_lbp_images(self) -> None:
        """
        Saves LBP images in parallel.
        Returns: None
        """

        rgb_images = file_reader(self.rgb_images_path, "jpg")

        with ThreadPoolExecutor(max_workers=self.max_worker) as executor:
            futures = []
            for img_path in tqdm(rgb_images, desc="LBP images"):
                output_name = "lbp_" + os.path.basename(img_path)
                output_file = os.path.join(self.lbp_images_path, output_name)
                bbox_images = cv2.imread(img_path, 0)
                future = executor.submit(self.process_lbp_image, bbox_images, str(output_file))
                futures.append(future)

            for future in futures:
                future.result()

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------ C R E A T E   L A B E L   D I R S -------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def create_label_dirs(self, rgb_path: str, contour_path: str, texture_path: str, lbp_path: str) -> None:
        """
        Create labeled directories for image files based on the given dataset.

            rgb_path: str, path to the directory containing RGB images.
            contour_path: str, path to the directory containing contour images.
            texture_path: str, path to the directory containing texture images.
            lbp_path: str, path to the directory containing LBP images.
        Returns: None
        """

        files_rgb = sorted(os.listdir(rgb_path))
        files_contour = sorted(os.listdir(contour_path))
        files_texture = sorted(os.listdir(texture_path))
        files_lbp = sorted(os.listdir(lbp_path))

        for idx, file_rgb in tqdm(enumerate(files_rgb), desc="Copying image files"):
            # dataset_type ogyeiv2 or synthetic
            if self.dataset_type == "ogyeiv2":
                if "_s_" in file_rgb:
                    match = re.search(r'^(.*?)_s_\d{3}\.jpg$', file_rgb)
                elif "_u_" in file_rgb:
                    match = re.search(r'^(.*?)_u_\d{3}\.jpg$', file_rgb)
                else:
                    match = None

                if match:
                    value = match.group(1)
                else:
                    raise ValueError(f"Wrong file name: {file_rgb}")
            elif self.dataset_type == "synthetic":
                match = re.search(r'[c|r]_(.*?)_(\d+)\.', file_rgb)
                if match:
                    value = match.group(1)
                else:
                    raise ValueError(f"Wrong file name: {file_rgb}")
            elif self.dataset_type == "nih":
                value = file_rgb.split("_")[0]
            elif self.dataset_type == "cure":
                splits = file_rgb.split("_")
                value = splits[0] + "_" + splits[1]
            elif self.dataset_type == "hunyuan2":
                value = re.search(r'(?<=object_)(\d+)_(\d+)', file_rgb).group(0)
            else:
                raise ValueError(f"Wrong dataset type: {self.dataset_type}")

            out_path_rgb = os.path.join(rgb_path, value)
            os.makedirs(out_path_rgb, exist_ok=True)
            try:
                shutil.move(os.path.join(rgb_path, file_rgb), out_path_rgb)
            except shutil.Error as se:
                logging.error(f"Error moving file: {se.args[0]}")

        class_paths = {}
        for idx, (file_contour, file_texture, file_lbp) in \
                tqdm(enumerate(zip( files_contour, files_texture, files_lbp)), desc="Copying image files"):

            # dataset_type ogyeiv2 or synthetic
            if self.dataset_type == "ogyeiv2":
                if "_s_" in file_contour:
                    match = re.search(r'^(.*?)_s_\d{3}\.jpg$', file_contour)
                elif "_u_" in file_contour:
                    match = re.search(r'^(.*?)_u_\d{3}\.jpg$', file_contour)
                else:
                    match = None

                if match:
                    value = match.group(1)
                else:
                    raise ValueError(f"Wrong file name: {files_contour}")
            elif self.dataset_type == "synthetic":
                if self.cfg.get("less_classes"):
                    match = re.search(r'(?:[rc]_)?(object_[a-zA-Z_]+)', file_contour)
                else:
                    match = re.search(r'contour_[c|r]_(.*?)_(\d+)\.', file_contour)

                if match:
                    value = match.group(1)
                else:
                    raise ValueError(f"Wrong file name: {file_contour}")
            elif self.dataset_type == "nih":
                value = file_contour.split("_")[1]
            elif self.dataset_type == "cure":
                splits= file_contour.split("_")
                value = splits[1] + "_" + splits[2]
            elif self.dataset_type == "hunyuan2":
                value = re.search(r'(?<=object_)(\d+)_(\d+)', file_contour).group(0)
            else:
                raise ValueError(f"Wrong dataset type: {self.dataset_type}")

            if value not in class_paths:
                class_paths[value] = {
                    "contour": [],
                    "texture": [],
                    "lbp": []
                }

            class_paths[value]["contour"].append(os.path.join(contour_path, file_contour))
            class_paths[value]["texture"].append(os.path.join(texture_path, file_texture))
            class_paths[value]["lbp"].append(os.path.join(lbp_path, file_lbp))

        if self.dataset_type == "synthetic" and self.cfg.get("less_classes"):
        # Assuming `class_paths` is already populated
            split_class_paths = {}

            # Iterate through each class
            for class_name, paths in class_paths.items():
                # Initialize class_01 and class_02 for the current class
                split_class_paths[f"{class_name}01"] = {"contour": [], "texture": [], "lbp": []}
                split_class_paths[f"{class_name}02"] = {"contour": [], "texture": [], "lbp": []}


                # Iterate through each data type (contour, texture, lbp)
                for data_type in ["contour", "texture", "lbp"]:
                    files = paths[data_type]
                    files.sort()  # Sort files for consistent splitting

                    # Split files into two halves
                    split_index = ceil(len(files) / 2)
                    split_class_paths[f"{class_name}01"][data_type] = files[:split_index]
                    split_class_paths[f"{class_name}02"][data_type] = files[split_index:]

        else:
            split_class_paths = class_paths

        # Example of accessing the split data
        for class_name, split_data in split_class_paths.items():
            print(f"Class: {class_name}")
            for data_type, data_files in split_data.items():
                print(f"\t{data_type}: {len(data_files)} files")


        for key, value in split_class_paths.items():
            out_path_contour = os.path.join(contour_path, key)
            out_path_texture = os.path.join(texture_path, key)
            out_path_lbp = os.path.join(lbp_path, key)

            os.makedirs(out_path_contour, exist_ok=True)
            os.makedirs(out_path_texture, exist_ok=True)
            os.makedirs(out_path_lbp, exist_ok=True)

            for file_contour, file_texture, file_lbp in zip(value["contour"], value["texture"], value["lbp"]):
                shutil.move(file_contour, out_path_contour)
                shutil.move(file_texture, out_path_texture)
                shutil.move(file_lbp, out_path_lbp)

    # ------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------- M A I N ----------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    @measure_execution_time
    def main(self) -> None:
        """
        Executes the functions to create the images.

        Returns: None
        """

        self.save_rgb_images()
        self.save_contour_images()
        self.save_texture_images()
        self.save_lbp_images()

        self.create_label_dirs(
            rgb_path=self.rgb_images_path,
            contour_path=self.contour_images_path,
            texture_path=self.texture_images_path,
            lbp_path=self.lbp_images_path
        )


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------- __M A I N__ ----------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    for operation in ["customer", "reference"]:
        logging.info(f"Operation: {operation}")
        try:
            proc_unet_images = CreateStreamImages(operation=operation)
            proc_unet_images.main()
        except KeyboardInterrupt as kie:
            logging.error(f'{kie}')
