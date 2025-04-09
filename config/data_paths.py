"""
File: data_paths.py
Author: Richárd Rádli
E-mail: radli.richard@mik.uni-pannon.hu
Date: Apr 12, 2023

Description: The program stores the const values of different variables. There is a main class, named _Const(), and 3
other classes are inherited from that (Images, Data, Dataset).
"""

import logging
import os

from utils.utils import setup_logger


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++ C O N S T ++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class _Const(object):
    # Setup logger
    setup_logger()

    # Select user and according paths
    user = os.getlogin()
    root_mapping = {
        "ubuntu": {
            "STORAGE_ROOT":
                "/home/ubuntu/dev/pill_detection/storage",
            "DATASET_ROOT":
                "/home/ubuntu/dev/pill_detection/datasets",
            "PROJECT_ROOT":
                "/home/ubuntu/dev/pill_detection",
        }
    }

    if user in root_mapping:
        root_info = root_mapping[user]
        STORAGE_ROOT = root_info["STORAGE_ROOT"]
        DATASET_ROOT = root_info["DATASET_ROOT"]
        PROJECT_ROOT = root_info["PROJECT_ROOT"]
    else:
        raise ValueError("Wrong user!")

    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------- C R E A T E   D I R C T O R I E S ---------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    @classmethod
    def create_directories(cls, dirs, root_type) -> None:
        """
        Class method that creates the missing directories.

        Args:
            dirs: These are the directories that the function checks.
            root_type: Either STORAGE or DATASET.

        Returns:
             None
        """

        for _, path in dirs.items():
            if root_type == "STORAGE":
                dir_path = os.path.join(cls.STORAGE_ROOT, path)
            elif root_type == "PROJECT":
                dir_path = os.path.join(cls.PROJECT_ROOT, path)
            elif root_type == "DATASET":
                dir_path = os.path.join(cls.DATASET_ROOT, path)
            else:
                raise ValueError("Wrong root type!")

            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                logging.info(f"Directory {dir_path} has been created")


class ConfigFilePaths(_Const):
 
    dirs_config_paths = {
        "config_augmentation":
            "config/json_files/augmentation_config.json",
        "config_schema_augmentation":
            "config/json_files/augmentation_config_schema.json",

        "config_fusion_net":
            "config/json_files/fusion_net_config.json",
        "config_schema_fusion_net":
            "config/json_files/fusion_net_config_schema.json",

        "config_stream_images":
            "config/json_files/stream_images_config.json",
        "config_schema_stream_images":
            "config/json_files/stream_images_config_schema.json",

        "config_streamnet":
            "config/json_files/streamnet_config.json",
        "config_schema_streamnet":
            "config/json_files/streamnet_config_schema.json"
    }

    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------- I N I T -----------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self):
        super().__init__()

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------ G E T   D A T A   P A T H ---------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def get_data_path(self, key):
        return os.path.join(self.PROJECT_ROOT, self.dirs_config_paths.get(key, ""))


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++ I M A G E S +++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class Images(_Const):
    dirs_images = {}
    for dataset in ["ogyeiv2", "synthetic", "nih", "cure", "hunyuan2"]:
        dirs_images.update({
        f"stream_images_{dataset}_anchor":
        f"images/{dataset}/stream_images/anchor",
        f"stream_images_{dataset}_pos_neg":
        f"images/{dataset}/stream_images/pos_neg",
        # ------------------------------------------------- A N C H O R ------------------------------------------------
        f"contour_stream_{dataset}_anchor":
        f"images/{dataset}/stream_images/anchor/contour",
        f"lbp_stream_{dataset}_anchor":
        f"images/{dataset}/stream_images/anchor/lbp",
        f"rgb_stream_{dataset}_anchor":
        f"images/{dataset}/stream_images/anchor/rgb",
        f"texture_stream_{dataset}_anchor":
        f"images/{dataset}/stream_images/anchor/texture",

        # ----------------------------------------------- P O S   N E G ------------------------------------------------
        f"contour_stream_{dataset}_pos_neg":
            f"images/{dataset}/stream_images/pos_neg/contour",
        f"lbp_stream_{dataset}_pos_neg":
            f"images/{dataset}/stream_images/pos_neg/lbp",
        f"rgb_stream_{dataset}_pos_neg":
            f"images/{dataset}/stream_images/pos_neg/rgb",
        f"texture_stream_{dataset}_pos_neg":
            f"images/{dataset}/stream_images/pos_neg/texture",

        # -------------------------------------------------- Q U E R Y -------------------------------------------------
        f"{dataset}_query":
            f"images/{dataset}/test/query",
        f"contour_stream_{dataset}_query":
            f"images/{dataset}/test/query/contour",
        f"lbp_stream_{dataset}_query":
            f"images/{dataset}/test/query/lbp",
        f"rgb_stream_{dataset}_query":
            f"images/{dataset}/test/query/rgb",
        f"texture_stream_{dataset}_query":
            f"images/{dataset}/test/query/texture",

        # ---------------------------------------------------- R E F ---------------------------------------------------
        f"{dataset}_ref":
            f"images/{dataset}/test/ref",
        f"contour_stream_{dataset}_ref":
            f"images/{dataset}/test/ref/contour",
        f"lbp_stream_{dataset}_ref":
            f"images/{dataset}/test/ref/lbp",
        f"rgb_stream_{dataset}_ref":
            f"images/{dataset}/test/ref/rgb",
        f"texture_stream_{dataset}_ref":
            f"images/{dataset}/test/ref/texture",

        # ------------------------------------ P L O T T I N G   S T R E A M   N E T -----------------------------------
        f"plotting_efficient_net_v2_{dataset}_hmtl":
            f"images/{dataset}/plotting/stream_net/efficient_net_v2/hmtl",
        f"plotting_efficient_net_v2_{dataset}_dmtl":
            f"images/{dataset}/plotting/stream_net/efficient_net_v2/dmtl",

        # ------------------------------------ P L O T T I N G   F U S I O N   N E T -----------------------------------
        f"plotting_fusion_network_efficient_net_v2_multihead_attention_{dataset}_hmtl":
            f"images/{dataset}/plotting/fusion_net/fusion_network_efficient_net_v2_multihead_attention/hmtl",
        f"plotting_fusion_network_efficient_net_v2_multihead_attention_{dataset}_dmtl":
            f"images/{dataset}/plotting/fusion_net/fusion_network_efficient_net_v2_multihead_attention/dmtl",
        })
    

    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------- I N I T -----------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self):
        super().__init__()
        self.create_directories(self.dirs_images, "STORAGE")

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------ G E T   D A T A   P A T H ---------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def get_data_path(self, key):
        return os.path.join(self.STORAGE_ROOT, self.dirs_images.get(key, ""))


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++ D A T A +++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class Data(_Const):
    dirs_data = {}
    for dataset in ["ogyeiv2", "synthetic", "nih", "cure", "hunyuan2"]:
        dirs_data.update({
        # ------------------------------------- W E I G H T S   S T R E A M   N E T ------------------------------------
        # EfficientNetV2 - StreamNetwork
        f"weights_efficient_net_v2_contour_{dataset}_hmtl":
            f"data/{dataset}/weights/stream_net/efficient_net_v2/contour/hmtl",
        f"weights_efficient_net_v2_lbp_{dataset}_hmtl":
            f"data/{dataset}/weights/stream_net/efficient_net_v2/lbp/hmtl",
        f"weights_efficient_net_v2_rgb_{dataset}_hmtl":
            f"data/{dataset}/weights/stream_net/efficient_net_v2/rgb/hmtl",
        f"weights_efficient_net_v2_texture_{dataset}_hmtl":
            f"data/{dataset}/weights/stream_net/efficient_net_v2/texture/hmtl",

        f"weights_efficient_net_v2_contour_{dataset}_dmtl":
            f"data/{dataset}/weights/stream_net/efficient_net_v2/contour/dmtl",
        f"weights_efficient_net_v2_lbp_{dataset}_dmtl":
            f"data/{dataset}/weights/stream_net/efficient_net_v2/lbp/dmtl",
        f"weights_efficient_net_v2_rgb_{dataset}_dmtl":
            f"data/{dataset}/weights/stream_net/efficient_net_v2/rgb/dmtl",
        f"weights_efficient_net_v2_texture_{dataset}_dmtl":
            f"data/{dataset}/weights/stream_net/efficient_net_v2/texture/dmtl",

        # ------------------------------------- W E I G H T S   F U S I O N   N E T ------------------------------------
        f"weights_fusion_network_efficient_net_v2_multihead_attention_{dataset}_hmtl":
            f"data/{dataset}/weights/fusion_net/efficient_net_v2_multihead_attention/hmtl",
        f"weights_fusion_network_efficient_net_v2_multihead_attention_{dataset}_dmtl":
            f"data/{dataset}/weights/fusion_net/efficient_net_v2_multihead_attention/dmtl",

        # --------------------------------------- L O G S   S T R E A M   N E T ----------------------------------------
        # EfficientNetV2
        f"logs_efficient_net_v2_contour_{dataset}_hmtl":
            f"data/{dataset}/logs/stream_net/efficient_net_v2/contour/hmtl",
        f"logs_efficient_net_v2_lbp_{dataset}_hmtl":
            f"data/{dataset}/logs/stream_net/efficient_net_v2/lbp/hmtl",
        f"logs_efficient_net_v2_rgb_{dataset}_hmtl":
            f"data/{dataset}/logs/stream_net/efficient_net_v2/rgb/hmtl",
        f"logs_efficient_net_v2_texture_{dataset}_hmtl":
            f"data/{dataset}/logs/stream_net/efficient_net_v2/texture/hmtl",

        f"logs_efficient_net_v2_contour_{dataset}_dmtl":
            f"data/{dataset}/logs/stream_net/efficient_net_v2/contour/dmtl",
        f"logs_efficient_net_v2_lbp_{dataset}_dmtl":
            f"data/{dataset}/logs/stream_net/efficient_net_v2/lbp/dmtl",
        f"logs_efficient_net_v2_rgb_{dataset}_dmtl":
            f"data/{dataset}/logs/stream_net/efficient_net_v2/rgb/dmtl",
        f"logs_efficient_net_v2_texture_{dataset}_dmtl":
            f"data/{dataset}/logs/stream_net/efficient_net_v2/texture/dmtl",

        # ---------------------------------------- L O G S   F U S I O N   N E T ---------------------------------------
        f"logs_fusion_network_efficient_net_v2_multihead_attention_{dataset}_hmtl":
            f"data/{dataset}/logs/fusion_net/efficient_net_v2_multihead_attention/hmtl",
        f"logs_fusion_network_efficient_net_v2_multihead_attention_{dataset}_dmtl":
            f"data/{dataset}/logs/fusion_net/efficient_net_v2_multihead_attention/dmtl",

        # -------------------------------- P R E D I C T I O N S    S T R E A M   N E T --------------------------------
        # Predictions
        f"predictions_efficient_net_v2_{dataset}_hmtl":
            f"data/{dataset}/predictions/stream_net/efficient_net_v2/hmtl",
        f"predictions_efficient_net_v2_{dataset}_dmtl":
            f"data/{dataset}/predictions/stream_net/efficient_net_v2/dmtl",

        # -------------------------------- P R E D I C T I O N S    F U S I O N   N E T --------------------------------
        # Predictions
        f"predictions_fusion_network_efficient_net_v2_multihead_attention_{dataset}_hmtl":
            f"data/{dataset}/predictions/fusion_net/efficient_net_v2_multihead_attention/hmtl",
        f"predictions_fusion_network_efficient_net_v2_multihead_attention_{dataset}_dmtl":
            f"data/{dataset}/predictions/fusion_net/efficient_net_v2_multihead_attention/dmtl",

        # -------------------------------------------- R E F   V E C T O R S -------------------------------------------
        f"reference_vectors_efficient_net_v2_{dataset}_hmtl":
            f"data/{dataset}/ref_vec/stream_net/efficient_net_v2/hmtl",
        f"reference_vectors_efficient_net_v2_{dataset}_dmtl":
            f"data/{dataset}/ref_vec/stream_net/efficient_net_v2/dmtl",

        # --------------------------------- R E F   V E C T O R S   F U S I O N   N E T --------------------------------
        f"ref_vec_fusion_network_efficient_net_v2_multihead_attention_{dataset}_hmtl":
            f"data/{dataset}/ref_vec/fusion_net/efficient_net_v2_multihead_attention/hmtl",
        f"ref_vec_fusion_network_efficient_net_v2_multihead_attention_{dataset}_dmtl":
            f"data/{dataset}/ref_vec/fusion_net/efficient_net_v2_multihead_attention/dmtl",

        # ---------------------------------------- H A R D E S T   S A M P L E S ---------------------------------------
        f"hardest_samples_efficient_net_v2_contour_{dataset}_hmtl":
            f"data/{dataset}/hardest_samples/efficient_net_v2/contour/hmtl",
        f"hardest_samples_efficient_net_v2_lbp_{dataset}_hmtl":
            f"data/{dataset}/hardest_samples/efficient_net_v2/lbp/hmtl",
        f"hardest_samples_efficient_net_v2_rgb_{dataset}_hmtl":
            f"data/{dataset}/hardest_samples/efficient_net_v2/rgb/hmtl",
        f"hardest_samples_efficient_net_v2_texture_{dataset}_hmtl":
            f"data/{dataset}/hardest_samples/efficient_net_v2/texture/hmtl",

        f"hardest_samples_efficient_net_v2_contour_{dataset}_dmtl":
        f"data/{dataset}/hardest_samples/efficient_net_v2/contour/dmtl",
        f"hardest_samples_efficient_net_v2_lbp_{dataset}_dmtl":
        f"data/{dataset}/hardest_samples/efficient_net_v2/lbp/dmtl",
        f"hardest_samples_efficient_net_v2_rgb_{dataset}_dmtl":
        f"data/{dataset}/hardest_samples/efficient_net_v2/rgb/dmtl",
        f"hardest_samples_efficient_net_v2_texture_{dataset}_dmtl":
        f"data/{dataset}/hardest_samples/efficient_net_v2/texture/dmtl",

        f"{dataset}_k_fold":
            f"data/{dataset}/k_fold"
    })

        

    def __init__(self):
        super().__init__()
        self.create_directories(self.dirs_data, "STORAGE")

    def get_data_path(self, key):
        return os.path.join(self.STORAGE_ROOT, self.dirs_data.get(key, ""))


class NLPData(_Const):
    nlp_data = {
        "pill_names":
            "nlp/csv/pill_names",
        "full_sentence_csv":
            "nlp/csv/full_sentence_csv",
        "vector_distances":
            "nlp/csv/distances",

        "nlp_vector":
            "nlp/npy/nlp_vector",

        "word_vector_vis":
            "nlp/plot/word_vector",
        "elbow":
            "nlp/plot/elbow",
        "silhouette":
            "nlp/plot/silhouette",

        "patient_information_leaflet_doc":
            "nlp/documents/patient_information_leaflet_doc",
        "patient_information_leaflet_docx":
            "nlp/documents/patient_information_leaflet_docx",
        "extracted_features_files":
            "nlp/documents/extracted_features_files"
    }

    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------- I N I T -----------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self):
        super().__init__()
        self.create_directories(self.nlp_data, "STORAGE")

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------ G E T   D A T A   P A T H ---------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def get_data_path(self, key):
        return os.path.join(self.STORAGE_ROOT, self.nlp_data.get(key, ""))


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++ D A T A S E T ++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class Datasets(_Const):
    dirs_dataset = {}
    for dataset in ["ogyeiv2", "synthetic", "nih", "cure", "hunyuan2"]:
        dirs_dataset.update({
        # ------------------------------------------------- O G Y E I --------------------------------------------------
        # CUSTOMER
        f"{dataset}_customer_images":
            f"{dataset}/Customer/images",
        f"{dataset}_customer_segmentation_labels":
            f"{dataset}/Customer/segmentation_labels",
        f"{dataset}_customer_mask_images":
            f"{dataset}/Customer/mask_images",

        # REFERENCE
        f"{dataset}_reference_images":
            f"{dataset}/Reference/images",
        f"{dataset}_reference_segmentation_labels":
            f"{dataset}/Reference/segmentation_labels",
        f"{dataset}_reference_mask_images":
            f"{dataset}/Reference/mask_images",

        # STREAM - Customer
        f"stream_images_{dataset}_customer":
            f"{dataset}/Customer/stream_images",
        f"stream_images_{dataset}_customer_contour":
            f"{dataset}/Customer/stream_images/contour",
        f"stream_images_{dataset}_customer_lbp":
            f"{dataset}/Customer/stream_images/lbp",
        f"stream_images_{dataset}_customer_rgb":
            f"{dataset}/Customer/stream_images/rgb",
        f"stream_images_{dataset}_customer_texture":
            f"{dataset}/Customer/stream_images/texture",

        # STREAM - Reference
        f"stream_images_{dataset}_reference":
            f"{dataset}/Reference/stream_images",
        f"stream_images_{dataset}_reference_contour":
            f"{dataset}/Reference/stream_images/contour",
        f"stream_images_{dataset}_reference_lbp":
            f"{dataset}/Reference/stream_images/lbp",
        f"stream_images_{dataset}_reference_rgb":
            f"{dataset}/Reference/stream_images/rgb",
        f"stream_images_{dataset}_reference_texture":
            f"{dataset}/Reference/stream_images/texture",

        # UNSPLITTED
        f"{dataset}_images":
            f"{dataset}/unsplitted/images",
        f"{dataset}_mask_images":
            f"{dataset}/unsplitted/gt_masks",
        f"{dataset}_segmentation_labels":
            f"{dataset}/unsplitted/labels",

        # SPLITTED
        f"{dataset}_train_images":
            f"{dataset}/splitted/train/images",
        f"{dataset}_train_mask_images":
            f"{dataset}/splitted/train/gt_train_masks",
        f"{dataset}_train_segmentation_labels":
            f"{dataset}/splitted/train/labels",

        f"{dataset}_valid_images":
            f"{dataset}/splitted/valid/images",
        f"{dataset}_valid_mask_images":
            f"{dataset}/splitted/valid/gt_valid_masks",
        f"{dataset}_valid_segmentation_labels":
            f"{dataset}/splitted/valid/labels",

        f"{dataset}_test_images":
            f"{dataset}/splitted/test/images",
        f"{dataset}_test_mask_images":
            f"{dataset}/splitted/test/gt_test_masks",
        f"{dataset}_test_segmentation_labels":
            f"{dataset}/splitted/test/labels",
        })

    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------- I N I T -----------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self):
        super().__init__()
        self.create_directories(self.dirs_dataset, "DATASET")

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------ G E T   D A T A   P A T H ---------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def get_data_path(self, key):
        return os.path.join(self.DATASET_ROOT, self.dirs_dataset.get(key, ""))


CONST: _Const = _Const()
JSON_FILES_PATHS: ConfigFilePaths = ConfigFilePaths()
IMAGES_PATH: Images = Images()
NLP_DATA_PATH: NLPData = NLPData()
DATA_PATH: Data = Data()
DATASET_PATH: Datasets = Datasets()
