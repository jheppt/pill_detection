from config.data_paths import DATA_PATH, DATASET_PATH, IMAGES_PATH


def dataset_images_path_selector(dataset_name: str):
    """
    Selects the correct directory paths based on the given operation string.

    Returns:
        A dictionary containing directory paths for images, masks, and other related files.

    Raises ValueError:
        If the operation string is not "train" or "test".
    """

    path_to_images = {}
    for dataset in ["bmd", "ogyeiv2", "synthetic", "nih"]:
        # -------------------------------------------------- O G Y E I -------------------------------------------------
        path_to_images[dataset] ={
            "customer": {
                "customer_images":
                    DATASET_PATH.get_data_path(f"{dataset}_customer_images"),
                "customer_segmentation_labels":
                    DATASET_PATH.get_data_path(f"{dataset}_customer_segmentation_labels"),
                "customer_mask_images":
                    DATASET_PATH.get_data_path(f"{dataset}_customer_mask_images")
            },

            "reference": {
                "reference_images":
                    DATASET_PATH.get_data_path(f"{dataset}_reference_images"),
                "reference_segmentation_labels":
                    DATASET_PATH.get_data_path(f"{dataset}_reference_segmentation_labels"),
                "reference_mask_images":
                    DATASET_PATH.get_data_path(f"{dataset}_reference_mask_images")
            },

            "unsplitted": {
                "images":
                    DATASET_PATH.get_data_path(f"{dataset}_images"),
                "mask_images":
                    DATASET_PATH.get_data_path(f"{dataset}_mask_images"),
                "segmentation_labels":
                    DATASET_PATH.get_data_path(f"{dataset}_segmentation_labels")
            },

            "train": {
                "images":
                    DATASET_PATH.get_data_path(f"{dataset}_train_images"),
                "mask_images":
                    DATASET_PATH.get_data_path(f"{dataset}_train_mask_images"),
                "segmentation_labels":
                    DATASET_PATH.get_data_path(f"{dataset}_train_segmentation_labels")
            },

            "valid": {
                "images":
                    DATASET_PATH.get_data_path(f"{dataset}_valid_images"),
                "mask_images":
                    DATASET_PATH.get_data_path(f"{dataset}_valid_mask_images"),
                "segmentation_labels":
                    DATASET_PATH.get_data_path(f"{dataset}_valid_segmentation_labels")
            },

            "test": {
                "images":
                    DATASET_PATH.get_data_path(f"{dataset}_test_images"),
                "mask_images":
                    DATASET_PATH.get_data_path(f"{dataset}_test_mask_images"),
                "segmentation_labels":
                    DATASET_PATH.get_data_path(f"{dataset}_test_segmentation_labels"),
            },

            "src_stream_images": {
                "reference": {
                    "stream_images":
                        DATASET_PATH.get_data_path(f"stream_images_{dataset}_reference"),
                    "stream_images_contour":
                        DATASET_PATH.get_data_path(f"stream_images_{dataset}_reference_contour"),
                    "stream_images_lbp":
                        DATASET_PATH.get_data_path(f"stream_images_{dataset}_reference_lbp"),
                    "stream_images_rgb":
                        DATASET_PATH.get_data_path(f"stream_images_{dataset}_reference_rgb"),
                    "stream_images_texture":
                        DATASET_PATH.get_data_path(f"stream_images_{dataset}_reference_texture"),
                },
                "customer": {
                    "stream_images":
                        DATASET_PATH.get_data_path(f"stream_images_{dataset}_customer"),
                    "stream_images_contour":
                        DATASET_PATH.get_data_path(f"stream_images_{dataset}_customer_contour"),
                    "stream_images_lbp":
                        DATASET_PATH.get_data_path(f"stream_images_{dataset}_customer_lbp"),
                    "stream_images_rgb":
                        DATASET_PATH.get_data_path(f"stream_images_{dataset}_customer_rgb"),
                    "stream_images_texture":
                        DATASET_PATH.get_data_path(f"stream_images_{dataset}_customer_texture"),
                }
            },

            "dst_stream_images": {
                'stream_images_anchor':
                    IMAGES_PATH.get_data_path(f"stream_images_{dataset}_anchor"),
                "stream_images_pos_neg":
                    IMAGES_PATH.get_data_path(f"stream_images_{dataset}_pos_neg"),
                'ref':
                    IMAGES_PATH.get_data_path(f"ref_{dataset}_ref"),
                'query':
                    IMAGES_PATH.get_data_path(f"{dataset}_query")
            },

            "other": {
                "k_fold":
                    DATA_PATH.get_data_path(f"{dataset}_k_fold")
            }
        }



    return path_to_images[dataset_name]
