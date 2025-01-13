from typing import Dict

from config.data_paths import DATA_PATH, IMAGES_PATH


def substream_paths() -> Dict:
    """
    Returns the dictionary containing the configuration details for the four different subnetworks
    (Contour, LBP, RGB, Texture) used in the StreamNetwork phase.

    Returns:
         A dictionary containing the configuration details for the four subnetworks.
    """

    network_config = {}
    # ----------------------------------------------- C O N T O U R ------------------------------------------------
    for stream in ["Contour", "LBP", "RGB", "Texture"]:
        network_config[stream] = {}
        for dataset in ["ogyeiv2", "synthetic", "nih"]:
            network_config[stream][dataset] = {}
            for network in ["EfficientNetV2"]:
                network_config[stream][dataset][network] = {
                "train": {
                    "anchor": IMAGES_PATH.get_data_path(f"{stream.lower()}_stream_{dataset}_anchor"),
                    "pos_neg": IMAGES_PATH.get_data_path(f"{stream.lower()}_stream_{dataset}_pos_neg")
                },
                "test": {
                    "ref": IMAGES_PATH.get_data_path(f"{stream.lower()}_stream_{dataset}_ref"),
                    "query": IMAGES_PATH.get_data_path(f"{stream.lower()}_stream_{dataset}_query")
                },
                "model_weights_dir": {
                    "hmtl": DATA_PATH.get_data_path(f"weights_efficient_net_v2_{stream.lower()}_{dataset}_hmtl"),
                    "dmtl": DATA_PATH.get_data_path(f"weights_efficient_net_v2_{stream.lower()}_{dataset}_dmtl")
                },
                "logs_dir": {
                    "hmtl": DATA_PATH.get_data_path(f"logs_efficient_net_v2_{stream.lower()}_{dataset}_hmtl"),
                    "dmtl": DATA_PATH.get_data_path(f"logs_efficient_net_v2_{stream.lower()}_{dataset}_dmtl"),
                },
                "hardest_samples": {
                    "hmtl": DATA_PATH.get_data_path(f"hardest_samples_efficient_net_v2_{stream.lower()}_{dataset}_hmtl"),
                    "dmtl": DATA_PATH.get_data_path(f"hardest_samples_efficient_net_v2_contour_{dataset}_dmtl")
                }
            }
    return network_config


# ------------------------------------------------------------------------------------------------------------------
# ---------------------------------- G E T   M A I N   N E T W O R K   C O N F I G ---------------------------------
# ------------------------------------------------------------------------------------------------------------------
def stream_network_backbone_paths(dataset_type, network_type) -> Dict:
    """
    Returns a dictionary containing the prediction, plotting, and reference vectors folder paths for different types of
    networks, organized first by dataset and then by network type.

    Args:
        dataset_type (str): The type of dataset to use.
        network_type (str): The type of network, e.g., 'CNN' or 'EfficientNetV2'.

    Returns:
         dict: Dictionary containing the prediction, plotting, and reference vectors folder paths.
    """
    network_configs ={}


    for dataset in ["ogyeiv2", "synthetic", "nih"]:
        network_configs[dataset] = {}
        for network in ["EfficientNetV2"]:
            network_configs[dataset][network] = {
            'prediction_folder': {
                "hmtl": DATA_PATH.get_data_path(f"predictions_efficient_net_v2_{dataset}_hmtl"),
                "dmtl": DATA_PATH.get_data_path(f"predictions_efficient_net_v2_{dataset}_dmtl")
            },
            'plotting_folder': {
                "hmtl": IMAGES_PATH.get_data_path(f"plotting_efficient_net_v2_{dataset}_hmtl"),
                "dmtl": IMAGES_PATH.get_data_path(f"plotting_efficient_net_v2_{dataset}_dmtl")
            },
            'ref_vectors_folder': {
                "hmtl": DATA_PATH.get_data_path(f"reference_vectors_efficient_net_v2_{dataset}_hmtl"),
                "dmtl": DATA_PATH.get_data_path(f"reference_vectors_efficient_net_v2_{dataset}_dmtl"),
            },
            'hard_sample': {
                "hmtl": {
                    "Contour": DATA_PATH.get_data_path(f"hardest_samples_efficient_net_v2_contour_{dataset}_hmtl"),
                    "LBP": DATA_PATH.get_data_path(f"hardest_samples_efficient_net_v2_lbp_{dataset}_hmtl"),
                    "RGB": DATA_PATH.get_data_path(f"hardest_samples_efficient_net_v2_rgb_{dataset}_hmtl"),
                    "Texture": DATA_PATH.get_data_path(f"hardest_samples_efficient_net_v2_texture_{dataset}_hmtl")
                },
                "dmtl": {
                    "Contour": DATA_PATH.get_data_path(f"hardest_samples_efficient_net_v2_contour_{dataset}_dmtl"),
                    "LBP": DATA_PATH.get_data_path(f"hardest_samples_efficient_net_v2_lbp_{dataset}_dmtl"),
                    "RGB": DATA_PATH.get_data_path(f"hardest_samples_efficient_net_v2_rgb_{dataset}_dmtl"),
                    "Texture": DATA_PATH.get_data_path(f"hardest_samples_efficient_net_v2_texture_{dataset}_dmtl")
                }
            }
        },

    return network_configs[dataset_type][network_type]


# ----------------------------------------------------------------------------------------------------------------------
# ------------------------ M A I N   N E T W O R K   C O N F I G   F U S I O N   T R A I N I N G -----------------------
# ----------------------------------------------------------------------------------------------------------------------
def fusion_network_paths(dataset_type: str, network_type: str) -> Dict:
    """
    Returns a dictionary containing the logs, weights, predictions, plotting, and reference vectors folder paths for
    different fusion networks based on the network_type parameter.

    Args:
        dataset_type (str): The type of dataset.
        network_type (str): The type of fusion network, e.g., 'CNNFusionNet' or 'EfficientNetSelfAttention'.

    Returns:
        dict: Dictionary containing the folder paths for logs, weights, predictions, plotting, and reference vectors.
    """

    network_configs = {}
    for dataset in ["ogyeiv2", "synthetic", "nih"]:
        network_configs[dataset] = {
            'EfficientNetV2MultiHeadAttention': {
                'logs_folder': {
                    "hmtl":
                        DATA_PATH.get_data_path(
                            f"logs_fusion_network_efficient_net_v2_multihead_attention_{dataset}_hmtl"
                        ),
                    "dmtl":
                        DATA_PATH.get_data_path(
                            f"logs_fusion_network_efficient_net_v2_multihead_attention_{dataset}_dmtl"
                        ),
                },
                'weights_folder': {
                    "hmtl":
                        DATA_PATH.get_data_path(
                            f"weights_fusion_network_efficient_net_v2_multihead_attention_{dataset}_hmtl"
                        ),
                    "dmtl":
                        DATA_PATH.get_data_path(
                            f"weights_fusion_network_efficient_net_v2_multihead_attention_{dataset}_dmtl"
                        )
                },
                'prediction_folder': {
                    "hmtl":
                        DATA_PATH.get_data_path(
                            f"predictions_fusion_network_efficient_net_v2_multihead_attention_{dataset}_hmtl"
                        ),
                    "dmtl":
                        DATA_PATH.get_data_path(
                            f"predictions_fusion_network_efficient_net_v2_multihead_attention_{dataset}_dmtl"
                        ),
                },
                'plotting_folder': {
                    "hmtl":
                        IMAGES_PATH.get_data_path(
                            f"plotting_fusion_network_efficient_net_v2_multihead_attention_{dataset}_hmtl"
                        ),
                    "dmtl":
                        IMAGES_PATH.get_data_path(
                            f"plotting_fusion_network_efficient_net_v2_multihead_attention_{dataset}_dmtl"
                        ),
                },
                'ref_vectors_folder': {
                    "hmtl":
                        DATA_PATH.get_data_path(
                            f"ref_vec_fusion_network_efficient_net_v2_multihead_attention_{dataset}_hmtl"
                        ),
                    "dmtl":
                        DATA_PATH.get_data_path(
                            f"ref_vec_fusion_network_efficient_net_v2_multihead_attention_{dataset}_dmtl"
                        )
                }
            }
        }
    return network_configs[dataset_type][network_type]
