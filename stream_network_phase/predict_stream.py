"""
File: predict_stream_network.py
Author: Richárd Rádli
E-mail: radli.richard@mik.uni-pannon.hu
Date: Apr 12, 2023

Description: This code implements the inference for the stream network phase.
"""
import random

import colorama
import logging
import os
import pandas as pd
import torch
import wandb

from torchvision import transforms
from tqdm.auto import tqdm
from PIL import Image

from config.json_config import json_config_selector
from config.networks_paths_selector import stream_network_backbone_paths, substream_paths
from stream_network_models.stream_network_selector import StreamNetworkFactory
from utils.utils import (create_timestamp, find_latest_file_in_latest_directory,  plot_ref_query_images, setup_logger,
                         use_gpu_if_available, load_config_json)


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++ P R E D I C T   S T R E A M  +++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class PredictStream:
    # ------------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------- __I N I T__ --------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, type_of_stream: str, dataset_type: str = None, model_type: str = None,
                 amount_of_classes:int=None, amount_of_images_per_class:int=None) -> None:
        # Setup logger

        setup_logger()

        # Load config
        self.cfg = (
            load_config_json(
                json_schema_filename=json_config_selector("stream_net").get("schema"),
                json_filename=json_config_selector("stream_net").get("config")
            )
        )
        # Set up tqdm colours
        colorama.init()

        self.amount_of_classes = amount_of_classes
        self.amount_of_images_per_class = amount_of_images_per_class

        # Create time stamp
        self.timestamp = create_timestamp()

        self.preprocess_rgb = None
        self.preprocess_con_tex_lbp = None

        self.accuracy_top5 = None
        self.accuracy_top1 = None
        self.num_correct_top1 = 0
        self.num_correct_top5 = 0
        self.top5_indices = []
        self.confidence_percentages = None

        # Load configs
        self.model_type = self.cfg.get("dataset_type") if model_type is None else model_type
        self.dataset_type = self.cfg.get("dataset_type") if dataset_type is None else dataset_type
        self.network_type = self.cfg.get("type_of_net")

        self.main_network_config = stream_network_backbone_paths(
                dataset_type=self.model_type,
                network_type=self.network_type

        )[0]

        # Select device
        self.device = use_gpu_if_available()

        # type of stream
        self.type_of_stream = type_of_stream

        # Load networks
        self.network = self.load_network()
        self.network.eval()


        self.network = self.network.to(self.device)

        image_size = self.cfg.get("networks").get(self.network_type).get("image_size")

        # Preprocess images
        self.preprocess_rgb = \
            transforms.Compose(
                [
                    transforms.Resize(
                        (
                            image_size,
                            image_size
                        )
                    ),
                    transforms.CenterCrop(
                        (
                            image_size,
                            image_size
                        )
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ]
            )

        self.preprocess_con_tex_lbp = \
            transforms.Compose(
                [
                    transforms.Resize(
                        (
                            image_size,
                            image_size
                        )
                    ),
                    transforms.CenterCrop(
                        (
                            image_size,
                            image_size
                        )
                    ),
                    transforms.Grayscale(),
                    transforms.ToTensor(),
                ]
            )

        self.plot_dir_folder = self.main_network_config.get('plotting_folder').get(self.cfg.get("type_of_loss_func"))
        self.results_folder = self.main_network_config.get('prediction_folder').get(self.cfg.get("type_of_loss_func"))

    # ------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------- L O A D   N E T W O R K S -------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def load_network(self) -> torch.nn.Module:
        """
        This function loads the pretrained networks, with the latest .pt files

        Return:
             The Contour, LBP, RGB, and Texture networks.
        """

        substream_network_cfg = self.cfg.get("streams").get(self.type_of_stream)


        weight_files_path = (
            substream_paths().get(self.type_of_stream).get(self.model_type).get(self.network_type).get("model_weights_dir").get(self.cfg.get("type_of_loss_func"))
        )

        latest_pt_file = find_latest_file_in_latest_directory(
            path=weight_files_path
        )
        wandb.config.update({"model_weights_dir": latest_pt_file}, allow_val_change=True)

        network_stream = StreamNetworkFactory.create_network(self.network_type, substream_network_cfg)


        network_stream.load_state_dict(torch.load(latest_pt_file))

        return network_stream



    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------ S A V E   R E F E R E N C E   V E C T O R -----------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def save_reference_vector(self, vectors, labels, images_path):
        """

            vectors:
            labels:
            images_path:

        Return:
            None
        """

        ref_save_dir = (
            os.path.join(
                self.main_network_config.get('ref_vectors_folder').get(self.cfg.dataset_type),
                f"{self.timestamp}_{self.cfg.type_of_loss_func}"
            )
        )
        os.makedirs(ref_save_dir, exist_ok=True)
        torch.save({'vectors': vectors,
                    'labels': labels,
                    'images_path': images_path},
                   os.path.join(ref_save_dir, "ref_vectors.pt"))

    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------- G E T   V E C T O R S ---------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def get_vector(self, images_dir: dict, operation: str):
        """
        Args:
            images_dirs:
                Dictionary containing paths to the images for different streams (e.g., 'con', 'lbp', 'rgb', 'tex').
            operation:
                String indicating the type of operation ('query' or 'reference').
            type_of_stream:
                String indicating the type of stream to process (e.g., 'con', 'lbp', 'rgb', 'tex').

        Returns:
            tuple containing dictionaries - vectors, labels, and image_tensors.
        """

        logging.info(f"Processing {operation} images for {self.type_of_stream} stream")
        color = colorama.Fore.BLUE if operation == "query" else colorama.Fore.RED
        medicine_classes = os.listdir(images_dir)

        # sort the classes and take the first amount_of_classes and amount_of_images_per_class
        if self.amount_of_classes is not None:
            medicine_classes = medicine_classes[:self.amount_of_classes]
        else:
            wandb.config.update(d={"amount_of_classes": len(medicine_classes)}, allow_val_change=True)

        vectors = {}
        labels = {}
        images_tensors= {}
        ground_truth_labels = []

        for image_name in tqdm(medicine_classes,
                               desc=color + f"\nProcessing {operation} images for {self.type_of_stream} stream", position=0,
                               leave=True):
            # Collect image paths for the selected stream
            image_paths = os.listdir(os.path.join(images_dir, image_name))

            vectors[image_name] = []
            labels[image_name] = []
            images_tensors[image_name] = []

            # random shuffle the images
            random.shuffle(image_paths)



            for idx, image_path in enumerate(image_paths):
                if self.amount_of_images_per_class is not None and idx >= self.amount_of_images_per_class:
                    break
                # Load and preprocess the image for the selected stream
                image = Image.open(os.path.join(images_dir, image_name, image_path))
                if self.type_of_stream == "RGB":
                    preprocessed_image = self.preprocess_rgb(image)
                else:
                    preprocessed_image = self.preprocess_con_tex_lbp(image)

                # Move to device
                preprocessed_image = preprocessed_image.unsqueeze(0).to(self.device)

                # Forward pass through the network for the selected stream
                with torch.no_grad():
                    vector = self.network(preprocessed_image)

                # Append results to the dictionary
                vectors[image_name].append(vector.cpu())
                images_tensors[image_name].append(preprocessed_image)
                ground_truth_labels.append(image_name)

        logging.info(f"Processing of {operation} images for {self.type_of_stream} stream is complete")
        return vectors, images_tensors, ground_truth_labels

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------- M E A S U R E   C O S S I M   A N D   E U C D I S T ------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def compare_query_and_reference_vectors(self, reference_vectors: dict, query_vectors: dict):
        """
        This method measures the similarity and distance between the query_vectors and all reference_vectors using
        Euclidean distance. It returns the predicted medicine labels based on the closest reference vector, and
        calculates top-1 and top-5 accuracy.

        Args:
            reference_vectors: a dictionary of embedded vectors for the reference set.
            query_vectors: a dictionary of embedded vectors for the query set.

        Returns:
            predicted_medicine_euc_dist: List of predicted labels for each query vector.
            similarity_scores_euc_dist: List of similarity scores for each query vector.
            most_similar_indices_euc_dist: List of indices of the most similar reference vectors.
            accuracy_top1: The top-1 accuracy.
            accuracy_top5: The top-5 accuracy.
        """

        logging.info("Comparing query and reference vectors")

        similarity_scores_euc_dist = []
        predicted_medicine_euc_dist = []
        most_similar_indices_euc_dist = []

        # Flatten all reference vectors into one tensor and track labels
        all_reference_vectors = []
        all_reference_labels = []

        for label, vectors in reference_vectors.items():
            all_reference_vectors.extend([vec.squeeze(0) for vec in vectors])
            all_reference_labels.extend([label] * len(vectors))  # Track the label for each vector

        all_reference_vectors_tensor = torch.stack(
            [torch.as_tensor(vec).to(self.device) for vec in all_reference_vectors])

        total_queries = 0

        for image_name, query_vector_list in tqdm(query_vectors.items(), desc="Comparing process", position=0, leave=True):

            total_queries += len(query_vector_list)
            query_vectors_tensor = torch.stack(
                [torch.as_tensor(vec).squeeze(0).to(self.device) for vec in query_vector_list]
            )

            for idx_query, query_vector in enumerate(query_vectors_tensor):
                scores_euclidean_distance = torch.norm(query_vector - all_reference_vectors_tensor, dim=1)

                # Get the index of the most similar reference vector
                most_similar_index = scores_euclidean_distance.argmin().item()
                most_similar_indices_euc_dist.append(most_similar_index)

                # Get the predicted medicine label (top-1 prediction)
                predicted_medicine = all_reference_labels[most_similar_index]
                predicted_medicine_euc_dist.append(predicted_medicine)

                # Check if the top-1 prediction is correct
                if predicted_medicine == image_name:
                    self.num_correct_top1 += 1

                # Get the top-5 predicted medicines
                top5_indices = torch.argsort(scores_euclidean_distance)[:5]
                top5_predicted_medicines = [all_reference_labels[i] for i in top5_indices]

                # Check if the correct label is in the top-5 predictions
                if image_name in top5_predicted_medicines:
                    self.num_correct_top5 += 1

                # Track the similarity scores for analysis if needed
                similarity_scores_euc_dist.append(scores_euclidean_distance.cpu().tolist())

        # Calculate accuracies
        self.accuracy_top1 = self.num_correct_top1 / total_queries
        self.accuracy_top5 = self.num_correct_top5 / total_queries

        return predicted_medicine_euc_dist

    # ------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------- D I S P L A Y   R E S U L T S ------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def display_results(self, query_vectors: dict, predicted_labels: list) -> None:
        """
        Display the results of the prediction.

        Args:
            query_vectors: dict, containing the query vectors (ground truth labels are the keys of the dictionary).
            predicted_labels: list, predicted labels for the queries.

        Returns:
            None
        """

        # Extract the ground truth labels from the query_vectors dictionary keys
        ground_truth_labels = [label for label, vectors in query_vectors.items() for _ in vectors]

        # Ensure the number of ground truth labels matches the predicted labels
        if len(ground_truth_labels) != len(predicted_labels):
            raise ValueError("The number of ground truth labels and predicted labels must match!")

        # Create dataframe with the results
        df = (
            pd.DataFrame(
                list(zip(ground_truth_labels, predicted_labels)),
                columns=['GT Medicine Name', 'Predicted Medicine Name (ED)']
            )
        )

        # Create statistics dataframe
        df_stat = [
            ["Correctly predicted (Top-1):", f'{self.num_correct_top1}'],
            ["Correctly predicted (Top-5):", f'{self.num_correct_top5}'],
            ["Miss predicted top 1:", f'{len(ground_truth_labels) - self.num_correct_top1}'],
            ["Miss predicted top 5:", f'{len(ground_truth_labels) - self.num_correct_top5}'],
            ['Accuracy (Top-1):', f'{self.accuracy_top1:.4%}'],
            ['Accuracy (Top-5):', f'{self.accuracy_top5:.4%}']
        ]
        wandb.log({"Accuracy (Top-1)": self.accuracy_top1,
                   "Accuracy (Top-5)": self.accuracy_top5,
                   "Miss predicted top 1": len(ground_truth_labels) - self.num_correct_top1,
                   "Miss predicted top 5": len(ground_truth_labels) - self.num_correct_top5,
                   "Correctly predicted (Top-1)": self.num_correct_top1,
                   "Correctly predicted (Top-5)": self.num_correct_top5})

        df_stat = pd.DataFrame(df_stat, columns=['Metric', 'Value'])

        # Set Pandas options for better visibility in logs or console
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)

        # Log and print the result
        logging.info(df)
        logging.info(df_stat)

        # Combine dataframes and save to a file
        df_combined = pd.concat([df, df_stat], ignore_index=True)
        df_combined.to_csv(
            os.path.join(self.results_folder,
                         f"{self.timestamp}_stream_network_prediction.txt"),
            sep='\t', index=True
        )

    # ------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------- M A I N ----------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def predict(self) -> None:
        """
        Executes the pipeline for prediction.

        Returns:
             None
        """

        query_dir = substream_paths().get(self.type_of_stream).get(self.dataset_type).get(self.network_type).get("test").get("query")

        reference_dir = substream_paths().get(self.type_of_stream).get(self.dataset_type).get(self.network_type).get("test").get("ref")


        query_vecs, query_image_tensors, query_lables = self.get_vector(query_dir, "query")
        reference_vecs, reference_image_tensors, _ = self.get_vector(reference_dir, "reference")

        predicted_medicines = self.compare_query_and_reference_vectors(reference_vecs, query_vecs)
        self.display_results(query_vecs, predicted_medicines)

        # Plot query and reference medicines
        plot_dir = (
            os.path.join(
                self.plot_dir_folder,
                f"{self.timestamp}"
            )
        )
        plot_ref_query_images(query_lables, predicted_medicines, query_image_tensors, reference_image_tensors, plot_dir, max_correct=0, max_incorrect=100, save_images=False)

if __name__ == '__main__':
    list_of_datasets = ["ogyeiv2", "nih", "cure"]
    list_of_models = ["ogyeiv2", "nih", "cure", "synthetic", "hunyuan2"]

    for type_of_stream in ["Contour", "LBP", "RGB", "Texture"]:
        for model_type in list_of_models:
            for dataset_type in list_of_datasets:
                project = "stream_network_predict_25classes"
                name = f"{type_of_stream}_{model_type}_on_{dataset_type}"
                amount_of_classes = 25
                amount_of_images_per_class = 2

                wandb.init(project=project,
                           name=name,
                           dir="../wandb_logging",
                           group=type_of_stream,
                           tags=[type_of_stream, model_type,f"on_{dataset_type}"],
                           allow_val_change=True,
                           config = {
                               "type_of_stream": type_of_stream,
                               "model_type": model_type,
                               "dataset_type": dataset_type,
                               "amount_of_classes": amount_of_classes,
                               "amount_of_images_per_class": amount_of_images_per_class,
                               "model_weights_dir": "",
                               "same_data_family": model_type == dataset_type
                           })


                stream = PredictStream(type_of_stream=type_of_stream, dataset_type=dataset_type, model_type=model_type, amount_of_classes=amount_of_classes, amount_of_images_per_class=amount_of_images_per_class)
                stream.predict()
                wandb.finish()
