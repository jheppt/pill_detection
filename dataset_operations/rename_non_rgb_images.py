import os
import re
from shutil import move
from math import ceil


def group_and_split_files(base_path, regex_pattern):
    """
    Gruppiert Dateien basierend auf einem Regex-Muster und teilt sie in _01 und _02 Ordner auf.

    :param base_path: Basisverzeichnis, das durchsucht wird.
    :param regex_pattern: Regex-Muster zum Gruppieren der Dateien.
    """
    # Dictionary für Gruppen
    grouped_files = {}

    # Durchsuche die Verzeichnisse und Dateien
    for root, _, files in os.walk(base_path):
        for file in files:
            # Regex-Matching für den Gruppennamen
            match = re.search(regex_pattern, file)
            if match:
                group_name = match.group(1)  # Gruppiere nach dem ersten Regex-Capture
                if group_name not in grouped_files:
                    grouped_files[group_name] = []
                grouped_files[group_name].append(os.path.join(root, file))

    # Verarbeite die Gruppen
    for group_name, file_paths in grouped_files.items():
        # Zielordner für _01 und _02 erstellen
        folder_01 = os.path.join(base_path, f"{group_name}_01")
        folder_02 = os.path.join(base_path, f"{group_name}_02")
        os.makedirs(folder_01, exist_ok=True)
        os.makedirs(folder_02, exist_ok=True)

        # Dateien sortieren und aufteilen
        file_paths.sort()  # Sortierung für Konsistenz
        split_index = ceil(len(file_paths) / 2)

        # Verschiebe Dateien in die entsprechenden Ordner
        for i, file_path in enumerate(file_paths):
            target_folder = folder_01 if i < split_index else folder_02
            target_path = os.path.join(target_folder, os.path.basename(file_path))
            move(file_path, target_path)

        print(
            f"Processed group '{group_name}': {len(file_paths[:split_index])} files in '_01', {len(file_paths[split_index:])} files in '_02'.")


# Beispielaufruf
base_directory = '/Pfad/zum/Basisverzeichnis'
base_directory = '/home/ubuntu/dev/pill_detection/datasets/synthetic/Reference/stream_images/contour'
pattern = r'(object_[a-zA-Z0-9_]+)_\d+$'  # Regex für allgemeine Gruppennamen
group_and_split_files(base_directory, pattern)


