import os
import re


def rename_files_in_path(path):
    # Gehe durch alle Dateien im angegebenen Pfad
    for filename in os.listdir(path):
        # Überprüfen, ob der Dateiname den String "capsule.XXX" enthält
        new_filename = re.sub(r'capsule\.\d{3}', 'capsule', filename)

        # Wenn sich der Dateiname ändert, benenne die Datei um
        if new_filename != filename:
            old_file_path = os.path.join(path, filename)
            new_file_path = os.path.join(path, new_filename)

            # Datei umbenennen
            os.rename(old_file_path, new_file_path)
            print(f'Renamed: {filename} -> {new_filename}')


# Beispielaufruf

for folder1 in ["Customer", "Reference"]:
    for folder2 in ["images","mask_images", "segmentation_labels"]:
        path_to_files = f'/home/ubuntu/dev/pill_detection/datasets/synthetic/{folder1}/{folder2}'
        rename_files_in_path(path_to_files)

