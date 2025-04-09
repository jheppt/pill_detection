import os
from PIL import Image
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor

def convert_image(input_path, output_path):
    with Image.open(input_path) as img:
        img.save(output_path, "PNG")

def convert_jpg_to_png(input_folder, output_folder):
    # Erstelle den Ausgabeordner, falls er nicht existiert
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with ProcessPoolExecutor() as executor:
        futures = []
        for filename in tqdm(os.listdir(input_folder), desc=f"Processing {input_folder}"):
            if filename.lower().endswith(".jpg") or filename.lower().endswith(".jpeg"):
                input_path = os.path.join(input_folder, filename)
                output_filename = os.path.splitext(filename)[0] + ".png"
                output_path = os.path.join(output_folder, output_filename)
                futures.append(executor.submit(convert_image, input_path, output_path))

        for future in tqdm(futures, desc="Converting images"):
            future.result()



for folder in ["Customer","Reference"]:
    # Beispielnutzung
    input_folder = f"/home/ubuntu/dev/pill_detection/datasets/nih/{folder}/mask_images_jpg"  # Pfad zum Ordner mit JPG-Bildern
    output_folder = f"/home/ubuntu/dev/pill_detection/datasets/nih/{folder}/mask_images"  # Pfad zum Ordner für PNG-Bilder

    convert_jpg_to_png(input_folder, output_folder)