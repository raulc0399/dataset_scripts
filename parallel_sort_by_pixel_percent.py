import os
import multiprocessing
import pyvips
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def calculate_black_pixel_percentage(input_image_path, black_threshold=0):
    image = pyvips.Image.new_from_file(input_image_path)
    if image.bands > 1:
        gray_image = image.colourspace('b-w')
    else:
        gray_image = image

    black_percentage = (gray_image <= black_threshold).avg() * 100
    return black_percentage

def main():
    num_processes = 8 # multiprocessing.cpu_count()
    
    base_folder = "../persons-photo-concept-bucket-images-to-train/"
    input_folder = os.path.join(base_folder, "pose")
    
    # Get list of image files
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

    # Create a multiprocessing pool
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Process images in parallel with progress bar
        list(tqdm(pool.imap_unordered(calculate_black_pixel_percentage, image_files), total=len(calculate_black_pixel_percentage), desc="generating histograms"))

if __name__ == "__main__":
    main()