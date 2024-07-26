import os
import multiprocessing
import pyvips
import shutil
from tqdm import tqdm

def calculate_black_pixel_percentage(args):
    input_image_path, black_threshold = args
    image = pyvips.Image.new_from_file(input_image_path)

    if image.bands > 1:
        gray_image = image.colourspace('b-w')
    else:
        gray_image = image

    black_percentage = (gray_image <= black_threshold).avg() * 100
    return (input_image_path, black_percentage)

def main():
    num_processes = multiprocessing.cpu_count()
    
    base_folder = "../persons-photo-concept-bucket-images-to-train/"
    input_folder = os.path.join(base_folder, "pose")
    output_folder = os.path.join(base_folder, "pose_filtered")
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Get list of pose image files
    pose_files = [f for f in os.listdir(input_folder) if f.lower().endswith('_pose.jpg')]
    
    # Prepare arguments for multiprocessing
    args = [(os.path.join(input_folder, f), 0) for f in pose_files]
    
    # Create a multiprocessing pool
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Process images in parallel with progress bar
        results = list(tqdm(pool.imap_unordered(calculate_black_pixel_percentage, args), total=len(args), desc="Calculating black pixel percentages"))
    
    # Sort results by black pixel percentage
    sorted_results = sorted(results, key=lambda x: x[1])
    
    # Copy the first 20,000 pairs of files with the least black pixels
    for i, (file_path, _) in enumerate(tqdm(sorted_results[:20000], desc="Copying files")):
        pose_file = os.path.basename(file_path)
        original_file = pose_file.replace('_pose.jpg', '.jpg')
        
        # Copy pose file
        shutil.copy2(file_path, os.path.join(output_folder, pose_file))
        
        # Copy original file
        original_file_path = os.path.join(input_folder, original_file)
        shutil.copy2(original_file_path, os.path.join(output_folder, original_file))

if __name__ == "__main__":
    main()