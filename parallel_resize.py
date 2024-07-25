import os
import multiprocessing
import pyvips
from tqdm import tqdm

def resize_image(args):
    input_path, output_path, resize_to = args
    try:
        # Load the image
        image = pyvips.Image.new_from_file(input_path, access='sequential')
        
        # Calculate the new dimensions
        width, height = image.width, image.height
        if width < height:
            new_width = resize_to
            scale = new_width / width
        else:
            new_height = resize_to
            scale = new_height / height
        
        # Resize the image
        resized = image.resize(scale)
        
        # Save the resized image
        resized.write_to_file(output_path)
    except Exception as e:
        print(f"Error processing {input_path}: {e}")

def main():
    num_processes = 8 # multiprocessing.cpu_count()
    resize_to = 1024

    base_folder = "./photo-concept-bucket-images/"
    input_folder = os.path.join(base_folder, "images")
    output_folder = os.path.join(base_folder, f"resized_{resize_to}")

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get list of image files
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

    # Prepare arguments for multiprocessing
    args_list = [(os.path.join(input_folder, f), os.path.join(output_folder, f), resize_to) for f in image_files]

    # Create a multiprocessing pool
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Process images in parallel with progress bar
        list(tqdm(pool.imap_unordered(resize_image, args_list), total=len(args_list), desc="Resizing images"))

if __name__ == "__main__":
    main()