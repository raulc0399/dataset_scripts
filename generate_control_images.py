from controlnet_aux.processor import CannyDetector, MidasDetector
from controlnet_aux.anyline import AnylineDetector
from controlnet_aux.teed import TEEDdetector
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue, Empty
import os
import threading

control_type = "canny"

class ProcessorFactory:
    @staticmethod
    def create_processor(processor_type):
        if processor_type == "canny":
            return CannyDetector()
        
        elif processor_type == "midas":
            return MidasDetector.from_pretrained("lllyasviel/Annotators")
        
        elif processor_type == "anyline":
            return AnylineDetector.from_pretrained(
               "TheMistoAI/MistoLine", filename="MTEED.pth", subfolder="Anyline"
            )
        
        elif processor_type == "teed":
            return TEEDdetector.from_pretrained("fal-ai/teed", filename="5_model.pth")
        
        else:
            raise ValueError("Invalid processor type")

class Processor:
    def __init__(self, processor_type):
        self.processor_type = processor_type
        self.processor = ProcessorFactory.create_processor(processor_type)

    def apply(self, image):
        if self.processor is None:
            return None
        
        output_image = None
        if self.processor_type == "canny":
            output_image = self.processor(image)
        
        elif self.processor_type == "midas":
            self.processor(image)
        
        elif self.processor_type == "anyline":
            self.processor(image, detect_resolution=1024)
        
        elif self.processor_type == "teed":
            self.processor(image, detect_resolution=1024)

        return output_image

class Counter:
    def __init__(self):
        self.value = 0
        self.lock = threading.Lock()

    def increment(self):
        with self.lock:
            self.value += 1
            return self.value

    def get_value(self):
        with self.lock:
            return self.value

def process_image(file_queue, folder, output_folder, processor, pbar, counter):
    while True:
        try:
            file_name = file_queue.get_nowait()
        except Empty:
            break

        file_path = os.path.join(folder, file_name)

        output_file_name = f"{os.path.splitext(file_name)[0]}_{control_type}.jpg"
        output_path = os.path.join(output_folder, output_file_name)

        try:
            img = Image.open(file_path)
            processed_image = processor.apply(img)
            if processed_image is not None:
                processed_image.save(output_path)
                # os.system(f"cp {file_path} {os.path.join(output_folder, file_name)}")
                
                processed_count = counter.increment()
                pbar.set_description(f"Control images: {processed_count}")

        except Exception as e:
            print(f"Error processing {file_name}: {str(e)}")
        
        pbar.update(1)
        file_queue.task_done()

def main(folder, output_folder, num_threads=8):
    files_to_process = []
    files_in_directory = os.listdir(folder)

    for file_name in files_in_directory:
        input_file_path = os.path.join(folder, file_name)

        if os.path.isfile(input_file_path) and input_file_path.lower().endswith(('.jpg', '.jpeg', '.png')) and not file_name.endswith(f"_{control_type}.jpg"):
            files_to_process.append(file_name)

    file_queue = Queue()
    for file in files_to_process:
        file_queue.put(file)

    count_files = len(files_to_process)

    print(f"Processing {count_files} images")

    # Create instances of the detector
    processors = [Processor(control_type) for _ in range(num_threads)]

    print(f"Using {len(processors)} threads")

    counter = Counter()
    with tqdm(total=count_files, desc="Processing images") as pbar:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for i in range(num_threads):
                future = executor.submit(
                    process_image,
                    file_queue,
                    folder,
                    output_folder,
                    processors[i],
                    pbar,
                    counter
                )
                futures.append(future)
            
            # Wait for all tasks to complete
            for future in as_completed(futures):
                future.result()

            print(f"Total processed images: {counter.get_value()}")

if __name__ == "__main__":
    # folder = "./test_images"
    base_folder = "../photo-concept-bucket-images/"
    folder = os.path.join(base_folder, "images")
    output_folder = os.path.join(base_folder, control_type)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    main(folder, output_folder)
