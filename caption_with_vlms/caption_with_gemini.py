from google import genai
from google.genai import types
from utils import get_image_files, load_or_create_metadata, process_images_with_tasks

model_name = "gemini-2.5-flash"
# model_name = "gemini-2.5-pro"
client = genai.Client()

def process_image(image_path: str, prompt: str, system_prompt: str = None) -> str:
    try:
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        
        config = None
        if system_prompt:
            config = types.GenerateContentConfig(
                system_instruction=system_prompt
            )
        
        # Generate content with Gemini
        response = client.models.generate_content(
            model=model_name,
            config=config,
            contents=[
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type='image/jpeg',
                ),
                prompt
            ]
        )

        output_text = response.text
        # print("output text:", output_text)
        
        return output_text.strip()
    
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return ""

if __name__ == "__main__":
    image_files = get_image_files("./images")
    metadata = load_or_create_metadata(image_files)
    
    process_images_with_tasks(image_files, metadata, process_image, "gemini", image_dir="./images")
