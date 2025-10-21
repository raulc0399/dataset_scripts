import openai
import base64
from utils import get_image_files, load_or_create_metadata, process_images_with_prompts

# model="o4-mini"
# model="gpt-4o-mini"
# model="gpt-5-mini"
model="gpt-5"

# Set up OpenAI client
client = openai.OpenAI()

def encode_image_to_base64(image_path: str) -> str:
    """Convert image to base64 string for OpenAI API"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def process_image(image_path: str, prompt: str) -> str:
    # Encode image to base64
    base64_image = encode_image_to_base64(image_path)
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "high"
                    }
                }
            ]
        }
    ]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages
        )
        
        output_text = response.choices[0].message.content
        # print("output text:", output_text)
        
        return output_text.strip()
    
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return ""

if __name__ == "__main__":
    image_files = get_image_files("./images")
    metadata = load_or_create_metadata(image_files)
    
    process_images_with_prompts(image_files, metadata, process_image, "gpt5", image_dir="./images")
