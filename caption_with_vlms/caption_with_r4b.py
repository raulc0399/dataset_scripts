from transformers import AutoModel, AutoProcessor
from PIL import Image
import torch
from utils import get_image_files, load_or_create_metadata, process_images_with_prompts

model_id = "YannQi/R-4B"

model = AutoModel.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
).to("cuda")

# Load processor
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

def process_image(image_path: str, prompt: str, system_prompt: str = None) -> str:
    # Define conversation messages
    messages = []
    
    # Add system message if provided
    if system_prompt:
        messages.append({
            "role": "system",
            "content": system_prompt
        })
    
    messages.append({
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image_path,
            },
            {"type": "text", "text": prompt},
        ],
    })

    # Apply chat template
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        thinking_mode="auto"
    )

    # print("chat template:", text)

    image = Image.open(image_path)

    # Process inputs
    inputs = processor(
        images=image,
        text=text,
        return_tensors="pt"
    ).to("cuda")

    # Generate output
    generated_ids = model.generate(**inputs, max_new_tokens=16384)
    output_ids = generated_ids[0][len(inputs.input_ids[0]):]

    # Decode output
    output_text = processor.decode(
        output_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    
    return output_text.strip()

if __name__ == "__main__":
    from utils import process_images_with_tasks
    
    image_files = get_image_files("./images")
    metadata = load_or_create_metadata(image_files)
    
    process_images_with_tasks(image_files, metadata, process_image, "r4b", image_dir="./images")
