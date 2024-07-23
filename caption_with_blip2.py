from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch

prompt = "this is an image of"

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
model.to(device)

image = Image.open("../architecture/231206_dataset_architecture_IG_p1/3058051006436847425_1483075987.jpg")

inputs = processor(
    images=image,
    text=prompt,
    return_tensors="pt"
).to(device, torch.float16)

generated_ids = model.generate(**inputs, max_new_tokens=500)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)

print(generated_text)

