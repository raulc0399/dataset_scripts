from PIL import Image
import os
import numpy as np

home_dir = os.path.expanduser("~")
img = Image.open(os.path.join(home_dir, "tmp/input_img.png"))

head_mask = np.zeros_like(img)
head_mask[90:900,80:1500] = 255
mask_image = Image.fromarray(head_mask)

mask_image.save(os.path.join(home_dir, "tmp/mask_generated.png"))