from controlnet_aux.processor import MidasDetector
from PIL import Image

midas = MidasDetector.from_pretrained("lllyasviel/Annotators")

img = Image.open("~/tmp/spaceship.png")
depth_image = midas(img)

depth_image.save("~/tmp/spaceship_depth.png")
