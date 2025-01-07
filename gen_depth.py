from controlnet_aux.processor import MidasDetector, ZoeDetector, Processor
from PIL import Image
import os

home_dir = os.path.expanduser("~")
img = Image.open(os.path.join(home_dir, "tmp/input_img.png"))

midas = MidasDetector.from_pretrained("lllyasviel/Annotators")
depth_image = midas(img)
depth_image.save(os.path.join(home_dir, "tmp/controlnet_img_midas.png"))

zoe = ZoeDetector.from_pretrained("lllyasviel/Annotators")
depth_image1 = zoe(img)
depth_image1.save(os.path.join(home_dir, "tmp/controlnet_img_zoe.png"))

p = Processor("depth_leres")
depth_image2 = p(img)
depth_image2.save(os.path.join(home_dir, "tmp/controlnet_img_leres.png"))

p1 = Processor("depth_leres++")
depth_image3 = p1(img)
depth_image3.save(os.path.join(home_dir, "tmp/controlnet_img_lerespp.png"))
