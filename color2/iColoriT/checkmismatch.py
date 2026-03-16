import os

val_data_path = "/home/sapanagupta/PycharmProjects/color2/iColoriT/docs/imgs/img.jpeg"
hint_dirs = "/home/sapanagupta/PycharmProjects/color2/iColoriT/hint_dirs/h2-n0"

val_images = sorted(os.listdir(val_data_path))
hint_images = sorted(os.listdir(hint_dirs))

assert len(val_images) == len(hint_images), "Mismatch between images and hints."
print("All files match!")
