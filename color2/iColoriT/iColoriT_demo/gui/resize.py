# from PIL import Image
# import os
#
# # Input and output paths
# input_path = "/home/sapanagupta/ICOLORIT_INPUTS/custom_test/I_2605.JPG"
# output_path = "/home/sapanagupta/ICOLORIT_INPUTS/custom_test/imgs/I_2605_224.jpg"
#
# os.makedirs(os.path.dirname(output_path), exist_ok=True)
# img = Image.open(input_path)
# img_224 = img.resize((224, 224), Image.BICUBIC)
# img_224.save(output_path)
# print(f"Saved: {output_path}")
########################################


#import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

img = np.array(Image.open("/home/sapanagupta/ICOLORIT_INPUTS/custom_test/imgs/I_2605_224.jpg"))
#plt.imshow(img)
hints = [
    (161,117,35,54,87),
    (93,93,54,72,96),
    (160,71,35,49,76),
    (152,88,49,67,103),
    (119,69,36,54,76),
    (85,143,49,56,74),
    (69,128,50,65,86),
    (44,167,61,70,85),
    (91,13,88,99,101),
    (67,90,191,200,199),
    (151,177,87,94,100),
    (149,108,97,107,109),
    (59,158,147,149,146),
    (114,55,89,92,85),
    (84,59,156,158,157),
    (47,139,133,122,102),
    (141,216,74,27,9),
    (110,216,64,26,13),
    (67,217,92,44,24),
    (172,214,79,42,23),
]
for x, y, r, g, b in hints:
    plt.scatter(x, y, color=np.array([r,g,b])/255, s=60, edgecolors='black')
plt.title("Hint locations (color=hint RGB)")
plt.show()
