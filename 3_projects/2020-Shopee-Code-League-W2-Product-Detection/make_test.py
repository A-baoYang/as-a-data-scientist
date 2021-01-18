import cv2
import os
# Helper libraries
import pandas as pd
import numpy as np
from tqdm import tqdm


### Read Test Data
all_pic = []
test_images = []
resize_to = 100
channel = 3

filepath = f'test/test_resize_{resize_to}x{resize_to}_channel{channel}/'
all_pic = [filepath + f for f in os.listdir(filepath)]

for pic in tqdm(all_pic):
    pic = cv2.imread(pic)
    test_images.append(pic)

test_images = np.array(test_images)
### scaling
test_images = test_images / 255.0
print('test_images shape:', test_images.shape)
test_images = np.asarray(test_images)
np.save(f'test_images__{resize_to}x{resize_to}_channel{channel}.npy', test_images)
print('All job done.')


