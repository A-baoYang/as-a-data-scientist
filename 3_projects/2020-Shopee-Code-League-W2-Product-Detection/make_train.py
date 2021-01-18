import cv2
import os
# Helper libraries
import numpy as np
from tqdm import tqdm


### Read Train Data
train_images = []
train_labels = []
all_pic = []
resize_to = 100
channel = 3
for cate in range(0, 42):
    filepath = f'train/train_resize_{resize_to}x{resize_to}_channel{channel}/{cate}/'
#    pics = glob.glob(f'D:/JY/2020_Shopee_Code_League/Week2_20200620/train/train_resize/{cate}/*.png')
    for f in os.listdir(filepath):
        all_pic.append(filepath + f)
        train_labels.append(cate)
#    all_pic = [filepath + f for f in os.listdir(filepath)]

#slide = int(len(all_pic)/5)

#for round in range(0, 6):
#    for pic in tqdm(all_pic[round*slide:(round+1)*slide]):
for pic in tqdm(all_pic):
    pic = cv2.imread(pic)
    train_images.append(pic)

train_images = np.array(train_images)
train_images = train_images / 255.0
train_images = np.asarray(train_images)
#np.save(f'train_images_{round}.npy', train_images)
np.save(f'train_images__{resize_to}x{resize_to}_channel{channel}.npy', train_images)
print(len(train_images))
#train_images = []

train_labels = np.array(train_labels)
train_labels = np.asarray(train_labels)
np.save('train_labels.npy', train_labels)
print(len(train_labels))
print('All job done.')



