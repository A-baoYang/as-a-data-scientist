import cv2
import os
import pandas as pd
from tqdm import tqdm

resize_to = 100
channel = 3

dims = list()
for cate in tqdm(range(0, 42)):
    if cate <= 9:
        filepath = f'train/train/0{cate}/'
    else:
        filepath = f'train/train/{cate}/'

    # create folder for resize pics
    try:
        os.makedirs(f'train/train_resize_{resize_to}x{resize_to}_channel{channel}/{cate}')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    all_pic = [filepath + f for f in os.listdir(filepath)]

    for pic_path in tqdm(all_pic):
        img = cv2.imread(pic_path)
        print(img.shape)
        h, w = img.shape[:2]
        name = pic_path.split('/')[3]
        dims.append((name, h, w, cate))

        resized = cv2.resize(img, (resize_to, resize_to), interpolation=cv2.INTER_AREA)
        cv2.imwrite(f'train/train_resize_{resize_to}x{resize_to}_channel{channel}/{cate}/{name}.jpg', resized)
        print(f'resized picture: {cate}/{name}.jpg stored to file.')

#df_dims = pd.DataFrame(dims, columns=['filename', 'height', 'width', 'cate'])
#df_dims.to_csv('dims_records_all_train_pic.csv')
print('All job done.')

