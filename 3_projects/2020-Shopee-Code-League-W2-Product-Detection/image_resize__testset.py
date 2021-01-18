import cv2
import os
import pandas as pd
from tqdm import tqdm

resize_to = 100
channel = 3


dims = list()
filepath = 'test/test/'
# create folder for resize pics
try:
    os.makedirs(f'test/test_resize_{resize_to}x{resize_to}_channel{channel}')
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

all_pic = [filepath + f for f in os.listdir(filepath)]
test_check = pd.read_csv('test.csv')
invalid = []

for pic_path in tqdm(all_pic):
    print(pic_path)
    if pic_path.split('/')[2] in test_check.filename.values:
        img = cv2.imread(pic_path, 3)
        print(img.shape)
        h, w = img.shape[:2]
        name = pic_path.split('/')[2]
        dims.append((name, h, w))

        resized = cv2.resize(img, (resize_to, resize_to), interpolation=cv2.INTER_AREA)
        cv2.imwrite(f'test/test_resize_{resize_to}x{resize_to}_channel{channel}/{name}.jpg', resized)
        print(f'resized picture: {name}.jpg stored to file.')
    else:
        invalid.append(pic_path)
        print(f'{pic_path} not used in test.csv')

#df_dims = pd.DataFrame(dims, columns=['filename', 'height', 'width'])
#df_dims.to_csv('dims_records_all_test_pic.csv', index=False)
print('All job done.')
print(invalid)
