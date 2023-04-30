import cv2
import os
import pydicom
from tqdm import tqdm

# Replace this line with your choice: 'train' or 'test'
conversion_type = 'test'

if conversion_type == 'train':
    print('Converting train images from .dcm to .jpg...')
    inputdir = 'dataset/stage_2_train_images/'
    outdir = 'dataset/images'
elif conversion_type == 'test':
    print('Converting test images from .dcm to .jpg...')
    inputdir = 'dataset/stage_2_test_images/'
    outdir = 'dataset/samples'
os.makedirs(outdir, exist_ok=True)

train_list = [f for f in os.listdir(inputdir)]

for i, f in tqdm(enumerate(train_list[:]), total=len(train_list)):
    ds = pydicom.read_file(inputdir + f)  # read dicom image
    img = ds.pixel_array  # get image array
    # img = cv2.resize(img, (416, 416))
    cv2.imwrite(os.path.join(outdir, f.replace('.dcm', '.jpg')), img)  # write jpg image