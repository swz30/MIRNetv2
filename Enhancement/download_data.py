## Learning Enriched Features for Fast Image Restoration and Enhancement
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
## IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)
## https://www.waqaszamir.com/publication/zamir-2022-mirnetv2/


## Download training and testing data for image enhancement task
import os
# import gdown
import shutil

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, required=True, help='train, test, val or train-test')
parser.add_argument('--dataset', type=str, default='Lol', help='all, Lol, FiveK')
args = parser.parse_args()

### Google drive IDs ######
Lol_train = '1K29vsPfMUsAkYvmNLcaUgiOEYGMxFydd'      ## https://drive.google.com/file/d/1K29vsPfMUsAkYvmNLcaUgiOEYGMxFydd/view?usp=sharing
Lol_test = '1jUGpsih3T-1H7t3gqpEdj7ZD5GcU_v0m'      ## https://drive.google.com/file/d/1jUGpsih3T-1H7t3gqpEdj7ZD5GcU_v0m/view?usp=sharing
FiveK_val = '13A8XA8Gqb2O-Z4mEXo0yL_hyA0Av7kvF'      ## https://drive.google.com/file/d/13A8XA8Gqb2O-Z4mEXo0yL_hyA0Av7kvF/view?usp=sharing
FiveK_test = '1sdB0DcZ5hodFHMxJxxY16DZwA6JwuwFy'     ## https://drive.google.com/file/d/1sdB0DcZ5hodFHMxJxxY16DZwA6JwuwFy/view?usp=sharing 

dataset = args.dataset

for data in args.data.split('-'):
    if data == 'train':
        print('Lol Training Data!')
        # gdown.download(id=Lol_train, output='Datasets/train.zip', quiet=False)
        os.system(f'gdrive download {Lol_train} --path Datasets/')
        print('Extracting Lol data...')
        shutil.unpack_archive('Datasets/train.zip', 'Datasets')
        os.remove('Datasets/train.zip')
    
    if data == 'val':
        if dataset == 'FiveK':
            print('FiveK validation data used during training!')
            # gdown.download(id=FiveK_val, output='Datasets/val.zip', quiet=False)
            os.system(f'gdrive download {FiveK_val} --path Datasets/')
            print('Extracting FiveK val Data...')
            shutil.unpack_archive('Datasets/val.zip', 'Datasets')
            os.remove('Datasets/val.zip')


    if data == 'test':
        if dataset == 'all' or dataset == 'Lol':
            print('Lol Testing Data!')
            # gdown.download(id=Lol_test, output='Datasets/test.zip', quiet=False)
            os.system(f'gdrive download {Lol_test} --path Datasets/')
            print('Extracting GoPro Data...')
            shutil.unpack_archive('Datasets/test.zip', 'Datasets')
            os.remove('Datasets/test.zip')

        if dataset == 'all' or dataset == 'FiveK':
            print('FiveK Testing Data!')
            # gdown.download(id=FiveK_test, output='Datasets/test.zip', quiet=False)
            os.system(f'gdrive download {FiveK_test} --path Datasets/')
            print('Extracting FiveK Data...')
            shutil.unpack_archive('Datasets/test.zip', 'Datasets')
            os.remove('Datasets/test.zip')

       


# print('Download completed successfully!')
