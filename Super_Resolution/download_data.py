## Learning Enriched Features for Fast Image Restoration and Enhancement
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
## IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)
## https://www.waqaszamir.com/publication/zamir-2022-mirnetv2/


## Download training and testing data for real image super-resolution
import os
# import gdown
import shutil

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, required=True, help='train, test or train-test')
args = parser.parse_args()

### Google drive IDs ######
realSR_train = '1gPWWkoNm2BHJLvfZIFPWM1xZ30rvFbLb'   ## https://drive.google.com/file/d/1gPWWkoNm2BHJLvfZIFPWM1xZ30rvFbLb/view?usp=sharing
realSR_test  = '1xMgyAm-P2KksQs57Dy6deXC5MwXfhgr5'   ## https://drive.google.com/file/d/1xMgyAm-P2KksQs57Dy6deXC5MwXfhgr5/view?usp=sharing

for data in args.data.split('-'):
    if data == 'train':
        print('RealSR Training Data!')
        os.makedirs(os.path.join('Datasets', 'Downloads'), exist_ok=True)
        # gdown.download(id=realSR_train, output='Datasets/train.zip', quiet=False)
        os.system(f'gdrive download {realSR_train} --path Datasets/Downloads/')
        print('Extracting RealSR data...')
        shutil.unpack_archive('Datasets/Downloads/train.zip', 'Datasets/Downloads')
        os.remove('Datasets/Downloads/train.zip')

    if data == 'test':
        print('Download RealSR Testing Data')
        # gdown.download(id=realSR_test, output='Datasets/test.zip', quiet=False)
        os.system(f'gdrive download {realSR_test} --path Datasets/')
        print('Extracting test data...')
        shutil.unpack_archive('Datasets/test.zip', 'Datasets')
        os.remove('Datasets/test.zip')

   
# print('Download completed successfully!')


