## Learning Enriched Features for Fast Image Restoration and Enhancement
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
## IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)
## https://www.waqaszamir.com/publication/zamir-2022-mirnetv2/

## Download training and testing data for Image Denoising task


import os
# import gdown
import shutil

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, required=True, help='train, test or train-test')
parser.add_argument('--dataset', type=str, default='SIDD', help='all or SIDD or DND')
args = parser.parse_args()

### Google drive IDs ######
SIDD_train = '1UHjWZzLPGweA9ZczmV8lFSRcIxqiOVJw'      ## https://drive.google.com/file/d/1UHjWZzLPGweA9ZczmV8lFSRcIxqiOVJw/view?usp=sharing
SIDD_val   = '1Fw6Ey1R-nCHN9WEpxv0MnMqxij-ECQYJ'      ## https://drive.google.com/file/d/1Fw6Ey1R-nCHN9WEpxv0MnMqxij-ECQYJ/view?usp=sharing
SIDD_test  = '11vfqV-lqousZTuAit1Qkqghiv_taY0KZ'      ## https://drive.google.com/file/d/11vfqV-lqousZTuAit1Qkqghiv_taY0KZ/view?usp=sharing
DND_test   = '1CYCDhaVxYYcXhSfEVDUwkvJDtGxeQ10G'      ## https://drive.google.com/file/d/1CYCDhaVxYYcXhSfEVDUwkvJDtGxeQ10G/view?usp=sharing


for data in args.data.split('-'):
    if data == 'train':
        print('SIDD Training Data!')
        os.makedirs(os.path.join('Datasets', 'Downloads'), exist_ok=True)
        # gdown.download(id=SIDD_train, output='Datasets/Downloads/train.zip', quiet=False)
        os.system(f'gdrive download {SIDD_train} --path Datasets/Downloads/')
        print('Extracting SIDD Data...')
        shutil.unpack_archive('Datasets/Downloads/train.zip', 'Datasets/Downloads')
        os.rename(os.path.join('Datasets', 'Downloads', 'train'), os.path.join('Datasets', 'Downloads', 'SIDD'))
        os.remove('Datasets/Downloads/train.zip')

        print('SIDD Validation Data!')
        # gdown.download(id=SIDD_val, output='Datasets/val.zip', quiet=False)
        os.system(f'gdrive download {SIDD_val} --path Datasets/')
        print('Extracting SIDD Data...')
        shutil.unpack_archive('Datasets/val.zip', 'Datasets')
        os.remove('Datasets/val.zip')

    if data == 'test':
        if args.dataset == 'all' or args.dataset == 'SIDD':
            print('SIDD Testing Data!')
            # gdown.download(id=SIDD_test, output='Datasets/test.zip', quiet=False)
            os.system(f'gdrive download {SIDD_test} --path Datasets/')
            print('Extracting SIDD Data...')
            shutil.unpack_archive('Datasets/test.zip', 'Datasets')
            os.remove('Datasets/test.zip')

        if args.dataset == 'all' or args.dataset == 'DND':
            print('DND Testing Data!')
            # gdown.download(id=DND_test, output='Datasets/test.zip', quiet=False)
            os.system(f'gdrive download {DND_test} --path Datasets/')
            print('Extracting DND data...')
            shutil.unpack_archive('Datasets/test.zip', 'Datasets')
            os.remove('Datasets/test.zip')


# print('Download completed successfully!')
