
## Learning Enriched Features for Fast Image Restoration and Enhancement
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
## IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)
## https://www.waqaszamir.com/publication/zamir-2022-mirnetv2/

## Download training and testing data for Defocus Deblurring task
import os
# import gdown
import shutil

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, required=True, help='train, test or train-test')
args = parser.parse_args()

### Google drive IDs ######
dpdd_train = '1bl5i1cDQNvkgVA_x37QdhvvFk1R80kfe'  ## https://drive.google.com/file/d/1bl5i1cDQNvkgVA_x37QdhvvFk1R80kfe/view?usp=sharing
dpdd_val   = '1KRAmBzluu-IG9-BOsuakB5rjY5_f-kiR'  ## https://drive.google.com/file/d/1KRAmBzluu-IG9-BOsuakB5rjY5_f-kiR/view?usp=sharing
dpdd_test  = '1dDWUQ_D93XGtcywoUcZE1HOXCV4EuLyw'  ## https://drive.google.com/file/d/1dDWUQ_D93XGtcywoUcZE1HOXCV4EuLyw/view?usp=sharing

for data in args.data.split('-'):
    if data == 'train':
        print('DPDD Training Data!')
        os.makedirs(os.path.join('Datasets', 'Downloads', 'DPDD'), exist_ok=True)
        # gdown.download(id=dpdd_train, output='Datasets/Downloads/DPDD/train.zip', quiet=False)
        os.system(f'gdrive download {dpdd_train} --path Datasets/Downloads/DPDD/')
        print('Extracting DPDD data...')
        shutil.unpack_archive('Datasets/Downloads/DPDD/train.zip', 'Datasets/Downloads/DPDD')
        os.remove('Datasets/Downloads/DPDD/train.zip')

        print('DPDD Validation Data!')
        # gdown.download(id=dpdd_val, output='Datasets/Downloads/DPDD/val.zip', quiet=False)
        os.system(f'gdrive download {dpdd_val} --path Datasets/Downloads/DPDD/')
        print('Extracting DPDD val set...')
        shutil.unpack_archive('Datasets/Downloads/DPDD/val.zip', 'Datasets/Downloads/DPDD')
        os.remove('Datasets/Downloads/DPDD/val.zip')

    if data == 'test':
        print('DPDD Testing Data!')
        # gdown.download(id=dpdd_test, output='Datasets/test.zip', quiet=False) 
        os.system(f'gdrive download {dpdd_test} --path Datasets/')
        print('Extracting DPDD test set...')
        shutil.unpack_archive('Datasets/test.zip', 'Datasets')
        os.rename(os.path.join('Datasets', 'test'), os.path.join('Datasets', 'DPDD'))
        os.makedirs(os.path.join('Datasets', 'test'))
        shutil.move(os.path.join('Datasets', 'DPDD'), os.path.join('Datasets', 'test', 'DPDD'))
        os.remove('Datasets/test.zip')

# print('Download completed successfully!')
