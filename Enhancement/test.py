## Learning Enriched Features for Fast Image Restoration and Enhancement
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
## IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)
## https://www.waqaszamir.com/publication/zamir-2022-mirnetv2/

import numpy as np
import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
import torch.nn.functional as F
import utils

from natsort import natsorted
from glob import glob
from basicsr.models.archs.mirnet_v2_arch import MIRNet_v2
from skimage import img_as_ubyte
from pdb import set_trace as stx
from skimage import metrics


parser = argparse.ArgumentParser(description='Image Enhancement using MIRNet-v2')

parser.add_argument('--input_dir', default='./Datasets/', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/', type=str, help='Directory for results')
parser.add_argument('--weights', default='./pretrained_models/enhancement_lol.pth', type=str, help='Path to weights')
parser.add_argument('--dataset', default='Lol', type=str, help='Test Dataset') # ['FiveK', 'Lol']

args = parser.parse_args()


####### Load yaml #######
if args.dataset=='FiveK':
    yaml_file = 'Options/Enhancement_MIRNet_v2_FiveK.yml'
    weights = 'pretrained_models/enhancement_fivek.pth'
elif args.dataset=='Lol':
    yaml_file = 'Options/Enhancement_MIRNet_v2_Lol.yml'
    weights = 'pretrained_models/enhancement_lol.pth'
else: 
    print("Wrong dataset name")

import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)

s = x['network_g'].pop('type')
##########################

model_restoration = MIRNet_v2(**x['network_g'])

checkpoint = torch.load(weights)
model_restoration.load_state_dict(checkpoint['params'])
print("===>Testing using weights: ",weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()


factor = 4
dataset = args.dataset
result_dir  = os.path.join(args.result_dir, dataset)
os.makedirs(result_dir, exist_ok=True)

input_dir = os.path.join(args.input_dir, 'test', dataset, 'input')
input_paths = natsorted(glob(os.path.join(input_dir, '*.png')) + glob(os.path.join(input_dir, '*.jpg')))

target_dir = os.path.join(args.input_dir, 'test', dataset, 'target')
target_paths = natsorted(glob(os.path.join(target_dir, '*.png')) + glob(os.path.join(target_dir, '*.jpg')))


psnr = []
with torch.inference_mode():
    for inp_path, tar_path in tqdm(zip(input_paths,target_paths), total=len(target_paths)):
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()

        img = np.float32(utils.load_img(inp_path))/255.
        target = np.float32(utils.load_img(tar_path))/255.

        img = torch.from_numpy(img).permute(2,0,1)
        input_ = img.unsqueeze(0).cuda()

        # Padding in case images are not multiples of 4
        h,w = input_.shape[2], input_.shape[3]
        H,W = ((h+factor)//factor)*factor, ((w+factor)//factor)*factor
        padh = H-h if h%factor!=0 else 0
        padw = W-w if w%factor!=0 else 0
        input_ = F.pad(input_, (0,padw,0,padh), 'reflect')

        restored = model_restoration(input_)

        # Unpad images to original dimensions
        restored = restored[:,:,:h,:w]

        restored = torch.clamp(restored,0,1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

        psnr.append(utils.PSNR(target, restored))
        

        utils.save_img((os.path.join(result_dir, os.path.splitext(os.path.split(inp_path)[-1])[0]+'.png')), img_as_ubyte(restored))

psnr = np.mean(np.array(psnr))
print("PSNR: %.2f " %(psnr))

