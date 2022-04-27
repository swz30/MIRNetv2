"""
## Learning Enriched Features for Fast Image Restoration and Enhancement
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
## IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)
## https://www.waqaszamir.com/publication/zamir-2022-mirnetv2/
"""
##--------------------------------------------------------------
##------- Demo file to test MIRNet-V2 on your own images---------
## Example usage on directory containing several images:   python demo.py --task real_denoising --input_dir './demo/degraded/' --result_dir './demo/restored/'
## Example usage on a image directly: python demo.py --task real_denoising --input_dir './demo/degraded/noisy.png' --result_dir './demo/restored/'
## Example usage with tile option on a large image: python demo.py --task real_denoising --input_dir './demo/degraded/noisy.png' --result_dir './demo/restored/' --tile 720 --tile_overlap 32
##--------------------------------------------------------------

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import os
from runpy import run_path
from skimage import img_as_ubyte
from natsort import natsorted
from glob import glob
import cv2
from tqdm import tqdm
import argparse
from pdb import set_trace as stx
import numpy as np

parser = argparse.ArgumentParser(description='Test MIRNet-v2 on your own images')
parser.add_argument('--input_dir', default='./demo/degraded/', type=str, help='Directory of input images or path of single image')
parser.add_argument('--result_dir', default='./demo/restored/', type=str, help='Directory for restored results')
parser.add_argument('--task', required=True, type=str, help='Task to run', choices=['real_denoising',
                                                                                    'super_resolution',
                                                                                    'contrast_enhancement',
                                                                                    'lowlight_enhancement'])
parser.add_argument('--tile', type=int, default=None, help='Tile size (e.g 720). None means testing on the original resolution image')
parser.add_argument('--tile_overlap', type=int, default=32, help='Overlapping of different tiles')

args = parser.parse_args()

def load_img(filepath):
    return cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)

def save_img(filepath, img):
    cv2.imwrite(filepath,cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def get_weights_and_parameters(task, parameters):
    if task == 'real_denoising':
        weights = os.path.join('Real_Denoising', 'pretrained_models', 'real_denoising.pth')
        if not os.path.exists(weights):
            os.system('wget https://github.com/swz30/MIRNetv2/releases/download/v1.0.0/real_denoising.pth -P Real_Denoising/pretrained_models/')

    elif task == 'super_resolution':
        weights = os.path.join('Super_Resolution', 'pretrained_models', 'sr_x4.pth')
        parameters['scale'] =  4
        if not os.path.exists(weights):
            os.system('wget https://github.com/swz30/MIRNetv2/releases/download/v1.0.0/sr_x4.pth -P Super_Resolution/pretrained_models/')

    elif task == 'contrast_enhancement':
        weights = os.path.join('Enhancement', 'pretrained_models', 'enhancement_fivek.pth')
        if not os.path.exists(weights):
            os.system('wget https://github.com/swz30/MIRNetv2/releases/download/v1.0.0/enhancement_fivek.pth -P Enhancement/pretrained_models/')

    elif task == 'lowlight_enhancement':
        weights = os.path.join('Enhancement', 'pretrained_models', 'enhancement_lol.pth')
        if not os.path.exists(weights):
            os.system('wget https://github.com/swz30/MIRNetv2/releases/download/v1.0.0/enhancement_lol.pth -P Enhancement/pretrained_models/')

    return weights, parameters

task    = args.task
inp_dir = args.input_dir
out_dir = os.path.join(args.result_dir, task)

os.makedirs(out_dir, exist_ok=True)

extensions = ['jpg', 'JPG', 'png', 'PNG', 'jpeg', 'JPEG', 'bmp', 'BMP']

if any([inp_dir.endswith(ext) for ext in extensions]):
    files = [inp_dir]
else:
    files = []
    for ext in extensions:
        files.extend(glob(os.path.join(inp_dir, '*.'+ext)))
    files = natsorted(files)

if len(files) == 0:
    raise Exception(f'No files found at {inp_dir}')

# Get model weights and parameters
parameters = {
    'inp_channels':3,
    'out_channels':3, 
    'n_feat':80,
    'chan_factor':1.5,
    'n_RRG':4,
    'n_MRB':2,
    'height':3,
    'width':2,
    'bias':False,
    'scale':1,
    'task': task
    }



weights, parameters = get_weights_and_parameters(task, parameters)


load_arch = run_path(os.path.join('basicsr', 'models', 'archs', 'mirnet_v2_arch.py'))
model = load_arch['MIRNet_v2'](**parameters)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

checkpoint = torch.load(weights)
model.load_state_dict(checkpoint['params'])
model.eval()

img_multiple_of = 4

print(f"\n ==> Running {task} with weights {weights}\n ")

with torch.inference_mode():
    for file_ in tqdm(files):
        if torch.cuda.is_available():
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

        img = load_img(file_)

        input_ = torch.from_numpy(img).float().div(255.).permute(2,0,1).unsqueeze(0).to(device)

        # Pad the input if not_multiple_of 8
        height,width = input_.shape[2], input_.shape[3]
        H,W = ((height+img_multiple_of)//img_multiple_of)*img_multiple_of, ((width+img_multiple_of)//img_multiple_of)*img_multiple_of
        padh = H-height if height%img_multiple_of!=0 else 0
        padw = W-width if width%img_multiple_of!=0 else 0
        input_ = F.pad(input_, (0,padw,0,padh), 'reflect')

        if args.tile is None:
            ## Testing on the original resolution image
            restored = model(input_)
        else:
            # test the image tile by tile
            b, c, h, w = input_.shape
            tile = min(args.tile, h, w)
            assert tile % 4 == 0, "tile size should be multiple of 4"
            tile_overlap = args.tile_overlap

            stride = tile - tile_overlap
            h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
            w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
            E = torch.zeros(b, c, h, w).type_as(input_)
            W = torch.zeros_like(E)

            for h_idx in h_idx_list:
                for w_idx in w_idx_list:
                    in_patch = input_[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                    out_patch = model(in_patch)
                    out_patch_mask = torch.ones_like(out_patch)

                    E[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch)
                    W[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch_mask)
            restored = E.div_(W)

        restored = torch.clamp(restored, 0, 1)

        # Unpad the output
        restored = restored[:,:,:height,:width]

        restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
        restored = img_as_ubyte(restored[0])

        f = os.path.splitext(os.path.split(file_)[-1])[0]
        # stx()

        save_img((os.path.join(out_dir, f+'.png')), restored)

    print(f"\nRestored images are saved at {out_dir}")
