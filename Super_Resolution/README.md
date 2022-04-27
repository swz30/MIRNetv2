
## Training

1. Download training data (RealSR version 1), run
```
python download_data.py --data train
```

2. Generate image patches from full-resolution training images, run
```
python generate_patches.py --scale x2
python generate_patches.py --scale x3
python generate_patches.py --scale x4
```

3. To train MIRNet_v2 with default settings, run
```
cd MIRNetv2
./train.sh Super_Resolution/Options/SuperResolution_MIRNet_v2_scale2.yml
./train.sh Super_Resolution/Options/SuperResolution_MIRNet_v2_scale3.yml
./train.sh Super_Resolution/Options/SuperResolution_MIRNet_v2_scale4.yml
```

**Note:** The above training script uses 8 GPUs by default. To use any other number of GPUs, modify [Restormer/train.sh](../train.sh) and the yaml file correspondng to each SR scaling factor (e.g.,  [Super_Resolution/Options/SuperResolution_MIRNet_v2_scale2.yml](Options/SuperResolution_MIRNet_v2_scale2.yml))

## Evaluation

- Download the pre-trained models and place them in `./pretrained_models/`: 
```
wget https://github.com/swz30/MIRNetv2/releases/download/v1.0.0/sr_x2.pth -P pretrained_models/
wget https://github.com/swz30/MIRNetv2/releases/download/v1.0.0/sr_x3.pth -P pretrained_models/
wget https://github.com/swz30/MIRNetv2/releases/download/v1.0.0/sr_x4.pth -P pretrained_models/
```
- Download test datasets (for x2, x3, x4 scale factors), run 
```
python download_data.py --data test
```

- Testing
```
python test.py --scale x2
python test.py --scale x3
python test.py --scale x4
```

#### To reproduce PSNR/SSIM scores of Table 4, run

```
evaluate_PSNR_SSIM.m 
```
