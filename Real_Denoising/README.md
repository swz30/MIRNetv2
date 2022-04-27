
# Real Image Denoising

## Training

- Download SIDD training data, run
```
python download_data.py --data train --noise real
```

- Generate image patches from full-resolution training images, run
```
python generate_patches_sidd.py 
```

- Train MIRNet_v2
```
cd MIRNetv2
./train.sh Real_Denoising/Options/RealDenoising_MIRNet_v2.yml
```

**Note:** This training script uses 8 GPUs by default. To use any other number of GPUs, modify [MIRNetv2/train.sh](../train.sh) and [Real_Denoising/Options/RealDenoising_MIRNet_v2.yml](Options/RealDenoising_MIRNet_v2.yml)

## Evaluation

- Download the pre-trained model and place it in `./pretrained_models/`:
```
wget https://github.com/swz30/MIRNetv2/releases/download/v1.0.0/real_denoising.pth -P pretrained_models/
```

#### Testing on SIDD dataset

- Download SIDD validation data, run 
```
python download_data.py --noise real --data test --dataset SIDD
```

- To obtain denoised results, run
```
python test_real_denoising_sidd.py --save_images
```

- To reproduce PSNR/SSIM scores on SIDD data (Table 3), run
```
evaluate_sidd.m
```

#### Testing on DND dataset

- Download the DND benchmark data, run 
```
python download_data.py --noise real --data test --dataset DND
```

- To obtain denoised results, run
```
python test_real_denoising_dnd.py --save_images
```

- To reproduce PSNR/SSIM scores (Table 3), upload the results to the [DND benchmark website](https://noise.visinf.tu-darmstadt.de/).
