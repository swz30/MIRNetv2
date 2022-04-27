## Training

- To download DPDD training data, run
```
python download_data.py --data train
```

- Generate image patches from full-resolution training images, run
```
python generate_patches_dpdd.py 
```

- To train MIRNetv2 on dual-pixel defocus deblurring task, run
```
cd MIRNetv2
./train.sh Defocus_Deblurring/Options/DefocusDeblur_DualPixel_16bit_MIRNet_v2.yml
```

**Note:** The above training scripts use 8 GPUs by default. To use any other number of GPUs, modify [Restormer/train.sh](../train.sh) and [DefocusDeblur_DualPixel_16bit_MIRNet_v2.yml](Options/DefocusDeblur_DualPixel_16bit_MIRNet_v2.yml) 


## Evaluation

- Download the pre-trained model and place it in `./pretrained_models/`:
```
wget https://github.com/swz30/MIRNetv2/releases/download/v1.0.0/dual_pixel_defocus_deblurring.pth -P pretrained_models/
```

- Download test dataset, run
```
python download_data.py --data test
```

- Testing 
```
python test_dual_pixel_defocus_deblur.py --save_images
```

This testing script will reproduce image quality scores of Table 2 in the paper. 