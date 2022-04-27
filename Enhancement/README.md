## Training
#### Training on LoL dataset
- Download LoL training and testing data, run
```
python download_data.py --data train-test --dataset Lol
```
- To train MIRNet_v2, run
```
cd MIRNetv2
./train.sh Enhancement/Options/Enhancement_MIRNet_v2_Lol.yml
```

#### Training on MIT-Adobe Fivek dataset
- For MIT-Adobe Fivek training data, download DNGs from https://data.csail.mit.edu/graphics/fivek/ and then follow this for data preparation [nothinglo/Deep-Photo-Enhancer#38](https://github.com/nothinglo/Deep-Photo-Enhancer/issues/38#issuecomment-449786636)

- Download Fivek mini validation data, run
```
python download_data.py --data val --dataset FiveK
```
- Generate image patches from full-resolution training images
```
python generate_patches_fivek.py 
```

- To train MIRNet_v2, run
```
cd MIRNetv2
./train.sh Enhancement/Options/Enhancement_MIRNet_v2_FiveK.yml
```

**Note:** The above training script uses 8 GPUs by default. To use any other number of GPUs, modify [Restormer/train.sh](../train.sh) and [Enhancement/Options/Enhancement_MIRNet_v2_FiveK.yml](Options/Enhancement_MIRNet_v2_FiveK.yml)

## Evaluation

#### Testing on LoL dataset
- Download the pre-trained model and place it in `./pretrained_models/`
```
wget https://github.com/swz30/MIRNetv2/releases/download/v1.0.0/enhancement_lol.pth -P pretrained_models/
```
- Download LoL testset, run
```
python download_data.py --data test --dataset Lol
```

- Testing
```
python test.py --dataset Lol
```

#### Testing on MIT-Adobe Fivek dataset
- Download the pre-trained model and place it in `./pretrained_models/`
```
wget https://github.com/swz30/MIRNetv2/releases/download/v1.0.0/enhancement_fivek.pth -P pretrained_models/
```
- Download MIT-Adobe Fivek testset, run
```
python download_data.py --data test --dataset FiveK
```

- Testing
```
python test.py --dataset FiveK
```

#### To reproduce PSNR/SSIM scores of the paper (Table 5) on LoL, run this MATLAB script

```
evaluate_PSNR_SSIM.m 
```
```
