



# Learning Enriched Features for Fast Image Restoration and Enhancement (TPAMI 2022)


[Syed Waqas Zamir](https://scholar.google.es/citations?user=WNGPkVQAAAAJ&hl=en), [Aditya Arora](https://adityac8.github.io/), [Salman Khan](https://salman-h-khan.github.io/), [Munawar Hayat](https://scholar.google.com/citations?user=Mx8MbWYAAAAJ&hl=en), [Fahad Shahbaz Khan](https://scholar.google.es/citations?user=zvaeYnUAAAAJ&hl=en), [Ming-Hsuan Yang](https://scholar.google.com/citations?user=p9-ohHsAAAAJ&hl=en), and [Ling Shao](https://scholar.google.com/citations?user=z84rLjoAAAAJ&hl=en)

[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://www.waqaszamir.com/publication/zamir-2022-mirnetv2/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1HN9Sd8UEqB1k_O8RpdRLL8ZUKcxh5LP8?usp=sharing)

#### News
- **April 27, 2022:** Codes and pre-trained models are released!

<hr />

> **Abstract:** * Given a degraded input image, image restoration aims to recover the missing high-quality image content. Numerous applications demand effective image restoration, e.g., computational photography, surveillance, autonomous vehicles, and remote sensing. Significant advances in image restoration have been made in recent years, dominated by convolutional neural networks (CNNs). The widely-used CNN-based methods typically operate either on full-resolution or on progressively low-resolution representations. In the former case, spatial details are preserved but the contextual information cannot be precisely encoded. In the latter case, generated outputs are semantically reliable but spatially less accurate. This paper presents a new architecture with a holistic goal of maintaining spatially-precise high-resolution representations through the entire network, and receiving complementary contextual information from the low-resolution representations. The core of our approach is a multi-scale residual block containing the following key elements: (a) parallel multi-resolution convolution streams for extracting multi-scale features, (b) information exchange across the multi-resolution streams, (c) non-local attention mechanism for capturing contextual information, and (d) attention based multi-scale feature aggregation. Our approach learns an enriched set of features that combines contextual information from multiple scales, while simultaneously preserving the high-resolution spatial details. Extensive experiments on six real image benchmark datasets demonstrate that our method, named as MIRNet-v2 , achieves state-of-the-art results for a variety of image processing tasks, including defocus deblurring, image denoising, super-resolution, and image enhancement.* 
<hr />

<details>
  <summary> <strong>Network Architecture</strong> (click to expand) </summary>
 
<p align="center">
  <img src = "https://i.imgur.com/sX8Gubx.png" width="700">
  <br/>
  <b> Overall Framework of MIRNet_v2 </b>
</p>

<table>
  <tr>
    <td> <img src = "https://i.imgur.com/npRdnUx.png" width="600"> </td>
    <td> <img src = "https://i.imgur.com/UswooC4.png" width="600"> </td>
  </tr>
  <tr>
    <td><p align="center"><b>Selective Kernel Feature Fusion (SKFF)</b></p></td>
    <td><p align="center"> <b>Residual Contextual Block (RCB)</b></p></td>
  </tr>
</table>
    
</details>

## Installation

See [INSTALL.md](INSTALL.md) for the installation of dependencies required to run MIRNet_v2.

## Demo

To test the pre-trained MIRNet_v2 models of Real Denoising, Dual-Pixel Defocus Deblurring, Super-Resolution,  and Image Enhancement on your own images,you can either use Google Colab [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1HN9Sd8UEqB1k_O8RpdRLL8ZUKcxh5LP8?usp=sharing), or command line as following
```
python demo.py --task Task_Name --input_dir path_to_images --result_dir save_images_here
```
Example usage to perform Image Denoising on a directory of images:
```
python demo.py --task real_denoising --input_dir './demo/degraded/' --result_dir './demo/restored/'
```
Example usage to perform Image Denoising on an image directly:
```
python demo.py --task real_denoising --input_dir './demo/degraded/noisy.png' --result_dir './demo/restored/'
```

## Training and Evaluation

Training and Testing instructions for Real Denoising, Defocus Deblurring, Super-Resolution, and Image Enhancement are provided in their respective directories. Here is a summary table containing hyperlinks for easy navigation:

<table>
  <tr>
    <th align="left">Task</th>
    <th align="center">Training Instructions</th>
    <th align="center">Testing Instructions</th>
    <th align="center">MIRNetv2's Visual Results</th>
  </tr>
  <tr>
    <td align="left">Real Denoising</td>
    <td align="center"><a href="Real_Denoising/README.md#training">Link</a></td>
    <td align="center"><a href="Real_Denoising/README.md#evaluation">Link</a></td>
    <td align="center"><a href="https://drive.google.com/drive/folders/1h1_UxesAxVNqBLtOdZ_cLMCr3XRSqg91?usp=sharing">Download</a></td>
  </tr>
  <tr>
    <td>Defocus Deblurring</td>
    <td align="center"><a href="Defocus_Deblurring/README.md#training">Link</a></td>
    <td align="center"><a href="Defocus_Deblurring/README.md#evaluation">Link</a></td>
    <td align="center"><a href="https://drive.google.com/drive/folders/1_3S4LK-BbMbqLhq3vbcn8V2PsctO_cqP?usp=sharing">Download</a></td>
  </tr>
  <tr>
    <td>Super-Resolution</td>
    <td align="center"><a href="Super_Resolution/README.md#training">Link</a></td>
    <td align="center"><a href="Super_Resolution/README.md#evaluation">Link</a></td>
    <td align="center"><a href="https://drive.google.com/drive/folders/1rvc8Bio0GmdIf-w4iIdEmqnli0HHM6nS?usp=sharing">Download</a></td>
  </tr>
  <tr>
    <td>Image Enhancement</td>
    <td align="center"><a href="Enhancement/README.md#training-1">Link</a></td>
    <td align="center"><a href="Enhancement/README.md#evaluation-1">Link</a></td>
    <td align="center"><a href="https://drive.google.com/drive/folders/18l7SSl-wT9-BMZL4j_dNzDeccUB0T0ci?usp=sharing">Download</a></td>
  </tr>
</table>

## Results
Experiments are performed for different image processing tasks.

<details>
<summary><strong>Real Denoising</strong> (click to expand) </summary>
<p align="center">
<img src = "https://imgur.com/jV5K8Ji.png" width="450"> 
</p>
</details>

<details>
<summary><strong>Defocus Deblurring</strong> (click to expand) </summary>

<img src = "https://imgur.com/y5itTxY.png"> 
</details>


<details>
<summary><strong>Super-Resolution</strong> (click to expand) </summary>
<p align="center">
<img src = "https://imgur.com/u1H237x.png" width="450"> 
</p>
</details>

<details>
<summary><strong>Image Enhancement</strong> (click to expand) </summary>
    
<img src = "https://imgur.com/2VOIXNP.png">
</details>

## Citation
If you use MIRNet_v2, please consider citing:

    @article{Zamir2022MIRNetv2,
    title={Learning Enriched Features for Fast Image Restoration and Enhancement}, 
    author={Syed Waqas Zamir and Aditya Arora and Salman Khan and Munawar Hayat 
            and Fahad Shahbaz Khan and Ming-Hsuan Yang and Ling Shao},
    journal={IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
    year={2022}
    }


## Contact
Should you have any question, please contact waqas.zamir@inceptioniai.org


**Acknowledgment:** This code is based on the [BasicSR](https://github.com/xinntao/BasicSR) toolbox. 

## Our Related Works
- Restormer: Efficient Transformer for High-Resolution Image Restoration, CVPR 2022. [Paper](https://arxiv.org/abs/2111.09881) | [Code](https://github.com/swz30/Restormer)
- Multi-Stage Progressive Image Restoration, CVPR 2021. [Paper](https://arxiv.org/abs/2102.02808) | [Code](https://github.com/swz30/MPRNet)
- Learning Enriched Features for Real Image Restoration and Enhancement, ECCV 2020. [Paper](https://arxiv.org/abs/2003.06792) | [Code](https://github.com/swz30/MIRNet)
- CycleISP: Real Image Restoration via Improved Data Synthesis, CVPR 2020. [Paper](https://arxiv.org/abs/2003.07761) | [Code](https://github.com/swz30/CycleISP)
