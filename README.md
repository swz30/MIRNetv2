# Learning Enriched Features for Fast Image Restoration and Enhancement (TPAMI 2022)

[Syed Waqas Zamir](https://scholar.google.es/citations?user=WNGPkVQAAAAJ&hl=en), [Aditya Arora](https://adityac8.github.io/), [Salman Khan](https://salman-h-khan.github.io/), [Munawar Hayat](https://scholar.google.com/citations?user=Mx8MbWYAAAAJ&hl=en), [Fahad Shahbaz Khan](https://scholar.google.es/citations?user=zvaeYnUAAAAJ&hl=en), [Ming-Hsuan Yang](https://scholar.google.com/citations?user=p9-ohHsAAAAJ&hl=en), and [Ling Shao](https://scholar.google.com/citations?user=z84rLjoAAAAJ&hl=en)

[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2111.09881)



# Code will be released before April 20. 

<hr />

> **Abstract:** *Given a degraded input image, image restoration aims to recover the missing high-quality image content. Numerous
applications demand effective image restoration, e.g., computational photography, surveillance, autonomous vehicles, and remote
sensing. Significant advances in image restoration have been made in recent years, dominated by convolutional neural networks
(CNNs). The widely-used CNN-based methods typically operate either on full-resolution or on progressively low-resolution
representations. In the former case, spatial details are preserved but the contextual information cannot be precisely encoded. In the
latter case, generated outputs are semantically reliable but spatially less accurate. This paper presents a new architecture with a
holistic goal of maintaining spatially-precise high-resolution representations through the entire network, and receiving complementary
contextual information from the low-resolution representations. The core of our approach is a multi-scale residual block containing the
following key elements: (a) parallel multi-resolution convolution streams for extracting multi-scale features, (b) information exchange
across the multi-resolution streams, (c) non-local attention mechanism for capturing contextual information, and (d) attention based
multi-scale feature aggregation. Our approach learns an enriched set of features that combines contextual information from multiple
scales, while simultaneously preserving the high-resolution spatial details. Extensive experiments on six real image benchmark
datasets demonstrate that our method, named as MIRNet-v2 , achieves state-of-the-art results for a variety of image processing tasks,
including defocus deblurring, image denoising, super-resolution, and image enhancement.* 
<hr />


## Citation
If you use MIRNet_v2, please consider citing:

    @article{Zamir2022MIRNetv2,
        title={Restormer: Efficient Transformer for High-Resolution Image Restoration}, 
        author={Syed Waqas Zamir and Aditya Arora and Salman Khan and Munawar Hayat 
                and Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao},
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
