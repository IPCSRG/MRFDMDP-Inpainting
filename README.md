# MRFDMDP-Inpainting

Code for Multi-receptive fields and dynamic matching of damaged patterns for image inpainting.

## Prerequisites

- Python 3.7
- NVIDIA GPU + CUDA cuDNN 10.1
- PyTorch 1.8.1



## Run
**1.   Training**

To train our model, modify the model config file [train.py](train.py). You may need to change the path of dataset or the parameters of the networks etc. Then run python train.py \

**2.   Testing**

To output the generated results of the inputs, you can use the [test_and_save.py](test_and_save.py).  

## Download Datasets

We use [Places2](http://places2.csail.mit.edu/), [CelebA-HQ](https://github.com/switchablenorms/CelebAMask-HQ), and [Paris Street-View](https://github.com/pathak22/context-encoder) datasets. [Liu et al.](https://arxiv.org/abs/1804.07723) provides 12k [irregular masks](https://nv-adlr.github.io/publication/partialconv-inpainting) as the testing mask.



## Citation

to be published. 

## Acknowledgments

This code based on [LGNet](https://github.com/weizequan/LGNet).
The evaluation code is borrowed from [TFill](https://github.com/lyndonzheng/TFill).
Please consider to cite their papers.
```
@ARTICLE{9730792,
  author={Quan, Weize and Zhang, Ruisong and Zhang, Yong and Li, Zhifeng and Wang, Jue and Yan, Dong-Ming},
  journal={IEEE Transactions on Image Processing}, 
  title={Image Inpainting With Local and Global Refinement}, 
  year={2022},
  volume={31},
  pages={2405-2420}
}
@InProceedings{Zheng_2022_CVPR,
    author    = {Zheng, Chuanxia and Cham, Tat-Jen and Cai, Jianfei and Phung, Dinh},
    title     = {Bridging Global Context Interactions for High-Fidelity Image Completion},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {11512-11522}
}
```
