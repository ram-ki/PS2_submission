
## Our approach
-> We implemented the Deep-learning based Watermark-Decomposition Network for Visible Watermark Removal paper for PS2

-> The DL model is trained on the CLWD dataset, which has pairs of images with and without watermark

-> The DL model aims at detecting, removing and inpainting the watermark with refined features

-> We use the pre-trained os the WDNet to assist our watermark-removal tool 

-> The reason to use WDNet was its ability to generalise to different types and colors of watermark logos without the need to look at mask at inferernce

## Steps to run the code
-> Download the pre-trained models and paste in './Pretrained_WDNet/'

-> run ```python test.py```

-> the model results are stored in './results/'

-> if you wan to remove text/numbers in the image run ```python submission.py```

-> the model results are stored in './results_2/'


## write-up
-> The solution is the implementation of the paper WDNet

-> The WDNet uses a sequence of generator and discriminator for watermark-removal

## Citation
Please cite the related works in your publications if it helps your research:
```
@InProceedings{Liu_2021_WACV,
author = {Yang Liu and
          Zhen Zhu and
          Xiang Bai},
title = {WDNet: Watermark-Decomposition Network for Visible Watermark Removal},
booktitle = { 2021 {IEEE/CVF} Winter Conference on Applications of Computer Vision (WACV)},
publisher = {{IEEE}},
page = {3685-3693},
year = {2021}
}
```
## Dataset CLWD
[CLWD](https://drive.google.com/file/d/17y1gkUhIV6rZJg1gMG-gzVMnH27fm4Ij/view?usp=sharing)

## Pretraied Model
Thanks for the help of @[ChaiHuanhuan](https://github.com/ChaiHuanhuan), who trained the WDNet and provided a pretrained [WDNet model](https://drive.google.com/drive/folders/1UYOtWmYZQQmCPMLVrstVxhPYW4Jngo-g?usp=sharing). This model is trained for 50 epoches.

