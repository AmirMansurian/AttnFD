## Attention as Geometric Transformation: Revisiting Feature Distillation for Semantic Segmentation (WACV26 - LENS Workshop)
[![arXiv](https://img.shields.io/badge/arXiv-2403.05451-<COLOR>.svg)](https://arxiv.org/abs/2403.05451)
![Stars](https://img.shields.io/github/stars/AmirMansurian/AttnFD?style=social)
![Forks](https://img.shields.io/github/forks/AmirMansurian/AttnFD?style=social)

The source code of [(Attention as Geometric Transformation: Revisiting Feature Distillation for Semantic Segmentation)](https://arxiv.org/abs/2403.05451).
 
 Also, see our other works:
 - [Adaptive Inter-Class Similarity Distillation for Semantic Segmentation](https://github.com/AmirMansurian/AICSD)
 - [A Comprehensive Survey on Knowledge Distillation](https://github.com/IPL-sharif/KD_Survey)

<p align="center">
 <img src="https://raw.githubusercontent.com/AmirMansurian/AttnFD/refs/heads/main/Images/new_diagram.png"  width="800" height="400"/>
</p>

### Requirements

- Python3
- PyTorch (> 0.4.1)
- torchvision
- numpy
- scipy
- tqdm
- matplotlib 
- pillow

### Datasets and Models
- Datasets: [[PascalVoc]](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) [[Cityscapes]](https://www.cityscapes-dataset.com/)
- Teacher model: [[ResNet101-DeepLabV3+]](https://drive.google.com/file/d/1_TM1p38Ev-e-P68YUQGMo7YpkK_-AUFq/view?usp=sharing)

Download the datasets and teacher models. Put the teacher model in ```pretrained/``` and set the path to the datasets in ```mypath.py```.


### Training
- Without distillation
  ```shell
  python train.py --backbone resnet18 --dataset pascal --nesterov --epochs 120 --batch-size 6
  ```

- Distillation
  ```shell
  python train_kd.py --backbone resnet18 --dataset pascal --nesterov --epochs 120 --batch-size 6 --attn_lambda 2
  ```


### Experimental Results


Comparison of results on the PascalVOC dataset.

| Method                               | mIoU(%)            | Params(M) |
| ------------------------------------ | ------------------ | --------- |
| Teacher: Deeplab-V3 + (ResNet-101)   | 77.85              | 59.3      |
| Student: Deeplab-V3 + (ResNet-18)   | 67.50              | 16.6      |
| Student + KD                        | 69.13 ± 0.11       | 16.6      |
| Student + Overhaul                      | 70.67 ± 0.25       | 16.6      |
| Student + DistKD                        | 69.84 ± 0.11     | 5.9       |
| Student + CIRKD                        | 71.02 ± 0.11      | 5.9       |
| Student + LAD                        | 71.42 ± 0.09      | 5.9       |
| **Student + AttnFD (ours)**              | **73.09 ± 0.06**   | 5.9       |



Comparison of results on the Cityscapes dataset.

| Method            | mIoU(%)  | Accuracy(%) |
| ----------------- | -------- | ----------- |
| Teacher: ResNet101      | 77.66    | 84.05       |
| Student: ResNet18      | 64.09    | 74.8        |
| Student + KD           | 65.21 (+1.12) | 76.32 (+1.74) |
| Student + Overhaul           | 70.31 (+6.22) | 80.10 (+5.3) |
| Student + DistKD           | 71.81 (+7.72) | 80.73 (+5.93) |
| Student + CIRKD         | 70.49 (+6.40) | 79.99 (+5.19) |
| Student + LAD         | 71.37 (+7.28) | 80.93 (+6.13)   |
| **Student + AttnFD (ours)** | **73.04 (+8.95)** | **83.01 (+8.21)** |

 
 ## Citation
If you use this repository for your research or wish to refer to our distillation method, please use the following BibTeX entries:
```bibtex
@article{mansourian2024attention,
  title={Attention-guided Feature Distillation for Semantic Segmentation},
  author={Mansourian, Amir M and Jalali, Arya and Ahmadi, Rozhan and Kasaei, Shohreh},
  journal={arXiv preprint arXiv:2403.05451},
  year={2024}
}

@article{mansourian2025enriching,
  title={Enriching Knowledge Distillation with Cross-Modal Teacher Fusion},
  author={Mansourian, Amir M and Babaei, Amir Mohammad and Kasaei, Shohreh},
  journal={arXiv preprint arXiv:2511.09286},
  year={2025}
}

@article{mansourian2025a,
title={A Comprehensive Survey on Knowledge Distillation},
author={Amir M. Mansourian and Rozhan Ahmadi and Masoud Ghafouri and Amir Mohammad Babaei and Elaheh Badali Golezani and Zeynab yasamani ghamchi and Vida Ramezanian and Alireza Taherian and Kimia Dinashi and Amirali Miri and Shohreh Kasaei},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2025}
}

@article{mansourian2025aicsd,
  title={AICSD: Adaptive inter-class similarity distillation for semantic segmentation},
  author={Mansourian, Amir M and Ahamdi, Rozhan and Kasaei, Shohreh},
  journal={Multimedia Tools and Applications},
  pages={1--20},
  year={2025},
  publisher={Springer}
}
```

### Acknowledgement
This codebase is heavily borrowed from [A Comprehensive Overhaul of Feature Distillation ](https://github.com/clovaai/overhaul-distillation). Thanks for their excellent work.
