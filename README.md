## Adaptive Inter-Class Similarity Distillation for Semantic Segmentation 
 This repository contains the source code of AICSD [(Adaptive Inter-Class Similarity Distillation for Semantic Segmentation )](https://arxiv.org/abs/2308.04243).

<p align="center">
 <img src="https://raw.githubusercontent.com/AmirMansurian/AICSD/main/Images/pull_figure_main.png"  width="500" height="500"/>
</p>

 Intra-class distributions for each class. Distributions are created by applying softmax to spatial dimension of output prediction of last layer. Similarities between each pair of intra-class distributions have good potential for distillation. Distributions are created from the PASCAL VOC 2012 dataset with 21 category classes.

### Method Diagram
<img src="https://raw.githubusercontent.com/AmirMansurian/AICSD/main/Images/Method_Diagram.png"  width="700" height="300" />

**Overall diagram of the proposed AICSD**. Network outputs are flattened into 1D vectors, followed by application of a softmax function to create intra-class distributions. KL divergence is then calculated between each distribution to create inter-class similarity matrices. An MSE loss function is then defined between the ICS matrices of the teacher and student. Also, KL divergence is calculated between the logits of the teacher and student for pixel-wise distillation. To mitigate the negative effects of teacher network, an adaptive weighting loss strategy is used to scale two distillation losses and ross-entropy loss of semantic segmentation. During training, hyperparameter $\alpha$ undergoes adaptive changes and progressively increases with epoch number.

### Performance on PascalVOC2012

| Method                               | mIoU(%)            | Params(M) |
| ------------------------------------ | ------------------ | --------- |
| Teacher: Deeplab-V3 + (ResNet-101)   | 77.85              | 59.3      |
| Student1: Deeplab-V3 + (ResNet-18)   | 67.50              | 16.6      |
| Student2: Deeplab-V3 + (MobileNet-V2)| 63.92              | 5.9       |
| Student1 + KD                        | 69.13 ± 0.11       | 16.6      |
| Student1 + AD                        | 68.95 ± 0.26       | 16.6      |
| Student1 + SP                        | 69.04 ± 0.10       | 16.6      |
| Student1 + ICKD                      | 69.13 ± 0.17       | 16.6      |
| Student1 + AICSD (ours)              | **70.03 ± 0.13**  | 16.6      |
| Student2 + KD                        | 66.39 ± 0.21       | 5.9       |
| Student2 + AD                        | 66.27 ± 0.17       | 5.9       |
| Student2 + SP                        | 66.32 ± 0.05       | 5.9       |
| Student2 + ICKD                      | 67.01 ± 0.10       | 5.9       |
| Student2 + AICSD (ours)              | **68.05 ± 0.24**   | 5.9       |


### Performance on CityScapes
| Method            | mIoU(%)  | Accuracy(%) |
| ----------------- | -------- | ----------- |
| T: ResNet101      | 77.66    | 84.05       |
| S1: ResNet18      | 64.09    | 74.8        |
| S2: MobileNet v2  | 63.05    | 73.38       |
| S1 + KD           | 65.21 (+1.12) | 76.32 (+1.74) |
| S1 + AD           | 65.29 (+1.20) | 76.27 (+1.69) |
| S1 + SP           | 65.64 (+1.55) | 76.90 (+2.05) |
| S1 + ICKD         | 66.98 (+2.89) | 77.48 (+2.90) |
| S1 + AICSD (ours) | **68.46 (+4.37)** | **78.30 (+3.72)** |
| S2 + KD           | 64.03 (+0.98) | 75.34 (+1.96)   |
| S2 + AD           | 63.72 (+0.67) | 74.79 (+1.41)   |
| S2 + SP           | 64.22 (+1.17) | 75.28 (+1.90)   |
| S2 + ICKD         | 65.55 (+2.50) | 76.48 (+3.10)   |
| S2 + AICSD (ours) | **66.53** (+3.48) | **76.96 (+3.58)** |

### Visualization
<img src="https://raw.githubusercontent.com/AmirMansurian/AICSD/main/Images/visualization_2.png"   width="700" height="400"/>

### How to run
For ICSD method:
  ```shell
  python train_kd.py --backbone resnet18 --dataset pascal  --pa_lambda 9500
  ```

For AICSD method:
  ```shell
  python train_kd.py --backbone resnet18 --dataset pascal  --pa_lambda 9500 --pi_lambda 10 --ALW
  ```

### Teacher model
Download following pre-trained teacher network and put it into ```pretrained/``` directory
- [ResNet101-DeepLabV3+](https://drive.google.com/file/d/1REgApngVChDZbXrkbYkdI8ziCpApEdot/view?usp=sharing)

 measure performance on **test** set with [Pascal VOC evaluation server](http://host.robots.ox.ac.uk/pascal/VOC/).
 
 ## Citation
If you use this repository for your research or wish to refer to our distillation method, please use the following BibTeX entry:
```bibtex
@article{mansourian2023aicsd,
  title={AICSD: Adaptive Inter-Class Similarity Distillation for Semantic Segmentation},
  author={Mansourian, Amir M and Ahmadi, Rozhan and Kasaei, Shohreh},
  journal={arXiv preprint arXiv:2308.04243},
  year={2023}
}
```

### Acknowledgement
This codebase is heavily borrowed from [A Comprehensive Overhaul of Feature Distillation ](https://github.com/clovaai/overhaul-distillation). Thanks for their excellent work.
