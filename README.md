# PINet

[1] Bo Hu, Tuoxun Zhao, Jia Zheng, Yan Zhang, Leida Li, Weisheng Li, and Xinbo Gao, "Blind image quality assessment with coarse-grained perception construction and fine-grained interaction learning," IEEE Transactions on Broadcasting, 2023.

# Abstract

Image Quality Assessment (IQA) plays an important role in the field of computer vision. However, most of the existing metrics for Blind IQA (BIQA) adopt an end-to-end way and do not adequately simulate the process of human subjective evaluation, which limits further improvements in model performance. In the process of perception, people first give a preliminary impression of the distortion type and relative quality of the images, and then give a specific quality score under the influence of the interaction of the two. Although some methods have attempted to explore the effects of distortion type and relative quality, the relationship between them has been neglected. In this paper, we propose a BIQA with coarse-grained perception construction and fine-grained interaction learning, called PINet for short. The fundamental idea is to learn from the two-stage human perceptual process. Specifically, in the pre-training stage, the backbone initially processes a pair of synthetic distorted images with pseudo-subjective scores, and the multi-scale feature extraction module integrates the deep information and delivers it to the coarse-grained perception construction module, which performs the distortion discrimination and the quality ranking. In the fine-tuning stage, we propose a fine-grained interactive learning module to interact with the two pieces of information to further improve the performance of the proposed PINet. The experimental results prove that the proposed PINet not only achieves competing performances on synthetic distortion datasets but also performs better on authentic distortion datasets.

# Requirements

- numpy
- openpyxl
- pandas
- Pillow
- scipy
- torch
- torchvision

More information please check the requirements.txt.