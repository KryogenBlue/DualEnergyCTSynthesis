### PyTorch Lightning Implementation: U-Net for Dual Energy CT Synthesis

#### Project Overview
This repository contains a PyTorch Lightning implementation of U-Net for Dual Energy Computed Tomography (DECT) synthesis, replicating the approach described in the paper by Wei Zhao, et al.:
- "Dual-Energy Computed Tomography Imaging from Contrast-Enhanced Single-Energy Computed Tomography." Cornell University - arXiv, October 2020. ([https://doi.org/10.48550/arXiv.2010.13253])

#### Credits and Acknowledgments
- This implementation is recomposed from the `Image_Segmentation` project by LeeJunHyun available on GitHub: [https://github.com/LeeJunHyun/Image_Segmentation](https://github.com/LeeJunHyun/Image_Segmentation).
- This project also utilizes GitHub Copilot and ChatGPT-4 for code suggestions and debugging assistance.

#### Dataset
- The code is designed to be compatible with any DECT Pair Dataset. Model hyperparameters should be fine-tuned on the data set to achieve optimal accuracy.
- **Note:** The private dataset PLAData scanned at Nanjing General Hospital of PLA, is not authorized for public distribution. 

#### Contact Information
For more information, please contact:
- **Email:** medphyxhli@buaa.edu.cn

#### How to Use This Repository
For more details of DECT synthesis approach, read more papers by Wei Zhao, et al.:
- “A Deep Learning Approach for Dual-Energy CT Imaging Using a Single-Energy CT Data.” 15th International Meeting on Fully Three-Dimensional Image Reconstruction in Radiology and Nuclear Medicine, 2019. ([https://doi.org/10.1117/12.2534433])
- “A Deep Learning Approach for Virtual Monochromatic Spectral CT Imaging with a Standard Single Energy CT Scanner.” Cornell University - arXiv, May 2020. ([https://doi.org/10.48550/arXiv.2005.09859])
- “Estimating Dual-Energy CT Imaging from Single-Energy CT Data with Material Decomposition Convolutional Neural Network.” Medical Image Analysis, May 2021. ([https://doi.org/10.1016/j.media.2021.102001])
- “Obtaining Dual-Energy Computed Tomography (CT) Information from a Single-Energy CT Image for Quantitative Imaging Analysis of Living Subjects by Using Deep Learning.” Biocomputing 2020, 2019. ([https://doi.org/10.1142/9789811215636_0013])