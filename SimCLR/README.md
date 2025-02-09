# PyTorch SimCLR: A Simple Framework for Contrastive Learning of Visual Representations
[![DOI](https://zenodo.org/badge/241184407.svg)](https://zenodo.org/badge/latestdoi/241184407)


### Blog post with full documentation: [Exploring SimCLR: A Simple Framework for Contrastive Learning of Visual Representations](https://sthalles.github.io/simple-self-supervised-learning/)

![Image of SimCLR Arch](https://sthalles.github.io/assets/contrastive-self-supervised/cover.png)

### See also [PyTorch Implementation for BYOL - Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning](https://github.com/sthalles/PyTorch-BYOL).

### **Modifications by [Liu Zhihui]**
The following code is modified from [sthalles/SimCLR](https://github.com/sthalles/SimCLR)
Original code is located at [simclr](https://github.com/sthalles/SimCLR/blob/master)
This repository includes modifications to the original SimCLR implementation. Key changes include:
- Added new dataset about mammal hair scales. [10.5281/zenodo.14835824]
- Added support for saving the best model based on Top1 accuracy.
- Improved training progress visualization with `tqdm` progress bars.
- Added real-time logging of loss and Top1 accuracy during training.
- Updated the learning rate scheduler to use `get_last_lr()` instead of `get_lr()`.
