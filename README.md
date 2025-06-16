-The code in SimCLR is modified from [sthalles/SimCLR](https://github.com/sthalles/SimCLR)  

-Our dataset and trained networks on mammalian hair scales  [[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14863215.svg)](https://doi.org/10.5281/zenodo.14863215)](https://doi.org/10.5281/zenodo.15668655)

### dataset
ori_all_training.tar Contains all raw data outside the test set

2025_85_720[_per**}.tar The dataset comprises data excluded from the test set, which have been processed through SAM-based semantic segmentation followed by a series of augmentation techniques, including random contrast adjustment, cropping, noise addition, mirroring, and rotation.

2025_15_100.tar Test set

### semantic segmentation
seg.tar Semantic Segmentation Models and Annotated Datasets
