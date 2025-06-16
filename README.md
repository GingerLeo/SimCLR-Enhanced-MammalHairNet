-The code in SimCLR is modified from [sthalles/SimCLR](https://github.com/sthalles/SimCLR)  

-Our dataset and trained networks on mammalian hair scales  [[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14863215.svg)](https://doi.org/10.5281/zenodo.14863215)](https://doi.org/10.5281/zenodo.15668655)

### dataset
ori_all_training.tar Contains all raw data outside the test set

hair_2025_per15.tar Contains all raw data in the test set

SAM_not_in_per15_all_filter_rename.tar Contains all the data outside the test set, after SAM semantic segmentation

sam_2025_85_rgb_720_per100_50_30.tar The dataset comprises data excluded from the test set, which have been processed through SAM-based semantic segmentation followed by a series of augmentation techniques, including random contrast adjustment, cropping, noise addition, mirroring, and rotation.

SAM_15_2025_100_oriname.tar & sam_2025_15_100_rgb.tar Note that these two test sets only have some name adjustments(Orangutan->Bornean_orangutan;Red_deer->Wapiti;Wolf->Grey_wolf;Chinese_Water_deer->Water_deer), but the images are the same.  The images within these two compressed packages vary in resolution; however, all the image sizes are adjusted to [224 224] when classifying.

SimCLR.tar ->sim_25 & sim_33

### semantic segmentation
seg.tar Semantic Segmentation Models and Annotated Datasets

### classification
code&model.tar Contains five folders Evaluation_Metrics, fine_tuning, img_cropped_and_augmented, resnet, and SimCLR
