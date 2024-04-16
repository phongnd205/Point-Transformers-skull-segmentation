# Pytorch Implementation of Point Transformers

## Part Segmentation
### Data Preparation
Prepare data with point cloud and corresponding label

### Run
Change which method to use in `config/partseg.yaml` and run
```
python train_partseg.py
```
### Results
Currently only Hengshuang's method is implemented.

### Miscellaneous
This code is modified from https://github.com/yanx27/Pointnet_Pointnet2_pytorch and https://github.com/qq456cvb/Point-Transformers
