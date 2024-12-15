# iENE Prediction Inference
This is a pipeline for predicting iENE from a dataset of head and neck CTs. 

## Getting Started
The pipeline consists of the following steps:

0. (optional) `process`: This step is used to preprocess DICOM datasets into ML-ready NIfTI files.
1. `prepare`: This step will create a symbolic directory structure compatible with nnUNet.
   * This step expects a directory structure like this:
   ```
   dataset
   ├── patient_one
   │   └── CT
   │       └── CT.nii.gz
   └── patient_two
       └── CT
           └── CT.nii.gz
   ```
2. `larynx`: This step will run the nnUNet larynx segmentation model on the prepared data.
   * This step expects an INPUT directory structure like this. Each file must end with `_0000.nii.gz`.
   ```
   dataset_ready
   ├── patient_one_0000.nii.gz
   ├── patient_two_0000.nii.gz
   └── patient_three_0000.nii.gz
   ```
   * While using the nnUNet larynx segmentation model is recommended, you can also prepare your own segmentations with the following directory structure:
   ```
   dataset_larynx_masks
   ├── patient_one.nii.gz
   ├── patient_two.nii.gz
   └── patient_three.nii.gz
   ```
3. `predict`: This step will run the iENE model on the prepared data.
4. `average`: This step will average the predictions from the iENE model to create a single prediction file.

## Model Checkpoints
You have two types of model checkpoints to download. Download them here: [https://drive.google.com/drive/folders/1U_js4aYkxT5EgMaD3sM22_50VOkjxiVw?usp=drive_link](https://drive.google.com/drive/folders/1U_js4aYkxT5EgMaD3sM22_50VOkjxiVw?usp=drive_link)
1. nnUNet checkpoint: Save `checkopint_best.pth` into `src/nnunet/nnUNet_results/Dataset001_Larynx/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0`
2. iENE prediction checkpoints: Save `fold_1.ckpt` through `fold_4.ckpt` into `src/models`