import os, glob, pathlib
from typing import Tuple

import numpy as np
import pandas as pd
import SimpleITK as sitk

import torch
import torchio as tio
from torch.utils.data import Dataset
from _utils import find_centroid, crop_centroid, clean_path

from imgtools.ops import Resample, Resize

# def find_centroid(mask: sitk.Image) -> np.ndarray:
#     """Find the centroid of a binary image in image
#     coordinates.

#     Parameters
#     ----------
#     mask
#         The bimary mask image.

#     Returns
#     -------
#     np.ndarray
#         The (x, y, z) coordinates of the centroid
#         in image space.
#     """
#     mask_uint = sitk.Cast(mask, sitk.sitkUInt8)
#     stats = sitk.LabelShapeStatisticsImageFilter()
#     stats.Execute(mask_uint)
#     centroid_coords = stats.GetCentroid(1)
#     centroid_idx = mask.TransformPhysicalPointToIndex(centroid_coords)
#     return np.asarray(centroid_idx, dtype=np.float32)

# def crop_centroid(image: sitk.Image, centroid, input_size) -> sitk.Image:
#     min_x = int(centroid[0] - input_size[0] // 2)
#     max_x = int(centroid[0] + input_size[0] // 2)
#     min_y = int(centroid[1] - input_size[1] // 2)
#     max_y = int(centroid[1] + input_size[1] // 2)
    
#     # tuning where in the neck to crop
#     min_z = int(centroid[2] - input_size[2] // 2)
#     max_z = int(centroid[2] + input_size[2] // 2)

#     img_x, img_y, img_z = image.GetSize()

#     if min_x < 0: 
#         min_x, max_x = 0, input_size[0]
#     elif max_x > img_x: # input_size[0]:
#         min_x, max_x = img_x - input_size[0], img_x

#     if min_y < 0:
#         min_y, max_y = 0, input_size[1]
#     elif max_y > img_y: # input_size[1]:
#         min_y, max_y = img_y - input_size[1], img_y

#     if min_z < 0:
#         min_z, max_z = 0, input_size[2]
#     elif max_z > img_z: # input_size[2]:
#         min_z, max_z = img_z - input_size[2], img_z
    
#     return image[min_x:max_x, min_y:max_y, min_z:max_z]

# def clean_path(path: str):
#     return pathlib.Path(path).as_posix()

class ExternalDataset(Dataset):
    """Dataset class used in simple CNN baseline training.

    The images are loaded using SimpleITK, preprocessed and cached for faster
    retrieval during training.
    """
    def __init__(self,
                 root_directory: str,
                 mask_directory: str,
                 input_size: tuple=(256, 256, 128),
                 num_workers: int=-1,
                 acsconv: bool=True,
                 testaug: str=None):
        """Initialize the class.

        If the cache directory does not exist, the dataset is first
        preprocessed and cached.

        Parameters
        ----------
        root_directory
            Path to directory containing the training and test images and
            segmentation masks.
        clinical_data_path
            Path to a CSV file with subject metadata and clinical information.
        input_size
            The size of input volume to extract around the tumour centre.
        train
            Whether to load the training or test set.
        transform
            Callable used to transform the images after preprocessing.
        num_workers
            Number of parallel processes to use for data preprocessing.
        """
        self.root_directory = pathlib.Path(root_directory).as_posix() 
        self.image_list = sorted(glob.glob(os.path.join(self.root_directory, "*.nii.gz")))
        self.image_list = [clean_path(i) for i in self.image_list]
        self.input_size = input_size
        self.num_workers = num_workers
        self.acsconv = acsconv
        self.testaug = testaug
        if mask_directory is None:
            self.mask_directory = self.root_directory
        else:
            self.mask_directory = pathlib.Path(mask_directory).as_posix()

        self.resample = Resample(spacing=(1., 1., 1.))
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get an input-target pair from the dataset.

        The images are assumed to be preprocessed and cached.

        Parameters
        ----------
        idx
            The index to retrieve (note: this is not the subject ID).

        Returns
        -------
        tuple of torch.Tensor and torch.Tensor
            The input-target pair.
        """
        
        path = self.image_list[idx]
        subject_id = path.split("/")[-1].split("_0000")[0]

        # images/masks are in different folders and folder structures
        image = sitk.ReadImage(clean_path(os.path.join(self.root_directory, f"{subject_id}_0000.nii.gz")))
        image = self.resample(image)

        mask = sitk.ReadImage(clean_path(os.path.join(self.mask_directory, f"{subject_id}.nii.gz")))
        mask = self.resample(mask)

        try: # crop around larynx
            larynx_centre = find_centroid(mask)
        except Exception as e: # if no masks found...
            print(idx, "has no Larynx mask. Exception: ", e)
            larynx_centre = (250, 180, 80)
        
        if self.testaug is not None:
            tol = np.subtract(larynx_centre, np.divide(self.input_size, 2))
            shape = image.GetSize()
            tol2 = np.subtract(shape, np.add(larynx_centre, np.divide(self.input_size, 2)))
            translate = [0, 0, 0]
            tr_n = 12

            if self.testaug == 'x-': 
                translate[0] = -tr_n
            elif self.testaug == 'x+': 
                translate[0] = tr_n
            elif self.testaug == 'y-': 
                translate[1] = -tr_n
            elif self.testaug == 'y+': 
                translate[1] = tr_n
            elif self.testaug == 'z-': 
                translate[2] = -tr_n
            elif self.testaug == 'z+': 
                translate[2] = tr_n
            
            if tol[0] <= tr_n or tol2[0] <= tr_n:
                translate[0] = 0
            if tol[1] <= tr_n or tol2[1] <= tr_n:
                translate[1] = 0
            if tol[2] <= tr_n or tol2[2] <= tr_n:
                translate[2] = 0
            
            # translate
            larynx_centre = np.add(larynx_centre, translate)
        
        # print("\n\n\n", idx, self.testaug, larynx_centre, "\n\n")
        image = crop_centroid(image, larynx_centre, self.input_size)

        # window image intensities to [-500, 1000] HU range
        image = sitk.Clamp(image, sitk.sitkFloat32, -500, 1000)
        image = tio.ScalarImage.from_sitk(sitk.Cast(image, sitk.sitkFloat32)).data[0]
        image = torch.unsqueeze(image, 0)
        
        # random rotate data augmentation
        # (this is added post-crop to maintain rotation centered around larynx)
        # if self.dataaug:
        #     image = self.rotate(image)

        # normalize intensities with mean = 0. and SD = 1.
        image = (image - -500.) / 1500.
                  

        assert image[0].shape == self.input_size, f"{idx} failed cuz {image[0].shape}"

            
        if self.acsconv:
            image = torch.stack((image[0], image[0], image[0]))
        
        return image, 0.

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.image_list)
