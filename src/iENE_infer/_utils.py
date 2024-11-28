import SimpleITK as sitk
import numpy as np
from imgtools.ops import Resample, Resize

def find_bbox(mask: sitk.Image) -> np.ndarray:
    mask_uint = sitk.Cast(mask, sitk.sitkUInt8)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(mask_uint)
    xstart, ystart, zstart, xsize, ysize, zsize = stats.GetBoundingBox(1)
    
    # Prevent the following ITK Error from SmoothingRecursiveGaussianImageFilter: 
    # The number of pixels along dimension 2 is less than 4. This filter requires a minimum of four pixels along the dimension to be processed.
    if xsize < 4:
        xsize = 4
    if ysize < 4:
        ysize = 4
    if zsize < 4:
        zsize = 4

    xend, yend, zend = xstart + xsize, ystart + ysize, zstart + zsize
    return xstart, xend, ystart, yend, zstart, zend

def all_to_nifti(path: str):
    import os
    if path.endswith("nii") or path.endswith("nii.gz"):
        return path
    else:
        dir_path, file_path = os.path.split(path)
        new_file_path = file_path.split(".")[0] + ".nii.gz"
        new_path = os.path.join(dir_path, new_file_path)

        # i know this is messy let me be
        if new_file_path in os.listdir(dir_path):
            return new_path
        
        img = sitk.ReadImage(path)
        sitk.WriteImage(img, new_path)
        return new_path
    

def crop_bbox(image: sitk.Image, bbox_coords, input_size) -> sitk.Image:
    min_x, max_x, min_y, max_y, min_z, max_z = bbox_coords
    img_x, img_y, img_z = image.GetSize()

    if min_x < 0: 
        min_x, max_x = 0, input_size[0]
    elif max_x > img_x: # input_size[0]:
        min_x, max_x = img_x - input_size[0], img_x

    if min_y < 0:
        min_y, max_y = 0, input_size[1]
    elif max_y > img_y: # input_size[1]:
        min_y, max_y = img_y - input_size[1], img_y

    if min_z < 0:
        min_z, max_z = 0, input_size[2]
    elif max_z > img_z: # input_size[2]:
        min_z, max_z = img_z - input_size[2], img_z
    
    img_crop = image[min_x:max_x, min_y:max_y, min_z:max_z]
    img_crop = Resize(input_size)(img_crop)
    return img_crop

def find_centroid(mask: sitk.Image) -> np.ndarray:
    """Find the centroid of a binary image in image
    coordinates.

    Parameters
    ----------
    mask
        The bimary mask image.

    Returns
    -------
    np.ndarray
        The (x, y, z) coordinates of the centroid
        in image space.
    """
    mask_uint = sitk.Cast(mask, sitk.sitkUInt8)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(mask_uint)
    centroid_coords = stats.GetCentroid(1)
    centroid_idx = mask.TransformPhysicalPointToIndex(centroid_coords)
    return np.asarray(centroid_idx, dtype=np.float32)

def crop_centroid(image: sitk.Image, centroid, input_size) -> sitk.Image:
    min_x = int(centroid[0] - input_size[0] // 2)
    max_x = int(centroid[0] + input_size[0] // 2)
    min_y = int(centroid[1] - input_size[1] // 2)
    max_y = int(centroid[1] + input_size[1] // 2)
    
    # tuning where in the neck to crop
    min_z = int(centroid[2] - input_size[2] // 2)
    max_z = int(centroid[2] + input_size[2] // 2)

    img_x, img_y, img_z = image.GetSize()

    if min_x < 0: 
        min_x, max_x = 0, input_size[0]
    elif max_x > img_x: # input_size[0]:
        min_x, max_x = img_x - input_size[0], img_x

    if min_y < 0:
        min_y, max_y = 0, input_size[1]
    elif max_y > img_y: # input_size[1]:
        min_y, max_y = img_y - input_size[1], img_y

    if min_z < 0:
        min_z, max_z = 0, input_size[2]
    elif max_z > img_z: # input_size[2]:
        min_z, max_z = img_z - input_size[2], img_z
    
    return image[min_x:max_x, min_y:max_y, min_z:max_z]

def crop_centroid_arr(image: np.ndarray, centroid, input_size) -> sitk.Image:
    min_x = int(centroid[0] - input_size[0] // 2)
    max_x = int(centroid[0] + input_size[0] // 2)
    min_y = int(centroid[1] - input_size[1] // 2)
    max_y = int(centroid[1] + input_size[1] // 2)
    
    # tuning where in the neck to crop
    min_z = int(centroid[2] - input_size[2] // 2)
    max_z = int(centroid[2] + input_size[2] // 2)

    img_x, img_y, img_z = image.GetSize()

    if min_x < 0: 
        min_x, max_x = 0, input_size[0]
    elif max_x > img_x: # input_size[0]:
        min_x, max_x = img_x - input_size[0], img_x

    if min_y < 0:
        min_y, max_y = 0, input_size[1]
    elif max_y > img_y: # input_size[1]:
        min_y, max_y = img_y - input_size[1], img_y

    if min_z < 0:
        min_z, max_z = 0, input_size[2]
    elif max_z > img_z: # input_size[2]:
        min_z, max_z = img_z - input_size[2], img_z
    
    return image[min_x:max_x, min_y:max_y, min_z:max_z]

def binarize_mask(mask: sitk.Image) -> sitk.Image:
    mask_arr = sitk.GetArrayFromImage(mask)
    mask_arr[mask_arr > 0.5] = 1

    mask_new = sitk.GetImageFromArray(mask_arr, isVector=False) #isVector flag to allow 4D image conversion
    mask_new.CopyInformation(mask) 
    return mask_new