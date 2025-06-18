import os, glob
import SimpleITK as sitk
from argparse import ArgumentParser

parser = ArgumentParser(description="Validate NRRD images")
parser.add_argument("--root_directory", type=str, required=True, help="Path to the root directory containing NRRD images")
args = parser.parse_args()

for path in glob.glob(os.path.join(args.root_directory, "*.nrrd")):
    image = sitk.ReadImage(path)
    # Perform validation on the image

    print(image.GetSize(), image.GetSpacing())