import os
import pathlib
import argparse

def create_symlinks(input_parent, output_parent):
    # Ensure the output directory exists
    os.makedirs(output_parent, exist_ok=True)

    # Iterate over each child directory in the input parent directory
    for child_dir in os.listdir(input_parent):
        child_path = os.path.join(input_parent, child_dir)
        if os.path.isdir(child_path):
            ct_file_path = os.path.join(child_path, 'CT', 'CT.nii.gz')
            if os.path.isfile(ct_file_path):
                symlink_name = f"{child_dir}_0000.nii.gz"
                symlink_path = os.path.join(output_parent, symlink_name)
                # Create the symbolic link
                pathlib.Path(symlink_path).symlink_to(ct_file_path)
                print(f"Created symlink: {symlink_path} -> {ct_file_path}")

def main():
    parser = argparse.ArgumentParser()

    # Model hyperparameters
    parser.add_argument("input_dir", type=str, help="The input directory.")
    parser.add_argument("output_dir", type=str, help="The output directory.")
    args = parser.parse_known_args()[0]

    create_symlinks(args.input_dir, args.output_dir)


if __name__ == '__main__':
    main()