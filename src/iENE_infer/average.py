import os, glob
import re
import pandas as pd
from argparse import ArgumentParser


def average_floats_in_csv_files(prediction_path):
    directory, file_path = os.path.split(prediction_path)
    file_start, file_end = file_path.split(".")
    print(directory, file_start, file_end)

    # Filter files that match the regex
    files = [os.path.join(directory, file) for file in os.listdir(directory) if file.startswith(file_start) and file.endswith(file_end) and not "AVERAGE" in file]
    print(os.listdir(directory))
    
    new_df = None
    
    # Read and collect floats from each matched file
    for file in files: 
        df = pd.read_csv(file)
        suffix = file.split(file_start)[1].split(".csv")[0]
        if new_df is None:
            new_df = df
        else:
            new_df = pd.merge(new_df, df, on="ID", how="outer")

    new_df.set_index("ID")
    new_df['ENE_average'] = new_df.mean(axis=1)
    new_df['ENE_sd'] = new_df.std(axis=1)
    new_df.reset_index(inplace=True)

    # save aggregated results
    new_prediction_path = prediction_path.replace(".csv", "_AVERAGE.csv")
    new_df[['ID', 'ENE_average', 'ENE_sd']].to_csv(new_prediction_path, index=False)
    
    print("Cleaning up intermediate files...")
    # Remove the original TTA predictions
    for file in files:
        os.remove(file)

if __name__ == "__main__":
    parser = ArgumentParser()

    # Model hyperparameters
    parser.add_argument("pred_save_path", type=str, help="The saved predictions files.")
    args = parser.parse_known_args()[0]
    
    average_floats_in_csv_files(args.pred_save_path)