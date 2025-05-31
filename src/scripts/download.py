#!/usr/bin/env python
from nbiatoolkit import NBIAClient
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
import os

def download_one(client, patient, downloadDir, filePattern):
    print(f"Downloading {patient}")
        
    # Get series for the patient
    serieses = client.getSeries(
        Collection='RADCURE',
        PatientID=patient,
    )
    print(serieses)

    for series in serieses:
        print(f"Downloading {series['SeriesInstanceUID']}")
        # Download the series
        client.downloadSeries(
            series['SeriesInstanceUID'],
            downloadDir=downloadDir,
            filePattern=filePattern,
            nParallel=2,
        )
    
    print(f"Finished downloading {patient}")
    print("========================================")

def main():
    filePattern = '%PatientID/%StudyInstanceUID/%SeriesInstanceUID/%InstanceNumber.dcm'
    downloadDir = os.environ['SAVE_DIR']

    df = pd.read_csv(os.environ['IENE_CSV_PATH'])
    patients = df[df.split == 'test']['RADCURE_ID']

    client = NBIAClient(username=os.environ['TCIA_USERNAME'],
                        password=os.environ['TCIA_PASSWORD'])
        
    Parallel(n_jobs=8)(delayed(download_one)(client, patient, downloadDir, filePattern) for patient in patients)
    print("Finished downloading all patients")
    print("========================================")


if __name__ == "__main__":
    main()