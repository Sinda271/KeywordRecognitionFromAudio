import os
import json
import librosa
# import numpy as np

DATASET_PATH = "dataset"  # this is the path the dataset folder which is in the same folder as the prepare_data.py file
JSON_PATH = "data.json"  # this is the json file where we are going to put our structured data
SAMPLE_RATE = 22050   # number of samples in 1 sec


def prepare_dataset(dataset_path, json_path, n_mfcc=13, hop_length=512, n_fft=2048):
    # Create a data dictionary to store all the data we need
    data = {
        "mappings": [],  # to map the keywords into numbers
        "labels": [],  # the target outputs expected
        "MFCCs": [],  # the MFCC vector for each audio file
        "files": []  # the paths to each audio file
    }

    # loop through all the sub folders in the dataset folder
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        # we need to ensure that we are not in the root level
        if dirpath is not dataset_path:
            # update the mappings
            category = dirpath.split("/")[-1]  # 'dataset/down' -> ['dataset', 'down'] -> 'down'
            data["mappings"].append(category)
            print(f" Processing {category}")

            # loop through all the filenames and extract MFCCs
            for f in filenames:
                # get filepath
                file_path = os.path.join(dirpath, f)

                # load audio file
                signal, sr = librosa.load(file_path)

                # ensure that the audio is at least 1 sec long
                if len(signal) >= SAMPLE_RATE:
                    # enforce 1 sec long to each signal including those that are greater that 1 sec
                    signal = signal[:SAMPLE_RATE]

                    # extracting MFCCs
                    mfccs = librosa.feature.mfcc(signal, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)

                    # storing data in the data dictionary
                    data["labels"].append(i - 1)  # i the sub folders' indexes used in the for enumerate loop
                    data["MFCCs"].append(mfccs.T.tolist())  # from an array to a list
                    data["files"].append(file_path)
                    print(f"{file_path}: {i - 1}")

    # store data dictionary in the json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


if __name__ == "__main__":
    prepare_dataset(DATASET_PATH, JSON_PATH)

