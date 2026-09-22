import json
import numpy as np
import os
from pathlib import Path
import re
from typing import Dict, Any

from ..domain.entities import ECGRecord, ECGDataset
from ..domain.value_objects import LeadName, SamplingRate

def read_directory(dirpath: str, dataset_id: str = None) -> ECGDataset:
    # Load all ECG files from a directory
    dirpath = Path(dirpath)
    dataset_id = dataset_id or dirpath.name

    dataset = ECGDataset(dataset_id = dataset_id)

    for filepath in sorted(dirpath.glob("*")):
        if _is_ecg_file(filepath) == '.json':
            record = read_ecg_from_json(str(filepath))
            dataset.add(record)

    return dataset

def read_file_list(filepaths: list[str], dataset_id: str = "batch") -> ECGDataset:
    # Load specific files into a dataset
    dataset = ECGDataset(dataset_id=dataset_id)

    for filepath in filepaths:
        record = read_ecg_from_json(filepath)
        dataset.add(record)

    return dataset

def read_ecg_from_json(filepath: str):
    """
    Creates an ECGRecord object from a .json file
    containing the lead information.

    Parameters:
    -----------
    filepath: str
        Path to the folder/file which contains the ECG waveforms.

    Returns:
    --------
    ECGRecord:
        Custom dataclass object
    """

    if filepath[-5:] == '.json':
        lead_arrays = lead_arrays(filepath)
        fs = get_fs(filepath)
        filetype = '.json'

    return ECGRecord(
        record_id = os.path.basename(filepath),
        leads = lead_arrays,
        sampling_rate = SamplingRate(hz=fs),
        metadata={
            'source_file': filepath,
            'original_format': filetype 
        }
    )

# Internals

def _is_ecg_file(filepath: Path) -> str:
    return filepath.suffix.lower()

def lead_arrays(data):
    with open(data, 'r') as f:
        my_dict = json.load(f)
    
    fs = None
    #finding sampling rate
    for key in my_dict['header']:
        name = key.lower()
        if name in ['fs', 'samplingrate', 'sampling_rate', 'sampling rate', 'sampling frequency', 'frequency']:
            fs = my_dict['header'][key]
            break
    if fs is None:
        raise ValueError(f'Sampling rate not found in header of {data}')
    
    #creating lead arrays
    ecg = my_dict['ecg']
    
    ecg_arr = np.asarray(ecg)  # shape (N, 8)
    ecg_arr = ecg_arr.astype(float)
    lead_1, lead_2 = ecg_arr[:,0], ecg_arr[:,1]
    v1,v2,v3,v4,v5,v6 = ecg_arr[:,2], ecg_arr[:,3], ecg_arr[:,4], ecg_arr[:,5], ecg_arr[:,6], ecg_arr[:,7]
    lead_3 = lead_1 - lead_2
    lead_aVR = -(lead_1 + lead_2) / 2
    lead_aVL = lead_1 - lead_2 / 2
    lead_aVF = lead_2 - lead_1 / 2
    t = np.arange(len(ecg_arr)) / float(fs)

    
    leads = {'I':lead_1,
            'II':lead_2,
            'III':lead_3,
            'AVF':lead_aVF,
            'AVL':lead_aVL,
            'AVR':lead_aVR,
            'V1':v1,
            'V2':v2,
            'V3':v3,
            'V4':v4,
            'V5':v5,
            'V6':v6
            }

    
    return leads

def get_fs(file):
    with open(file, 'r') as f:
        my_dict = json.open(f)

    fs = None
    #finding sampling rate
    for key in my_dict['header']:
        name = key.lower()
        if name in ['fs', 'samplingrate', 'sampling_rate', 'sampling rate', 'sampling frequency', 'frequency']:
            fs = my_dict['header'][key]
            break
    if fs is None:
        raise ValueError(f'Sampling rate not found in header of {os.path.basename(file)}')
    else:
        return fs

def file_upload(root_folder):
    file_map = {}

    # Normalize the initial input path
    starting_path = double_single_backslashes(root_folder)

    # Check if it's a single file
    if os.path.isfile(starting_path):
        name = os.path.basename(starting_path)
        file_map[name] = starting_path
        return file_map

    # Check if it's a directory
    if os.path.isdir(starting_path):
        for root, dirs, files in os.walk(starting_path):
            for name in files:
                full_path = os.path.join(root, name)
                file_map[name] = full_path
        return file_map

    # If neither file nor directory, raise an error
    raise ValueError(f"Path does not exist: {starting_path}")

def double_single_backslashes(text):
    # r'' for the regex pattern to keep it readable.
    # (?<!\\) - Look behind: not a backslash
    # \\      - The literal backslash we want to match
    # (?!\\)  - Look ahead: not a backslash
    return re.sub(r'(?<!\\)\\(?!\\)', r'\\\\', text)