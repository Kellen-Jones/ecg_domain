from .domain.entities import ECGDataset, ECGRecord
from .domain.value_objects import LeadName, SamplingRate
from .infrastructure.readers import (
    read_ecg_from_json,
    read_directory,
    read_file_list
)

def load_record(filepath: str) -> ECGRecord:
    return read_ecg_from_json

def load_dataset(dirpath: str, dataset_id: str = None) -> ECGDataset:
    return read_directory(dirpath, dataset_id)

def load_file_list(filepaths: list[str], dataset_id: str = "batch") -> ECGDataset:
    return read_file_list