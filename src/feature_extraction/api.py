from typing import Optional

from preprocessing import ECGRecord, ECGDataset

from .domain.entities import (
    DelineatedRecord,
    SubjectFeatures,
    DatasetFeatures,
)

from .domain.services import FeatureExtractionService

from .infrastructure.factory import (
    create_default_pipeline,
    create_minimal_pipeline,
    create_nn_pipeline,
    create_custom_pipeline
)

_default_pipeline: Optional[FeatureExtractionService] = None

def _get_default_pipeline() -> FeatureExtractionService:
    global _default_pipeline
    if _default_pipeline is None:
        _default_pipeline = create_default_pipeline()
    return _default_pipeline

def extract_record_features(
        record: ECGRecord,
        pipeline: Optional[FeatureExtractionService] = None,
        store_delineation: bool = True
) -> tuple[SubjectFeatures, Optional[DelineatedRecord]]:
    """
    Extract features from a single ECG record.

    Usage:
        from preprocessing import load_record
        from feature_extraction import extract_record_features
        
        record = load_record("data/patient_001.dat")
        features, delineation = extract_record_features(record)
    """

    p = pipeline or _get_default_pipeline()
    return p.process_record(record, store_delineation)

def extract_dataset_features(
        dataset: ECGDataset,
        pipeline: Optional[FeatureExtractionService] = None,
        dataset_id: Optional[str] = None,
        store_delineations: bool = True
) -> DatasetFeatures:
    """
    Extract features from an entire dataset.
    
    Usage:
        from preprocessing import load_dataset
        from feature_extraction import extract_dataset_features
        
        dataset = load_dataset("data/raw/study_001/")
        features = extract_dataset_features(dataset)
        df = features.dataframe
    """

    p = pipeline or _get_default_pipeline()
    return p.process_dataset(dataset, dataset_id, store_delineations)