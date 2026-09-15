from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from preprocessing import ECGRecord, ECGDataset, LeadName, SamplingRate

from .entities import (
    DelineatedBeat,
    DelineatedRecord,
    SubjectFeatures,
    DatasetFeatures
)

from .value_objects import FiducialPoints, WaveformWindow, BeatAnalysisContext

# STAGE 1: Fiducial Extraction

class FiducialExtractor(ABC):
    """
    Detects R-peaks and delineates waveform boundaries.

    Abstraction which allows for a Neurokit-agnostic
    usage later in the pipeline.

    Each implementation handles:
        - R-peak detection (per-lead)
        - R-peak reconciliation across leads
        - Wave delineation
        - Beat segmentation
    """

    @abstractmethod
    def extract_fiducial_points(self, record: ECGRecord) -> DelineatedRecord:
        """
        Full delineation of a single ECG record.
        """
        pass

class RPeakDetector(ABC):
    """
    Detects R-peaks on a single lead.

    Separated from FiducialExtractor to leave optionality for:
        - different detectors per lead
        - ensemble multiple detectors
        - test detectors independently
    """

    @abstractmethod
    def detect_rpeaks(
        self,
        signal: np.ndarray,
        sampling_rate: SamplingRate,
    ) -> np.ndarray:
        """
        Returns array of R-peak sample indices
        """
        pass

class RPeakReconciler(ABC):
    """
    Takes per-lead R-peak detections and produces a single
    R-peak series.
    """

    @abstractmethod
    def reconcile_rpeaks(
        self,
        per_lead_peaks: Dict[LeadName, np.ndarray],
        sampling_rate: SamplingRate,
        reference_lead: Optional[LeadName] = None,
    ) -> Tuple[np.ndarray, LeadName]:
        """
        Returns:
            consensus_peaks: Reconciled R-peak positions
            reference_lead: Which lead was used as reference
        """
        pass

class WaveDelineator(ABC):
    """
    Given a signal and a series of R-peaks, finds the P,
    QRS, and T wave boundaries
    """

    @abstractmethod
    def delineate_wave(
        self,
        signal: np.ndarray,
        r_peaks: np.ndarray,
        sampling_rate: SamplingRate
    ) -> List[FiducialPoints]:
        """
        Returns one FiducialPoints object per beat (absolute position).
        """
        pass

class BeatSegmenter(ABC):
    """
    Segments a continuous signal into individual beat waveforms using
    R-peak positions.
    """

    @abstractmethod
    def segment_beat(
        self,
        signal: np.ndarray,
        r_peaks: np.ndarray,
        sampling_rate: SamplingRate
    ) -> List[WaveformWindow]:
        """
        Returns one WaveformWindow object per beat.
        """
        pass

# STAGE 2: Feature Computation

class BeatLevelComputer(ABC):
    """
    Computes features from a single beat on a single lead.
    """

    @abstractmethod
    def compute(
        self,
        ctx: BeatAnalysisContext
    ) -> Dict[str, Optional[float]]:
        """
        Returns {feature_name: value} for this beat.
        """
        pass

    @abstractmethod
    def feature_names(self) -> List[str]:
        """
        All feature names this computer can produce
        for there can be validated consistency across subjects.
        """
        pass

class MultiBeatComputer(ABC):
    """
    Computes features across multiple beats on a single lead.
    Called once per lead per subject.
    """

    @abstractmethod
    def compute(
        self,
        beats: List[DelineatedBeat],
        r_peaks: np.ndarray,
        sampling_rate: SamplingRate
    ) -> dict[str, Optional[float]]:
        """
        Returns {feature_name: value} computed across all beats.
        """
        pass

    @abstractmethod
    def feature_names(self) -> List[str]:
        pass

    @property
    def minimum_beats(self) -> int:
        """
        Minimum number of beats reuqired for meaningful computation.
        """
        return 2

class MultiLeadComputer(ABC):
    """
    Computes features across multiple leads for a single beat.
    Called once per beat
    """

    @abstractmethod
    def compute(
        self,
        beat_across_leads: Dict[LeadName, DelineatedBeat],
        sampling_rate: SamplingRate
    ) -> dict[str, Optional[float]]:
        """
        Returns {feature_name: value} computed across leads for a single beat.
        """
        pass

    @abstractmethod
    def feature_names(self) -> List[str]:
        pass

    @property
    def required_leads(self) -> Optional[List[LeadName]]:
        """
        Number of leads which must be present
        """
        return None

# ORCHESTRATOR (ties extraction and computation)

class FeatureExtractionService:
    """
    Domain service which orchestrates the full pipeline:
    ECGRecord -> DelineatedRecord -> SubjectFeatures

    Where ordering and logic lives. Infrastructure implementations are injected.
    """

    def __init__(
            self,
            extractor: FiducialExtractor,
            beat_computers: List[BeatLevelComputer],
            multi_beat_computers: List[MultiBeatComputer],
            multi_lead_computers: List[MultiLeadComputer],
            min_fiducial_completeness: float = 0.5
    ):
        self._extractor = extractor
        self._beat_computers = beat_computers
        self._multi_beat_computers = multi_beat_computers
        self._multi_lead_computers = multi_lead_computers
        self._min_completeness = min_fiducial_completeness

        # Pre-compute full feature name list for validation
        self._all_feature_names = self._collect_features_names()

    def _collect_feature_names(self) -> List[str]:
        names = []
        for c in self._beat_computers:
            names.extend(c.feature_names())
        for c in self._multi_beat_computers:
            names.extend(c.feature_names())
        for c in self._multi_lead_computers:
            names.extend(c.feature_names())

        # Check for duplicates
        seen = set()
        dupes = set()
        for name in names:
            if name in seen:
                dupes.add(name)
            seen.add(name)
        if dupes:
            raise ValueError(
                f"Duplicate feature names across computers: {dupes}. "
                f"Prefix your feature names to avoid collisions."
            )

        return names

    # Single record processing

    def process_record(
            self,
            record: ECGRecord,
            store_delineation: bool = True
    ) -> Tuple[SubjectFeatures, Optional[DelineatedRecord]]:
        """
        Full pipeline for one subject.
        Returns both the features and optionally the delineation
        for access in NN pipelines.
        """
        # 1. Fiducial extraction
        delineation = self._extractor.extract(record)

        # 2. Feature computation
        features = self._compute_features(delineation)

        delineation_out = delineation if store_delineation else None
        return features, delineation_out

    def _compute_features(
            self, delineation: DelineatedRecord
    ) -> SubjectFeatures:
        """
        Returns all three levels of feature computation and
        merges into a single DataFrame.
        """
        rows: List[Dict[str, Any]] = []
        dropped_beats: List[int] = []
        dropped_reasons: Dict[int, str] = {}

        for beat_id, beat_map in delineation.iter_by_beat_id():

            # Check if this beat has sufficient fiducial data
            if not self._beat_passes_quality(beat_map):
                dropped_beats.append(beat_id)
                dropped_reasons[beat_id] = "insufficient_fiducials"
                continue

            # Compute multi-lead features (once per beat)
            multi_lead_features = self._compute_multi_lead(
                beat_map, delineation.sampling_rate
            )

            # Compute per-lead features
            for lead, beat in beat_map.items():
                row = {
                    "subject_id": delineation.record_id,
                    "lead": lead.value,
                    "beat_id": beat_id
                }

                # Beat-level features
                for computer in self._beat_computers:
                    beat_features = computer.compute(
                        beat, delineation.sampling_rate
                    )
                    row.update(beat_features)

                # Multi-lead features (broadcast to every lead)
                row.update(multi_lead_features)
                rows.append(row)

        # Compute multi-beat features (once per lead, broadcast)
        df = pd.DataFrame(rows)
        if not df.empty:
            df = self._apply_multi_beat_features(df, delineation)

        return SubjectFeatures(
            subject_id = delineation.record_id,
            features = df,
            dropped_beats = dropped_beats,
            dropped_reasons = dropped_reasons,
            computation_metadata={
                "n_beat_computers": len(self._beat_computers),
                "n_multi_beat_computers": len(self._multi_beat_computers),
                "n_multi_lead_computers": len(self._multi_lead_computers),
                "min_fiducial_completeness": self._min_completeness,
                "total_feature_names": len(self._all_feature_names),
            }
        )

    def _beat_passes_quality(
            self, beat_map: Dict[LeadName, DelineatedBeat]
    ) -> bool:
        # Check if enough fiducial data exists for feature computation
        completeness_scores = [
            beat.fiducials.completness for beat in beat_map.values()
        ]
        avg_completeness = sum(completeness_scores) / len(completeness_scores)
        return avg_completeness >= self._min_completeness

    def _compute_multi_lead(
            self,
            beat_map: Dict[LeadName, DelineatedBeat],
            sampling_rate: SamplingRate
    ) -> Dict[str, Optional[float]]:
        # Run all multi-lead comptuers for one beat.
        combined = {}
        available_leads = set(beat_map.keys())

        for computer in self._multi_lead_computers:
            required = computer.required_leads
            if required and not all(l in available_leads for l in required):
                # Fill with None - required leads missing
                combined.update(
                    {name: None for name in computer.feature_names()}
                )
            else:
                combined.update(
                    computer.compute(beat_map, sampling_rate)
                )

        return combined

    def _apply_multi_beat_features(
            self,
            df: pd.DataFrame,
            delineation: DelineatedRecord
    ) -> pd.DataFrame:
        """
        Run multi-beat comptuers per lead and merge results into the existing DataFrame.
        Multi-beat features are broadcast: same value for every beat on the same
        lead.
        """
        for computer in self._multi_beat_computers:
            feature_cols = computer.feature_names()

            for lead in delineation.leads:
                beats = delineation.get_lead_beats(lead)

                if len(beats) < computer.minimum_beats:
                    # Not enough beats - fill with None
                    for col in feature_cols:
                        mask = df["lead"] == lead.value
                        df.loc[mask, col] = None
                    continue

                multi_beat_values = computer.compute(
                    beats,
                    delineation.consensus_r_peaks,
                    delineation.sampling_rate
                )

                # Broadcast to all rows for this lead
                mask = df["lead"] == lead.value
                for col, val in multi_beat_values.items():
                    df.loc[mask, col] = val

        return df

    # Dataset-level processing
    def process_dataset(
            self,
            dataset: ECGDataset,
            dataset_id: str = None,
            store_delineations: bool = True
    ) -> DatasetFeatures:
        # Process an entire dataset of ECG records
        
        result = DatasetFeatures(
            dataset_id=dataset_id or dataset.dataset_id
        )

        for record in dataset:
            try:
                features, delineation = self.process_record(
                    record, store_delineation=store_delineations
                )
                result.add_subject(features, delineation)
            except Exception as e:
                # Log by don't crash the whole batch
                print(
                    f"WARNING: Failed to process {record.record_id}: {e}"
                )
                continue

        return result

    
