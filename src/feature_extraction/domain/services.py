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
            refiner: "FiducialRefiner",
            beat_computers: List[BeatLevelComputer],
            multi_beat_computers: List[MultiBeatComputer],
            multi_lead_computers: List[MultiLeadComputer],
            min_fiducial_completeness: float = 0.5
    ):
        self._extractor = extractor
        self._refiner = refiner
        self._beat_computers = beat_computers
        self._multi_beat_computers = multi_beat_computers
        self._multi_lead_computers = multi_lead_computers
        self._min_completeness = min_fiducial_completeness
        self._all_feature_names = self._collect_feature_names()

    def _collect_feature_names(self) -> List[str]:
        names = []
        for c in self._beat_computers:
            names.extend(c.feature_names())
        for c in self._multi_beat_computers:
            names.extend(c.feature_names())
        for c in self._multi_lead_computers:
            names.extend(c.feature_names())

        seen = set()
        