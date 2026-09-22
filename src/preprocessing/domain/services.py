from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import logging

from preprocessing import ECGRecord, ECGDataset, LeadName, SamplingRate

from .entities import (
    DelineatedBeat,
    DelineatedRecord,
    SubjectFeatures,
    DatasetFeatures
)

from .value_objects import FiducialPoints, WaveformWindow, BeatAnalysisContext
from ..infrastructure.fiducial_refiner import FiducialRefiner

log = logging.getLogger(__name__)

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

    def __init__(
            self,
            extractor: FiducialExtractor,
            refiner: "FiducialRefiner",
            beat_computers: List[BeatLevelComputer],
            multi_beat_computers: List[MultiBeatComputer],
            multi_lead_computers: List[MultiLeadComputer],
            min_fiducial_completeness: float = 0.4
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
            record: "ECGRecord",
            store_delineation: bool = True
    ) -> Tuple["SubjectFeatures", Optional["DelineatedRecord"]]:
        delineation = self._extractor.extract(record)
        features = self._compute_features(delineation)
        delineation_out = delineation if store_delineation else None
        return features, delineation_out

    def _compute_features(
            self, delineation: "DelineatedRecord"
    ) -> "SubjectFeatures":
        """
        Full feature computation pipeline:
          1. For each beat, across all leads:
             a. Build BeatAnalysisContext (waveform, derivatives, baseline)
             b. Refine fiducials (fill in missing points)
             c. Run beat-level computers
             d. Run multi-lead computers
          2. For each lead:
             a. Run multi-beat computers
          3. Merge everything into a single DataFrame
        """
        rows: List[Dict[str, Any]] = []
        dropped_beats: List[int] = []
        dropped_reasons: Dict[int, str] = {}

        for beat_id, beat_map in delineation.iter_by_beat_id():

            # Quality gate: check raw fiducial completeness
            if not self._beat_passes_quality(beat_map):
                dropped_beats.append(beat_id)
                dropped_reasons[beat_id] = "insufficient_fiducials"
                continue

            # Refine fiducials and build contexts for each lead
            contexts: Dict["LeadName", "BeatAnalysisContext"] = {}
            refined_beat_map: Dict["LeadName", "DelineatedBeat"] = {}

            for lead, beat in beat_map.items():
                ctx = BeatAnalysisContext(
                    waveform=beat.waveform_for_analysis.samples,
                    sampling_rate=delineation.sampling_rate.hz,
                    fiducials=beat.fiducials
                )

                # Refine fiducials
                refined_fiducials = self._refiner.refine(
                    beat.fiducials, ctx
                )

                # Rebuild context with refined fiducials
                ctx = BeatAnalysisContext(
                    waveform=beat.waveform_for_analysis.samples,
                    sampling_rate=delineation.sampling_rate.hz,
                    fiducials=refined_fiducials
                )

                contexts[lead] = ctx
                refined_beat_map[lead] = beat

            # Multi-lead features
            multi_lead_features = self._compute_multi_Lead(
                refined_beat_map,
                delineation.sampling_rate
            )

            # Per-lead features
            for lead, ctx in contexts.items():
                row: Dict[str, Any] = {
                    "subject_id": delineation.record_id,
                    "lead": lead.value,
                    "beat_id": beat_id
                }

                # Run for each beat-level computer
                for computer in self._beat_computers:
                    try:
                        beat_features = computer.compute(ctx)
                        row.update(beat_features)
                    except Exception as e:
                        log.warning(
                            f"Computer {computer.__class__.__name__} "
                            f"failed on {delineation.record_id}/"
                            f"{lead.value}/beat_{beat_id}: {e}"
                        )
                        # Fill with None for this computer's features
                        row.update(
                            {name: None for name in computer.feature_names()}
                        )

                # Broadcast multi-lead features to every lead row
                row.update(multi_lead_features)

                rows.append(row)

        # Build DataFrame from beat-level rows
        df = pd.DataFrame(rows)

        # Multi-beat features (once per lead, broadcast)
        if not df.empty:
            df = self._apply_multi_beat_features(df, delineation)

        return SubjectFeatures(
            subject_id=delineation.record_id,
            features=df,
            dropped_beats=dropped_beats,
            dropped_reasons=dropped_reasons,
            computation_metadata={
                "n_beat_computers": len(self._beat_computers),
                "n_multi_beat_computers": len(self._multi_beat_computers),
                "n_multi_lead_computers": len(self._multi_lead_computers),
                "min_fiducial_completeness": self._min_completeness,
                "total_feature_names": len(self._all_feature_names),
            },
        )

    def _beat_passes_quality(
            self, beat_map: Dict["LeadName", "DelineatedBeat"]
    ) -> bool:
        completeness_scores = [
            beat.fiducials.completeness for beat in beat_map.values()
        ]
        avg_completeness = sum(completeness_scores) / len(completeness_scores)
        return avg_completeness >= self._min_completeness

    def _computer_multi_lead(
            self,
            beat_map: Dict["LeadName", "DelineatedBeat"],
            sampling_rate: "SamplingRate"
    ) -> Dict[str, Optional[float]]:
        combined: Dict[str, Optional[float]] = {}
        available_leads = set(beat_map.keys())

        for computer in self._multi_lead_computers:
            required = computer.required_leads
            if required and not all(l in available_leads for l in required):
                combined.update(
                    {name: None for name in computer.feature_names()}
                )
            else:
                try:
                    combined.update(
                        computer.computer(beat_map, sampling_rate)
                    )
                except Exception as e:
                    log.warning(
                        f"Multi-lead computer "
                        f"{computer.__class__.__name__} failed: {e}"
                    )
                    combined.update(
                        {name: None for name in computer.feature_names()}
                    )

        return combined


    def _apply_multi_beat_features(
            self,
            df: pd.DataFrame,
            delineation: "DelineatedRecord"
    ) -> pd.DataFrame:
        for computer in self._multi_beat_computers:
            feature_cols = computer.feature_names()

            for lead in delineation.leads:
                beats = delineation.get_lead_beats(lead)
                mask = df["lead"] == lead.value

                if len(beats) < computer.minimum_beats:
                    for col in feature_cols:
                        df.loc[mask, col] = None
                    continue

                try:
                    multi_beat_values = computer.compute(
                        beats,
                        delineation.consensus_r_peaks,
                        delineation.sampling_rate
                    )
                    for col, val in multi_beat_values.items():
                        df.loc[mask, col] = val

                except Exception as e:
                    log.warning(
                        f"Multi-beat computer "
                        f"{computer.__class__.__name__} failed on "
                        f"lead {lead.value}: {e}"
                    )
                    for col in feature_cols:
                        df.loc[mask, col] = None

        return df

    # Dataset-level processing

    def process_dataset(
            self,
            dataset: "ECGDataset",
            dataset_id: str = None,
            store_delineations: bool = True
    ) -> "DatasetFeatures":
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
                log.error(
                    f"Failed to process {record.record_id}: {e}"
                )
                continue

        return result