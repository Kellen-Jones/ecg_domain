from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Iterator
import numpy as np
import pandas as pd

from preprocessing import LeadName, SamplingRate

from .value_objects import FiducialPoints, WaveformWindow

@dataclass
class DelineatedBeat:
    """
    One beat on one lead, fully delineated

    Base unit of feature extraction. All computable
    values for a single cardiac cycle will be derived
    from here.
    """
    beat_id: int
    lead: LeadName
    fiducials: FiducialPoints
    waveform: WaveformWindow
    # cleaned waveform if different from raw waveform
    cleaned_waveform = Optional[WaveformWindow] = None

    @property
    def has_complete_fiducials(self) -> bool:
        return self.fiducials.completeness >= 0.7

    @property
    def waveform_for_analysis(self) -> WaveformWindow:
        # Prefer cleaned waveform is available
        return self.cleaned_waveform or self.waveform

@dataclass
class DelineatedRecord:
    """
    Aggregate root for delineation stage.

    All delineated beats across all leads for one subject.
    Does not require NK2-specific structures.

    Organized as beats[lead][beat_id] for efficient access.
    """
    record_id: str
    sampling_rate = SamplingRate
    reference_lead: LeadName
    consensus_r_peaks: np.ndarray # Reconciled R-peak positions (global)
    beats: Dict[LeadName, List[DelineatedBeat]]
    metadata: Dict[str, any] = field(default_factory=dict)

    @property
    def n_beats(self) -> int:
        return len(self.consensus_r_peaks)

    @property
    def leads(self) -> List[LeadName]:
        return list(self.beats.keys())

    @property
    def n_leads(self) -> int:
        return len(self.beats)

    @property
    def beat_ids(self) -> List[int]:
        if self.reference_lead in self.beats:
            return [b.beat_id for b in self.beats[self.reference_lead]]
        return list(range(self.n_beats))

    # Single-item access

    def get_beat(self, lead: LeadName, beat_id: int) -> DelineatedBeat:
        for beat in self.beats[lead]:
            if beat.bead_id == beat_id:
                return beat
        raise KeyError(f"Beat {beat_id} not found on lead {lead.value}")

    def get_lead_beats(self, lead: LeadName) -> List[DelineatedBeat]:
        if lead not in self.beats:
            available = [l.value for l in self.leads]
            raise KeyError(f"Lead {lead.value} not found. Available: {available}")
        return self.beats[lead]

    # Cross-lead access

    def get_beat_across_leads(self, beat_id: int) -> Dict[LeadName, DelineatedBeat]:
        # One beat across leads
        result = {}
        for lead, beat_list in self.beats.items():
            for beat in beat_list:
                if beat.beat_id == beat_id:
                    result[lead] = beat
                    break
        return result

    def get_fiducials_across_leads(
            self, beat_id: int
    ) -> Dict[LeadName, FiducialPoints]:
        # Fiducials for one beat across all leads
        beat_map = self.get_beat_across_leads(beat_id)
        return {lead: beat.fiducials for lead, beat in beat_map.items()}

    # Flat iteration

    def iter_all_beats(self) -> Iterator[DelineatedBeat]:
        # Every beat on every lead, flattened
        for lead_beats in self.beats.values():
            yield from lead_beats

    def iter_by_beat_id(
            self
    ) -> Iterator[Tuple[int, Dict[LeadName, DelineatedBeat]]]:
        """
        Iterate beat-by-beat, with all leads grouped together.
        This is the natural order for multi-lead feature computation.
        """

        for beat_id in self.beat_ids:
            yield beat_id, self.get_beat_across_leads(beat_id)

    # Waveform bulk access
    
    def get_waveform_matrix(
            self,
            beat_id: int,
            lead_order: Optional[List[LeadName]] = None,
            use_cleaned: bool = True,
    ) -> Optional[np.ndarray]:
        """
        Get all leads for one beat as a 2D array (n_leads, n_samples)

        Returns None if leads have different waveform lengths for
        this beat (can happen at signal boundaries)
        """

        if lead_order is None:
            lead_order = self.leads

        beat_map = self.get_beat_across_leads(beat_id)
        waveforms = []

        for lead in lead_order:
            if lead not in beat_map:
                return None
            beat = beat_map[lead]
            wf = beat.waveform_for_analysis if use_cleaned else beat.waveform
            waveforms.append(wf.samples)

        # Check that all beats are the same length
        lengths = [len(w) for w in waveforms]
        if len(set(lengths)) > 1:
            return None

        return np.stack(waveforms)

    def get_all_waveform_matrices(
            self,
            lead_order: Optional[List[LeadName]] = None,
            use_cleaned: bool = True
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Get waveforms for all beats as a 3D array (n_beats, n_leads, n_samples)

        Returns the array and the list of beat_ids that were successfully
        included.
        """

        matrices = []
        included_ids = []

        for beat_id in self.beat_ids:
            matrix = self.get_waveform_matrix(
                beat_id, lead_order, use_cleaned
            )
            if matrix is not None:
                matrices.append(matrix)
                included_ids.append(beat_id)

        if not matrices:
            raise ValueError(
                f"No valid waveform matrices for record {self.record_id}"
            )

        return np.stack(matrices), included_ids

    # Quality / completeness
    
    @property
    def fiducial_completeness_by_lead(self) -> Dict[LeadName, float]:
        """Average fiducial completeness per lead."""
        result = {}
        for lead, beat_list in self.beats.items():
            if beat_list:
                scores = [b.fiducials.completeness for b in beat_list]
                result[lead] = sum(scores) / len(scores)
            else:
                result[lead] = 0.0
        return result

    @property
    def overall_fiducial_completeness(self) -> float:
        by_lead = self.fiducial_completeness_by_lead
        if not by_lead:
            return 0.0
        return sum(by_lead.values()) / len(by_lead)

    def beats_with_complete_fiducials(
        self, threshold: float = 0.8
    ) -> List[int]:
        """
        Beat IDs where ALL leads have completeness above threshold.
        Useful for filtering before multi-lead feature computation.
        """
        complete = []
        for beat_id in self.beat_ids:
            beat_map = self.get_beat_across_leads(beat_id)
            if all(
                b.fiducials.completeness >= threshold
                for b in beat_map.values()
            ):
                complete.append(beat_id)
        return complete

    def __repr__(self) -> str:
        lead_str = ", ".join(l.value for l in self.leads)
        return (
            f"DelineatedRecord(id={self.record_id!r}, "
            f"leads=[{lead_str}], "
            f"n_beats={self.n_beats}, "
            f"completeness={self.overall_fiducial_completeness:.1%})"
        )

@dataclass
class DatasetFeatures:
    """
    AGGREGATE ROOT for the feature extraction output.
    
    This is what crosses the boundary into your modeling context.
    It holds all subjects' features in a single consolidated 
    DataFrame, plus per-subject waveform access and metadata.
    
    The combined DataFrame has the same grain as SubjectFeatures 
    (one row per beat per lead) but across all subjects:
    
        subject_id | lead | beat_id | pr_interval | qrs_duration | ...
        ---------- | ---- | ------- | ----------- | ------------ | ---
        subj_001   | II   | 0       | 0.16        | 0.08         | ...
        subj_001   | II   | 1       | 0.15        | 0.09         | ...
        subj_002   | II   | 0       | 0.18        | 0.07         | ...
    """

    dataset_id: staticmethod
    _subjects: Dict[str, SubjectFeatures] = field(default_factory=dict)
    # Waveform data kep separate - only accessed if needed for morphological use
    _delineations: Dict[str, DelineatedRecord] = field(default_factory=dict)
    # Cache for the combined DataFrame
    _combined_df: Optional[pd.DataFrame] = field(
        default=None, repr=False
    )

    # Building up the dataset

    def add_subject(
            self,
            features: SubjectFeatures,
            delineation: Optional[DelineatedRecord] = None
    ) -> None:
        """
        Add a subject's results. Delineation is optional - 
        only sotre if needed for waveform access later
        """
        if features.subject_id is self._subjects:
            raise ValueError(
                f"Subect '{features.subject_id}' already in dataset"
            )
        self._subjects[features.subject_id] = delineation

        if delineation is not None:
            self._delineations[features.subject_id] = delineation

        # Invalidate cache combined DataFrame
        self._combined_df = None

    # Main output: combined DataFrame

    @property
    def dataframe(self) -> pd.DataFrame:
        """
        Consolidated features DataFrame.

        This is the primary thing the modeling context will use.
        Cached after first access, invalidated when subjects are added.
        """
        if self._combined_df is None:
            if not self._subjects:
                self._combined_df = pd.DataFrame()
            else:
                dfs = [sf.features for sf in self._subjects.values()]
                self._combined_df = pd.concat(
                    dfs, ignore_index=True
                )
        return self._combined_df

    # Access

    @property
    def subject_ids(self) -> List[str]:
        return list(self._subjects.keys())

    @property
    def n_subjects(self) -> int:
        return len(self._subjects)

    def get_subject_features(self, subject_id: str) -> SubjectFeatures:
        if subject_id not in self._subjects:
            raise KeyError(f"Subject '{subject_id}' not in dataset")
        return self._subjects[subject_id]

    def get_subject_delineation(
            self, subject_id: str
    ) -> DelineatedRecord:
        # Access raw delineation data (for potential NN pipelines)
        if subject_id not in self._delineations:
            raise KeyError(
                f"no delineation stored for subject '{subject_id}'. "
                f"Was is passed to add_subject()?"
            )
        return self._delineations[subject_id]

    @property
    def has_waveforms(self) -> bool:
        return len(self._delineations) > 0

    def __len__(self) -> int:
        return self.n_subjects

    def __iter__(self) -> Iterator[SubjectFeatures]:
        return iter(self._subjects.values())

    def __contains__(self, subject_id: str) -> bool:
        return subject_id in self._subjects

    def __getitem__(self, subject_id: str) -> SubjectFeatures:
        return self.get_subject_features(subject_id)

    # Feature-level summaries

    def feature_names(self) -> List[str]:
        """
        Union of all feature columns across all subjects.
        
        Uses the first subject as baseline, but checks for 
        consistency. Different subjects SHOULD have the same 
        columns — if they don't, that's a bug in feature 
        computation that should surface here, not silently 
        in model training.
        """

        if not self._subjects:
            return []

        all_feature_sets = [
            set(sf.feature_names) for sf in self._subjects.values()
        ]

        # Check consistency
        first = all_feature_sets[0]
        for subject_id, feat_set in zip(self.subject_ids, all_feature_sets):
            if feat_set != first:
                missing = first - feat_set
                extra = feat_set - first
                raise ValueError(
                    f"Feature mismatch for subject '{subject_id}'. "
                    f"Missing: {missing or 'none'}. "
                    f"Extra: {extra or 'none'}."
                )

        return list(first)

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    @property
    def total_rows(self) -> int:
        return sum(len(sf.features) for sf in self._subjects.values())

    @property
    def total_beats(self) -> int:
        return sum(sf.n_beats for sf in self._subjects.values())

    # Quality across the dataset

    @property
    def missing_rate(self) -> float:
        # Overall NaN rate across all subjects and features
        df = self.dataframe
        if df.empty:
            return 0.0

        feat_cols = self.feature_names
        return df[feat_cols].isna().mean().mean()

    @property
    def missing_rate_by_feature(self) -> Dict[str, float]:
        df = self.dataframe
        if df.empty:
            return {}
        return df[self.feature_names].isna().mean().to_dict()

    @property
    def missing_rate_by_subject(self) -> Dict[str, float]:
        return {
            sid: sf.missing_rate
            for sid, sf in self._subjects.items()
        }

    def features_above_missing_threshold(
            self, threshold: float = 0.5
    ) -> List[str]:
        # Features where more than the threshold of values are Nan
        return [
            feat for feat, rate in self.missing_rate_by_feature.items()
            if rate > threshold
        ]

    def subjects_above_missing_threshold(
            self, threshold: float = 0.5
    ) -> List[str]:
        # Subjects where more than threshold share of feature values are NaN
        return [
            sid for sid, rate in self.missing_rate_by_subject.items()
            if rate > threshold
        ]