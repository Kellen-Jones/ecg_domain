from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Iterator, Callable
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
class SubjectFeatures:

    subject_id: str
    features: pd.DataFrame
    dropped_beats: List[int] = field(default_factory=list)
    dropped_reasons: Dict[int, str] = field(default_factory=dict)
    computation_metadata: Dict[str, Any] = field(default_factory=dict)

    # Required columns which must exist
    _INDEX_COLUMNS = ["subject_id", "lead", "beat_id"]

    def __post_init__(self):
        if self.features.empty:
            return

        # Validate required columns exist
        missing = [
            col for col in self._INDEX_COLUMNS
            if col not in self.features.columns
        ]
        if missing:
            raise ValueError(
                f"Features DataFrame missing required columns: {missing}. "
                f"Got columns: {list(self.features.columns)}"
            )

        # Validate subject_id consistency
        subjects_in_df = self.features["subject_id"].unique()
        if len(subjects_in_df) > 1:
            raise ValueError(
                f"SubjectFeatures should contain one subject, "
                f"found: {subjects_in_df}."
            )
        if subjects_in_df[0] != self.subject_id:
            raise ValueError(
                f"subject_id mismatch: entity says '{self.subject_id}', "
                f"DataFrame contains '{subjects_in_df[0]}'"
            )

    # Access

    @property
    def n_beats(self) -> int:
        return self.features["beat_id"].nunique()

    @property
    def n_leads(self) -> int:
        return self.features["lead"].nunique()

    @property
    def feature_names(self) -> List[str]:
        # Just computed feature columns, not the index columns
        return [
            col for col in self.features.columns
            if col not in self._INDEX_COLUMNS
        ]

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    @property
    def leads_present(self) -> List[str]:
        return sorted(self.features["leads"].unique().tolist())

    @property
    def beat_ids_present(self, lead: LeadName) -> List[int]:
        # All beats for one lead
        mask = self.features["lead"] == lead.value
        result = self.features[mask]
        if result.empty:
            available =self.leads_present
            raise KeyError(
                f"No features for lead {lead.value}. "
                f"Available: {available}"
            )
        return result

    def get_beat_features(self, beat_id: int) -> pd.DataFrame:
        # All leads for one beat
        mask = self.features["beat_id"] == beat_id
        result = self.features[mask]
        if result.empty:
            raise KeyError(f"No features for beat_id {beat_id}")
        return result

    def get_feature_vector(
            self, lead: LeadName, beat_id: int
    ) -> pd.Series:
        # Single row, one beat, one lead.
        mask = (
            (self.features["lead"] == lead.value)
            & (self.features["beat_id"] == beat_id)
        )
        result = self.features[mask]
        if len(result) != 1:
            raise KeyError(
                f"Expected 1 row for lead={lead.value}, "
                f"beat_id={beat_id}, got {len(result)}"
            )
        return result.iloc[0]

    # Quality

    @property
    def missing_rate(self) -> float:
        # Fraction of NaN values across all feature columns.
        feat_cols = self.feature_names
        if not feat_cols:
            return 0.0
        return self.features[feat_cols].isna().mean().mean()

    @property
    def missing_rate_by_feature(self) -> Dict[str, float]:
        # NaN rate per feature column
        return self.features[self.feature_names].isna().mean().to_dict()

    @property
    def n_dropped_beats(self) -> int:
        return len(self.dropped_beats)

    def features_above_missing_threshold(
            self, threshold: float = 0.7
    ) -> List[str]:
        # Features with more than 'threshold' fraction missing
        return [
            feat for feat, rate in self.missing_rate_by_feature.items()
            if rate > threshold
        ]

    def __repr__(self) -> str:
        return (
            f"SubjectFeatures(id={self.subject_id!r}, "
            f"n_beats={self.n_beats}, "
            f"n_features={self.n_features}, "
            f"missing={self.missing_rate:.1%}, "
            f"dropped_beats={self.n_dropped_beats})"
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

    # Dropped beat tracking

    @property
    def total_dropped_beats(self) -> int:
        return sum(sf.n_dropped_beats for sf  in self._subjects.values())

    @property
    def drop_rate(self) -> float:
        # fraction of beats dropped across the entire dataset
        total_original = self.total_beats + self.total_dropped_beats
        if total_original == 0:
            return 0.0

        return self.total_dropped_beats / total_original

    @property
    def dropped_beats_by_subject(self) -> Dict[str, List[int]]:
        return {
            sid: sf.dropped_beats
            for sid, sf in self._subjects.items()
            if sf.dropped_beats
        }

    # Filtering / subsetting

    def filter_subjects(
            self,
            predicate: Callable[[SubjectFeatures], bool],
            dataset_id: str = None
    ) -> "DatasetFeatures":
        
        """
        Return a new DatasetFeatures keeping only subjects
        which pass the predicate.

        Usage:
            clean = dataset.filter_subjects(
                lambda sf: sf.missing_rate < 0.1
            )
            enough-beats = dataset.filter_subjects(
                lambda sf: sf.n_beats >= 20
            )
        """
        filtered = DatasetFeatures(
            dataset_id=dataset_id or f"{self.dataset_id}_filtered"
        )
        for sid, sf in self._subjects.items():
            if predicate(sf):
                delineation = self._delineations.get(sid)
                filtered.add_subject(sf, delineation)

        return filtered

    def drop_features(
            self,
            feature_names: List[str],
            dataset_id: str = None
    ) -> "DatasetFeatures":
        """
        Return a new DatasetFeatures with specific feature
        columns removed from every subject.

        Usage:
            # Drop features that are mostly NaN
            bad_features = dataset.features_above_missing_threshold(0.5)
            cleaned = dataset.drop_features(bad_features)
        """
        result = DatasetFeatures(
            dataset_id=dataset_id or f"{self.dataset_id}pruned"
        )
        for sid, sf in self._subjects.items():
            cols_to_keep = [
                c for c in sf.features.columns
                if c not in feature_names
            ]
            pruned_df = sf.features[cols_to_keep].copy()

            pruned_sf = SubjectFeatures(
                subject_id=sf.subject_id,
                features=pruned_df,
                dropped_beats=sf.dropped_beats,
                dropped_reasons=sf.dropped_reasons,
                computation_metadata= {
                    **sf.computation_metadata,
                    "dropped_features": feature_names
                }
            )
            delineation = self._delineations.get(sid)
            result.add_subject(pruned_sf, delineation)

        return result

    def subset_leads(
            self,
            leads: List[LeadName],
            dataset_id: str = None
    ) -> "DatasetFeatures":
        # Return a new DatasetFeatures keeping only specific leads
        lead_values = {l.value for l in leads}
        result = DatasetFeatures(
            dataset_id=dataset_id or f"{self.dataset_id}_leads_subset"
        )
        for sid, sf in self._subjects.items():
            mask = sf.features["lead"].isin(lead_values)
            subset_df = sf.features[mask].copy()

            subset_sf = SubjectFeatures(
                subject_id=sf.subject_id,
                features=subset_df,
                dropped_beats=sf.dropped_beats,
                dropped_reasons=sf.dropped_reasons,
                computation_metadata={
                    **sf.computation_metadata,
                    "lead_subset": [l.value for l in leads]
                }
            )
            delineation = self._delineations.get(sid)
            result.add_subject(subset_sf, delineation)
        return result

    # Waveform bulk access for NN pipelines

    def get_all_waveform_tensors(
            self,
            lead_order: Optional[List[LeadName]] = None,
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Build a dataset-wide waveform tensor for neural network training.

        Returns:
        --------
            waveforms: (total_beats, n_leads, n_samples) array
            index: DataFrame with subject_id and beat_id for each
                row in the waveforms array to align with the 
                features DataFrame
        """
        if not self._delineations:
            raise ValueError(
                "No delineation data stored. Pass delineation "
                "to add_subject() to enable waveform access."
            )

        all_matrices = []
        index_rows = []

        for sid, delin in self._delineations.items():
            try:
                matrices, beat_ids = delin.get_all_waveform_matrices(
                    lead_order=lead_order,
                )
                for bid in beat_ids:
                    index_rows.append({
                        "subject_id": sid,
                        "beat_id": bid
                    })
            except ValueError:
                continue

        if not all_matrices:
            raise ValueError("No valid waveform matrices in dataset")

        # all matrices must be the same dimensions
        sample_counts = {m.shape[2] for m in all_matrices}
        if len(sample_counts) > 1:
            # truncating to shortest count/subject
            min_samples = min(sample_counts)
            all_matrices = [m[:, :, :min_samples] for m in all_matrices]

        combined = np.concatenate(all_matrices, axis=0)
        index_df = pd.DataFrame(index_rows)

        return combined, index_df

    # Summary

    def summary(self) -> dict[str, Any]:
        result = {
            "dataset_id": self.dataset_id,
            "n_subjects": self.n_subjects,
            "n_features": self.n_features,
            "total_beats": self.total_beats,
            "total_rows": self.total_rows,
            "missing_rate": self.missing_rate,
            "total_dropped_beats": self.total_dropped_beats,
            "drop_rate": self.drop_rate,
            "has_waveforms": self.has_waveforms
        }

        if self._subjects:
            beats_per_subject = [sf.n_beats for sf in self._subjects.values()]
            result["beats_per_subject"] = {
                "min": min(beats_per_subject),
                "max": max(beats_per_subject),
                "mean": sum(beats_per_subject) / len(beats_per_subject)
            }

            problematic_features = self.features_above_missing_threshold(0.3)
            if problematic_features:
                result["high_missing_features"] = problematic_features

            problematic_subjects = self.subjects_above_missing_threshold(0.3)
            if problematic_subjects:
                result["high_missing_subjects"] = problematic_subjects

        return result

    def __repr__(self) -> str:
        return (
            f"DatasetFeatures(id={self.dataset_id}, "
            f"n_subjects={self.n_subjects}, "
            f"n_features={self.n_features}, "
            f"total_beats={self.total_beats}, "
            f"missing={self.missing_rate:.1%}, "
            f"waveforms={'yes' if self.self.has_waveforms else 'no'})"
        )
    