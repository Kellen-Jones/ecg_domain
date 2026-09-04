from dataclasses import dataclass
from typing import Optional
import numpy as np

# Import from preprocessing (will work on lab desktop)
from preprocessing import SamplingRate, LeadName

@dataclass(frozen=True)
class FiducialPoints:
    """
    Agnostic representation of fiducial points for a single beat.

    All values are sample indices relative to the beat onset,
    not absolute positions in the full signal. Values are then portable - 
    not dependent on where the beat sits in the original recording.

    r_peak is the only required point, as the determination of the segment
    as a beat is dependent on an R-peak being present. All other points are
    Optional.

    All extractors must populate this extractor.
    """

    r_peak: int

    p_onset: Optional[int] = None
    p_peak: Optional[int] = None
    p_offset: Optional[int] = None

    qrs_onset: Optional[int] = None
    qrs_offset: Optional[int] = None
    q_peak: Optional[int] = None
    s_peak: Optional[int] = None

    t_onset: Optional[int] = None
    t_peak: Optional[int] = None
    t_offset: Optional[int] = None

    def __post_init__(self):
        for field_name in self.__annotations__:
            val = getattr(self, field_name)

            if self._is_nan(val):
                if field_name == "rpeak":
                    raise ValueError("r_peak cannot be NaN")
                object.__setattr____(self, field_name, None)

            elif val is not None:
                int_val = int(val)
                if int_val < 0:
                    raise ValueError(
                        f"{field_name} cannot be negative, got {int_val}"
                    )
                object.__setattr__(self, field_name, int_val)

        self._validate_ordering()

    @staticmethod
    def _is_nan(val) -> bool:
        if val is None:
            return False
        try:
            return np.isnan(val)
        except (TypeError, ValueError):
            return False

    def _validate_ordering(self):
        ordered_points = [
            ("p_onset", self.p_onset),
            ("p_peak", self.p_peak),
            ("p_offset", self.p_offset),
            ("qrs_onset", self.qrs_onset),
            ("q_peak", self.q_peak),
            ("r_peak", self.r_peak),
            ("s_peak", self.s_peak),
            ("qrs_offset", self.qrs_offset),
            ("t_onset", self.t_onset),
            ("t_peak", self.t_peak),
            ("t_offset", self.t_offset),   
        ]
        present = [(name, val) for name, val in ordered_points if val is not None]

        for i in range(1, len(present)):
            if present[i][1] < present[i-1][1]:
                raise ValueError(
                    f"Fiducial ordering violation: "
                    f"{present[i-1][0]}={present[i-1][1]} is after "
                    f"{present[i][0]}={present[i][1]}"
                )
    
    @property
    def has_p_wave(self) -> bool:
        return self.p_peak is not None

    @property
    def has_t_wave(self) -> bool:
        return self.t_peak is not None

    @property
    def has_full_qrs(self) -> bool:
        return all(v is not None for v in [
            self.qrs_onset, self.q_peak, self.s_peak, self.qrs_offset
        ])

    @property
    def completeness(self) -> float:
        # Fraction of fiducial points that were successfully detected
        all_points = [
            self.p_onset, self.p_peak, self.p_offset,
            self.qrs_onset, self.q_peak, self.s_peak, self.qrs_offset,
            self.t_onset, self.t_peak, self.t_offset
        ]
        found = sum(1 for p in all_points if p is not None)
        return found / len(all_points)

    def pr_interval_samples(self) -> Optional[int]:
        # PR interval in samples. Returns None if points missing.
        if self.p_onset is not None and self.qrs_offset is not None:
            return self.qrs_onset - self.p_onset
        return None

    def qrs_duration_samples(self) -> Optional[int]:
        if self.qrs_onset is not None and self.qrs_offset is not None:
            return self.qrs_offset - self.qrs_onset
        return None

    def qt_interval_samples(self) -> Optional[int]:
        if self.qrs_onset is not None and self.t_offset is not None:
            return self.t_offset - self.qrs_onset
        return None

    def to_dict(self) -> dict:
        # Easy conversion to build DataFrames
        return {
            "r_peak": self.r_peak,
            "p_onset": self.p_onset,
            "p_peak": self.p_peak,
            "p_offset": self.p_offset,
            "qrs_onset": self.qrs_onset,
            "q_peak": self.q_peak,
            "s_peak": self.s_peak,
            "qrs_offset": self.qrs_offset,
            "t_onset": self.t_onset,
            "t_peak": self.t_peak,
            "t_offset": self.t_offset,
        }

@dataclass(frozen=True)
class WaveformWindow:
    """
    Segment of signal corresponding to one beat on one lead.

    Kept separate from FiducialPoints due to:
        - Feature derivation requires fiducial features
        - Future neural networks require waveforms
    """
    samples: np.ndarray # 1D array, raw waveform
    onset_sample: int # Absolute position in original signal
    offset_sample: int

    @property
    def n_smaples(self) -> int:
        return len(self.samples)

    def __post_init__(self):
        if self.samples.ndim != 1:
            raise ValueError(
                f"WaveformWindow must be 1D, got shape {self.samples.shape}"
            )

@dataclass(frozen=True)
class FeatureLevel:
    # Identifies what scope a feature was computed at
    BEAT_LEAD = "beat_lead" # One beat, one lead
    BEAT_MULTILEAD = "beat_multilead" # One beat, across leads
    SUBJECT_LEAD = "subject_lead" # across beat, one lead
    SUBJECT = "subject" # multi-beat, multi-lead data