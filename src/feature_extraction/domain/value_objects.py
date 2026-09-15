from dataclasses import dataclass, field
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

    j_point: Optional[int] = None
    q_onset: Optional[int] = None
    st_midpoint: Optional[int] = None

    # Estimation flags - which points were estimated vs detected
    estimated_points: tuple = field(default_factory=tuple)

    def __post_init__(self):
        for field_name, annotation in self.__annotations__.items():
            if field_name in ('estimated_points'):
                continue

            val = getattr(self, field_name)

            if self._is_nan(val):
                if field_name == "r_peak":
                    raise ValueError("r_peak cannot be NaN")
                object.__setattr__(self, field_name, None)
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
            ("j_point", self.j_point),
            ("st_midpoint", self.st_midpoint),
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

    def is_estimated(self, point_name: str) -> bool:
        return point_name in self.estimated_points

    def with_updates(self, **kwargs) -> "FiducialPoints":
        """
        Create a new FiducialPoints with some fields changed.
        
        Since it's frozen, this is how refinement works:
        the refiner produces a NEW FiducialPoints with 
        filled-in values rather than mutating the original.
        """
        current = {
            f: getattr(self, f)
            for f in self.__annotations__
        }
        current.update(kwargs)
        return FiducialPoints(**current)
    
    
    # Interval helpers
    def pr_interval_samples(self) -> Optional[int]:
        if self.p_onset is not None and self.qrs_onset is not None:
            return self.qrs_onset - self.p_onset
        return None

    def qrs_duration_samples(self) -> Optional[int]:
        if self.qrs_onset is not None and self.qrs_offset is not None:
            return self.qrs_offset - self.qrs_onset
        return None

    def qt_interval_samples(self) -> Optional[int]:
        onset = self.q_onset or self.qrs_onset
        if onset is not None and self.t_offset is not None:
            return self.t_offset - onset
        return None

    def st_segment_samples(self) -> Optional[int]:
        start = self.j_point or self.qrs_offset
        if start is not None and self.t_onset is not None:
            return self.t_onset - start
        return None

    @property
    def completeness(self) -> float:
        core_points = [
            self.p_onset, self.p_peak, self.p_offset,
            self.qrs_onset, self.q_peak, self.s_peak, self.qrs_offset,
            self.t_onset, self.t_peak, self.t_offset
        ]
        found = sum(1 for p in core_points if p is not None)
        return found / len(core_points)

    def to_dict(self) -> dict:
        return {
            "r_peak": self.r_peak,
            "p_onset": self.p_onset,
            "p_peak": self.p_peak,
            "p_offset": self.p_offset,
            "qrs_onset": self.qrs_onset,
            "qrs_offset": self.qrs_offset,
            "q_peak": self.q_peak,
            "q_onset": self.q_onset,
            "s_peak": self.s_peak,
            "j_point": self.j_point,
            "st_midpoint": self.st_midpoint,
            "t_onset": self.t_onset,
            "t_peak": self.t_peak,
            "t_offset": self.t_offset,
            "estimated_points": self.estimated_points,
        }

@dataclass
class BeatAnalysisContext:
    """
    Pre-computed shared data that all feature computers need.
    
    NOT frozen because it's a working context, not a domain value.
    Created fresh for each beat, discarded after.
    """
    waveform: np.ndarray # Cleaned signal for this beat
    sampling_rate: float # Hz
    fiducials: FiducialPoints # refined fiducials

    # Pre-computed
    baseline: float = 0.0
    first_derivative: Optional[np.ndarray] = None
    second_derivative: Optional[np.ndarray] = None
    time_array: Optional[np.ndarray] = None

    def __post_init__(self):
        n = len(self.waveform)
        self.time_array = np.arange(n) / self.sampling_rate

        # Baseline: mean of first and last samples
        self.baseline = float(
            np.mean([self.waveform[0], self.waveform[-1]])
        )

        # Derivatives with respect to time
        if n > 1:
            self.first_derivative = np.gradient(
                self.first_derivative, self.time_array
            )
            self.second_derivative = np.gradient(
                self.first_derivative, self.time_array
            )
        else:
            self.first_derivative = np.zeros(n)
            self.second_derivative = np.zeros(n)

    @property
    def n_samples(self) -> int:
        return len(self.waveform)

    @property
    def beat_duration_s(self) -> float:
        return self.time_array[-1] if len(self.time_array) > 0 else 0.0

    def is_valid_idx(self, idx: Optional[int]) -> bool:
        return idx is not None and 0 <= idx < self.n_samples

    def amplitude_at(self, idx: Optional[int]) -> Optional[float]:
        if not self.is_valid_idx(idx):
            return None
        return float(self.waveform[idx])

    def amplitude_relative_to_baseline(
            self, idx: Optional[int]
    ) -> Optional[float]:
        amp = self.amplitude_at(idx)
        if amp is None:
            return None
        return amp - self.baseline

    def time_between(
            self, idx_a: Optional[int], idx_b: Optional[int]
    ) -> Optional[float]:
    # Time difference in seconds between two indices
        if not self.is_valid_idx(idx_a) or not self.is_valid_idx(idx_b):
            return None
        return abs(idx_b - idx_a) / self.sampling_rate

    def slope_between(
            self, idx_a: Optional[int], idx_b: Optional[int]
    ) -> Optional[float]:
        # Amplitude slope between two points
        if not self.is_valid_idx(idx_a) or not self.is_valid_idx(idx_b):
            return None
        if idx_a == idx_b:
            return None
        dy = self.waveform[idx_b] - self.waveform[idx_a]
        dt = (idx_b - idx_a) / self.sampling_rate
        return float(dy / dt)

    def safe_ratio(
            self, numerator: Optional[float], denominator: Optional[float]
    ) -> Optional[float]:
        if numerator is None or denominator is None:
            return None
        if denominator == 0:
            return None
        return float(numerator / denominator)


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
