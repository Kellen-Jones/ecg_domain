from dataclasses import dataclass
from enum import Enum
from typing import Tuple

class LeadName(Enum):
    # All leads which may possibly be present
    I = 'I'
    II = 'II'
    III = 'III'
    AVR = 'aVR'
    AVL = 'aVL'
    AVF = 'aVF'
    V1 = 'V1'
    V2 = 'V2'
    V3 = 'V3'
    V4 = 'V4'
    V5 = 'V5'
    V6 = 'V6'

    @classmethod
    def standard_12_lead(cls) -> list ["LeadName"]:
        return [
            cls.I, cls.II, cls.III,
            cls.AVR, cls.AVL, cls.AVF,
            cls.V1, cls.V2, cls.V3,
            cls.V4, cls.V5, cls.V6
        ]

    @classmethod
    def limb_leads(cls) -> list["LeadName"]:
        return [cls.I, cls.II, cls.III, cls.AVR, cls.AVL, cls.AVL]

    @classmethod
    def precordial_leads(cls) -> list["LeadName"]:
        return [cls.V1, cls.V2, cls.V3, cls.V4, cls.V5, cls.V6]

@dataclass(frozen=True)
class SamplingRate:
    """
    Self-validating sampling rate.

    Once created, sampling rate is valid everywhere it is passed.
    """
    hz: float

    def __post_init__(self):
        if self.hz <= 0:
            raise ValueError(f"Sampling rate must be positive, got {self.hz}")
        if self.hz < 50:
            raise ValueError(f"Sampling rate {self.hz}Hz is too low for ECG analysis")

    def samples_to_seconds(self, n_samples: int) -> float:
        return n_samples / self.hz

    def seconds_to_samples(self, seconds: float) -> int:
        return int(round(self.hz * seconds))

    def ms_to_samples(self, ms: float) -> int:
        return int(round(self.hz * ms / 1000.0))