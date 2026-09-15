from typing import Dict, List, Optional
import numpy as np

from preprocessing import SamplingRate

from ...domain.entities import DelineatedBeat
from ...domain.services import BeatLevelComputer




class IntervalComputer(BeatLevelComputer):
    """
    Computes time-interval features from fiducial points.
    PR interval, QT interval, QRS duration, etc.
    """

    

    def feature_names(self) -> List[str]:
        return [
            "ST_duration", "T_wave_peak_time", "PQ_duration",
            "QR_duration", "RS_duration", "ST_duration", "PR_duration", "QS_duration",
            "QT_duration", "R_wave_peak_time", "P_duration", "T_duration", "Tpeak_Tend",
            "QRS_endT_duration"
        ]

    def compute(
            self,
            beat: DelineatedBeat,
            sampling_rate: SamplingRate,
    ) -> Dict[str, Optional[float]]:
        fp = beat.fiducials
        fs = sampling_rate.hz

        def to_ms(samples: Optional[int]) -> Optional[float]:
            if samples is None:
                return None
            return (samples / fs) * 1000.0

        pr = to_ms(fp.pr_interval_samples())
        qrs = to_ms(fp.qrs_duration_samples())
        qt = to_ms(fp.qt_interval_samples())

        # P-wave duration
        p_dur = None
        if fp.p_onset is not None and fp.p_offset is not None:
            p_dur = to_ms(fp.p_offset - fp.p_onset)

        # T-wave duration
        t_dur = None
        if fp.p_onset is not None and fp.p_offset is not None:
            p_dur = to_ms(fp.t_offset - fp.t_onset)

        # ST segment: QRS offset to T onset
        st_seg = None
        if fp.qrs_offset is not None and fp.t_onset is not None:
            st_seg = to_ms(fp.t_onset - fp.qrs_offset)

        