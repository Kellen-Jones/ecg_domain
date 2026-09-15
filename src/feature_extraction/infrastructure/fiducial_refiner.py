from typing import Optional, List
import numpy as np
import logging

from ..domain.value_objects import FiducialPoints, BeatAnalysisContext

log = logging.getLogger(__name__)

class FiducialRefiner:
    """
    Takes raw fiducial detections (from any extractor)
    and fills in missing points using derivative-based calculators.

    Produces a refined FiducialPoints with estimation flags
    set for any points which were estimated instead of detected.
    """

    def refine(
            self,
            fiducials: FiducialPoints,
            ctx: BeatAnalysisContext
    ) -> FiducialPoints:
        """
        Run all refinement steps in order.
        
        Each step receives the current fiducials and returns 
        updated fiducials. Immutable — each step creates a 
        new FiducialPoints via with_updates().
        
        Order matters:
          1. P-wave onset/offset (needed by Q-onset detection)
          2. T-wave peak (needed by T-onset/offset)
          3. T-wave onset/offset (needed by J-point, ST features)
          4. Q-onset (needs P-offset)
          5. J-point (needs S-peak)
          6. ST midpoint (needs J-point and T-onset)
        """

        estimated: List[str] = list(fiducials.estimated_points)

        fp = fiducials

        # Section 4: P-wave onset estimation
        fp, est = self._estimate_p_onset(fp, ctx)
        estimated.extend(est)

        # Section 4: P-wave offset estimation
        fp, est = self._estimate_p_offset(fp, ctx)
        estimated.extend(est)

        # Section 6: T-wave peak estimation
        fp, est = self._estimate_t_peak(fp, ctx)
        estimated.extend(est)

        # Section 7: T-wave onset estimation
        fp, est = self._estimate_t_onset(fp, ctx)
        estimated.extend(est)

        # Section 7: T-wave offset estimation
        fp, est = self._estimate_t_offset(fp, ctx)
        estimated.extend(est)

        # Section 5: Q-onset detection
        fp, est = self._detect_q_onset(fp, ctx)
        estimated.extend(est)

        # Section 9: J-point detection
        fp, est = self._detect_j_point(fp, ctx)
        estimated.extend(est)

        # Section 10: ST midpoint calculation
        fp, est = self._compute_st_midpoint(fp, ctx)
        estimated.extend(est)

        # Update the estimated_points tuple
        fp = fp.with_updates(estimated_points=tuple(estimated))

        return fp

    # Section 4: P-wave Onset/Offset Estimation

    def _estimate_p_onset(
        self,
        fp: FiducialPoints,
        ctx: BeatAnalysisContext,
    ) -> tuple[FiducialPoints, List[str]]:
        """
        Estimate P-wave onset using the second derivative of the 
        signal between beat start and P-peak.
        
        The inflection point (max second derivative) before the 
        P-peak indicates where the P-wave begins.
        """
        if fp.p_onset is not None or fp.p_peak is None:
            return fp, []

        p_idx = fp.p_peak
        if p_idx <= 0:
            return fp, []

        try:
            signal_segment = ctx.waveform[0:p_idx]
            time_segment = ctx.time_array[0:p_idx]

            if len(signal_segment) < 2 or len(time_segment) < 2:
                return fp, []

            dp_dt = np.gradient(signal_segment, time_segment)
            d2p_dt2 = np.gradient(dp_dt, time_segment)

            p_onset_idx = int(np.argmax(d2p_dt2))

            return fp.with_updates(p_onset=p_onset_idx), ["p_onset"]

        except Exception as e:
            log.debug(f"P-onset estimation failed: {e}")
            return fp, []

    def _estimate_p_offset(
        self,
        fp: FiducialPoints,
        ctx: BeatAnalysisContext,
    ) -> tuple[FiducialPoints, List[str]]:
        """
        Estimate P-wave offset using the second derivative of the 
        signal in a window after the P-peak.
        """
        if fp.p_offset is not None or fp.p_peak is None:
            return fp, []

        p_idx = fp.p_peak
        end_idx = min(p_idx + 20, ctx.n_samples)

        if end_idx <= p_idx:
            return fp, []

        try:
            signal_segment = ctx.waveform[p_idx:end_idx]
            time_segment = ctx.time_array[p_idx:end_idx]

            if len(signal_segment) < 2 or len(time_segment) < 2:
                return fp, []

            dp_dt = np.gradient(signal_segment, time_segment)
            d2p_dt2 = np.gradient(dp_dt, time_segment)

            p_offset_idx = p_idx + int(np.argmax(d2p_dt2))

            return fp.with_updates(p_offset=p_offset_idx), ["p_offset"]

        except Exception as e:
            log.debug(f"P-offset estimation failed: {e}")
            return fp, []

    # Section 5: Q-wave onset detection

    def _detect_q_onset(
        self,
        fp: FiducialPoints,
        ctx: BeatAnalysisContext,
    ) -> tuple[FiducialPoints, List[str]]:
        """
        Detect Q-wave onset using the second derivative between 
        P-offset and Q-peak.
        
        The minimum second derivative in this region indicates 
        where the Q-wave begins departing from baseline.
        """
        if fp.q_onset is not None:
            return fp, []

        if fp.q_peak is None or fp.p_offset is None:
            return fp, []

        p_offset_idx = fp.p_offset
        q_idx = fp.q_peak
        start = p_offset_idx + 2

        if start < 0 or q_idx <= start:
            return fp, []

        try:
            q_onset_region = ctx.second_derivative[start:q_idx]

            if len(q_onset_region) == 0:
                return fp, []

            q_onset_idx = start + int(np.argmin(q_onset_region))

            return fp.with_updates(q_onset=q_onset_idx), ["q_onset"]

        except Exception as e:
            log.debug(f"Q-onset detection failed: {e}")
            return fp, []

    # Section 6: T-Wave Point estimations

    def _estimate_t_peak(
        self,
        fp: FiducialPoints,
        ctx: BeatAnalysisContext,
    ) -> tuple[FiducialPoints, List[str]]:
        """
        Estimate T-wave peak as the maximum absolute amplitude 
        after the S-peak.
        
        Uses absolute value because T-waves can be inverted.
        """
        if fp.t_peak is not None or fp.s_peak is None:
            return fp, []

        s_idx = fp.s_peak
        if not ctx.is_valid_idx(s_idx):
            return fp, []

        try:
            window = ctx.waveform[s_idx:]

            if len(window) == 0:
                return fp, []

            t_relative_idx = int(np.argmax(np.abs(window)))
            t_peak_idx = s_idx + t_relative_idx

            return fp.with_updates(t_peak=t_peak_idx), ["t_peak"]

        except Exception as e:
            log.debug(f"T-peak estimation failed: {e}")
            return fp, []

    def _estimate_t_onset(
        self,
        fp: FiducialPoints,
        ctx: BeatAnalysisContext,
    ) -> tuple[FiducialPoints, List[str]]:
        """
        Estimate T-wave onset using the second derivative between 
        S-peak and T-peak.
        """
        if fp.t_onset is not None:
            return fp, []

        if fp.s_peak is None or fp.t_peak is None:
            return fp, []

        s_idx = fp.s_peak
        t_idx = fp.t_peak

        if s_idx < 0 or t_idx <= s_idx:
            return fp, []

        try:
            time_segment = ctx.time_array[s_idx:t_idx]
            signal_segment = ctx.waveform[s_idx:t_idx]

            if len(time_segment) < 2:
                return fp, []

            dst_dt = np.gradient(signal_segment, time_segment)
            d2st_dt2 = np.gradient(dst_dt, time_segment)

            t_onset_idx = s_idx + int(np.argmax(d2st_dt2))

            return fp.with_updates(t_onset=t_onset_idx), ["t_onset"]

        except Exception as e:
            log.debug(f"T-onset estimation failed: {e}")
            return fp, []

    def _estimate_t_offset(
        self,
        fp: FiducialPoints,
        ctx: BeatAnalysisContext,
    ) -> tuple[FiducialPoints, List[str]]:
        """
        Estimate T-wave offset using the second derivative 
        in a window after the T-peak.
        """
        if fp.t_offset is not None or fp.t_peak is None:
            return fp, []

        t_idx = fp.t_peak
        end_idx = min(t_idx + 40, ctx.n_samples)

        if end_idx <= t_idx:
            return fp, []

        try:
            time_segment = ctx.time_array[t_idx:end_idx]
            signal_segment = ctx.waveform[t_idx:end_idx]

            if len(time_segment) < 2:
                return fp, []

            dt_dt = np.gradient(signal_segment, time_segment)
            d2t_dt2 = np.gradient(dt_dt, time_segment)

            t_offset_idx = t_idx + int(np.argmax(d2t_dt2))

            return fp.with_updates(t_offset=t_offset_idx), ["t_offset"]

        except Exception as e:
            log.debug(f"T-offset estimation failed: {e}")
            return fp, []

    # Section 7: J-point Detection

    def _detect_j_point(
        self,
        fp: FiducialPoints,
        ctx: BeatAnalysisContext,
    ) -> tuple[FiducialPoints, List[str]]:
        """
        Detect J-point as the first zero-crossing of the first 
        derivative after the S-peak.
        
        Falls back to a fixed offset from R-peak (80ms) if 
        S-peak is not available.
        """
        if fp.j_point is not None:
            return fp, []

        s_idx = fp.s_peak
        r_idx = fp.r_peak

        j_idx = None
        estimated_tag = "j_point"

        # Primary: search after S-peak
        if ctx.is_valid_idx(s_idx):
            search_start = s_idx + 5
            search_end = min(s_idx + 30, ctx.n_samples)

            if search_end > search_start:
                try:
                    deriv_segment = ctx.first_derivative[search_start:search_end]
                    if len(deriv_segment) > 0:
                        j_idx = search_start + int(
                            np.argmin(np.abs(deriv_segment))
                        )
                except Exception as e:
                    log.debug(f"J-point primary detection failed: {e}")

        # Fallback: fixed offset from R-peak
        if j_idx is None and ctx.is_valid_idx(r_idx):
            offset_samples = int(0.08 * ctx.sampling_rate)
            j_idx = min(r_idx + offset_samples, ctx.n_samples - 1)
            estimated_tag = "j_point_fallback"

        if j_idx is not None:
            return fp.with_updates(j_point=j_idx), [estimated_tag]

        return fp, []

    def _compute_st_midpoint(
        self,
        fp: FiducialPoints,
        ctx: BeatAnalysisContext,
    ) -> tuple[FiducialPoints, List[str]]:
        """
        Compute ST segment midpoint.
        
        Primary: midpoint between S-peak and T-onset (or J-point and T-onset).
        Fallback: fixed offset from R-peak (60ms).
        """
        if fp.st_midpoint is not None:
            return fp, []

        st_mid = None
        tag = "st_midpoint"

        # Primary: midpoint of ST segment
        st_start = fp.j_point or fp.s_peak
        t_onset = fp.t_onset

        if ctx.is_valid_idx(st_start) and ctx.is_valid_idx(t_onset):
            st_mid = (st_start + t_onset) // 2

        # Fallback: offset from R-peak
        if st_mid is None and ctx.is_valid_idx(fp.r_peak):
            offset_samples = int(0.06 * ctx.sampling_rate)
            st_mid = min(fp.r_peak + offset_samples, ctx.n_samples - 1)
            tag = "st_midpoint_fallback"

        if st_mid is not None:
            return fp.with_updates(st_midpoint=st_mid), [tag]

        return fp, []