from typing import Dict, List, Optional, Tuple
import numpy as np
import neurokit2 as nk

from preprocessing import ECGRecord, LeadName, SamplingRate

from ...domain.services import (
    FiducialExtractor,
    RPeakDetector,
    RPeakReconciler,
    WaveDelineator,
    BeatSegmenter
)

from ...domain.entities import (
    DelineatedBeat,
    DelineatedRecord
)

from ...domain.value_objects import FiducialPoints, WaveformWindow

class NeurokitRPeakDetector(RPeakDetector):
    # R-Peak detection using Neurokit2
    
    def __init__(self, method: str = "neurokit"):
        self._method = method

    def detect(
            self,
            signal: np.ndarray,
            sampling_rate: SamplingRate,
    ) -> np.ndarray:
        _, info = nk.ecg_peaks(
            signal, sampling_rate = int(sampling_rate.hz), method=self._method
        )
        return info["ECG_R_Peaks"]

class NeurokitWaveDelineator(WaveDelineator):

    def __init__(self, method: str = "cwt"):
        self._method = method

    def delineate(
            self,
            signal: np.ndarray,
            r_peaks: np.ndarray,
            sampling_rate: SamplingRate
    ) -> List[FiducialPoints]:
        _, waves = nk.ecg_delineate(
            signal, rpeaks = r_peaks,
            sampling_rate=int(sampling_rate.hz),
            method=self._method
        )

        n_beats = len(r_peaks)
        fiducials = []

        for i in range(n_beats):
            fp = FiducialPoints(
                r_peak=int(r_peaks[i]),
                p_onset=self._safe_index(waves, "ECG_P_Onsets", i),
                p_peak=self._safe_index(waves, "ECG_P_Peaks", i),
                p_offset=self._safe_index(waves, "ECG_P_Offsets", i),
                qrs_onset=self._safe_index(waves, "ECG_R_Onsets", i),
                qrs_offset=self._safe_index(waves, "ECG_R_Offsets", i),
                q_peak=self._safe_index(waves, "ECG_Q_Peaks", i),
                s_peak=self._safe_index(waves, "ECG_S_Peaks", i),
                t_onset=self._safe_index(waves, "ECG_T_Onsets", i),
                t_peak=self._safe_index(waves, "ECG_T_Peaks", i),
                t_offset=self._safe_index(waves, "ECG_T_Offsets", i),
            )
            fiducials.append(fp)

        return fiducials

    @staticmethod
    def _safe_index(
        waves: dict, key: str, index: int
    ) -> Optional[int]:
        """
        NK2 returns NaN for undetected points.
        FiducialPoints.__post_init__ handles NaN -> None converions,
        this creates additional guards against missing keys or 
        mismatched arrays.
        """
        if key not in waves:
            return None
        arr = waves[key]
        if index >= len(arr):
            return None
        return arr[index]

class NeurokitBeatSegmenter(BeatSegmenter):
    """
    Segments continuous signal into individiual beat waveforms.

    Uses R-R intervals to determine beat boundaries:
    each beat runs from the midpoint of the preceding R-R interval
    to the midpoint of the following R-R interval.

    First and last beats use extrapolation from the nearest R-R interval.
    """

    def segment(
            self,
            signal: np.ndarray,
            r_peaks: np.ndarray,
            sampling_rate: SamplingRate
    ) -> List[WaveformWindow]:
        if len(r_peaks) < 2:
            # Can't determine boundaries with fewer than 2 beats
            return self._single_beat_fallback(signal, r_peaks, sampling_rate)

        boundaries = self._compute_boundaries(r_peaks, len(signal))
        waveforms = []

        for i, (onset, offset) in enumerate(boundaries):
            wf = WaveformWindow(
                samples=signal[onset:offset].copy(),
                onset_sample=onset,
                offset_sample=offset
            )
            waveforms.append(wf)

        return waveforms

    def _computer_boundaries(
            self,
            r_peaks: np.ndarray,
            signal_length: int
    ) -> List[Tuple[int, int]]:
        # Compute beat onset/offset as midpoints between consecutive R-peaks
        boundaries = []

        for i in range(len(r_peaks)):
            # Onset: midpoint to previous R-peak (or signal start)
            if i == 0:
                if len(r_peaks) > 1:
                    rr_interval = r_peaks[1] - r_peaks[0]
                    onset = max(0, r_peaks[0] - rr_interval // 2)
                else:
                    onset = 0

                # Offset: midpoint to next R-peak (or signal end)
                if i == len(r_peaks) - 1:
                    if len(r_peaks) > 1:
                        rr_interval = r_peaks[-1] - r_peaks[-2]
                        offset = min(signal_length, r_peaks[i] + rr_interval // 2)
                    else:
                        offset = signal_length
                else:
                    offset = (r_peaks[i] + r_peaks[i + 1]) // 2

                boundaries.append((onset, offset))

            return boundaries

        def _single_beat_fallback(
                self,
                signal: np.ndarray,
                r_peaks: np.ndarray,
                sampling_rate: SamplingRate
        ) -> List[WaveformWindow]:
            """
            Fallback for signals with 0 or 1 detected beats.
            Uses a fixed window around the R-peak.
            """
            if len(r_peaks) == 0:
                return []

            # Use a 1-second window centered on the R-peak
            half_window = sampling_rate.min_samples(500)
            r = r_peaks[0]
            onset = max(0, r - half_window)
            offset = min(len(signal), r + half_window)

            return [WaveformWindow(
                samples=signal[onset:offset].copy(),
                onset_sample=onset,
                offset_sample=offset
            )]

class DefaultRPeakReconciler(RPeakReconciler):
    """
    Reconciles R-peaks detected independently on each lead 
    into a single consensus series.
    
    Strategy:
      1. Use the reference lead's detections as the anchor
      2. For each anchor peak, find the closest peak on 
         every other lead within a tolerance window
      3. Keep anchor peaks where a majority of leads agree
      4. Refine position by taking the median across leads
    """

    def __init__(
            self,
            tolerance_ms: float = 50.0,
            agreement_ratio: float = 0.5,
            preferred_reference: Optional[LeadName] = None
    ):
        self._tolerance_ms = tolerance_ms
        self._agreement_ratio = agreement_ratio
        self._preferred_reference = preferred_reference

    def reconcile(
            self,
            per_lead_peaks: Dict[LeadName, np.ndarray],
            sampling_rate: SamplingRate,
            reference_lead: Optional[LeadName] = None
    ) -> Tuple[np.ndarray, LeadName]:
        if not per_lead_peaks:
            raise ValueError("No per-lead peaks provided")

        # Pick reference lead
        ref_lead = self._pick_reference(per_lead_peaks, reference_lead)
        anchor_peaks = per_lead_peaks[ref_lead]
        other_leads = {
            k: v for k, v in per_lead_peaks.items() if k != ref_lead
        }

        if not other_leads:
            return anchor_peaks, ref_lead

        tolerance_samples = sampling_rate.ms_to_samples(self._tolerance_ms)
        min_agreeing = max(
            1, int(len(per_lead_peaks) * self._agreement_ratio)
        )
        consensus = []

        for anchor in anchor_peaks:
            # Collect the closest peak on each lead within tolerance
            nearby_peaks = [anchor] # Include the anchor itself

            for lead, peaks in other_leads.items():
                if len(peaks) == 0:
                    continue
                distances = np.abs(peaks - anchor)
                closest_idx = np.argmin(distances)
                if distances[closest_idx] <= tolerance_samples:
                    nearby_peaks.append(peaks[closest_idx])

                # Keep if enough leads agree
                if len(nearby_peaks) >= min_agreeing:
                    # Refine position is the median
                    consensus.append(int(np.median(nearby_peaks)))

        return np.array(consensus, dtype=int), ref_lead

    def _pick_reference(
            self,
            per_lead_peaks: Dict[LeadName, np.ndarray],
            explicit_reference: Optional[LeadName]
    ) -> LeadName:
        """
        Pick the best reference lead.
        
        Priority:
          1. Explicitly requested lead
          2. Preferred lead from config (typically Lead II)
          3. Lead with the most detected R-peaks
        """

        if explicit_reference and explicit_reference in per_lead_peaks:
            return explicit_reference

        if (
            self._preferred_reference
            and self._preferred_reference in per_lead_peaks
        ):
            return self._preferred_reference

        # Fall back to lead with most detections
        return max(per_lead_peaks, key=lambda k: len(per_lead_peaks[k]))

class NeurokitFiducialExtractor(FiducialExtractor):
    """
    Complete fiducial extraction pipeline using NeuroKit2 components.
    
    Composes the individual pieces:
      RPeakDetector → RPeakReconciler → WaveDelineator → BeatSegmenter
    
    This is the class you instantiate and inject into 
    FeatureExtractionService.
    """

    def __init__(
        self,
        peak_detector: Optional[NeurokitRPeakDetector] = None,
        reconciler: Optional[DefaultRPeakReconciler] = None,
        delineator: Optional[NeurokitWaveDelineator] = None,
        segmenter: Optional[NeurokitBeatSegmenter] = None,
    ):
        self._detector = peak_detector or NeurokitRPeakDetector()
        self._reconciler = reconciler or DefaultRPeakReconciler(
            preferred_reference=LeadName.II
        )
        self._delineator = delineator or NeurokitWaveDelineator()
        self._segmenter = segmenter or NeurokitBeatSegmenter()

    def extract(self, record: ECGRecord) -> DelineatedRecord:
        # Step 1: Detect R-peaks on every lead independently
        per_lead_peaks = {}
        for lead in record.lead_names:
            signal = record.get_lead(lead)
            try:
                peaks = self._detector.detect(signal, record.sampling_rate)
                if len(peaks) > 0:
                    per_lead_peaks[lead] = peaks
            except Exception as e:
                print(
                    f"WARNING: R-peak detection failed on "
                    f"{record.record_id}/{lead.value}: {e}"
                )
                continue

        if not per_lead_peaks:
            raise ValueError(
                f"R-peak detection failed on ALL leads "
                f"for record {record.record_id}"
            )

        # Step 2: reconcile across leads
        consensus_peaks, ref_lead = self._reconciler.reconcile(
            per_lead_peaks, record.sampling_rate
        )

        # Step 3: delineate and segment each lead
        beats_by_lead: Dict[LeadName, List[DelineatedBeat]] = {}

        for lead in record.lead_names:
            signal = record.get_lead(lead)

            try:
                fiducials_list = self._delineator.delineate(
                    signal, consensus_peaks, record.sampling_rate
                )

                # Segment waveform into beats
                waveforms = self._segmenter.segment(
                    signal, consensus_peaks, record.sampling_rate
                )

                # Pair fiducials with waveforms into DelineatedBeats
                lead_beats = self._assemble_beats(
                    lead=lead,
                    fiducials_list=fiducials_list,
                    waveforms=waveforms,
                    consensus_peaks=consensus_peaks
                )

                beats_by_lead[lead] = lead_beats

            except Exception as e:
                print(
                    f"WARNING: Delineation failed on "
                    f"{record.record_id}/{lead.value}: {e}"
                )
                continue

        if not beats_by_lead:
            raise ValueError(
                f"Delineation failed on ALL leads "
                f"for record {record.record_id}"
            )

        return DelineatedRecord(
            record_id=record.record_id,
            sampling_rate=record.sampling_rate,
            reference_lead=ref_lead,
            consensus_r_peaks=consensus_peaks,
            beats=beats_by_lead,
            metadata={
                "n_leads_with_peaks": len(per_lead_peaks),
                "n_leads_delineated": len(beats_by_lead),
                "detector_method": self._detector._method,
                "delineator_method": self._delineator._method,
            },
        )

    def _assemble_beats(
            self,
            lead: LeadName,
            fiducials_list: List[FiducialPoints],
            waveforms: List[WaveformWindow],
            consensus_peaks: np.ndarray
    ) -> List[DelineatedBeat]:
        """
        Pair up fiducials and waveforms into DelineatedBeat objects.
        
        Handles mismatches between the number of fiducials and 
        waveforms (which can happen at signal boundaries).
        """
        n_beats = min(len(fiducials_list), len(waveforms), len(consensus_peaks))

        beats = []
        for i in range(n_beats):
            # Convert absolute fiducial positions ot beat-relative
            relative_fiducials = self._to_relative_fiducials(
                fiducials_list[i],
                waveforms[i].onset_sample
            )

            beat = DelineatedBeat(
                beat_id = i,
                lead=lead,
                fiducials=relative_fiducials,
                waveform=waveforms[i]
            )
            beats.append(beat)

        return beats

    @staticmethod
    def _to_relative_fiducials(
        absolute: FiducialPoints,
        beat_onset: int
    ) -> FiducialPoints:
        """
        Convert absolute sample positions to positions relative 
        to the beat onset.
        
        If a beat starts at sample 1000 and the P-onset is at 
        sample 1020, the relative P-onset is 20.
        """
        def _relative(val: Optional[int]) -> Optional[int]:
            if val is None:
                return None
            result = val - beat_onset
            # Clamp to non-negative (fiducial before beat window
            # can happen at boundaries)
            return max(0, result)

        return FiducialPoints(
            r_peak=_relative(absolute.r_peak),
            p_onset=_relative(absolute.p_onset),
            p_peak=_relative(absolute.p_peak),
            p_offset=_relative(absolute.p_offset),
            qrs_onset=_relative(absolute.qrs_onset),
            qrs_offset=_relative(absolute.qrs_offset),
            q_peak=_relative(absolute.q_peak),
            s_peak=_relative(absolute.s_peak),
            t_onset=_relative(absolute.t_onset),
            t_peak=_relative(absolute.t_peak),
            t_offset=_relative(absolute.t_offset),
        )