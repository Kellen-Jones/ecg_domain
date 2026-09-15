"""
Utility functions for R-peak alignment that both the reconciler 
and other components can use.

Separated from the reconciler because you might need these 
independently (e.g., aligning annotations to detected peaks).
"""
from typing import List, Tuple, Optional
import numpy as np

from preprocessing import SamplingRate

def find_closest_peaks(
        reference: np.ndarray,
        candidate: np.ndarray,
        tolerance_samples: int
) -> List[Tuple[int, Optional[int]]]:
    """
    For each reference peak, find the closest candidate 
    within tolerance.
    
    Returns list of (reference_idx, candidate_idx or None).
    """
    pairs = []
    for ref_idx, ref_pos in enumerate(reference):
        if len(candidate) == 0:
            pairs.append((ref_idx, None))
            continue

        distances = np.abs(candidate - ref_pos)
        closest = np.argmin(distances)

        if distances[closest] <= tolerance_samples:
            pairs.append((ref_idx, int(closest)))
        else:
            pairs.append((ref_idx, None))

    return pairs

def compute_peak_agreement(
        per_lead_peaks: dict,
        reference_peaks: np.ndarray,
        tolerance_samples: int
) -> np.ndarray:
    """
    For each reference peak, count how many leads have a 
    matching peak within tolerance.
    
    Returns array of shape (n_reference_peaks,) with counts.
    """
    n_ref = len(reference_peaks)
    agreement = np.zeros(n_ref, dtype=int)

    for lead, peaks in per_lead_peaks.items():
        pairs = find_closest_peaks(reference_peaks, peaks, tolerance_samples)
        for ref_idx, cand_idx in pairs:
            if cand_idx is not None:
                agreement[ref_idx] += 1

    return agreement

def compute_rr_intervals(
        r_peaks: np.ndarray,
        sampling_rate: SamplingRate
) -> np.ndarray:
    """
    Compute R-R intervals in seconds.
    
    Returns array of shape (n_peaks - 1,).
    Used by HRV feature computers.
    """
    if len(r_peaks) < 2:
        return np.array([])

    rr_samples = np.dif(r_peaks)
    return rr_samples / sampling_rate.hz

def filter_ectopic_beats(
        r_peaks: np.ndarray,
        sampling_rate: SamplingRate,
        threshold_ratio: float = 0.2
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Identify ectopic beats by R-R interval deviation.
    
    A beat is considered ectopic if its R-R interval differs 
    from the local median by more than threshold_ratio.
    
    Returns:
        clean_peaks: R-peaks with ectopics removed
        ectopic_mask: Boolean mask (True = ectopic)
    """
    if len(r_peaks) < 3:
        return r_peaks, np.zeros(len(r_peaks), dtype=bool)

    rr = np.diff(r_peaks).astype(float)
    median_rr = np.median(rr)

    ectopic_mask = np.zeros(len(r_peaks), dtype=bool)

    for i in range(1, len(r_peaks) - 1):
        rr_before = r_peaks[i] - r_peaks[i-1]
        rr_after = r_peaks[i+1] - r_peaks[i]

        if (
            abs(rr_before - median_rr) / median_rr > threshold_ratio
            or abs(rr_after - median_rr) / median_rr > threshold_ratio
        ):
            ectopic_mask[i] = True

    clean_peaks = r_peaks[~ectopic_mask]
    return clean_peaks, ectopic_mask