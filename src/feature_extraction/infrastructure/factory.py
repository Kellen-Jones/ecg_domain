"""
Factory which assembes the full feature extraction pipeline.

All concrete implementations are chosen and wired together.
Everything else works with abstractions.
"""

from typing import Optional, List

from preprocessing import LeadName

from ..domain.services import (
    FeatureExtractionService,
    BeatLevelComputer,
    MultibeatComputer,
    MultiLeadComputer,
)
from .extractors.neurokit_extractor import (
    NeurokitFiducialExtractor,
    NeurokitRPeakDetector,
    NeurokitWaveDelineator,
    NeurokitBeatSegmenter,
    DefaultRPeakReconciler,
)
from .fiducial_refiner import FiducialRefiner
from .features.single_beat import (
    TimingComputer,
    SlopeComputer,
    AreaComputer,
    STMorphologyComputer,
    QRSMorphologyComputer,
    AmplitudeComputer,
    BeatNormalizedComputer,
    EstimationFlagComputer,
)
from .features.multi_beat import (
    HRVComputer,
    BeatToBeatVariabilityComputer,
    QTcComputer,
    RRDynamicsComputer,
)
from .features.multi_lead import (
    QRSAxisComputer,
    LeadConcordanceComputer,
    SpatialComplexityComputer,
)

def create_default_pipeline(
        peak_method: str = "neurokit",
        delineate_method: str = "dwt",
        reference_lead: LeadName = LeadName.II,
        reconciler_tolerance_ms: float = 0.5,
        min_fiducial_completeness: float = 0.5,
        include_estimation_flags: bool = True,
        include_rr_dynamics: bool = True,
        include_spatial_complexity: bool = True,
        extra_beat_computers: Optional[List[BeatLevelComputer]] = None,
        extra_multi_beat_computers: Optional[List[MultibeatComputer]] = None,
        extra_multi_lead_computers: Optional[List[MultiLeadComputer]] = None
) -> FeatureExtractionService:
    """
    Creates a fully configured feature extraction pipeline.

    Usage:
        pipeline = create_default_pipeline()
        result = pipeline.process_dataset(ecg_dataset)

    Customize:
        pipeline = create_default_pipeline(
        peak_method="hamilton",
        min_fiducial_completeness = 0.4,
        include_spatial_complexity = False,
        extra_beat_computers=[CustomComputer()]
        )

    """

    # Fiducial extractor
    extractor = NeurokitFiducialExtractor(peak_detector=NeurokitRPeakDetector(method=peak_method),
            reconciler=DefaultRPeakReconciler(
                tolerance_ms=reconciler_tolerance_ms,
                preferred_reference=reference_lead
            ),
            delineator=NeurokitWaveDelineator(method=delineate_method),
            segmenter=NeurokitBeatSegmenter()
            )
    

    # Fiducial refiner
    refiner = FiducialRefiner()

    # Beat-leavel computers
    beat_computers: List[BeatLevelComputer] = [
        TimingComputer(),
        SlopeComputer(),
        AreaComputer(),
        STMorphologyComputer(),
        QRSMorphologyComputer(),
        AmplitudeComputer(),
        BeatNormalizedComputer()
    ]

    if include_estimation_flags:
        beat_computers.append(EstimationFlagComputer())

    if extra_beat_computers:
        beat_computers.extend(extra_beat_computers)

    # Multi-beat computers
    multi_beat_computers: List[MultibeatComputer] = [
        HRVComputer(),
        BeatToBeatVariabilityComputer(),
        QTcComputer()
    ]

    if include_rr_dynamics:
        multi_beat_computers.append(RRDynamicsComputer())

    if extra_multi_beat_computers:
        multi_beat_computers.extend(extra_multi_beat_computers)

    # Multi-lead computers
    multi_lead_computers: List[MultiLeadComputer] = [
        QRSAxisComputer(),
        LeadConcordanceComputer()
    ]

    if include_spatial_complexity:
        multi_lead_computers.append(SpatialComplexityComputer())

    if extra_multi_lead_computers:
        multi_lead_computers.extend(extra_multi_lead_computers)

    # Assemble full pipeline
    return FeatureExtractionService(
        extractor=extractor,
        refiner=refiner,
        beat_computers=beat_computers,
        multi_beat_computers=multi_beat_computers,
        multi_lead_computers=multi_lead_computers,
        min_fiducial_completeness=min_fiducial_completeness
    )

def create_minimal_pipeline(
        peak_method: str = "neurokit",
        reference_lead: LeadName = LeadName.II
) -> FeatureExtractionService:
    """
    Minimal pipeline with only core features.

    Useful for quick iteration, debugging, or when 
    just intervals and amplitudes are needed without the
    full feature set
    """
    return create_default_pipeline(
        peak_method=peak_method,
        reference_lead=reference_lead,
        include_estimation_flags=False,
        include_rr_dynamics=False,
        include_spatial_complexity=False
    )

def create_nn_pipeline(
        peak_method: str = "neurokit",
        reference_lead: LeadName = LeadName.II,
        min_fiducial_completeness: float = 0.3
) -> FeatureExtractionService:
    """
    Pipeline configured for neural network use cases.
    """
    return create_default_pipeline(
        peak_method=peak_method,
        reference_lead=reference_lead,
        min_fiducial_completeness=min_fiducial_completeness,
        include_estimation_flags=True,
        include_rr_dynamics=True,
        include_spatial_complexity=True
    )

def create_custom_pipeline(
        beat_computers: List[BeatLevelComputer],
        multi_beat_computers: Optional[List[MultibeatComputer]] = None,
        multi_lead_computers: Optional[List[MultiLeadComputer]] = None,
        peak_method: str = "neurokit",
        delineate_method: str = "dwt",
        reference_lead: LeadName = LeadName.II,
        reconciler_tolerance_ms: float = 50.0,
        min_fiducial_completeness: float = 0.5
) -> FeatureExtractionService:
    """
    Pipeline to specify exactly which computers to use, without defaults.
    """
    extractor = NeurokitFiducialExtractor(
        peak_detector=NeurokitRPeakDetector(method=peak_method),
        reconciler=DefaultRPeakReconciler(
            tolerance_ms=reconciler_tolerance_ms,
            preferred_reference=reference_lead,
        ),
        delineator=NeurokitWaveDelineator(method=delineate_method),
        segmenter=NeurokitBeatSegmenter(),
    )

    refiner = FiducialRefiner()

    return FeatureExtractionService(
        extractor=extractor,
        refiner=refiner,
        beat_computers=beat_computers,
        multi_beat_computers=multi_beat_computers or [],
        multi_lead_computers=multi_lead_computers or [],
        min_fiducial_completeness=min_fiducial_completeness,
    )