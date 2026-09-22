from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List, Iterator, Callable
import numpy as np

from .value_objects import LeadName, SamplingRate

@dataclass
class ECGRecord:
    """
    Entity the rest of the pipeline uses.

    - Lead names are defined and canonical.
    - All lead arrays are defined as the same length.
    - Sampling rate is valid and attached to the data.
    - Leads can be accessed via enum.
    """
    record_id: str
    leads: Dict[LeadName, np.ndarray]
    sampling_rate: SamplingRate
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.leads:
            raise ValueError("PreECGRecord must contain at least one lead")

        # Ensure all leads have the same number of samples
        lengths = {name: len(arr) for name, arr in self.leads.items()}
        unique_lengths = set(lengths.values())
        if len(unique_lengths) != 1:
            raise ValueError(
                f"All leads must have the same number of samples. "
                f"Got: {lengths}"
            )

    # Core access

    @property
    def n_samples(self) -> int:
        first_lead = next(iter(self.leads.values()))
        return len(first_lead)

    @property
    def duration_seconds(self) -> float:
        return self.sampling_rate.samples_to_seconds(self.n_samples)

    @property
    def n_leads(self) -> int:
        return len(self.leads)

    def get_lead(self, lead: LeadName) -> np.ndarray:
        # Get a single lead's data, raising KeyError is the lead doesn't exist.
        if lead not in self.leads:
            available = [l.value for l in self.leads.keys()]
            raise KeyError(
                f"Lead {lead.value} not found. "
                f"Available leads: {available}"
            )

        return self.leads[lead]

    def has_lead(self, lead: LeadName) -> bool:
        return lead in self.leads

    # Bulk access when arrays are required

    def to_array(self, lead_order: Optional[List[LeadName]] = None) -> np.ndarray:
        """
        Convert to a 2D numpy array (n_leads, n_samples).

        Useful for downstream ML contexts which may require a 2D matrix,
        with guaranteed lead order.
        """
        if lead_order is None:
            lead_order = self.lead_names

        return np.stack([self.leads[lead] for lead in lead_order])

    # Convenience checks

    @property
    def is_12_lead(self) -> bool:
        return self.n_leads == 12

    @property
    def has_limb_leads(self) -> bool:
        return all(self.has_lead(l) for l in LeadName.limb_leads())

    @property
    def has_precordial_leads(self) -> bool:
        return all(self.had_lead(l) for l in LeadName.precordial_leads())

    def __repr__(self) -> str:
        leads_str = ", ".join(l.value for l in self.lead_names)
        return (
            f"PreECGRecord(id={self.record_id!r}, "
            f"leads=[{leads_str}], "
            f"fs={self.sampling_rate.hz}Hz, "
            f"duration={self.duration_seconds:.1f}s)"
        )

@dataclass
class ECGDataset:
    """
    Collection of ECGRecord objects moving through the pipeline as a group.
    """

    dataset_id: str
    _records: Dict[str, ECGRecord] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # If records were pass in, validate them
        if self._records:
            self._validate_consistency()

    # Adding records
    
    def add(self, record: ECGRecord) -> None:
        if record.record_id in self._records:
            raise ValueError(
                f"Record '{record.record_id}' already exists in dataset."
            )
        self._records[record.record_id] = record

    def add_many(self, records: list[ECGRecord]) -> None:
        for record in records:
            self.add(record)

    # Access

    def get(self, record_id: str) -> ECGRecord:
        if record_id not in self._records:
            available = list(self._records.keys())[:5]
            raise KeyError(
                f"Record '{record_id}' not found. "
                f"First few IDs: {available}"
            )
        return self._records[record_id]

    @property
    def record_ids(self) -> list[str]:
        return list(self._records.keys())

    @property
    def records(self) -> list[ECGRecord]:
        return list(self._records.values())

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[ECGRecord]:
        return iter(self._records.values())

    def __contains__(self, record_id: str) -> bool:
        return record_id in self._records

    def __getitem__(self, record_id: str) -> ECGRecord:
        return self.geet(record_id)

    # Filtering

    def filter(
            self, predicate: Callable[[ECGRecord], bool], dataset_id: str = None
    ) -> "ECGDataset":
        """
        Return a new dataset with only records matching the predicate.

        Usage:
            good_quality = dataset.filter(lambda r: r.duration_seconds >= 10)
            twelve_lead_only = dataset.filter(lambda r: r.is_12_lead)
        """
        filtered = ECGDataset(
            dataset_id=dataset_id or f"{self.dataset_id}_filtered"
        )
        for record in self:
            if predicate(record):
                filtered.add(record)
        return filtered

    # Consistency checks

    def _validate_consistency(self) -> None:
        if len(self._records) < 2:
            return

        self._check_same_leads()

    @property
    def sampling_rates(self) -> set[float]:
        return {r.sampling_rate.hz for r in self}

    @property
    def is_uniform_sampling_rate(self) -> bool:
        return len(self.sampling_rates) == 1

    @property
    def common_leads(self) -> set[LeadName]:
        # leads present in all records in the ECGDataset
        if not self._records:
            return set()
        lead_sets = [set(r.lead_names) for r in self]
        return set.intersection(*lead_sets)

    # Summary

    def summary(self) -> dict[str, Any]:
        if not self._records:
            return {"dataset_id": self.dataset_id, "n_records": 0}

        durations = [r.duration_seconds for r in self]
        return {
            "dataset_id": self.dataset_id,
            "n_records": len(self),
            "sampling_rates": self.sampling_rates,
            "common_leads": [l.value for l in self.common_leads],
            "duration_range_s": (min(durations), max(durations)),
            "total_duration_min": sum(durations) / 60.0
        }

    def __repr(self) -> str:
        return (
            f"ECGDataset(id={self.dataset_id!r}, "
            f"n_records={len(self)}, "
            f"sampling_rates={self.sampling_rates})"
        )