"""
Vocabulary of the modeling context.

Immutable, self-validatin building blocks which define
how splitting, training, optimization, and evaluation work.

Key concepts:
    - Subjects are split into train/val/test by subject
    - Beats within a subject are grouped to inflate sample size
    - The genetic algorithm optimizes both hyperparameters and feature selection
    - Fitness can target any classification metrics
    - Classification can be multi-class or binary
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Dict, List, Optional, Tuple, Any,
    FrozenSet, Union, Set
)
import numpy as np

# ENUMS

class ModelType(Enum):
    RANDOM_FOREST = "random_forest"
    NEURAL_NET = "neural_net"
    CNN = "cnn"
    ENSEMBLE = "ensemble"

class ClassificationType(Enum):
    BINARY = "binary"
    MULTICLASS = "multiclass"

    @property
    def is_binary(self) -> bool:
        return self == ClassificationType.BINARY

class MetricName(Enum):
    """
    All metrics the system can compute and optimize for.

    The genetic algorithm's fitness function will target one.
    Evaluation will compute all metrics.
    """

    ACCURACY = "accuracy"
    BALANCED_ACCURACY = "balanced_accuracy"
    PRECISION_MACRO = "precision_macro"
    PRECISION_WEIGHTED = "precision_weighted"
    RECALL_MACRO = "recall_macro"
    RECALL_WEIGHTED = "recall_weighted"
    F1_MACRO = "f1_macro"
    F1_BINARY = "f1_binary"
    AUC_OVR = "auc_ovr"           # one-vs-rest for multiclass
    AUC_OVO = "auc_ovo"           # one-vs-one for multiclass
    AUC_BINARY = "auc_binary"
    LOG_LOSS = "log_loss"
    COHEN_KAPPA = "cohen_kappa"
    MCC = "mcc"       # Matthews correlation coefficient

    @property
    def higher_is_better(self) -> bool:
        return self not in {MetricName.LOG_LOSS}

    @property
    def requires_binary(self) -> bool:
        return self in {MetricName.F1_BINARY, MetricName.AUC_BINARY}

    @property
    def requires_probabilities(self) -> bool:
        return self in {
            MetricName.AUC_OVR, MetricName.AUC_OVO,
            MetricName.AUC_BINARY, MetricName.LOG_LOSS
        }

class SplitName(Enum):
    TRAIN = "train"
    VALIDATION = "validation" or "val"
    TEST = "test"

# SPLITTING

@dataclass(frozen=True)
class SplitRatios:
    """
    Train/validation/test split proportions.

    These are ratios of SUBJECTS, not beats or groups.
    Subject-aware splitting enforced at the splitter level,
    but the ratios are defined here.
    """
    train: float
    validation: float
    test: float

    def __post_init__(self):
        total = self.train + self.validation + self.test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Split ratios must sum to 1.0, got "
                f"{self.train} + {self.validation} + {self.test} = {total}"
            )
        for name, val in [
            ("train", self.train),
            ("validation", self.validation),
            ("test", self.test)
        ]:
            if val < 0:
                raise ValueError(f"{name} ratio cannot be negative: {val}")
            if val > 1:
                raise ValueError(f"{name} ratio cannot exceed 1.0: {val}")

        if self.train < 0.1:
            raise ValueError(
                f"Training ratio {self.train} is suspiciously low"
            )

    @classmethod
    def default(cls) -> "SplitRatios":
        return cls(train=0.7, validation=0.15, test=0.15)

    @classmethod
    def no_validation(cls) -> "SplitRatios":
        return cls(train=0.8, validation=0.0, test=0.2)

    @property
    def has_validation(self) -> bool:
        return self.validation > 0

@dataclass(frozen=True)
class BeatGroupConfig:
    """
    Defines how beats within a subject are grouped to
    inflate sample size without data leakage.

    Since groups are formed within a subject, and 
    subjects are split before grouping, data leakage is avoided.
    """
    group_size: int
    min_group_size: int = 1
    overlap: int = 0
    drop_incomplete: bool = True
    aggregation: str = "median" # how to aggreagte features within a group

    def __post_init__(self):
        if self.group_size < 1:
            raise ValueError(
                f"group_size must be >= 1, got {self.group_size}"
            )
        if self.min_group_size < 1:
            raise ValueError(
                f"min_group_size must be >= 1, got {self.min_group_size}"
            )
        if self.min_group_size > self.group_size:
            raise ValueError(
                f"min_group_size ({self.min_group_size}) cannot exceed "
                f"group_size ({self.group_size})"
            )
        if self.overlap < 0:
            raise ValueError(
                f"overlap must be >= 0, got {self.overlap}"
            )
        if self.overlap >= self.group_size:
            raise ValueError(
                f"overlap ({self.overlap}) must be less than "
                f"group_size ({self.group_size})"
            )
        if self.aggregation not in ("mean", "median", "first", "none"):
            raise ValueError(
                f"aggregation must be one of 'mean', 'median', "
                f"'first', 'none', got '{self.aggregation}'"
            )

    @property
    def step_size(self) -> int:
        # How many beats to advance between groups
        return self.group_size - self.overlap

    def n_groups_for_subject(self, n_beats: int) -> int:
        # How many groups a subject with n_beats wiht produce
        if n_beats < self.min_group_size:
            return 0

        if self.drop_incomplete:
            if n_beats < self.group_size:
                return 0
            return (n_beats - self.group_size) // self.step_size + 1
        else:
            return max(1, (n_beats - self.min_group_size) // self.step_size + 1)

    @classmethod
    def single_beat(cls) -> "BeatGroupConfig":
        # Each beat is its own group (no aggregation)
        return cls(group_size=1, aggregation="none")

    @classmethod
    def default(cls) -> "BeatGroupConfig":
        return cls(group_size=10, min_group_size=5, overlap=0)

@dataclass(frozen=True)
class SplitConfig:
    """
    Complete configuration for how data is split.
    Combines subject-level splitting with beat grouping.
    """
    ratios: SplitRatios
    beat_group_config: BeatGroupConfig
    random_seed: int = 42
    stratify_by_label: bool = True

    @classmethod
    def default(cls) -> "SplitConfig":
        return cls(
            ratios=SplitRatios.default(),
            beat_group_config=BeatGroupConfig.default()
        )

# LABELS

@dataclass(frozen=True)
class ClassLabel:
    """
    A single class label with its encoding.

    Keeps the human-readable name attached to the integer
    so 0, 1, 2, etc. are always verified.
    """
    name: str
    code: str

    def __repr__(self) -> str:
        return f"ClassLabel({self.name}={self.code})"

@dataclass(frozen=True)
class LabelSchema:
    """
    Defines the complete label space for a classification task.

    Immutable mapping between class names and integer codes.
    Every entity which touches labels references this to stay 
    consistent.
    """
    labels: Tuple[ClassLabel, ...]
    classification_type: ClassificationType
    positive_class: Optional[str] = None # for binary, which class is "positive"

    def __post_init__(self):
        if len(self.labels) < 2:
            raise ValueError("Need at least 2 classes")

        # Check for duplicate names
        names = [l.name for l in self.labels]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate class names: {names}")

        # check for duplicate codes
        codes = [l.code for l in self.labels]
        if len(codes) != len(set(codes)):
            raise ValueError(f"Duplicate class codes: {codes}")

        # binary validation
        if self.classification_type == ClassificationType.BINARY:
            if len(self.labels) != 2:
                raise ValueError(
                    f"Binary classification requires exactly 2 classes, "
                    f"got {len(self.labels)}"
                )

            if self.positive_class is not None:
                if self.positive_class not in names:
                    raise ValueError(
                        f"positive_class '{self.positive_class}' "
                        f"not in labels: {names}"
                    )

    @property
    def n_classes(self) -> int:
        return len(self.labels)

    @property
    def class_names(self) -> List[str]:
        return [l.name for l in self.labels]

    @property
    def class_codes(self) -> List[int]:
        return [l.code for l in self.labels]

    def name_to_code(self, name: str) -> int:
        for label in self.labels:
            if label.name == name:
                return label.code
        raise ValueError(f"Unknown class name: '{name}'")

    def code_to_name(self, code: int) -> str:
        for label in self.labels:
            if label.code == code:
                return label.name
                return label.name
        raise ValueError(f"Unknown class code: {code}")

    def to_binary(self, positive_class: str) -> "LabelSchema":
        """
        Reduce a multi-class schema to binary.
        Everything not in positive_class becomes 'other'.
        """
        if positive_class not in self.class_names:
            raise ValueError(
                f"'{positive_class}' not in {self.class_names}"
            )
        return LabelSchema(
            labels=(
                ClassLabel(name=positive_class, code=1),
                ClassLabel(name="other", code=0)
            ),
            classification_type=ClassificationType.BINARY,
            positive_class=positive_class
        )

    @classmethod
    def from_array(
        cls,
        labels: np.ndarray,
        classification_type: ClassificationType = ClassificationType.MULTICLASS
    ) -> "LabelSchema":
        # Infer schema from an array of string labels.
        unique = sorted(set(labels))
        class_labels = tuple(
            ClassLabel(name=str(name), code=i)
            for i, name in enumerate(unique)
        )
        return cls(
            labels=class_labels,
            classification_type=classification_type
        )

# HYPERPARAMETERS
