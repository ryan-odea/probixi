from dataclasses import dataclass, field


@dataclass
class DriftDiagnostics:
    """Time series of how much the noise model moves as it updates.

    Attributes:
        step: Frame count at which each sample was recorded.
        mean_shift: Mean absolute change in the per-pixel mean since last sample.
        var_ratio_log: Mean log-ratio of new to old variance.
        kl_gaussian: Mean per-pixel KL of the new Gaussian from the old.
        effective_n: Effective sample size backing the estimate at that step.
        n_masked: Pixels currently masked out as invalid.
    """

    step: list[int] = field(default_factory=list)
    mean_shift: list[float] = field(default_factory=list)
    var_ratio_log: list[float] = field(default_factory=list)
    kl_gaussian: list[float] = field(default_factory=list)
    effective_n: list[float] = field(default_factory=list)
    n_masked: list[int] = field(default_factory=list)
