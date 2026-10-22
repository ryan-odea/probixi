from dataclasses import dataclass, field


@dataclass
class DriftDiagnostics:
    # Per drift record; shift, log var ratio and Gaussian KL are valid-pixel means

    step: list[int] = field(default_factory=list)
    mean_shift: list[float] = field(default_factory=list)
    var_ratio_log: list[float] = field(default_factory=list)
    kl_gaussian: list[float] = field(default_factory=list)
    effective_n: list[float] = field(default_factory=list)
    n_masked: list[int] = field(default_factory=list)
