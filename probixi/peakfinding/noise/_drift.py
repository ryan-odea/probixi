from dataclasses import dataclass, field


@dataclass
class DriftDiagnostics:
    # Time series of noise-model drift per update: mean shift, log var-ratio,
    # per-pixel Gaussian KL, effective sample size, and masked-pixel count.

    step: list[int] = field(default_factory=list)
    mean_shift: list[float] = field(default_factory=list)
    var_ratio_log: list[float] = field(default_factory=list)
    kl_gaussian: list[float] = field(default_factory=list)
    effective_n: list[float] = field(default_factory=list)
    n_masked: list[int] = field(default_factory=list)
