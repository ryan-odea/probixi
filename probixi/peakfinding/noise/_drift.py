from dataclasses import dataclass, field

import torch


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

    def latest(self) -> dict:
        if not self.step:
            return {}
        return {
            "step": self.step[-1],
            "mean_shift": self.mean_shift[-1],
            "var_ratio_log": self.var_ratio_log[-1],
            "kl_gaussian": self.kl_gaussian[-1],
            "effective_n": self.effective_n[-1],
            "n_masked": self.n_masked[-1],
        }

    def as_arrays(self) -> dict:
        return {k: torch.tensor(v) for k, v in self.__dict__.items()}
