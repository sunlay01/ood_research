"""Adaptive SAM under the common CMNIST harness."""

from __future__ import annotations

from .sam import SAMAlgorithm


class ASAMAlgorithm(SAMAlgorithm):
    name = "ASAM"
    formula_id = "CMNIST_ASAM_ADAPTIVE_TWO_STEP_SOURCE_RISK_V1"
    reference_id = "KWON_2021_ADAPTIVE_SHARPNESS_AWARE_MINIMIZATION"
    variant_id = "ASAM_RHO0P5_ETA0P01_COMMON_ADAM"
    admission_role = "FLATNESS_GEOMETRY"

    @property
    def rho(self) -> float:
        return float(self.config["asam"]["rho"])

    @property
    def adaptive(self) -> bool:
        return True

    @property
    def eta(self) -> float:
        return float(self.config["asam"]["eta"])
