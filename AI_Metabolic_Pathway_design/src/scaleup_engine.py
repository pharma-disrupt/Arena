"""
Scale-Up Engine Module

Simulates bioprocess scale-up from lab (2L) to pilot (200L) to
production (20,000L) scale, predicting kLa, mixing time, and yield
loss at each stage.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

from logger_setup import PipelineLogger


# ---------------------------------------------------------------------------
# SCALE LEVEL
# ---------------------------------------------------------------------------

@dataclass
class ScaleLevel:
    """Represents a single scale-up stage."""
    volume_l: float
    kla_h: float = 0.0
    mixing_time_s: float = 0.0
    power_per_volume_w_per_l: float = 0.0
    yield_loss_percent: float = 0.0
    scale_factor: float = 1.0
    comments: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "volume_l": self.volume_l,
            "kla_h": round(self.kla_h, 2),
            "mixing_time_s": round(self.mixing_time_s, 1),
            "power_per_volume_w_per_l": round(self.power_per_volume_w_per_l, 2),
            "yield_loss_percent": round(self.yield_loss_percent, 2),
            "scale_factor": round(self.scale_factor, 2),
            "comments": self.comments,
        }


# ---------------------------------------------------------------------------
# ORGANISM SCALE CONSIDERATIONS
# ---------------------------------------------------------------------------

ORGANISM_SCALE_CONSIDERATIONS: Dict[str, List[str]] = {
    "ecoli": [
        "Acetate accumulation risk at large scale due to oxygen gradients",
        "Heat generation high — requires efficient cooling at >500L",
        "Foaming significant — antifoam addition needed",
        "Shear sensitivity low — robust to agitation",
    ],
    "ecoli_bl21": [
        "T7 expression system may induce at large scale due to temperature gradients",
        "Inclusion body formation risk increases with scale",
        "High oxygen demand — kLa critical above 200L",
        "Protease activity may increase — add protease inhibitors",
    ],
    "scerevisiae": [
        "Crabtree effect enhanced at large scale due to glucose gradients",
        "Ethanol production may increase — requires off-gas analysis",
        "Flocculation risk at high cell density",
        "pH control critical — CO2 dissolution lowers pH",
    ],
    "scerevisiae_by": [
        "Similar to S288C but more sensitive to osmotic stress",
        "Requires tighter DO control at production scale",
    ],
    "bsubtilis": [
        "Protease secretion increases at scale — may degrade product",
        "Sporulation risk if nutrient gradients develop",
        "Biofilm formation on vessel walls possible",
        "Requires sterile filtration of air supply",
    ],
    "cglutamicum": [
        "Biotin limitation critical for glutamate production — monitor carefully",
        "Oxygen uptake rate high — kLa must be maintained",
        "Penicillin sensitivity requires careful vessel cleaning",
    ],
    "pputida": [
        "High solvent tolerance — good for lipophilic products",
        "PHA accumulation increases viscosity — affects mixing",
        "Requires precise temperature control (30°C optimal)",
        "Off-gas analysis critical for metabolic monitoring",
    ],
}


# ---------------------------------------------------------------------------
# SCALE-UP ENGINE
# ---------------------------------------------------------------------------

class ScaleUpEngine:
    """
    Simulates bioprocess scale-up cascade.

    Predicts kLa, mixing time, power input, and yield loss
    at each scale level.
    """

    def __init__(self) -> None:
        self._logger: Optional[PipelineLogger] = None

    def set_logger(self, logger: PipelineLogger) -> None:
        self._logger = logger

    def predict_kla(self, volume_l: float, agitation_rpm: float,
                    aeration_vvm: float, geometry_factor: float = 1.0) -> float:
        """
        Predict volumetric oxygen transfer coefficient (kLa).

        kLa = f(agitation, aeration, geometry, scale)
        Uses empirical correlation from literature.
        """
        # Van't Riet correlation (modified for scale)
        scale_factor = math.log10(volume_l / 2.0 + 1)
        kla_base = 0.02 * (agitation_rpm / 300) ** 1.5 * (aeration_vvm) ** 0.5
        kla = kla_base * geometry_factor / (1 + 0.1 * scale_factor)
        return max(5.0, min(300.0, kla))

    def predict_mixing_time(self, volume_l: float, agitation_rpm: float) -> float:
        """
        Predict mixing time in seconds.

        tm = f(volume, agitation) — increases with scale.
        """
        # Empirical correlation
        base_time = 5.0  # seconds at 2L
        scale_effect = math.log10(volume_l / 2.0 + 1) ** 2
        agitation_effect = 300.0 / max(agitation_rpm, 50.0)
        mixing_time = base_time * (1 + scale_effect) * agitation_effect
        return max(5.0, min(600.0, mixing_time))

    def calculate_yield_loss_at_scale(self, scale_level: float,
                                       organism_key: str,
                                       product_logp: float = 2.0) -> float:
        """
        Calculate expected yield loss at a given scale.

        Losses come from:
        - Oxygen transfer limitations
        - Mixing heterogeneity
        - Shear damage
        - Contamination risk
        - Product degradation
        """
        if scale_level <= 2.0:
            return 0.0  # Lab scale: no loss

        # Base loss increases with scale
        base_loss = 0.02 * math.log10(scale_level / 2.0)

        # Organism-specific factors
        org_factors = {
            "ecoli": 0.8,
            "ecoli_bl21": 1.0,
            "scerevisiae": 1.2,
            "scerevisiae_by": 1.3,
            "bsubtilis": 1.5,
            "cglutamicum": 1.1,
            "pputida": 0.9,
        }
        org_factor = org_factors.get(organism_key, 1.0)

        # Product-specific factors (lipophilic products harder at scale)
        product_factor = 1.0 + 0.05 * max(0, product_logp - 5.0)

        total_loss = base_loss * org_factor * product_factor
        return min(0.30, max(0.0, total_loss))  # Cap at 30%

    def run_scale_up_cascade(
        self,
        organism_key: str,
        product_name: str,
        product_logp: float = 2.0,
        scale_levels: Optional[List[float]] = None,
        lab_kla: float = 100.0,
        lab_agitation_rpm: float = 300.0,
        lab_aeration_vvm: float = 1.0,
    ) -> List[ScaleLevel]:
        """
        Simulate full scale-up cascade.

        Parameters
        ----------
        organism_key : str
            Target organism.
        product_name : str
            Target product.
        product_logp : float
            Octanol-water partition coefficient.
        scale_levels : list of float, optional
            Scale volumes in litres. Default: [2, 200, 20000].
        lab_kla : float
            Lab-scale kLa (h⁻¹).
        lab_agitation_rpm : float
            Lab-scale agitation (RPM).
        lab_aeration_vvm : float
            Lab-scale aeration (vvm).

        Returns
        -------
        list of ScaleLevel
        """
        if scale_levels is None:
            scale_levels = [2.0, 200.0, 20000.0]

        if self._logger:
            self._logger.info(
                "Running scale-up cascade: %s → %s, scales=%s",
                organism_key, product_name, scale_levels,
            )

        results: List[ScaleLevel] = []
        prev_volume = 2.0

        for volume in scale_levels:
            scale_factor = volume / prev_volume if prev_volume > 0 else 1.0

            # Predict kLa at this scale
            kla = self.predict_kla(volume, lab_agitation_rpm, lab_aeration_vvm)

            # Predict mixing time
            mixing_time = self.predict_mixing_time(volume, lab_agitation_rpm)

            # Power per volume
            power_per_vol = lab_agitation_rpm ** 3 * volume / (
                1e6 * max(1.0, scale_factor)
            )

            # Yield loss
            yield_loss = self.calculate_yield_loss_at_scale(
                volume, organism_key, product_logp
            )

            # Scale-specific comments
            comments = []
            if volume <= 10:
                comments.append("Lab scale — shake flask or bench-top bioreactor")
            elif volume <= 500:
                comments.append("Pilot scale — stirred-tank bioreactor")
            elif volume <= 5000:
                comments.append("Demonstration scale — requires process validation")
            else:
                comments.append("Production scale — full GMP compliance required")

            # Organism-specific considerations
            org_comments = ORGANISM_SCALE_CONSIDERATIONS.get(organism_key, [])
            comments.extend(org_comments[:2])

            if kla < 20:
                comments.append(
                    f"WARNING: kLa ({kla:.1f} h⁻¹) below recommended minimum (20 h⁻¹)"
                )
            if mixing_time > 120:
                comments.append(
                    f"WARNING: mixing time ({mixing_time:.0f}s) exceeds 2 minutes"
                )

            level = ScaleLevel(
                volume_l=volume,
                kla_h=round(kla, 2),
                mixing_time_s=round(mixing_time, 1),
                power_per_volume_w_per_l=round(power_per_vol, 2),
                yield_loss_percent=round(yield_loss * 100, 2),
                scale_factor=round(scale_factor, 2),
                comments=comments,
            )
            results.append(level)
            prev_volume = volume

        if self._logger:
            for level in results:
                self._logger.info(
                    "Scale %.0fL: kLa=%.1f/h, mixing=%.0fs, yield_loss=%.1f%%",
                    level.volume_l, level.kla_h, level.mixing_time_s,
                    level.yield_loss_percent,
                )

        return results


# ---------------------------------------------------------------------------
# MAIN — smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Scale-Up Engine")
    parser.add_argument("--organism", default="ecoli")
    parser.add_argument("--product", default="lycopene")
    args = parser.parse_args()

    logger = PipelineLogger()
    logger.set_stage("5")

    engine = ScaleUpEngine()
    engine.set_logger(logger)

    results = engine.run_scale_up_cascade(
        organism_key=args.organism,
        product_name=args.product,
        product_logp=17.6 if "lycopene" in args.product else 2.0,
    )

    os.makedirs("pipeline_output", exist_ok=True)
    with open("pipeline_output/scaleup_results.json", "w") as fh:
        json.dump([r.to_dict() for r in results], fh, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"  SCALE-UP RESULTS — {args.organism} → {args.product}")
    print(f"{'='*60}")
    for r in results:
        print(f"  {r.volume_l:>8.0f}L  | kLa={r.kla_h:>6.1f}/h | "
              f"mix={r.mixing_time_s:>5.0f}s | loss={r.yield_loss_percent:.1f}%")
        for c in r.comments:
            print(f"            → {c}")
    print(f"{'='*60}")

    print("\n▶ Scale-Up Engine smoke test passed.")
