"""
Toxicity Predictor Module

Predicts toxicity of pathway intermediates and final products for
industrial microorganisms. Assesses ROS risk, membrane disruption,
and organism-specific toxicity thresholds.
"""

from __future__ import annotations

import logging
import math
import os
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from logger_setup import PipelineLogger


# ---------------------------------------------------------------------------
# ORGANISM-SPECIFIC TOXICITY THRESHOLDS
# ---------------------------------------------------------------------------

ORGANISM_TOXICITY_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "ecoli": {
        "max_lipophilicity_logp": 5.0,
        "max_reactive_oxygen_score": 0.6,
        "max_membrane_disruption": 0.5,
        "max_solvent_tolerance_g_per_l": 50.0,
        "toxic_intermediates": ["acetaldehyde", "formaldehyde", "acrolein"],
    },
    "ecoli_bl21": {
        "max_lipophilicity_logp": 5.0,
        "max_reactive_oxygen_score": 0.55,
        "max_membrane_disruption": 0.45,
        "max_solvent_tolerance_g_per_l": 45.0,
        "toxic_intermediates": ["acetaldehyde", "formaldehyde"],
    },
    "scerevisiae": {
        "max_lipophilicity_logp": 6.0,
        "max_reactive_oxygen_score": 0.7,
        "max_membrane_disruption": 0.6,
        "max_solvent_tolerance_g_per_l": 80.0,
        "toxic_intermediates": ["acetaldehyde"],
    },
    "scerevisiae_by": {
        "max_lipophilicity_logp": 5.5,
        "max_reactive_oxygen_score": 0.65,
        "max_membrane_disruption": 0.55,
        "max_solvent_tolerance_g_per_l": 75.0,
        "toxic_intermediates": ["acetaldehyde"],
    },
    "bsubtilis": {
        "max_lipophilicity_logp": 4.5,
        "max_reactive_oxygen_score": 0.5,
        "max_membrane_disruption": 0.4,
        "max_solvent_tolerance_g_per_l": 30.0,
        "toxic_intermediates": ["acrolein", "hydrogen_peroxide"],
    },
    "cglutamicum": {
        "max_lipophilicity_logp": 5.0,
        "max_reactive_oxygen_score": 0.6,
        "max_membrane_disruption": 0.5,
        "max_solvent_tolerance_g_per_l": 60.0,
        "toxic_intermediates": ["ammonia", "glutamate_excess"],
    },
    "pputida": {
        "max_lipophilicity_logp": 7.0,
        "max_reactive_oxygen_score": 0.75,
        "max_membrane_disruption": 0.7,
        "max_solvent_tolerance_g_per_l": 100.0,
        "toxic_intermediates": [],  # P. putida is very robust
    },
}


# ---------------------------------------------------------------------------
# TOXICITY PREDICTOR
# ---------------------------------------------------------------------------

class ToxicityPredictor:
    """
    Predicts toxicity risk of metabolic pathway intermediates.

    Uses simulated models for:
    - Intermediate toxicity scoring
    - Reactive oxygen species (ROS) risk
    - Membrane disruption potential
    - Organism-specific toxicity thresholds
    """

    def __init__(self) -> None:
        self._logger: Optional[PipelineLogger] = None
        self._thresholds = ORGANISM_TOXICITY_THRESHOLDS

    def set_logger(self, logger: PipelineLogger) -> None:
        self._logger = logger

    def predict_intermediate_toxicity(
        self,
        intermediate_name: str,
        logp: Optional[float] = None,
        mw: Optional[float] = None,
        functional_groups: Optional[List[str]] = None,
    ) -> float:
        """
        Predict toxicity score for a pathway intermediate.

        Score range: 0.0 (non-toxic) to 1.0 (highly toxic).

        Parameters
        ----------
        intermediate_name : str
            Name of the intermediate metabolite.
        logp : float, optional
            Octanol-water partition coefficient.
        mw : float, optional
            Molecular weight.
        functional_groups : list of str, optional
            Chemical functional groups present.

        Returns
        -------
        float
            Toxicity score between 0 and 1.
        """
        if self._logger:
            self._logger.debug(
                "Predicting toxicity for intermediate: %s", intermediate_name
            )

        # Use logp as primary toxicity predictor if available
        if logp is not None:
            # Higher logp → higher toxicity (lipophilic compounds more toxic)
            logp_score = min(1.0, max(0.0, logp / 10.0))
        else:
            logp_score = 0.3  # Default moderate toxicity

        # Check for known toxic functional groups
        toxic_groups = {"aldehyde", "epoxide", "acyl_chloride", "nitro", "isocyanate"}
        functional_groups = functional_groups or []
        group_score = sum(0.15 for g in functional_groups if g.lower() in toxic_groups)

        # Molecular weight effect (very large molecules less toxic)
        mw_score = 0.0
        if mw is not None:
            if mw > 1000:
                mw_score = 0.1
            elif mw > 500:
                mw_score = 0.2
            elif mw > 200:
                mw_score = 0.3
            else:
                mw_score = 0.4

        # Combine scores
        toxicity = 0.5 * logp_score + 0.3 * group_score + 0.2 * mw_score

        # Add organism-specific noise
        random.seed(hash(f"{intermediate_name}_{logp}_{mw}"))
        noise = random.gauss(0, 0.05)

        return round(min(1.0, max(0.0, toxicity + noise)), 4)

    def calculate_ros_score(
        self,
        intermediates: List[str],
        organism_key: str = "ecoli",
    ) -> Dict[str, float]:
        """
        Calculate reactive oxygen species (ROS) risk for each intermediate.

        ROS generation is a common side-effect of metabolic engineering,
        especially for pathways involving:
        - Cytochrome P450 enzymes
        - Oxidative reactions
        - Electron transport chain perturbations

        Returns
        -------
        dict
            Maps intermediate name → ROS score (0–1).
        """
        if self._logger:
            self._logger.info("Calculating ROS scores for %d intermediates", len(intermediates))

        ros_scores: Dict[str, float] = {}

        # Known ROS-generating reactions
        ros_generating = {
            "cytochrome_p450": 0.8,
            "monoxygenase": 0.7,
            "oxidase": 0.6,
            "dehydrogenase": 0.4,
            "peroxidase": 0.5,
            "superoxide_dismutase": 0.3,
        }

        for intermediate in intermediates:
            random.seed(hash(f"{intermediate}_{organism_key}"))

            # Check if intermediate is associated with ROS-generating enzymes
            max_ros = 0.0
            for enzyme_type, base_score in ros_generating.items():
                if enzyme_type in intermediate.lower():
                    max_ros = max(max_ros, base_score)

            # If not directly associated, estimate from organism thresholds
            thresholds = self._thresholds.get(organism_key, {})
            max_ros_allowed = thresholds.get("max_reactive_oxygen_score", 0.6)

            # Simulate ROS score
            if max_ros > 0:
                ros_score = min(max_ros_allowed, max_ros * random.uniform(0.8, 1.2))
            else:
                ros_score = random.uniform(0.05, max_ros_allowed * 0.5)

            ros_scores[intermediate] = round(min(1.0, max(0.0, ros_score)), 4)

        return ros_scores

    def check_membrane_disruption(
        self,
        intermediates: List[str],
        logp_values: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Check membrane disruption potential of intermediates.

        Lipophilic compounds (high logP) can disrupt cell membranes,
        leading to leakage and reduced viability.

        Returns
        -------
        dict
            Maps intermediate name → membrane disruption score (0–1).
        """
        if self._logger:
            self._logger.info(
                "Checking membrane disruption for %d intermediates", len(intermediates)
            )

        disruption_scores: Dict[str, float] = {}
        logp_values = logp_values or {}

        for intermediate in intermediates:
            logp = logp_values.get(intermediate, 2.0)  # Default moderate logP

            # Membrane disruption correlates with logP
            # logP < 2: low disruption
            # logP 2-5: moderate disruption
            # logP > 5: high disruption
            if logp < 2.0:
                disruption = logp / 10.0
            elif logp < 5.0:
                disruption = 0.2 + (logp - 2.0) / 10.0
            else:
                disruption = 0.5 + (logp - 5.0) / 10.0

            random.seed(hash(f"{intermediate}_membrane"))
            noise = random.gauss(0, 0.05)

            disruption_scores[intermediate] = round(
                min(1.0, max(0.0, disruption + noise)), 4
            )

        return disruption_scores

    def assess_overall_toxicity(
        self,
        organism_key: str,
        pathway_intermediates: List[Dict[str, Any]],
        final_product_name: str = "product",
        final_product_logp: float = 2.0,
    ) -> Dict[str, Any]:
        """
        Perform comprehensive toxicity assessment for a pathway.

        Parameters
        ----------
        organism_key : str
            Target organism.
        pathway_intermediates : list of dict
            Each dict should have keys: name, logp (optional), mw (optional),
            functional_groups (optional).
        final_product_name : str
            Name of the final product molecule.
        final_product_logp : float
            LogP of the final product.

        Returns
        -------
        dict
            Keys: intermediate_toxicity_scores, overall_toxicity_risk,
                  flagged_intermediates, recommendations.
        """
        if self._logger:
            self._logger.info(
                "Assessing overall toxicity for organism=%s, product=%s",
                organism_key, final_product_name,
            )

        thresholds = self._thresholds.get(organism_key, {})
        max_logp = thresholds.get("max_lipophilicity_logp", 5.0)
        max_ros = thresholds.get("max_reactive_oxygen_score", 0.6)
        max_membrane = thresholds.get("max_membrane_disruption", 0.5)

        # Score each intermediate
        intermediate_scores: Dict[str, float] = {}
        ros_scores: Dict[str, float] = {}
        membrane_scores: Dict[str, float] = {}
        flagged: List[str] = []

        for inter in pathway_intermediates:
            name = inter.get("name", "unknown")

            # Toxicity score
            tox_score = self.predict_intermediate_toxicity(
                intermediate_name=name,
                logp=inter.get("logp"),
                mw=inter.get("mw"),
                functional_groups=inter.get("functional_groups", []),
            )
            intermediate_scores[name] = tox_score

            # ROS score
            inter_ros = self.calculate_ros_score([name], organism_key)
            ros_scores[name] = inter_ros.get(name, 0.0)

            # Membrane disruption
            logp_for_mem = inter.get("logp") if inter.get("logp") is not None else 2.0
            inter_mem = self.check_membrane_disruption([name], {name: logp_for_mem})
            membrane_scores[name] = inter_mem.get(name, 0.0)

            # Flag if any score exceeds threshold
            if (
                tox_score > max_logp / 10.0
                or ros_scores[name] > max_ros
                or membrane_scores[name] > max_membrane
            ):
                flagged.append(name)

        # Score final product
        product_tox = self.predict_intermediate_toxicity(
            intermediate_name=final_product_name,
            logp=final_product_logp,
        )
        intermediate_scores[final_product_name] = product_tox

        if product_tox > max_logp / 10.0:
            flagged.append(final_product_name)

        # Overall risk assessment
        all_scores = list(intermediate_scores.values())
        avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
        max_score = max(all_scores) if all_scores else 0.0

        if max_score > 0.7:
            overall_risk = "HIGH"
        elif max_score > 0.4:
            overall_risk = "MEDIUM"
        else:
            overall_risk = "LOW"

        result = {
            "intermediate_toxicity_scores": intermediate_scores,
            "ros_scores": ros_scores,
            "membrane_disruption_scores": membrane_scores,
            "overall_toxicity_risk": overall_risk,
            "flagged_intermediates": flagged,
            "average_toxicity_score": round(avg_score, 4),
            "max_toxicity_score": round(max_score, 4),
            "recommendations": self._generate_recommendations(
                overall_risk, flagged, organism_key
            ),
        }

        if self._logger:
            self._logger.info(
                "Toxicity assessment complete: risk=%s, flagged=%d, avg_score=%.3f",
                overall_risk, len(flagged), avg_score,
            )

        return result

    def _generate_recommendations(
        self,
        risk_level: str,
        flagged_intermediates: List[str],
        organism_key: str,
    ) -> List[str]:
        """Generate actionable recommendations based on toxicity assessment."""
        recommendations: List[str] = []

        if risk_level == "HIGH":
            recommendations.append(
                f"Consider switching to a more solvent-tolerant host organism"
            )
            recommendations.append(
                f"Implement in-situ product removal (ISPR) strategy"
            )
            recommendations.append(
                f"Add efflux pumps for toxic intermediates"
            )

        if flagged_intermediates:
            for inter in flagged_intermediates[:3]:  # Top 3
                recommendations.append(
                    f"Monitor {inter} concentration during fermentation"
                )

        if organism_key in ("bsubtilis", "cglutamicum"):
            recommendations.append(
                f"Consider co-culture with a robust host for toxic intermediate processing"
            )

        recommendations.append(
            f"Perform small-scale toxicity validation before scale-up"
        )

        return recommendations


# ---------------------------------------------------------------------------
# MAIN — smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Toxicity Predictor")
    parser.add_argument("--organism", default="ecoli")
    parser.add_argument("--product", default="lycopene")
    args = parser.parse_args()

    logger = PipelineLogger()
    logger.set_stage("3")

    predictor = ToxicityPredictor()
    predictor.set_logger(logger)

    # Simulate pathway intermediates
    intermediates = [
        {"name": "pyruvate", "logp": -0.5, "mw": 88.06, "functional_groups": ["ketone", "carboxylic_acid"]},
        {"name": "acetyl_coa", "logp": 1.5, "mw": 809.57, "functional_groups": ["thioester", "phosphate"]},
        {"name": "geranyl_geranyl_pyrophosphate", "logp": 6.0, "mw": 450.47, "functional_groups": ["phosphate", "olefin"]},
        {"name": "phytoene", "logp": 12.0, "mw": 544.88, "functional_groups": ["olefin"]},
        {"name": "lycopene", "logp": 17.6, "mw": 536.85, "functional_groups": ["olefin"]},
    ]

    assessment = predictor.assess_overall_toxicity(
        organism_key=args.organism,
        pathway_intermediates=intermediates,
        final_product_name=args.product,
        final_product_logp=17.6,
    )

    logger.info("Overall toxicity risk: %s", assessment["overall_toxicity_risk"])
    logger.info("Flagged intermediates: %s", assessment["flagged_intermediates"])
    logger.info("Recommendations: %s", assessment["recommendations"])

    os.makedirs("pipeline_output", exist_ok=True)
    with open("pipeline_output/toxicity_assessment.json", "w") as fh:
        json.dump(assessment, fh, indent=2, default=str)

    logger.info("Toxicity assessment saved to pipeline_output/toxicity_assessment.json")
    print(
        f"\n▶ Toxicity Predictor smoke test passed. "
        f"Risk: {assessment['overall_toxicity_risk']}"
    )
