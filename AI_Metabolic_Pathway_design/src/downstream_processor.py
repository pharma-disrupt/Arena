"""
Downstream Processor Module

Simulates downstream processing for bioproduct purification including
product classification, chromatography mode selection, purification
train simulation, and cost estimation.
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
# PURIFICATION STEP
# ---------------------------------------------------------------------------

@dataclass
class PurificationStep:
    """Represents a single downstream purification step."""
    step_number: int
    step_name: str
    step_type: str  # "harvest", "clarification", "chromatography", "filtration", "drying"
    recovery_percent: float
    purity_percent: float
    cost_per_kg: float
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_number": self.step_number,
            "step_name": self.step_name,
            "step_type": self.step_type,
            "recovery_percent": round(self.recovery_percent, 2),
            "purity_percent": round(self.purity_percent, 2),
            "cost_per_kg": round(self.cost_per_kg, 2),
            "details": self.details,
        }


# ---------------------------------------------------------------------------
# PRODUCT PROPERTY DATABASE
# ---------------------------------------------------------------------------

PRODUCT_PROPERTIES: Dict[str, Dict[str, Any]] = {
    "lycopene": {
        "location": "intracellular",
        "molecular_weight": 536.85,
        "logp": 17.6,
        "water_solubility_mg_per_l": 0.003,
        "charge_at_ph7": "neutral",
        "thermal_stability": "low",
        "shear_sensitivity": "moderate",
        "recommended_chromatography": "normal_phase",
        "harvest_method": "centrifugation + cell disruption",
    },
    "vanillin": {
        "location": "secreted",
        "molecular_weight": 152.15,
        "logp": 1.21,
        "water_solubility_mg_per_l": 10000.0,
        "charge_at_ph7": "neutral",
        "thermal_stability": "moderate",
        "shear_sensitivity": "low",
        "recommended_chromatography": "reverse_phase",
        "harvest_method": "filtration + liquid-liquid extraction",
    },
    "artemisinic_acid": {
        "location": "secreted",
        "molecular_weight": 234.33,
        "logp": 3.8,
        "water_solubility_mg_per_l": 50.0,
        "charge_at_ph7": "negative",
        "thermal_stability": "moderate",
        "shear_sensitivity": "low",
        "recommended_chromatography": "reverse_phase",
        "harvest_method": "filtration + solvent extraction",
    },
    "lysine": {
        "location": "secreted",
        "molecular_weight": 146.19,
        "logp": -3.0,
        "water_solubility_mg_per_l": 1500000.0,
        "charge_at_ph7": "positive",
        "thermal_stability": "high",
        "shear_sensitivity": "low",
        "recommended_chromatography": "ion_exchange",
        "harvest_method": "centrifugation + crystallisation",
    },
    "glutamate": {
        "location": "secreted",
        "molecular_weight": 147.13,
        "logp": -3.0,
        "water_solubility_mg_per_l": 860000.0,
        "charge_at_ph7": "negative",
        "thermal_stability": "high",
        "shear_sensitivity": "low",
        "recommended_chromatography": "ion_exchange",
        "harvest_method": "centrifugation + crystallisation",
    },
    "threonine": {
        "location": "secreted",
        "molecular_weight": 119.12,
        "logp": -3.1,
        "water_solubility_mg_per_l": 250000.0,
        "charge_at_ph7": "zwitterion",
        "thermal_stability": "high",
        "shear_sensitivity": "low",
        "recommended_chromatography": "ion_exchange",
        "harvest_method": "centrifugation + crystallisation",
    },
    "pha": {
        "location": "intracellular",
        "molecular_weight": 86.09,
        "logp": 1.5,
        "water_solubility_mg_per_l": 13000.0,
        "charge_at_ph7": "neutral",
        "thermal_stability": "moderate",
        "shear_sensitivity": "low",
        "recommended_chromatography": "none_required",
        "harvest_method": "centrifugation + solvent extraction + precipitation",
    },
    "hyaluronic_acid": {
        "location": "secreted",
        "molecular_weight": 776.65,
        "logp": -5.0,
        "water_solubility_mg_per_l": 1000000.0,
        "charge_at_ph7": "negative",
        "thermal_stability": "low",
        "shear_sensitivity": "high",
        "recommended_chromatography": "size_exclusion",
        "harvest_method": "tangential flow filtration + precipitation",
    },
    "riboflavin": {
        "location": "secreted",
        "molecular_weight": 376.36,
        "logp": -1.4,
        "water_solubility_mg_per_l": 100.0,
        "charge_at_ph7": "neutral",
        "thermal_stability": "low",
        "shear_sensitivity": "low",
        "recommended_chromatography": "reverse_phase",
        "harvest_method": "filtration + crystallisation",
    },
}

# Organism-specific harvest strategies
ORGANISM_HARVEST: Dict[str, Dict[str, Any]] = {
    "ecoli": {
        "cell_disruption": "high-pressure homogenisation",
        "disruption_efficiency": 0.95,
        "debris_removal": "depth filtration",
        "notes": "Gram-negative — outer membrane requires mechanical disruption",
    },
    "ecoli_bl21": {
        "cell_disruption": "high-pressure homogenisation",
        "disruption_efficiency": 0.93,
        "debris_removal": "depth filtration",
        "notes": "BL21 strain — lower protease activity than K-12",
    },
    "scerevisiae": {
        "cell_disruption": "bead milling",
        "disruption_efficiency": 0.90,
        "debris_removal": "centrifugation + filtration",
        "notes": "Thick cell wall — requires harsh disruption; secreted products easier",
    },
    "scerevisiae_by": {
        "cell_disruption": "bead milling",
        "disruption_efficiency": 0.88,
        "debris_removal": "centrifugation + filtration",
        "notes": "BY4741 — auxotrophic markers may affect secretion",
    },
    "bsubtilis": {
        "cell_disruption": "enzymatic lysis (lysozyme)",
        "disruption_efficiency": 0.92,
        "debris_removal": "centrifugation",
        "notes": "Gram-positive — thick peptidoglycan; secretes proteases",
    },
    "cglutamicum": {
        "cell_disruption": "bead milling",
        "disruption_efficiency": 0.91,
        "debris_removal": "centrifugation + filtration",
        "notes": "Gram-positive, mycolic acid layer — robust but disruptible",
    },
    "pputida": {
        "cell_disruption": "high-pressure homogenisation",
        "disruption_efficiency": 0.94,
        "debris_removal": "depth filtration",
        "notes": "Gram-negative — similar to E. coli but more solvent tolerant",
    },
}


# ---------------------------------------------------------------------------
# DOWNSTREAM PROCESSOR
# ---------------------------------------------------------------------------

class DownstreamProcessor:
    """
    Simulates downstream processing for bioproduct purification.
    """

    def __init__(self) -> None:
        self._logger: Optional[PipelineLogger] = None

    def set_logger(self, logger: PipelineLogger) -> None:
        self._logger = logger

    def classify_product_location(self, product_name: str,
                                   organism_key: str) -> str:
        """
        Classify product as intracellular, secreted, or periplasmic.

        Returns
        -------
        str
            One of: "intracellular", "secreted", "periplasmic"
        """
        props = PRODUCT_PROPERTIES.get(product_name, {})
        return props.get("location", "intracellular")

    def select_chromatography_mode(self, product_name: str,
                                    organism_key: str) -> Dict[str, Any]:
        """
        Select appropriate chromatography mode based on product properties.

        Returns
        -------
        dict
            Chromatography configuration.
        """
        props = PRODUCT_PROPERTIES.get(product_name, {})
        recommended = props.get("recommended_chromatography", "reverse_phase")

        modes = {
            "reverse_phase": {
                "resin": "C18 silica",
                "mobile_phase_a": "0.1% TFA in water",
                "mobile_phase_b": "0.1% TFA in acetonitrile",
                "flow_rate_ml_per_min": 2.0,
                "expected_recovery": 0.85,
                "expected_purity": 0.95,
                "cost_per_run": 150.0,
            },
            "ion_exchange": {
                "resin": "SP Sepharose" if props.get("charge_at_ph7") == "positive" else "Q Sepharose",
                "buffer_a": "20 mM phosphate pH 7.0",
                "buffer_b": "20 mM phosphate + 1M NaCl pH 7.0",
                "flow_rate_ml_per_min": 5.0,
                "expected_recovery": 0.90,
                "expected_purity": 0.92,
                "cost_per_run": 80.0,
            },
            "size_exclusion": {
                "resin": "Sephadex G-50",
                "buffer": "50 mM Tris pH 7.5, 150 mM NaCl",
                "flow_rate_ml_per_min": 1.0,
                "expected_recovery": 0.80,
                "expected_purity": 0.98,
                "cost_per_run": 200.0,
            },
            "normal_phase": {
                "resin": "Silica gel",
                "mobile_phase_a": "Hexane",
                "mobile_phase_b": "Ethyl acetate",
                "flow_rate_ml_per_min": 3.0,
                "expected_recovery": 0.75,
                "expected_purity": 0.90,
                "cost_per_run": 120.0,
            },
            "affinity": {
                "resin": "Protein A/G (if applicable)",
                "binding_buffer": "PBS pH 7.4",
                "elution_buffer": "0.1 M glycine pH 2.5",
                "flow_rate_ml_per_min": 1.5,
                "expected_recovery": 0.95,
                "expected_purity": 0.99,
                "cost_per_run": 500.0,
            },
            "none_required": {
                "resin": "N/A",
                "method": "Solvent extraction + precipitation",
                "expected_recovery": 0.70,
                "expected_purity": 0.85,
                "cost_per_run": 50.0,
            },
        }

        mode_config = modes.get(recommended, modes["reverse_phase"])
        mode_config["mode"] = recommended
        return mode_config

    def simulate_purification_train(self, product_name: str,
                                     organism_key: str,
                                     starting_titer_g_per_l: float,
                                     starting_volume_l: float,
                                     ) -> Dict[str, Any]:
        """
        Simulate a complete downstream purification train.

        Returns
        -------
        dict
            Purification results including per-step recovery and purity,
            overall yield, and cost.
        """
        if self._logger:
            self._logger.info(
                "Simulating purification train: %s in %s, "
                "titer=%.1f g/L, volume=%.1f L",
                product_name, organism_key, starting_titer_g_per_l, starting_volume_l,
            )

        props = PRODUCT_PROPERTIES.get(product_name, {})
        location = props.get("location", "intracellular")
        harvest_info = ORGANISM_HARVEST.get(organism_key, ORGANISM_HARVEST["ecoli"])
        chrom_config = self.select_chromatography_mode(product_name, organism_key)

        steps: List[PurificationStep] = []
        current_recovery = 1.0
        current_purity = 0.10  # Crude broth ~10% purity

        # Step 1: Harvest
        if location == "secreted":
            step_recovery = 0.95
            step_purity = 0.15
            step_name = "Microfiltration / Centrifugation"
        else:
            step_recovery = harvest_info["disruption_efficiency"]
            step_purity = 0.05
            step_name = f"Cell disruption ({harvest_info['cell_disruption']})"

        current_recovery *= step_recovery
        current_purity = step_purity
        steps.append(PurificationStep(
            step_number=1,
            step_name=step_name,
            step_type="harvest",
            recovery_percent=round(step_recovery * 100, 1),
            purity_percent=round(current_purity * 100, 1),
            cost_per_kg=50.0,
            details=harvest_info,
        ))

        # Step 2: Clarification
        clarification_recovery = 0.92
        clarification_purity = current_purity * 1.5
        current_recovery *= clarification_recovery
        current_purity = min(0.99, clarification_purity)
        steps.append(PurificationStep(
            step_number=2,
            step_name="Depth filtration / Centrifugation",
            step_type="clarification",
            recovery_percent=round(clarification_recovery * 100, 1),
            purity_percent=round(current_purity * 100, 1),
            cost_per_kg=30.0,
        ))

        # Step 3: Chromatography
        chrom_recovery = chrom_config.get("expected_recovery", 0.85)
        chrom_purity = chrom_config.get("expected_purity", 0.95)
        current_recovery *= chrom_recovery
        current_purity = chrom_purity
        steps.append(PurificationStep(
            step_number=3,
            step_name=f"Chromatography ({chrom_config.get('mode', 'unknown')})",
            step_type="chromatography",
            recovery_percent=round(chrom_recovery * 100, 1),
            purity_percent=round(current_purity * 100, 1),
            cost_per_kg=chrom_config.get("cost_per_run", 150.0),
            details=chrom_config,
        ))

        # Step 4: Polishing (if needed)
        if current_purity < 0.98:
            polish_recovery = 0.90
            polish_purity = 0.99
            current_recovery *= polish_recovery
            current_purity = polish_purity
            steps.append(PurificationStep(
                step_number=4,
                step_name="Polishing chromatography",
                step_type="chromatography",
                recovery_percent=round(polish_recovery * 100, 1),
                purity_percent=round(current_purity * 100, 1),
                cost_per_kg=200.0,
            ))

        # Step 5: Concentration / Drying
        concentration_recovery = 0.95
        concentration_purity = current_purity
        current_recovery *= concentration_recovery
        current_purity = min(0.999, concentration_purity * 1.01)
        steps.append(PurificationStep(
            step_number=len(steps) + 1,
            step_name="Ultrafiltration / Lyophilisation",
            step_type="drying",
            recovery_percent=round(concentration_recovery * 100, 1),
            purity_percent=round(current_purity * 100, 1),
            cost_per_kg=100.0,
        ))

        # Calculate totals
        total_product_g = starting_titer_g_per_l * starting_volume_l
        recovered_product_g = total_product_g * current_recovery
        total_cost = sum(s.cost_per_kg for s in steps) * (recovered_product_g / 1000.0)
        cost_per_kg = total_cost / max(0.001, recovered_product_g / 1000.0)

        results = {
            "product_name": product_name,
            "organism": organism_key,
            "starting_titer_g_per_l": starting_titer_g_per_l,
            "starting_volume_l": starting_volume_l,
            "total_product_g": round(total_product_g, 2),
            "recovered_product_g": round(recovered_product_g, 2),
            "overall_recovery_percent": round(current_recovery * 100, 2),
            "final_purity_percent": round(current_purity * 100, 2),
            "total_cost_usd": round(total_cost, 2),
            "cost_per_kg_usd": round(cost_per_kg, 2),
            "steps": [s.to_dict() for s in steps],
        }

        if self._logger:
            self._logger.info(
                "Purification train complete: recovery=%.1f%%, purity=%.1f%%, "
                "cost=$%.2f/kg",
                current_recovery * 100, current_purity * 100, cost_per_kg,
            )

        return results

    def organism_specific_harvest_strategy(self, organism_key: str,
                                            product_name: str) -> Dict[str, Any]:
        """
        Recommend organism-specific harvest and downstream strategy.

        Returns
        -------
        dict
            Harvest strategy configuration.
        """
        harvest_info = ORGANISM_HARVEST.get(organism_key, ORGANISM_HARVEST["ecoli"])
        product_props = PRODUCT_PROPERTIES.get(product_name, {})
        location = product_props.get("location", "intracellular")

        strategy = {
            "organism": organism_key,
            "product": product_name,
            "product_location": location,
            "harvest_method": harvest_info.get("cell_disruption", "unknown"),
            "disruption_efficiency": harvest_info.get("disruption_efficiency", 0.9),
            "debris_removal": harvest_info.get("debris_removal", "filtration"),
            "notes": harvest_info.get("notes", ""),
        }

        if location == "secreted":
            strategy["harvest_method"] = "Centrifugation + microfiltration"
            strategy["disruption_efficiency"] = 1.0
            strategy["notes"] += " | Product is secreted — no cell disruption needed"

        return strategy


# ---------------------------------------------------------------------------
# MAIN — smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Downstream Processor")
    parser.add_argument("--organism", default="ecoli")
    parser.add_argument("--product", default="lycopene")
    parser.add_argument("--titer", type=float, default=5.0)
    parser.add_argument("--volume", type=float, default=200.0)
    args = parser.parse_args()

    logger = PipelineLogger()
    logger.set_stage("5")

    processor = DownstreamProcessor()
    processor.set_logger(logger)

    # Classify product
    location = processor.classify_product_location(args.product, args.organism)
    logger.info("Product location: %s", location)

    # Select chromatography
    chrom = processor.select_chromatography_mode(args.product, args.organism)
    logger.info("Chromatography mode: %s", chrom.get("mode", "unknown"))

    # Simulate purification train
    results = processor.simulate_purification_train(
        product_name=args.product,
        organism_key=args.organism,
        starting_titer_g_per_l=args.titer,
        starting_volume_l=args.volume,
    )

    # Harvest strategy
    strategy = processor.organism_specific_harvest_strategy(
        args.organism, args.product
    )

    os.makedirs("pipeline_output", exist_ok=True)
    with open("pipeline_output/downstream_results.json", "w") as fh:
        json.dump({
            "product_location": location,
            "chromatography": chrom,
            "purification_train": results,
            "harvest_strategy": strategy,
        }, fh, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"  DOWNSTREAM RESULTS — {args.organism} → {args.product}")
    print(f"{'='*60}")
    print(f"  Location    : {location}")
    print(f"  Chromatog.  : {chrom.get('mode', 'N/A')}")
    print(f"  Recovery    : {results['overall_recovery_percent']:.1f}%")
    print(f"  Purity      : {results['final_purity_percent']:.1f}%")
    print(f"  Cost        : ${results['cost_per_kg_usd']:.2f}/kg")
    print(f"  Steps       : {len(results['steps'])}")
    for step in results['steps']:
        print(f"    {step['step_number']}. {step['step_name']} "
              f"(recovery={step['recovery_percent']}%, purity={step['purity_percent']}%)")
    print(f"{'='*60}")

    print("\n▶ Downstream Processor smoke test passed.")
