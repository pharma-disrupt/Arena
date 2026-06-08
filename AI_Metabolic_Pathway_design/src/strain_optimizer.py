"""
Strain Optimizer Module

Implements in-silico strain design algorithms including simulated OptKnock
knockout identification, multi-objective NSGA-III optimisation, metabolic
burden scoring, and genetic stability prediction.
"""

from __future__ import annotations

import logging
import math
import os
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import differential_evolution

from logger_setup import PipelineLogger


# ---------------------------------------------------------------------------
# GENETIC MODIFICATION DATACLASS
# ---------------------------------------------------------------------------

@dataclass
class GeneticModification:
    """Represents a single genetic engineering intervention."""

    gene_name: str
    modification_type: str  # "knockout", "overexpression", "insertion"
    expected_effect: str
    confidence_score: float
    literature_evidence: bool = False
    strain_improvement_history: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gene_name": self.gene_name,
            "modification_type": self.modification_type,
            "expected_effect": self.expected_effect,
            "confidence_score": self.confidence_score,
            "literature_evidence": self.literature_evidence,
            "strain_improvement_history": self.strain_improvement_history,
        }


# ---------------------------------------------------------------------------
# STRAIN OPTIMIZER
# ---------------------------------------------------------------------------

class StrainOptimizer:
    """
    Multi-objective strain design optimiser.

    Combines:
    - Simulated OptKnock for knockout target identification
    - NSGA-III-inspired multi-objective optimisation
    - Metabolic burden scoring
    - Genetic stability prediction
    """

    # Known knockout targets per organism for common products
    KNOCKOUT_DATABASE: Dict[str, Dict[str, List[str]]] = {
        "ecoli": {
            "lycopene": ["ldhA", "adhE", "frdA", "poxB", "ackA"],
            "vanillin": ["fadE", "fadD", "acs", "poxB"],
            "lysine": ["mdh", "sucA", "frdA", "ldhA", "adhE"],
            "riboflavin": ["purR", "ribR", "ribS"],
            "pha": ["fadR", "atoC", "atoDA"],
        },
        "ecoli_bl21": {
            "lycopene": ["ldhA", "adhE", "poxB"],
            "vanillin": ["fadE", "acs"],
            "lysine": ["mdh", "frdA"],
        },
        "scerevisiae": {
            "lycopene": ["ROX1", "UPC2", "ERG9"],
            "artemisinic_acid": ["ROX1", "ERG9", "HMG2"],
            "vanillin": ["ADH1", "PDC1", "ALD6"],
        },
        "scerevisiae_by": {
            "lycopene": ["ROX1", "UPC2"],
            "vanillin": ["ADH1", "PDC1"],
        },
        "bsubtilis": {
            "riboflavin": ["ykvC", "ywnB", "purR"],
            "pha": ["ackA", "poxB", "ldh"],
        },
        "cglutamicum": {
            "lysine": ["hom", "thrB", "dapA"],
            "glutamate": ["icd", "odx", "pqo"],
        },
        "pputida": {
            "pha": ["phaZ", "phaD"],
            "vanillin": ["catA", "catB", "pcaHG"],
        },
    }

    OVEREXPRESSION_DATABASE: Dict[str, Dict[str, List[str]]] = {
        "ecoli": {
            "lycopene": ["dxs", "idi", "ispDF", "dxr"],
            "vanillin": ["sam5", "vanAB", "calA"],
            "lysine": ["dapA", "lysC", "asd"],
            "riboflavin": ["ribA", "ribB", "ribC"],
            "pha": ["phaA", "phaB", "phaC"],
        },
        "scerevisiae": {
            "lycopene": ["ERG20", "tHMG1", "IDI1"],
            "artemisinic_acid": ["tHMG1", "ERG20", "IDI1"],
        },
        "bsubtilis": {
            "riboflavin": ["ribAB", "ribD", "ribE"],
        },
        "cglutamicum": {
            "lysine": ["dapA", "asd", "hom"],
            "glutamate": ["gltB", "gltD", "gdhA"],
        },
        "pputida": {
            "pha": ["phaC1", "phaC2"],
        },
    }

    def __init__(self) -> None:
        self._logger: Optional[PipelineLogger] = None

    def set_logger(self, logger: PipelineLogger) -> None:
        self._logger = logger

    def run_optknock_simulated(
        self,
        organism_key: str,
        molecule_key: str,
        max_knockouts: int = 3,
        essential_genes: Optional[List[str]] = None,
    ) -> List[GeneticModification]:
        """
        Simulate OptKnock algorithm to identify optimal knockout targets.

        Uses a biologically-curated database of known knockouts combined
        with FBA-based coupling analysis simulation.

        Parameters
        ----------
        organism_key : str
            Target organism (e.g., "ecoli").
        molecule_key : str
            Target product (e.g., "lycopene").
        max_knockouts : int
            Maximum number of knockout targets to recommend.
        essential_genes : list, optional
            Genes that cannot be knocked out.

        Returns
        -------
        list of GeneticModification
        """
        if self._logger:
            self._logger.info(
                "Running simulated OptKnock: organism=%s, molecule=%s, max_KO=%d",
                organism_key, molecule_key, max_knockouts,
            )

        essential = set(essential_genes or [])

        # Get candidate knockouts from database
        org_kos = self.KNOCKOUT_DATABASE.get(organism_key, {})
        mol_kos = org_kos.get(molecule_key, [])

        # Filter out essential genes
        candidates = [g for g in mol_kos if g.lower() not in essential]

        # Score each candidate (simulated coupling analysis)
        scored_kos: List[Tuple[str, float]] = []
        for gene in candidates:
            # Simulated growth-product coupling score
            random.seed(hash(f"{organism_key}_{molecule_key}_{gene}"))
            coupling_score = random.uniform(0.5, 0.95)
            scored_kos.append((gene, coupling_score))

        # Sort by coupling score (highest first)
        scored_kos.sort(key=lambda x: x[1], reverse=True)

        # Select top N
        selected = scored_kos[:max_knockouts]

        modifications: List[GeneticModification] = []
        for gene, score in selected:
            mod = GeneticModification(
                gene_name=gene,
                modification_type="knockout",
                expected_effect=f"Reduces flux to competing pathway; increases {molecule_key} yield",
                confidence_score=round(score, 4),
                literature_evidence=True,
                strain_improvement_history=[
                    f"Simulated OptKnock coupling score: {score:.3f}"
                ],
            )
            modifications.append(mod)

        if self._logger:
            self._logger.info(
                "OptKnock identified %d knockout targets: %s",
                len(modifications), [m.gene_name for m in modifications],
            )

        return modifications

    def run_nsga3_optimization(
        self,
        organism_key: str,
        molecule_key: str,
        knockouts: List[str],
        overexpressions: List[str],
        n_populations: int = 50,
        n_generations: int = 30,
    ) -> List[Dict[str, Any]]:
        """
        Simulate NSGA-III multi-objective optimisation.

        Objectives:
        1. Maximise product yield (mol/mol glucose)
        2. Maximise growth rate (h⁻¹)
        3. Minimise metabolic burden

        Returns a Pareto front of strain designs.
        """
        if self._logger:
            self._logger.info(
                "Running NSGA-III optimisation: organism=%s, molecule=%s, "
                "KO=%d, OE=%d",
                organism_key, molecule_key, len(knockouts), len(overexpressions),
            )

        n_objectives = 3
        n_variables = len(knockouts) + len(overexpressions)

        if n_variables == 0:
            # No modifications: return baseline
            return [{
                "design_id": "baseline",
                "knockouts": [],
                "overexpressions": [],
                "objectives": {
                    "product_yield_mol_per_mol": 0.0,
                    "growth_rate_per_hour": 0.5,
                    "metabolic_burden": 0.0,
                },
                "dominated": False,
            }]

        # Simulate NSGA-III with differential evolution
        def objective_function(x: np.ndarray) -> np.ndarray:
            """Multi-objective function to optimise."""
            # Decode: first len(knockouts) are KO presence (binary),
            # rest are OE levels (continuous 0-1)
            ko_present = x[:len(knockouts)] > 0.5
            oe_levels = x[len(knockouts):]

            # Simulate yield improvement from knockouts
            base_yield = 0.10
            ko_benefit = sum(0.05 * (1 if present else 0) for present in ko_present)

            # Simulate yield improvement from overexpression
            oe_benefit = sum(0.03 * level for level in oe_levels)

            product_yield = min(0.80, base_yield + ko_benefit + oe_benefit)

            # Growth penalty from knockouts (lethality simulation)
            growth_penalty = sum(0.05 * (1 if present else 0) for present in ko_present)
            growth_rate = max(0.10, 0.80 - growth_penalty)

            # Metabolic burden from overexpression
            burden = sum(0.10 * level for level in oe_levels)
            burden = min(1.0, burden)

            return np.array([product_yield, growth_rate, burden])

        # Run differential evolution
        bounds = [(0, 1)] * n_variables
        result = differential_evolution(
            func=lambda x: -objective_function(x)[0],  # Single objective: maximise yield
            bounds=bounds,
            maxiter=n_generations,
            popsize=max(10, n_populations // max(1, n_variables)),
            seed=42,
        )

        # Generate Pareto front by sampling around optimal solution
        pareto_front: List[Dict[str, Any]] = []
        n_samples = min(20, n_populations)

        for i in range(n_samples):
            random.seed(i * 137 + 42)
            sample = result.x.copy()

            # Perturb sample
            for j in range(len(sample)):
                sample[j] += random.gauss(0, 0.1)
                sample[j] = max(0, min(1, sample[j]))

            obj = objective_function(sample)

            ko_present = [knockouts[j] for j in range(len(knockouts))
                         if sample[j] > 0.5]
            oe_active = [overexpressions[j - len(knockouts)]
                        for j in range(len(knockouts), n_variables)
                        if sample[j] > 0.3]

            pareto_front.append({
                "design_id": f"design_{i:03d}",
                "knockouts": ko_present,
                "overexpressions": oe_active,
                "objectives": {
                    "product_yield_mol_per_mol": round(float(obj[0]), 4),
                    "growth_rate_per_hour": round(float(obj[1]), 4),
                    "metabolic_burden": round(float(obj[2]), 4),
                },
                "dominated": False,  # Simplified: assume non-dominated
            })

        # Sort by product yield (primary objective)
        pareto_front.sort(
            key=lambda d: d["objectives"]["product_yield_mol_per_mol"],
            reverse=True,
        )

        if self._logger:
            self._logger.info(
                "NSGA-III complete: %d designs on Pareto front",
                len(pareto_front),
            )

        return pareto_front

    def score_metabolic_burden(
        self,
        organism_key: str,
        knockouts: List[str],
        overexpressions: List[str],
        heterologous_genes: int = 0,
    ) -> float:
        """
        Calculate metabolic burden score (0–1, lower is better).

        Burden comes from:
        - Gene knockouts disrupting native metabolism
        - Overexpression draining cellular resources
        - Heterologous gene expression
        - Plasmid copy number (simulated)
        """
        if self._logger:
            self._logger.debug(
                "Scoring metabolic burden: KO=%d, OE=%d, heterologous=%d",
                len(knockouts), len(overexpressions), heterologous_genes,
            )

        # Base burden from knockouts
        ko_burden = len(knockouts) * 0.05  # 5% per knockout

        # Burden from overexpression
        oe_burden = len(overexpressions) * 0.08  # 8% per OE

        # Heterologous gene burden
        het_burden = heterologous_genes * 0.10  # 10% per heterologous gene

        # Plasmid burden (simulated)
        plasmid_burden = 0.05 if heterologous_genes > 0 else 0.0

        # Total burden (capped at 1.0)
        total_burden = ko_burden + oe_burden + het_burden + plasmid_burden

        return round(min(1.0, max(0.0, total_burden)), 4)

    def predict_genetic_stability(
        self,
        organism_key: str,
        knockouts: List[str],
        overexpressions: List[str],
        heterologous_genes: int = 0,
    ) -> float:
        """
        Predict genetic stability score (0–1, higher is more stable).

        Factors:
        - Number of modifications (more = less stable)
        - Essential gene proximity
        - Repeated sequence elements
        - Organism-specific stability characteristics
        """
        if self._logger:
            self._logger.debug(
                "Predicting genetic stability: organism=%s, KO=%d, OE=%d, het=%d",
                organism_key, len(knockouts), len(overexpressions), heterologous_genes,
            )

        # Base stability per organism
        base_stability = {
            "ecoli": 0.85,
            "ecoli_bl21": 0.80,
            "scerevisiae": 0.90,
            "scerevisiae_by": 0.88,
            "bsubtilis": 0.75,
            "cglutamicum": 0.82,
            "pputida": 0.78,
        }.get(organism_key, 0.80)

        # Stability penalty per modification
        penalty_per_ko = 0.02
        penalty_per_oe = 0.015
        penalty_per_het = 0.03

        total_penalty = (
            len(knockouts) * penalty_per_ko
            + len(overexpressions) * penalty_per_oe
            + heterologous_genes * penalty_per_het
        )

        stability = base_stability - total_penalty

        return round(min(1.0, max(0.0, stability)), 4)

    def rank_strain_designs(
        self,
        pareto_front: List[Dict[str, Any]],
        organism_key: str,
        molecule_key: str,
    ) -> List[Dict[str, Any]]:
        """
        Rank strain designs using a weighted composite score.

        Weights:
        - Product yield: 40%
        - Growth rate: 25%
        - Metabolic burden: -20% (lower is better)
        - Genetic stability: 15%
        """
        if self._logger:
            self._logger.info(
                "Ranking %d strain designs for %s → %s",
                len(pareto_front), organism_key, molecule_key,
            )

        ranked: List[Dict[str, Any]] = []

        for design in pareto_front:
            obj = design["objectives"]

            # Normalise objectives to 0–1
            yield_score = min(1.0, obj["product_yield_mol_per_mol"] / 0.8)
            growth_score = min(1.0, obj["growth_rate_per_hour"] / 1.0)
            burden_score = 1.0 - obj["metabolic_burden"]  # Invert: lower burden = better
            stability_score = self.predict_genetic_stability(
                organism_key,
                design.get("knockouts", []),
                design.get("overexpressions", []),
                heterologous_genes=len(design.get("overexpressions", [])),
            )

            # Weighted composite
            composite = (
                0.40 * yield_score
                + 0.25 * growth_score
                + 0.20 * burden_score
                + 0.15 * stability_score
            )

            design["composite_score"] = round(composite, 4)
            design["yield_score"] = round(yield_score, 4)
            design["growth_score"] = round(growth_score, 4)
            design["burden_score"] = round(burden_score, 4)
            design["stability_score"] = round(stability_score, 4)

            ranked.append(design)

        # Sort by composite score
        ranked.sort(key=lambda d: d["composite_score"], reverse=True)

        if self._logger:
            self._logger.info(
                "Top design: %s (composite=%.3f, yield=%.3f, growth=%.3f)",
                ranked[0]["design_id"] if ranked else "N/A",
                ranked[0].get("composite_score", 0) if ranked else 0,
                ranked[0].get("yield_score", 0) if ranked else 0,
                ranked[0].get("growth_score", 0) if ranked else 0,
            )

        return ranked


# ---------------------------------------------------------------------------
# MAIN — smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Strain Optimizer")
    parser.add_argument("--organism", default="ecoli")
    parser.add_argument("--molecule", default="lycopene")
    args = parser.parse_args()

    logger = PipelineLogger()
    logger.set_stage("3")

    optimizer = StrainOptimizer()
    optimizer.set_logger(logger)

    # Run OptKnock
    essential_genes = ["dnaA", "rpoB", "ftsZ"]
    kos = optimizer.run_optknock_simulated(
        organism_key=args.organism,
        molecule_key=args.molecule,
        max_knockouts=3,
        essential_genes=essential_genes,
    )
    logger.info("OptKnock results: %d knockouts", len(kos))

    # Run NSGA-III
    ko_names = [k.gene_name for k in kos]
    oe_names = ["dxs", "idi", "crtE"]  # Simulated
    pareto = optimizer.run_nsga3_optimization(
        organism_key=args.organism,
        molecule_key=args.molecule,
        knockouts=ko_names,
        overexpressions=oe_names,
    )
    logger.info("NSGA-III: %d designs on Pareto front", len(pareto))

    # Rank designs
    ranked = optimizer.rank_strain_designs(
        pareto_front=pareto,
        organism_key=args.organism,
        molecule_key=args.molecule,
    )

    # Metabolic burden and stability
    top_design = ranked[0] if ranked else {}
    burden = optimizer.score_metabolic_burden(
        organism_key=args.organism,
        knockouts=top_design.get("knockouts", []),
        overexpressions=top_design.get("overexpressions", []),
        heterologous_genes=3,
    )
    stability = optimizer.predict_genetic_stability(
        organism_key=args.organism,
        knockouts=top_design.get("knockouts", []),
        overexpressions=top_design.get("overexpressions", []),
        heterologous_genes=3,
    )

    logger.info("Top design burden: %.3f, stability: %.3f", burden, stability)

    os.makedirs("pipeline_output", exist_ok=True)
    with open("pipeline_output/strain_design.json", "w") as fh:
        json.dump({
            "knockouts": [k.to_dict() for k in kos],
            "pareto_front": pareto[:5],
            "ranked_designs": ranked[:3],
            "burden_score": burden,
            "stability_score": stability,
        }, fh, indent=2, default=str)

    logger.info("Strain design saved to pipeline_output/strain_design.json")
    print(
        f"\n▶ Strain Optimizer smoke test passed. "
        f"Top composite score={ranked[0].get('composite_score', 'N/A') if ranked else 'N/A'}"
    )
