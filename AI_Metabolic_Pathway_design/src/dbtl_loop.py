"""
DBTL Loop Module

Implements the Design-Build-Test-Learn cycle with construct library
generation, simulated HTS screening, Bayesian optimisation, and
iterative strain improvement.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.stats import norm

from logger_setup import PipelineLogger


# ---------------------------------------------------------------------------
# DATACLASSES
# ---------------------------------------------------------------------------

@dataclass
class Construct:
    """Represents a single genetic construct in the DBTL library."""
    construct_id: str
    promoter: str
    rbs: str
    genes: List[str]
    predicted_titer: float
    actual_titer: Optional[float] = None
    cycle_tested: Optional[int] = None


@dataclass
class DBTLCycle:
    """Results from a single DBTL cycle."""
    cycle_number: int
    constructs_tested: int
    best_titer_g_per_l: float
    best_construct_id: str
    improvement_fold: float
    bo_next_candidates: List[str]


# ---------------------------------------------------------------------------
# DBTL ORCHESTRATOR
# ---------------------------------------------------------------------------

class DBTLOrchestrator:
    """
    Orchestrates Design-Build-Test-Learn cycles for strain optimisation.

    Each cycle:
    1. DESIGN: Generate construct library
    2. BUILD: Simulate construct assembly
    3. TEST: Simulate HTS screening results
    4. LEARN: Bayesian optimisation to select next batch
    """

    PROMOTER_POOL = [
        "Ptac", "Ptrc", "Plac", "ParaBAD", "PT7", "PlacUV5",
        "Pbad", "PrhaBAD", "PJ23100", "PJ23119",
    ]
    RBS_POOL = [
        "RBS_B0034", "RBS_B0030", "RBS_B0032", "RBS_B0031",
        "RBS_B0033", "RBS_B0064",
    ]

    def __init__(self) -> None:
        self._logger: Optional[PipelineLogger] = None
        self._all_constructs: List[Construct] = []
        self._screening_results: Dict[str, float] = {}

    def set_logger(self, logger: PipelineLogger) -> None:
        self._logger = logger

    def generate_construct_library(
        self,
        pathway_genes: List[str],
        base_promoter: str = "Ptac",
        base_rbs: str = "RBS_B0034",
        n_constructs: int = 48,
    ) -> List[Construct]:
        """
        Generate a library of genetic constructs with combinatorial
        promoter/RBS variations.
        """
        if self._logger:
            self._logger.info(
                "Generating construct library: %d constructs, genes=%s",
                n_constructs, pathway_genes,
            )

        constructs: List[Construct] = []
        for i in range(n_constructs):
            random.seed(i * 137)
            prom = random.choice(self.PROMOTER_POOL)
            rbs = random.choice(self.RBS_POOL)

            # Vary gene copy numbers
            gene_list = list(pathway_genes)
            if i % 3 == 0 and len(gene_list) > 1:
                gene_list.append(gene_list[0])  # Duplicate first gene

            # Predicted titer (simulated)
            base_titer = 1.0
            prom_strength = self.PROMOTER_POOL.index(prom) / len(self.PROMOTER_POOL)
            rbs_strength = self.RBS_POOL.index(rbs) / len(self.RBS_POOL)
            predicted = base_titer * (0.5 + prom_strength + rbs_strength) * random.uniform(0.5, 2.0)

            construct = Construct(
                construct_id=f"C{i+1:04d}",
                promoter=prom,
                rbs=rbs,
                genes=gene_list,
                predicted_titer=round(predicted, 3),
            )
            constructs.append(construct)

        self._all_constructs = constructs
        return constructs

    def simulate_screening_results(
        self,
        constructs: List[Construct],
        base_titer: float = 1.0,
        noise_std: float = 0.15,
    ) -> Dict[str, float]:
        """
        Simulate high-throughput screening results with realistic noise.
        """
        if self._logger:
            self._logger.info("Simulating HTS screening for %d constructs", len(constructs))

        results: Dict[str, float] = {}
        for construct in constructs:
            random.seed(hash(construct.construct_id))

            # Base titer influenced by promoter and RBS strength
            prom_idx = self.PROMOTER_POOL.index(construct.promoter)
            rbs_idx = self.RBS_POOL.index(construct.rbs)

            strength_factor = (prom_idx / len(self.PROMOTER_POOL)) * 0.6 + (rbs_idx / len(self.RBS_POOL)) * 0.4
            titer = base_titer * (0.5 + strength_factor) * construct.predicted_titer

            # Add experimental noise
            noise = random.gauss(1.0, noise_std)
            measured_titer = max(0.01, titer * noise)

            results[construct.construct_id] = round(measured_titer, 3)
            construct.actual_titer = results[construct.construct_id]

        self._screening_results = results

        if self._logger:
            best_id = max(results, key=results.get)
            self._logger.info(
                "Screening complete: best=%s (%.3f g/L), mean=%.3f g/L",
                best_id, results[best_id],
                sum(results.values()) / len(results),
            )

        return results

    def run_bayesian_optimization(
        self,
        constructs: List[Construct],
        results: Dict[str, float],
        n_candidates: int = 8,
    ) -> List[str]:
        """
        Use Bayesian optimisation with Expected Improvement (EI) acquisition
        function to select the next batch of constructs to test.
        """
        if self._logger:
            self._logger.info("Running Bayesian optimisation: %d candidates", n_candidates)

        if not results:
            return [c.construct_id for c in constructs[:n_candidates]]

        # Calculate EI for each construct
        ei_scores: Dict[str, float] = {}
        all_titers = list(results.values())
        f_best = max(all_titers)
        sigma = np.std(all_titers) if len(all_titers) > 1 else 0.1

        for construct in constructs:
            cid = construct.construct_id
            if cid not in results:
                # Unobserved: high uncertainty → high EI
                ei_scores[cid] = f_best + 2.0 * sigma
            else:
                # Observed: calculate EI
                mu = results[cid]
                if sigma > 0:
                    z = (mu - f_best) / sigma
                    ei = (mu - f_best) * norm.cdf(z) + sigma * norm.pdf(z)
                else:
                    ei = 0.0
                ei_scores[cid] = ei

        # Select top N by EI
        sorted_constructs = sorted(ei_scores.items(), key=lambda x: x[1], reverse=True)
        selected = [cid for cid, _ in sorted_constructs[:n_candidates]]

        if self._logger:
            self._logger.info("Selected %d candidates for next cycle", len(selected))

        return selected

    def run_dbtl_loop(
        self,
        pathway_genes: List[str],
        n_cycles: int = 3,
        constructs_per_cycle: int = 48,
        n_next_candidates: int = 8,
        base_titer: float = 1.0,
    ) -> List[DBTLCycle]:
        """
        Run the full DBTL loop for N cycles.

        Each cycle builds on the previous cycle's best results.
        """
        if self._logger:
            self._logger.info(
                "Starting DBTL loop: %d cycles, %d constructs/cycle",
                n_cycles, constructs_per_cycle,
            )

        cycles: List[DBTLCycle] = []
        current_base_titer = base_titer
        best_ever_titer = 0.0

        for cycle_num in range(1, n_cycles + 1):
            if self._logger:
                self._logger.info("=== DBTL Cycle %d ===", cycle_num)

            # DESIGN
            constructs = self.generate_construct_library(
                pathway_genes=pathway_genes,
                n_constructs=constructs_per_cycle,
            )

            # BUILD (simulated - just track construct IDs)
            if self._logger:
                self._logger.debug("Built %d constructs", len(constructs))

            # TEST
            results = self.simulate_screening_results(
                constructs=constructs,
                base_titer=current_base_titer,
                noise_std=0.15,
            )

            # LEARN
            next_candidates = self.run_bayesian_optimization(
                constructs=constructs,
                results=results,
                n_candidates=n_next_candidates,
            )

            # Track best
            best_id = max(results, key=results.get)
            best_titer = results[best_id]
            improvement = best_titer / best_ever_titer if best_ever_titer > 0 else 1.0
            best_ever_titer = max(best_ever_titer, best_titer)

            # Update base titer for next cycle
            current_base_titer = best_titer * 1.1  # 10% improvement expected

            cycle_result = DBTLCycle(
                cycle_number=cycle_num,
                constructs_tested=len(constructs),
                best_titer_g_per_l=round(best_titer, 3),
                best_construct_id=best_id,
                improvement_fold=round(improvement, 3),
                bo_next_candidates=next_candidates,
            )
            cycles.append(cycle_result)

            if self._logger:
                self._logger.info(
                    "Cycle %d: best=%s (%.3f g/L), improvement=%.2fx, next candidates=%d",
                    cycle_num, best_id, best_titer, improvement, len(next_candidates),
                )

        if self._logger:
            self._logger.info(
                "DBTL loop complete: %d cycles, best titer=%.3f g/L",
                n_cycles, best_ever_titer,
            )

        return cycles


# ---------------------------------------------------------------------------
# MAIN — smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DBTL Loop")
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--constructs", type=int, default=48)
    parser.add_argument("--genes", nargs="+", default=["crtE", "crtB", "crtI"])
    args = parser.parse_args()

    logger = PipelineLogger()
    logger.set_stage("4")

    orchestrator = DBTLOrchestrator()
    orchestrator.set_logger(logger)

    cycles = orchestrator.run_dbtl_loop(
        pathway_genes=args.genes,
        n_cycles=args.cycles,
        constructs_per_cycle=args.constructs,
    )

    os.makedirs("pipeline_output", exist_ok=True)
    with open("pipeline_output/dbtl_results.json", "w") as fh:
        json.dump([
            {
                "cycle_number": c.cycle_number,
                "constructs_tested": c.constructs_tested,
                "best_titer_g_per_l": c.best_titer_g_per_l,
                "best_construct_id": c.best_construct_id,
                "improvement_fold": c.improvement_fold,
                "bo_next_candidates": c.bo_next_candidates,
            }
            for c in cycles
        ], fh, indent=2, default=str)

    print(f"\n▶ DBTL Loop smoke test passed. {len(cycles)} cycles completed.")
    for c in cycles:
        print(f"  Cycle {c.cycle_number}: best={c.best_titer_g_per_l} g/L, improvement={c.improvement_fold:.2f}x")
