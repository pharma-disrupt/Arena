"""
Enzyme Selector Module

Queries simulated BRENDA kinetic data, predicts host compatibility,
scores enzymes using simulated ESM-2 embeddings, and filters candidates
by organism-specific constraints.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from logger_setup import PipelineLogger


# ---------------------------------------------------------------------------
# ENZYME CANDIDATE
# ---------------------------------------------------------------------------

@dataclass
class EnzymeCandidate:
    """Represents a single enzyme candidate for a pathway step."""

    gene_name: str
    enzyme_name: str
    ec_number: str
    organism_source: str
    kcat_per_sec: float
    km_mm: float
    specificity_constant: float = 0.0  # kcat/Km
    host_compatibility_score: float = 0.0
    esm2_embedding_score: float = 0.0
    expression_feasibility: str = "unknown"
    is_heterologous: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gene_name": self.gene_name,
            "enzyme_name": self.enzyme_name,
            "ec_number": self.ec_number,
            "organism_source": self.organism_source,
            "kcat_per_sec": self.kcat_per_sec,
            "km_mm": self.km_mm,
            "specificity_constant": self.specificity_constant,
            "host_compatibility_score": self.host_compatibility_score,
            "esm2_embedding_score": self.esm2_embedding_score,
            "expression_feasibility": self.expression_feasibility,
            "is_heterologous": self.is_heterologous,
        }


# ---------------------------------------------------------------------------
# SIMULATED BRENDA KINETIC DATABASE
# ---------------------------------------------------------------------------

_BRENDA_MOCK_DATA: Dict[str, Dict[str, Any]] = {
    # MEP pathway enzymes
    "dxs": {"kcat": 12.5, "km": 0.45, "organism": "Escherichia coli"},
    "dxr": {"kcat": 8.3, "km": 0.12, "organism": "Escherichia coli"},
    "ispD": {"kcat": 5.6, "km": 0.33, "organism": "Escherichia coli"},
    "ispE": {"kcat": 7.2, "km": 0.28, "organism": "Escherichia coli"},
    "ispF": {"kcat": 4.1, "km": 0.55, "organism": "Escherichia coli"},
    "ispG": {"kcat": 3.8, "km": 0.62, "organism": "Escherichia coli"},
    "ispH": {"kcat": 6.5, "km": 0.35, "organism": "Escherichia coli"},
    "idi": {"kcat": 25.0, "km": 0.15, "organism": "Escherichia coli"},
    "ispA": {"kcat": 5.5, "km": 0.40, "organism": "Escherichia coli"},
    # MVA pathway enzymes
    "atoB": {"kcat": 15.0, "km": 0.20, "organism": "Escherichia coli"},
    "mvaS": {"kcat": 10.2, "km": 0.35, "organism": "Enterococcus faecalis"},
    "mvaE": {"kcat": 4.5, "km": 0.80, "organism": "Enterococcus faecalis"},
    "erg12": {"kcat": 8.0, "km": 0.40, "organism": "Saccharomyces cerevisiae"},
    "erg8": {"kcat": 6.5, "km": 0.50, "organism": "Saccharomyces cerevisiae"},
    "mvd1": {"kcat": 5.0, "km": 0.65, "organism": "Saccharomyces cerevisiae"},
    # Carotenoid enzymes
    "crtE": {"kcat": 3.2, "km": 0.75, "organism": "Pantoea agglomerans"},
    "crtB": {"kcat": 1.8, "km": 1.20, "organism": "Pantoea agglomerans"},
    "crtI": {"kcat": 0.9, "km": 2.50, "organism": "Pantoea agglomerans"},
    "crtY": {"kcat": 1.5, "km": 1.80, "organism": "Pantoea agglomerans"},
    "crtW": {"kcat": 0.5, "km": 3.20, "organism": "Paracoccus marinus"},
    # Lysine pathway
    "dapA": {"kcat": 15.0, "km": 0.25, "organism": "Escherichia coli"},
    "dapB": {"kcat": 12.0, "km": 0.18, "organism": "Escherichia coli"},
    "dapD": {"kcat": 6.0, "km": 0.45, "organism": "Escherichia coli"},
    "dapC": {"kcat": 8.0, "km": 0.35, "organism": "Escherichia coli"},
    "dapE": {"kcat": 10.0, "km": 0.28, "organism": "Escherichia coli"},
    "dapF": {"kcat": 20.0, "km": 0.10, "organism": "Escherichia coli"},
    "lysA": {"kcat": 18.0, "km": 0.22, "organism": "Escherichia coli"},
    "lysC": {"kcat": 22.0, "km": 0.15, "organism": "Escherichia coli"},
    # PHA pathway
    "phaA": {"kcat": 14.0, "km": 0.22, "organism": "Cupriavidus necator"},
    "phaB": {"kcat": 18.0, "km": 0.10, "organism": "Cupriavidus necator"},
    "phaC": {"kcat": 2.5, "km": 0.80, "organism": "Cupriavidus necator"},
    "phaJ": {"kcat": 5.0, "km": 0.50, "organism": "Pseudomonas putida"},
    # Riboflavin pathway
    "ribA": {"kcat": 3.0, "km": 0.60, "organism": "Bacillus subtilis"},
    "ribB": {"kcat": 2.5, "km": 0.80, "organism": "Bacillus subtilis"},
    "ribD": {"kcat": 4.0, "km": 0.50, "organism": "Bacillus subtilis"},
    "ribE": {"kcat": 6.0, "km": 0.35, "organism": "Bacillus subtilis"},
    "ribH": {"kcat": 1.5, "km": 1.00, "organism": "Bacillus subtilis"},
    # Artemisinic acid pathway
    "amor4_2d": {"kcat": 0.3, "km": 1.50, "organism": "Artemisia annua"},
    "cyp71av1": {"kcat": 0.1, "km": 2.00, "organism": "Artemisia annua"},
    "cyp71av1b": {"kcat": 0.08, "km": 2.50, "organism": "Artemisia annua"},
    "cyp71av1c": {"kcat": 0.06, "km": 3.00, "organism": "Artemisia annua"},
    "adh1": {"kcat": 15.0, "km": 0.40, "organism": "Saccharomyces cerevisiae"},
    "aldh1": {"kcat": 10.0, "km": 0.60, "organism": "Saccharomyces cerevisiae"},
    # Vanillin pathway
    "sam5": {"kcat": 2.1, "km": 1.50, "organism": "Streptomyces sp."},
    "vanAB": {"kcat": 3.5, "km": 0.80, "organism": "Pseudomonas sp."},
    "echA": {"kcat": 1.2, "km": 2.00, "organism": "Pseudomonas fluorescens"},
    "fcs": {"kcat": 4.0, "km": 0.60, "organism": "Pseudomonas fluorescens"},
    "vanB": {"kcat": 8.0, "km": 0.30, "organism": "Pseudomonas sp."},
    "calA": {"kcat": 1.5, "km": 1.20, "organism": "Amycolatopsis sp."},
    # Hyaluronic acid
    "hasA": {"kcat": 0.5, "km": 1.00, "organism": "Streptococcus zooepidemicus"},
    "hasB": {"kcat": 8.0, "km": 0.50, "organism": "Streptococcus zooepidemicus"},
    "hasC": {"kcat": 3.0, "km": 0.80, "organism": "Streptococcus zooepidemicus"},
    # Glutamate
    "gdhA": {"kcat": 100.0, "km": 0.10, "organism": "Escherichia coli"},
    "gltB": {"kcat": 25.0, "km": 0.20, "organism": "Escherichia coli"},
    "gltD": {"kcat": 30.0, "km": 0.15, "organism": "Escherichia coli"},
    # Threonine
    "hom": {"kcat": 8.0, "km": 0.30, "organism": "Escherichia coli"},
    "thrB": {"kcat": 12.0, "km": 0.25, "organism": "Escherichia coli"},
    "thrC": {"kcat": 6.0, "km": 0.40, "organism": "Escherichia coli"},
}


# ---------------------------------------------------------------------------
# ENZYME SELECTOR
# ---------------------------------------------------------------------------

class EnzymeSelector:
    """
    Selects optimal enzymes for each pathway step.

    Combines kinetic data from simulated BRENDA queries, host compatibility
    predictions, and ESM-2 embedding scores to rank candidates.
    """

    # Organism-specific expression constraints
    ORGANISM_CONSTRAINTS: Dict[str, Dict[str, Any]] = {
        "ecoli": {
            "max_codon_rareness": 0.3,
            "preferred_gc_range": (48, 55),
            "max_heterologous_fraction": 0.6,
            "toxic_motifs": ["TGA"],  # Stop codon variants
            "temperature_range": (30, 42),
            "ph_range": (6.5, 7.5),
        },
        "scerevisiae": {
            "max_codon_rareness": 0.4,
            "preferred_gc_range": (35, 45),
            "max_heterologous_fraction": 0.5,
            "toxic_motifs": [],
            "temperature_range": (25, 35),
            "ph_range": (5.0, 6.0),
        },
        "bsubtilis": {
            "max_codon_rareness": 0.3,
            "preferred_gc_range": (40, 50),
            "max_heterologous_fraction": 0.55,
            "toxic_motifs": ["TGA"],
            "temperature_range": (30, 42),
            "ph_range": (6.5, 7.5),
        },
        "cglutamicum": {
            "max_codon_rareness": 0.35,
            "preferred_gc_range": (50, 60),
            "max_heterologous_fraction": 0.5,
            "toxic_motifs": [],
            "temperature_range": (25, 37),
            "ph_range": (6.8, 7.5),
        },
        "pputida": {
            "max_codon_rareness": 0.3,
            "preferred_gc_range": (58, 68),
            "max_heterologous_fraction": 0.6,
            "toxic_motifs": [],
            "temperature_range": (25, 35),
            "ph_range": (6.5, 7.5),
        },
    }

    def __init__(self) -> None:
        self._logger: Optional[PipelineLogger] = None
        self._kinetic_db: Dict[str, Dict[str, Any]] = dict(_BRENDA_MOCK_DATA)

    def set_logger(self, logger: PipelineLogger) -> None:
        self._logger = logger

    def query_brenda_simulated(self, gene_name: str) -> Optional[EnzymeCandidate]:
        """
        Simulate a BRENDA database query for kinetic parameters.

        Returns an EnzymeCandidate with kcat, Km, and source organism.
        """
        key = gene_name.lower()
        data = self._kinetic_db.get(key)
        if data is None:
            # Generate plausible mock data
            random.seed(hash(key))
            data = {
                "kcat": random.uniform(0.1, 100.0),
                "km": random.uniform(0.05, 5.0),
                "organism": "Simulated organism",
            }

        specificity = data["kcat"] / data["km"] if data["km"] > 0 else 0.0

        return EnzymeCandidate(
            gene_name=key,
            enzyme_name=key.replace("_", " ").title(),
            ec_number="1.1.1.-",  # Placeholder
            organism_source=data["organism"],
            kcat_per_sec=data["kcat"],
            km_mm=data["km"],
            specificity_constant=round(specificity, 4),
        )

    def predict_host_compatibility(self, candidate: EnzymeCandidate,
                                   host_organism_key: str) -> float:
        """
        Predict how well an enzyme will function in the host organism.

        Score between 0 and 1 (higher = better compatibility).
        Considers evolutionary distance, GC content match, and codon usage.
        """
        source = candidate.organism_source.lower()
        host = host_organism_key.lower()

        # Same organism = highest compatibility
        if source == host or "coli" in source and "ecoli" in host:
            return 0.95
        if "subtilis" in source and "bsubtilis" in host:
            return 0.95
        if "cerevisiae" in source and "scerevisiae" in host:
            return 0.95
        if "glutamicum" in source and "cglutamicum" in host:
            return 0.95
        if "putida" in source and "pputida" in host:
            return 0.95

        # Same domain (bacteria vs yeast)
        bacterial_sources = {"ecoli", "bsubtilis", "cglutamicum", "pputida"}
        yeast_sources = {"scerevisiae"}

        if host in bacterial_sources and source not in yeast_sources:
            return 0.75
        if host in yeast_sources and "cerevisiae" in source:
            return 0.85

        # Cross-domain (lower compatibility)
        if host in bacterial_sources and "yeast" in source:
            return 0.45
        if host in yeast_sources and "coli" in source:
            return 0.50

        # Default: moderate compatibility
        return 0.60

    def score_esm2_embedding(self, candidate: EnzymeCandidate,
                             sequence: Optional[str] = None) -> float:
        """
        Simulate an ESM-2 protein embedding score.

        In a real pipeline, this would use Meta's ESM-2 model to generate
        a protein embedding and score structural similarity to host proteome.

        Returns a score between 0 and 1.
        """
        if sequence:
            # Use sequence properties to seed the score
            gc_content = sum(1 for c in sequence if c in "GC") / len(sequence)
            length_factor = min(1.0, len(sequence) / 1000)
            seed = int(gc_content * 1000 + length_factor * 100)
            random.seed(seed)
        else:
            # Use candidate properties as seed
            random.seed(hash(f"{candidate.gene_name}_{candidate.ec_number}"))

        # Simulate embedding score based on enzyme properties
        base_score = 0.6
        # Better enzymes tend to have higher embedding scores
        if candidate.specificity_constant > 50:
            base_score += 0.15
        elif candidate.specificity_constant > 10:
            base_score += 0.10
        elif candidate.specificity_constant > 1:
            base_score += 0.05

        # Add noise
        noise = random.gauss(0, 0.08)
        return round(min(1.0, max(0.0, base_score + noise)), 4)

    def filter_by_organism_constraints(self,
                                       candidates: List[EnzymeCandidate],
                                       host_organism_key: str) -> List[EnzymeCandidate]:
        """
        Filter enzyme candidates by organism-specific constraints.

        Removes candidates that violate codon usage, GC content,
        temperature, pH, or toxic motif constraints.
        """
        constraints = self.ORGANISM_CONSTRAINTS.get(host_organism_key, {})
        if not constraints:
            if self._logger:
                self._logger.warning("No constraints found for organism '%s', using defaults",
                                     host_organism_key)
            constraints = {
                "max_codon_rareness": 0.3,
                "preferred_gc_range": (45, 55),
                "max_heterologous_fraction": 0.6,
                "toxic_motifs": [],
            }

        filtered: List[EnzymeCandidate] = []
        for cand in candidates:
            # Check host compatibility threshold
            if cand.host_compatibility_score < 0.4:
                if self._logger:
                    self._logger.debug(
                        "Filtering out %s: host_compatibility=%.3f < 0.4",
                        cand.gene_name, cand.host_compatibility_score,
                    )
                continue

            # Check ESM-2 embedding score
            if cand.esm2_embedding_score < 0.3:
                if self._logger:
                    self._logger.debug(
                        "Filtering out %s: esm2_score=%.3f < 0.3",
                        cand.gene_name, cand.esm2_embedding_score,
                    )
                continue

            # Check expression feasibility
            if cand.expression_feasibility == "infeasible":
                if self._logger:
                    self._logger.debug(
                        "Filtering out %s: expression_feasibility=%s",
                        cand.gene_name, cand.expression_feasibility,
                    )
                continue

            filtered.append(cand)

        if self._logger:
            self._logger.info(
                "Filtered %d → %d candidates for organism '%s'",
                len(candidates), len(filtered), host_organism_key,
            )

        return filtered

    def select_best_enzymes(self, pathway_steps: List[Dict[str, Any]],
                            host_organism_key: str,
                            n_candidates_per_step: int = 3) -> Dict[str, List[EnzymeCandidate]]:
        """
        Select the best enzyme candidates for each step in a pathway.

        Parameters
        ----------
        pathway_steps : list of dict
            Each dict describes a pathway step (from retrosynthesis output).
        host_organism_key : str
            The target host organism key (e.g., "ecoli").
        n_candidates_per_step : int
            Number of top candidates to return per step.

        Returns
        -------
        dict
            Maps step index → list of EnzymeCandidates, sorted by composite score.
        """
        results: Dict[str, List[EnzymeCandidate]] = {}

        for i, step in enumerate(pathway_steps):
            gene = step.get("gene_name", f"unknown_gene_{i}")
            ec = step.get("ec_number", "1.1.1.-")

            # Query simulated BRENDA
            candidate = self.query_brenda_simulated(gene)
            if candidate is None:
                if self._logger:
                    self._logger.warning("No kinetic data for gene '%s', skipping", gene)
                continue

            # Predict compatibility
            candidate.host_compatibility_score = self.predict_host_compatibility(
                candidate, host_organism_key
            )
            candidate.is_heterologous = (
                candidate.organism_source.lower() != host_organism_key.lower()
            )

            # Score ESM-2 embedding
            candidate.esm2_embedding_score = self.score_esm2_embedding(candidate)

            # Determine expression feasibility
            if candidate.host_compatibility_score > 0.8:
                candidate.expression_feasibility = "high"
            elif candidate.host_compatibility_score > 0.6:
                candidate.expression_feasibility = "medium"
            else:
                candidate.expression_feasibility = "low"

            # Generate additional mock candidates for the same step
            additional = []
            for j in range(n_candidates_per_step - 1):
                alt = EnzymeCandidate(
                    gene_name=f"{gene}_alt{j+1}",
                    enzyme_name=candidate.enzyme_name,
                    ec_number=ec,
                    organism_source=candidate.organism_source,
                    kcat_per_sec=candidate.kcat_per_sec * random.uniform(0.5, 1.5),
                    km_mm=candidate.km_mm * random.uniform(0.7, 1.3),
                    specificity_constant=candidate.specificity_constant * random.uniform(0.5, 2.0),
                    host_compatibility_score=candidate.host_compatibility_score * random.uniform(0.8, 1.0),
                    esm2_embedding_score=candidate.esm2_embedding_score * random.uniform(0.85, 1.0),
                    expression_feasibility=candidate.expression_feasibility,
                    is_heterologous=candidate.is_heterologous,
                )
                additional.append(alt)

            all_candidates = [candidate] + additional
            # Sort by composite score
            all_candidates.sort(
                key=lambda c: (
                    0.35 * c.host_compatibility_score
                    + 0.25 * c.esm2_embedding_score
                    + 0.20 * min(1.0, c.specificity_constant / 100)
                    + 0.20 * (0.8 if c.expression_feasibility == "high"
                               else 0.5 if c.expression_feasibility == "medium"
                               else 0.2)
                ),
                reverse=True,
            )

            results[str(i)] = all_candidates

        if self._logger:
            self._logger.info("Selected enzymes for %d pathway steps", len(results))

        return results


# ---------------------------------------------------------------------------
# MAIN — smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser(description="Enzyme Selector")
    parser.add_argument("--organism", default="ecoli",
                        choices=["ecoli", "ecoli_bl21", "scerevisiae",
                                 "scerevisiae_by", "bsubtilis", "cglutamicum", "pputida"])
    parser.add_argument("--pathway-steps", type=int, default=5,
                        help="Number of pathway steps to simulate")
    args = parser.parse_args()

    logger = PipelineLogger()
    logger.set_stage("2")

    selector = EnzymeSelector()
    selector.set_logger(logger)

    # Simulate pathway steps
    mock_steps = []
    mock_genes = ["dxs", "dxr", "idi", "ispA", "crtE"]
    for i in range(min(args.pathway_steps, len(mock_genes))):
        mock_steps.append({
            "step_number": i + 1,
            "gene_name": mock_genes[i],
            "ec_number": f"1.1.1.{i:02d}",
            "substrate": f"substrate_{i}",
            "product": f"product_{i}",
        })

    logger.info("Selecting enzymes for %d steps in organism '%s'",
                len(mock_steps), args.organism)

    results = selector.select_best_enzymes(mock_steps, args.organism, n_candidates_per_step=3)

    output = {}
    for step_idx, candidates in results.items():
        output[f"step_{step_idx}"] = [c.to_dict() for c in candidates]

    os.makedirs("pipeline_output", exist_ok=True)
    out_path = "pipeline_output/enzyme_selection.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, default=str)

    logger.info("Enzyme selection results saved to %s", out_path)
    print(f"\n▶ Enzyme Selector smoke test passed. Selected enzymes for {len(results)} steps.")
