"""
Promoter & RBS Designer Module

Provides organism-specific promoter/RBS libraries, designs expression
cassettes, and predicts expression levels (TPM estimates).
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from logger_setup import PipelineLogger


# ---------------------------------------------------------------------------
# PROMOTER LIBRARIES  (organism → list of promoters with relative strengths)
# ---------------------------------------------------------------------------

PROMOTER_LIBRARY: Dict[str, List[Dict[str, Any]]] = {
    "ecoli": [
        {"name": "Ptac", "strength": 1.0, "inducible": True, "inducer": "IPTG"},
        {"name": "Ptrc", "strength": 0.95, "inducible": True, "inducer": "IPTG"},
        {"name": "Plac", "strength": 0.3, "inducible": True, "inducer": "IPTG"},
        {"name": "ParaBAD", "strength": 0.8, "inducible": True, "inducer": "arabinose"},
        {"name": "PT7", "strength": 1.0, "inducible": True, "inducer": "T7 RNA pol"},
        {"name": "PlacUV5", "strength": 0.6, "inducible": True, "inducer": "IPTG"},
        {"name": "Pbad", "strength": 0.75, "inducible": True, "inducer": "arabinose"},
        {"name": "PrhaBAD", "strength": 0.7, "inducible": True, "inducer": "rhamnose"},
        {"name": "PJ23100", "strength": 0.4, "inducible": False},
        {"name": "PJ23119", "strength": 1.0, "inducible": False},
    ],
    "scerevisiae": [
        {"name": "PGAL1", "strength": 1.0, "inducible": True, "inducer": "galactose"},
        {"name": "PGAL10", "strength": 0.9, "inducible": True, "inducer": "galactose"},
        {"name": "PTEF1", "strength": 0.8, "inducible": False},
        {"name": "PGPD", "strength": 0.85, "inducible": False},
        {"name": "PADC1", "strength": 0.7, "inducible": False},
        {"name": "PCYC1", "strength": 0.65, "inducible": False},
        {"name": "PTPI1", "strength": 0.5, "inducible": True, "inducer": "glucose repression"},
        {"name": "PMET25", "strength": 0.6, "inducible": True, "inducer": "methionine"},
    ],
    "bsubtilis": [
        {"name": "Pveg", "strength": 0.8, "inducible": False},
        {"name": "P43", "strength": 0.9, "inducible": False},
        {"name": "Phyperspank", "strength": 1.0, "inducible": True, "inducer": "IPTG"},
        {"name": "Pgrac", "strength": 0.7, "inducible": True, "inducer": "IPTG"},
        {"name": "PspoVG", "strength": 0.3, "inducible": False},
        {"name": "PaprE", "strength": 0.6, "inducible": False},
        {"name": "PsacB", "strength": 0.5, "inducible": True, "inducer": "sucrose"},
    ],
    "cglutamicum": [
        {"name": "Ptac", "strength": 0.9, "inducible": True, "inducer": "IPTG"},
        {"name": "Psod", "strength": 0.7, "inducible": False},
        {"name": "PgapA", "strength": 0.85, "inducible": False},
        {"name": "PfdhF", "strength": 0.5, "inducible": True, "inducer": "formate"},
        {"name": "Ptuf", "strength": 0.75, "inducible": False},
        {"name": "Pcg0929", "strength": 0.6, "inducible": False},
        {"name": "Pncgl0049", "strength": 0.4, "inducible": False},
    ],
    "pputida": [
        {"name": "Ptac", "strength": 0.9, "inducible": True, "inducer": "IPTG"},
        {"name": "Pbad", "strength": 0.8, "inducible": True, "inducer": "arabinose"},
        {"name": "Plac", "strength": 0.5, "inducible": True, "inducer": "IPTG"},
        {"name": "Pm", "strength": 0.7, "inducible": True, "inducer": "m-toluate"},
        {"name": "Pu", "strength": 1.0, "inducible": True, "inducer": "m-xylene"},
        {"name": "Pm/XylS", "strength": 0.85, "inducible": True, "inducer": "m-toluate"},
        {"name": "PrpoD", "strength": 0.6, "inducible": False},
    ],
}

# RBS strength library (relative translation initiation rates)
RBS_LIBRARY: Dict[str, List[Dict[str, Any]]] = {
    "ecoli": [
        {"name": "RBS_B0034", "strength": 1.0},
        {"name": "RBS_B0030", "strength": 0.5},
        {"name": "RBS_B0032", "strength": 0.75},
        {"name": "RBS_B0031", "strength": 0.25},
        {"name": "RBS_B0033", "strength": 1.5},
        {"name": "RBS_B0064", "strength": 0.1},
    ],
    "scerevisiae": [
        {"name": "RBS_ScTEF1", "strength": 1.0},
        {"name": "RBS_ScPGK1", "strength": 0.8},
        {"name": "RBS_ScTDH3", "strength": 0.9},
    ],
    "bsubtilis": [
        {"name": "RBS_BsPveg", "strength": 1.0},
        {"name": "RBS_BsP43", "strength": 0.7},
        {"name": "RBS_BsPhyperspank", "strength": 0.85},
    ],
    "cglutamicum": [
        {"name": "RBS_CgPgapA", "strength": 1.0},
        {"name": "RBS_CgPsod", "strength": 0.75},
        {"name": "RBS_CgPtac", "strength": 0.9},
    ],
    "pputida": [
        {"name": "RBS_PpPtac", "strength": 1.0},
        {"name": "RBS_PpPu", "strength": 0.85},
        {"name": "RBS_PpPbad", "strength": 0.7},
    ],
}


# ---------------------------------------------------------------------------
# PROMOTER-RBS DESIGNER
# ---------------------------------------------------------------------------

class PromoterRBSDesigner:
    """
    Designs expression cassettes (promoter + RBS + gene + terminator)
    and predicts expression levels.
    """

    # Terminator sequences (shortened representations)
    TERMINATORS: Dict[str, str] = {
        "ecoli": "T7Te",
        "scerevisiae": "CYC1t",
        "bsubtilis": "B0015",
        "cglutamicum": "T7Te",
        "pputida": "B0015",
    }

    def __init__(self, organism_key: str = "ecoli") -> None:
        self._organism_key = organism_key
        self._promoters = PROMOTER_LIBRARY.get(organism_key, PROMOTER_LIBRARY["ecoli"])
        self._rbs_list = RBS_LIBRARY.get(organism_key, RBS_LIBRARY["ecoli"])
        self._terminator = self.TERMINATORS.get(organism_key, "B0015")
        self._logger: Optional[PipelineLogger] = None

    def set_logger(self, logger: PipelineLogger) -> None:
        self._logger = logger

    def design_expression_cassette(self,
                                   gene_name: str,
                                   promoter: Optional[str] = None,
                                   rbs: Optional[str] = None,
                                   desired_expression_tpm: float = 1000.0,
                                   ) -> Dict[str, Any]:
        """
        Design a complete expression cassette for a given gene.

        Parameters
        ----------
        gene_name : str
            Name of the gene to express.
        promoter : str, optional
            Specific promoter name. Auto-selected if None.
        rbs : str, optional
            Specific RBS name. Auto-selected if None.
        desired_expression_tpm : float
            Target TPM (Transcripts Per Million).

        Returns
        -------
        dict
            Cassette design JSON.
        """
        # Select promoter
        if promoter:
            prom_obj = next((p for p in self._promoters if p["name"] == promoter), None)
            if prom_obj is None:
                if self._logger:
                    self._logger.warning(
                        "Promoter '%s' not found for %s, auto-selecting",
                        promoter, self._organism_key,
                    )
                prom_obj = self._select_promoter(desired_expression_tpm)
        else:
            prom_obj = self._select_promoter(desired_expression_tpm)

        # Select RBS
        if rbs:
            rbs_obj = next((r for r in self._rbs_list if r["name"] == rbs), None)
            if rbs_obj is None:
                if self._logger:
                    self._logger.warning(
                        "RBS '%s' not found for %s, auto-selecting",
                        rbs, self._organism_key,
                    )
                rbs_obj = self._rbs_list[0]
        else:
            rbs_obj = self._rbs_list[0]

        # Predict expression level
        predicted_tpm = self.predict_expression_level(
            prom_obj["strength"], rbs_obj["strength"]
        )

        cassette = {
            "organism": self._organism_key,
            "gene_name": gene_name,
            "promoter": prom_obj["name"],
            "promoter_strength": prom_obj["strength"],
            "promoter_inducible": prom_obj.get("inducible", False),
            "promoter_inducer": prom_obj.get("inducer"),
            "rbs": rbs_obj["name"],
            "rbs_strength": rbs_obj["strength"],
            "terminator": self._terminator,
            "cassette_structure": (
                f"5'-[{prom_obj['name']}]--[RBS:{rbs_obj['name']}]--"
                f"[{gene_name}]--[Terminator:{self._terminator}]-3'"
            ),
            "predicted_expression_tpm": round(predicted_tpm, 1),
            "desired_expression_tpm": desired_expression_tpm,
            "expression_ratio": round(predicted_tpm / desired_expression_tpm, 3),
        }

        if self._logger:
            self._logger.debug(
                "Designed cassette for %s: promoter=%s (str=%.2f), "
                "RBS=%s (str=%.2f), predicted TPM=%.1f",
                gene_name, prom_obj["name"], prom_obj["strength"],
                rbs_obj["name"], rbs_obj["strength"], predicted_tpm,
            )

        return cassette

    def predict_expression_level(self, promoter_strength: float,
                                 rbs_strength: float) -> float:
        """
        Predict expression level in TPM based on promoter and RBS strengths.

        Uses a simplified model: TPM ∝ promoter_strength × RBS_strength
        with empirical scaling factor.
        """
        # Base expression level (simulated from calibration data)
        base_tpm = 5000.0

        # Promoter contribution (log-linear relationship)
        promoter_effect = math.log10(promoter_strength * 10 + 1) / math.log10(11)

        # RBS contribution
        rbs_effect = rbs_strength

        # Combined prediction with biological noise
        random.seed(hash(f"{promoter_strength}_{rbs_strength}_{self._organism_key}"))
        noise = random.gauss(1.0, 0.1)

        predicted_tpm = base_tpm * promoter_effect * rbs_effect * noise
        return max(10.0, min(50000.0, predicted_tpm))

    def _select_promoter(self, desired_tpm: float) -> Dict[str, Any]:
        """
        Auto-select a promoter whose strength matches the desired TPM.
        """
        if not self._promoters:
            return {"name": "Pdefault", "strength": 0.5, "inducible": False}

        # Normalise desired TPM to 0-1 range
        normalised = min(1.0, desired_tpm / 5000.0)

        # Find closest match
        best = min(self._promoters,
                   key=lambda p: abs(p["strength"] - normalised))
        return best


# ---------------------------------------------------------------------------
# MAIN — smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser(description="Promoter & RBS Designer")
    parser.add_argument("--organism", default="ecoli",
                        choices=["ecoli", "ecoli_bl21", "scerevisiae",
                                 "scerevisiae_by", "bsubtilis", "cglutamicum", "pputida"])
    parser.add_argument("--gene", default="crtE",
                        help="Target gene name")
    parser.add_argument("--desired-tpm", type=float, default=1000.0,
                        help="Desired expression level in TPM")
    args = parser.parse_args()

    logger = PipelineLogger()
    logger.set_stage("2")

    designer = PromoterRBSDesigner(args.organism)
    designer.set_logger(logger)

    cassette = designer.design_expression_cassette(
        gene_name=args.gene,
        desired_expression_tpm=args.desired_tpm,
    )

    logger.info("Expression cassette designed:")
    logger.info("  Organism   : %s", cassette["organism"])
    logger.info("  Gene       : %s", cassette["gene_name"])
    logger.info("  Promoter   : %s (strength=%.2f, inducible=%s)",
                cassette["promoter"], cassette["promoter_strength"],
                cassette["promoter_inducible"])
    logger.info("  RBS        : %s (strength=%.2f)",
                cassette["rbs"], cassette["rbs_strength"])
    logger.info("  Terminator : %s", cassette["terminator"])
    logger.info("  Predicted TPM: %.1f", cassette["predicted_expression_tpm"])
    logger.info("  Structure  : %s", cassette["cassette_structure"])

    os.makedirs("pipeline_output", exist_ok=True)
    with open("pipeline_output/cassette_design.json", "w") as fh:
        json.dump(cassette, fh, indent=2)

    logger.info("Cassette design saved to pipeline_output/cassette_design.json")
    print(f"\n▶ Promoter/RBS Designer smoke test passed. Predicted TPM={cassette['predicted_expression_tpm']:.1f}")
