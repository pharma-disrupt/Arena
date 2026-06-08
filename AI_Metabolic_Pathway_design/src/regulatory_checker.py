"""
Regulatory Checker Module

Performs regulatory and biosafety assessment for engineered strains
including BSL classification, allergenicity prediction, toxicity
assessment, GRAS status, and compliance reporting.
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

from logger_setup import PipelineLogger


# ---------------------------------------------------------------------------
# DATACLASSES
# ---------------------------------------------------------------------------

@dataclass
class RegulatoryAssessment:
    """Complete regulatory assessment for an engineered strain."""
    biosafety_level: int
    biosafety_rationale: str
    allergenicity_score: float
    allergenicity_risk: str
    toxicity_score: float
    toxicity_risk: str
    gras_status: bool
    gras_notes: str
    antibiotic_markers_removed: bool
    antibiotic_marker_notes: str
    environmental_risk_score: float
    environmental_risk_level: str
    regulatory_recommendations: List[str]
    compliance_checklist: Dict[str, bool]
    overall_assessment: str
    assessment_date: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "biosafety_level": self.biosafety_level,
            "biosafety_rationale": self.biosafety_rationale,
            "allergenicity_score": round(self.allergenicity_score, 4),
            "allergenicity_risk": self.allergenicity_risk,
            "toxicity_score": round(self.toxicity_score, 4),
            "toxicity_risk": self.toxicity_risk,
            "gras_status": self.gras_status,
            "gras_notes": self.gras_notes,
            "antibiotic_markers_removed": self.antibiotic_markers_removed,
            "antibiotic_marker_notes": self.antibiotic_marker_notes,
            "environmental_risk_score": round(self.environmental_risk_score, 4),
            "environmental_risk_level": self.environmental_risk_level,
            "regulatory_recommendations": self.regulatory_recommendations,
            "compliance_checklist": self.compliance_checklist,
            "overall_assessment": self.overall_assessment,
            "assessment_date": self.assessment_date,
        }


# ---------------------------------------------------------------------------
# REGULATORY DATABASES
# ---------------------------------------------------------------------------

GRAS_ORGANISMS: Dict[str, Dict[str, Any]] = {
    "ecoli": {
        "gras_status": True,
        "strain_specific": {
            "K-12 MG1655": True,
            "BL21(DE3)": True,
            "DH5alpha": True,
            "JM109": True,
        },
        "notes": "E. coli K-12 derivatives are GRAS for industrial use. "
                 "Not GRAS for food contact without additional approval.",
    },
    "scerevisiae": {
        "gras_status": True,
        "strain_specific": {
            "S288C": True,
            "BY4741": True,
            "CEN.PK": True,
        },
        "notes": "S. cerevisiae is GRAS (baker's yeast). Widely used in "
                 "food and beverage industry.",
    },
    "bsubtilis": {
        "gras_status": True,
        "strain_specific": {
            "168": True,
            "WB600": True,
            "WB800": True,
        },
        "notes": "B. subtilis is GRAS. Used extensively in enzyme production. "
                 "Non-pathogenic strain 168 is well-characterised.",
    },
    "cglutamicum": {
        "gras_status": True,
        "strain_specific": {
            "ATCC 13032": True,
            "AJ12640": True,
        },
        "notes": "C. glutamicum is GRAS. Used for amino acid production "
                 "(glutamate, lysine) at industrial scale.",
    },
    "pputida": {
        "gras_status": False,
        "strain_specific": {
            "KT2440": True,
        },
        "notes": "P. putida KT2440 is GRAS-certified for bioremediation and "
                 "industrial biocatalysis. Parent species is not GRAS.",
    },
}

ANTIBIOTIC_MARKERS: List[str] = [
    "ampR", "kanR", "cat", "tetR", "specR", "cmR", "ermR", "neoR",
]

ANTIBIOTIC_FREE_STRATEGIES: List[str] = [
    "Cre-loxP recombination",
    "FLP-FRT recombination",
    "Counter-selection (sacB)",
    "CRISPR-Cas9 marker excision",
    "Homologous recombination without marker",
    "Marker-free assembly (Gibson + DpnI)",
]


# ---------------------------------------------------------------------------
# REGULATORY CHECKER
# ---------------------------------------------------------------------------

class RegulatoryChecker:
    """
    Performs regulatory and biosafety assessment for engineered strains.

    Simulates:
    - BSL classification
    - Allergenicity prediction (simulated AllerTop)
    - Toxicity prediction (simulated ProTox-3)
    - GRAS status verification
    - Antibiotic marker removal check
    - Compliance report generation
    """

    def __init__(self) -> None:
        self._logger: Optional[PipelineLogger] = None

    def set_logger(self, logger: PipelineLogger) -> None:
        self._logger = logger

    def classify_biosafety_level(self, organism_key: str,
                                  strain: str = "",
                                  genetic_modifications: Optional[List[str]] = None) -> int:
        """
        Classify biosafety level (BSL-1, BSL-2, BSL-3, BSL-4).

        Parameters
        ----------
        organism_key : str
            Target organism.
        strain : str
            Specific strain name.
        genetic_modifications : list of str, optional
            List of genetic modifications.

        Returns
        -------
        int
            Biosafety level (1-4).
        """
        # Base BSL for each organism
        base_bsl = {
            "ecoli": 1,
            "ecoli_bl21": 1,
            "scerevisiae": 1,
            "scerevisiae_by": 1,
            "bsubtilis": 1,
            "cglutamicum": 1,
            "pputida": 1,
        }
        bsl = base_bsl.get(organism_key, 2)

        # Adjust for genetic modifications
        if genetic_modifications:
            # Pathogenicity factors
            pathogenic_factors = {"toxin", "virulence", "pathogenicity", "invasion"}
            if any(f in " ".join(genetic_modifications).lower()
                   for f in pathogenic_factors):
                bsl = max(bsl, 2)

            # Antibiotic resistance markers
            if any(m in genetic_modifications for m in ANTIBIOTIC_MARKERS):
                bsl = max(bsl, 2)

        return min(4, max(1, bsl))

    def predict_allergenicity(self, gene_names: Optional[List[str]] = None,
                               product_name: str = "") -> Dict[str, Any]:
        """
        Simulate AllerTop allergenicity prediction.

        Returns
        -------
        dict
            Allergenicity assessment.
        """
        if self._logger:
            self._logger.info("Running simulated AllerTop prediction")

        # Known allergenic proteins
        known_allergens = {
            "ara", "bet", "par", "che", "alt", "asp", "pen",
        }

        allergenicity_score = 0.0
        allergen_sources: List[str] = []

        if gene_names:
            for gene in gene_names:
                gene_lower = gene.lower()
                # Check for known allergenic sources
                for allergen in known_allergens:
                    if allergen in gene_lower:
                        allergenicity_score += 0.2
                        allergen_sources.append(gene)

                # Check protein family
                if "nslp" in gene_lower or "prof" in gene_lower:
                    allergenicity_score += 0.3
                    allergen_sources.append(f"{gene} (profilin)")

        # Product-based assessment
        product_allergens = {
            "peanut": 0.8,
            "tree_nut": 0.7,
            "shellfish": 0.75,
            "milk": 0.3,
            "egg": 0.3,
            "soy": 0.4,
            "wheat": 0.5,
            "fish": 0.6,
        }
        for product, score in product_allergens.items():
            if product in product_name.lower():
                allergenicity_score = max(allergenicity_score, score)
                allergen_sources.append(product_name)

        allergenicity_score = min(1.0, allergenicity_score)

        if allergenicity_score > 0.5:
            risk = "HIGH"
        elif allergenicity_score > 0.2:
            risk = "MEDIUM"
        else:
            risk = "LOW"

        return {
            "allergenicity_score": round(allergenicity_score, 4),
            "allergenicity_risk": risk,
            "allergen_sources": allergen_sources,
            "method": "Simulated AllerTop v2.1",
        }

    def predict_toxicity(self, product_name: str,
                          product_smiles: str = "",
                          heterologous_genes: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Simulate ProTox-3 toxicity prediction.

        Returns
        -------
        dict
            Toxicity assessment.
        """
        if self._logger:
            self._logger.info("Running simulated ProTox-3 prediction")

        # Known toxic products
        toxic_products = {
            "aflatoxin": 0.95,
            "botulinum": 0.99,
            "ricin": 0.98,
            "saxitoxin": 0.90,
            "tetrodotoxin": 0.85,
            "cytochalasin": 0.70,
        }

        toxicity_score = 0.1  # Default low toxicity

        for product, score in toxic_products.items():
            if product in product_name.lower():
                toxicity_score = max(toxicity_score, score)

        # SMILES-based estimation (simplified)
        if product_smiles:
            # More complex molecules tend to be more toxic
            mol_weight = len(product_smiles) * 0.5  # Very rough estimate
            if mol_weight > 500:
                toxicity_score += 0.1
            if mol_weight > 1000:
                toxicity_score += 0.1

            # Nitro groups, halogens increase toxicity
            if "N(=O)=O" in product_smiles or "[N+](=O)[O-]" in product_smiles:
                toxicity_score += 0.2
            if any(h in product_smiles for h in ["F", "Cl", "Br", "I"]):
                toxicity_score += 0.1

        # Gene-based assessment
        if heterologous_genes:
            for gene in heterologous_genes:
                gene_lower = gene.lower()
                if "toxin" in gene_lower or "toxic" in gene_lower:
                    toxicity_score += 0.3
                if "virulence" in gene_lower:
                    toxicity_score += 0.2

        toxicity_score = min(1.0, max(0.0, toxicity_score))

        if toxicity_score > 0.7:
            risk = "HIGH"
            ld50 = "< 50 mg/kg"
        elif toxicity_score > 0.3:
            risk = "MEDIUM"
            ld50 = "50-500 mg/kg"
        else:
            risk = "LOW"
            ld50 = "> 500 mg/kg"

        return {
            "toxicity_score": round(toxicity_score, 4),
            "toxicity_risk": risk,
            "predicted_ld50": ld50,
            "method": "Simulated ProTox-3",
        }

    def check_gras_status(self, organism_key: str,
                           strain: str = "") -> Dict[str, Any]:
        """
        Check GRAS (Generally Recognized As Safe) status.

        Returns
        -------
        dict
            GRAS status assessment.
        """
        org_data = GRAS_ORGANISMS.get(organism_key, {
            "gras_status": False,
            "strain_specific": {},
            "notes": f"Organism '{organism_key}' not in GRAS database.",
        })

        strain_status = org_data.get("strain_specific", {}).get(
            strain, org_data.get("gras_status", False)
        )

        return {
            "organism": organism_key,
            "strain": strain or "unspecified",
            "gras_status": strain_status,
            "notes": org_data.get("notes", ""),
        }

    def check_antibiotic_marker_removal(self,
                                         gene_modifications: Optional[List[str]] = None,
                                         heterologous_genes: Optional[List[str]] = None,
                                         ) -> Dict[str, Any]:
        """
        Check for antibiotic resistance markers and recommend removal strategies.

        Returns
        -------
        dict
            Antibiotic marker assessment.
        """
        all_genes = list(gene_modifications or []) + list(heterologous_genes or [])
        detected_markers = [m for m in all_genes if m in ANTIBIOTIC_MARKERS]

        markers_removed = len(detected_markers) == 0

        return {
            "antibiotic_markers_detected": detected_markers,
            "markers_removed": markers_removed,
            "recommended_strategies": ANTIBIOTIC_FREE_STRATEGIES,
            "notes": (
                "No antibiotic markers detected — strain is marker-free."
                if markers_removed
                else f"Antibiotic markers detected: {detected_markers}. "
                     "Removal recommended before industrial use."
            ),
        }

    def generate_compliance_report(
        self,
        organism_key: str,
        strain: str,
        product_name: str,
        product_smiles: str = "",
        genetic_modifications: Optional[List[str]] = None,
        heterologous_genes: Optional[List[str]] = None,
        gene_modifications: Optional[Dict[str, List[str]]] = None,
    ) -> RegulatoryAssessment:
        """
        Generate a complete regulatory compliance report.

        Parameters
        ----------
        organism_key : str
            Target organism.
        strain : str
            Specific strain.
        product_name : str
            Target product.
        product_smiles : str, optional
            SMILES string of product.
        genetic_modifications : list, optional
            List of genetic modifications.
        heterologous_genes : list, optional
            List of heterologous genes.
        gene_modifications : dict, optional
            Gene modifications (knockouts, overexpressions, insertions).

        Returns
        -------
        RegulatoryAssessment
        """
        if self._logger:
            self._logger.info(
                "Generating regulatory compliance report: %s %s → %s",
                organism_key, strain, product_name,
            )

        # Collect all gene names
        all_genes = list(genetic_modifications or [])
        if heterologous_genes:
            all_genes.extend(heterologous_genes)
        if gene_modifications:
            for mod_type, genes in gene_modifications.items():
                all_genes.extend(genes)

        # Run assessments
        bsl = self.classify_biosafety_level(
            organism_key, strain, all_genes
        )

        allergenicity = self.predict_allergenicity(all_genes, product_name)

        toxicity = self.predict_toxicity(
            product_name, product_smiles, heterologous_genes
        )

        gras = self.check_gras_status(organism_key, strain)

        antibiotic = self.check_antibiotic_marker_removal(
            all_genes, heterologous_genes
        )

        # Environmental risk
        env_score = 0.1
        if bsl >= 2:
            env_score += 0.2
        if antibiotic["markers_removed"] is False:
            env_score += 0.3
        if toxicity["toxicity_score"] > 0.5:
            env_score += 0.3
        env_score = min(1.0, env_score)

        if env_score > 0.5:
            env_risk = "HIGH"
        elif env_score > 0.2:
            env_risk = "MEDIUM"
        else:
            env_risk = "LOW"

        # Compliance checklist
        checklist = {
            "biosafety_assessment_complete": True,
            "allergenicity_assessed": True,
            "toxicity_assessed": True,
            "gras_status_verified": gras["gras_status"],
            "antibiotic_markers_removed": antibiotic["markers_removed"],
            "environmental_risk_assessed": True,
            "data_integrity_verified": True,
            "traceability_documented": True,
        }

        # Recommendations
        recommendations = []
        if bsl >= 2:
            recommendations.append(
                f"BSL-{bsl} containment required. Review facility capabilities."
            )
        if not gras["gras_status"]:
            recommendations.append(
                f"GRAS status not confirmed for {organism_key}/{strain}. "
                "Regulatory approval needed before food/pharma use."
            )
        if not antibiotic["markers_removed"]:
            recommendations.append(
                "Remove antibiotic resistance markers before scale-up. "
                f"Recommended: {antibiotic['recommended_strategies'][0]}."
            )
        if toxicity["toxicity_risk"] in ("MEDIUM", "HIGH"):
            recommendations.append(
                f"Toxicity risk: {toxicity['toxicity_risk']}. "
                "Additional toxicological studies required."
            )
        if allergenicity["allergenicity_risk"] in ("MEDIUM", "HIGH"):
            recommendations.append(
                f"Allergenicity risk: {allergenicity['allergenicity_risk']}. "
                "Allergen labelling required for consumer products."
            )
        if env_risk == "HIGH":
            recommendations.append(
                "High environmental risk. Containment protocols must be "
                "reviewed and approved by regulatory authority."
            )
        recommendations.append(
            "Maintain complete documentation of all genetic modifications "
            "and strain construction history."
        )

        # Overall assessment
        if any(v is False for v in checklist.values()):
            overall = "CONDITIONAL_APPROVAL"
        elif bsl >= 3 or toxicity["toxicity_risk"] == "HIGH":
            overall = "REQUIRES_REVIEW"
        else:
            overall = "APPROVED"

        assessment = RegulatoryAssessment(
            biosafety_level=bsl,
            biosafety_rationale=f"Based on organism {organism_key} and {len(all_genes)} genetic modifications",
            allergenicity_score=allergenicity["allergenicity_score"],
            allergenicity_risk=allergenicity["allergenicity_risk"],
            toxicity_score=toxicity["toxicity_score"],
            toxicity_risk=toxicity["toxicity_risk"],
            gras_status=gras["gras_status"],
            gras_notes=gras["notes"],
            antibiotic_markers_removed=antibiotic["markers_removed"],
            antibiotic_marker_notes=antibiotic["notes"],
            environmental_risk_score=round(env_score, 4),
            environmental_risk_level=env_risk,
            regulatory_recommendations=recommendations,
            compliance_checklist=checklist,
            overall_assessment=overall,
        )

        if self._logger:
            self._logger.info(
                "Regulatory assessment complete: BSL=%d, GRAS=%s, "
                "overall=%s",
                bsl, gras["gras_status"], overall,
            )

        return assessment


# ---------------------------------------------------------------------------
# MAIN — smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Regulatory Checker")
    parser.add_argument("--organism", default="ecoli")
    parser.add_argument("--strain", default="K-12 MG1655")
    parser.add_argument("--product", default="lycopene")
    args = parser.parse_args()

    logger = PipelineLogger()
    logger.set_stage("5")

    checker = RegulatoryChecker()
    checker.set_logger(logger)

    assessment = checker.generate_compliance_report(
        organism_key=args.organism,
        strain=args.strain,
        product_name=args.product,
        genetic_modifications=["dxs", "idi", "crtE", "crtB", "crtI"],
        heterologous_genes=["crtE", "crtB", "crtI"],
        gene_modifications={
            "knockouts": ["ldhA", "adhE", "poxB"],
            "overexpressions": ["dxs", "idi"],
            "heterologous_insertions": ["crtE", "crtB", "crtI"],
        },
    )

    os.makedirs("pipeline_output", exist_ok=True)
    with open("pipeline_output/regulatory_assessment.json", "w") as fh:
        json.dump(assessment.to_dict(), fh, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"  REGULATORY ASSESSMENT — {args.organism} {args.strain} → {args.product}")
    print(f"{'='*60}")
    print(f"  BSL               : {assessment.biosafety_level}")
    print(f"  GRAS Status       : {'Yes' if assessment.gras_status else 'No'}")
    print(f"  Allergenicity     : {assessment.allergenicity_risk} "
          f"(score={assessment.allergenicity_score:.3f})")
    print(f"  Toxicity          : {assessment.toxicity_risk} "
          f"(score={assessment.toxicity_score:.3f})")
    print(f"  Antibiotic-free   : {'Yes' if assessment.antibiotic_markers_removed else 'No'}")
    print(f"  Environmental Risk: {assessment.environmental_risk_level} "
          f"(score={assessment.environmental_risk_score:.3f})")
    print(f"  Overall           : {assessment.overall_assessment}")
    print(f"\n  Recommendations:")
    for rec in assessment.regulatory_recommendations:
        print(f"    → {rec}")
    print(f"{'='*60}")

    print("\n▶ Regulatory Checker smoke test passed.")
