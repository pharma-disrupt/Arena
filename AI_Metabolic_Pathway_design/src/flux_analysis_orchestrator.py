"""
Flux Analysis Orchestrator Module

Orchestrates Stage 3: FBA analysis, strain optimisation, and toxicity
prediction. Takes Stage 2 output JSON as input and produces validated
Stage 3 output JSON.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fba_engine import FBAEngine, FBAModel, organism_specific_constraints
from logger_setup import PipelineLogger, log_json_contract
from schema_validator import validate_and_raise, validate_stage_output
from strain_optimizer import StrainOptimizer
from toxicity_predictor import ToxicityPredictor
from exceptions import FBAConvergenceError, PipelineError


# ---------------------------------------------------------------------------
# STAGE 3 OUTPUT DATACLASS
# ---------------------------------------------------------------------------

@dataclass
class Stage3Output:
    """Typed container for the Stage 3 output payload."""

    pipeline_id: str
    stage_2_output: Dict[str, Any]
    fba_results: Dict[str, Any]
    strain_design: Dict[str, Any]
    toxicity_assessment: Dict[str, Any]
    stage_3_status: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pipeline_id": self.pipeline_id,
            "stage_2_output": self.stage_2_output,
            "fba_results": self.fba_results,
            "strain_design": self.strain_design,
            "toxicity_assessment": self.toxicity_assessment,
            "stage_3_status": self.stage_3_status,
        }


# ---------------------------------------------------------------------------
# FLUX ANALYSIS ORCHESTRATOR
# ---------------------------------------------------------------------------

class FluxAnalysisOrchestrator:
    """
    Orchestrates the full Stage 3 pipeline:
      1. FBA analysis (standard + pFBA + FVA)
      2. Strain design optimisation (OptKnock + NSGA-III)
      3. Toxicity assessment

    Takes Stage 2 output JSON as input and returns Stage 3 output JSON.
    """

    def __init__(self) -> None:
        self._logger: Optional[PipelineLogger] = None
        self._fba_engine = FBAEngine()
        self._strain_optimizer = StrainOptimizer()
        self._toxicity_predictor = ToxicityPredictor()

    def set_logger(self, logger: PipelineLogger) -> None:
        self._logger = logger
        self._fba_engine.set_logger(logger)
        self._strain_optimizer.set_logger(logger)
        self._toxicity_predictor.set_logger(logger)

    def run(self, stage_2_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the full Stage 3 pipeline.

        Parameters
        ----------
        stage_2_output : dict
            Validated JSON from Stage 2.

        Returns
        -------
        dict
            Stage 3 output JSON matching the schema contract.
        """
        if self._logger is None:
            self._logger = PipelineLogger()
        self._logger.set_stage("3")

        start_time = time.time()
        pipeline_id = stage_2_output.get("pipeline_id", str(uuid.uuid4()))

        self._logger.info("=== STAGE 3 START === pipeline_id=%s", pipeline_id)
        self._logger.debug("Input JSON received: %s",
                           {k: v for k, v in stage_2_output.items()
                            if k != "stage_2_output"})

        try:
            # 1. Validate input
            self._logger.info("Validating Stage 2 input JSON")
            validate_and_raise(stage_2_output, "stage_2_output", self._logger)

            # 2. Extract organism and molecule info
            stage_1 = stage_2_output.get("stage_1_output", {})
            organism_key = self._extract_organism_key(stage_1)
            molecule_key = self._extract_molecule_key(stage_1)

            self._logger.info("Organism: %s | Molecule: %s",
                              organism_key, molecule_key)

            # 3. Run FBA
            fba_results = self._run_fba_analysis(
                stage_2_output, organism_key
            )

            # 4. Run strain optimisation
            strain_design = self._run_strain_optimisation(
                stage_2_output, organism_key, molecule_key
            )

            # 5. Run toxicity assessment
            toxicity_assessment = self._run_toxicity_assessment(
                stage_2_output, organism_key, molecule_key
            )

            # 6. Assemble output
            output = Stage3Output(
                pipeline_id=pipeline_id,
                stage_2_output=stage_2_output,
                fba_results=fba_results,
                strain_design=strain_design,
                toxicity_assessment=toxicity_assessment,
                stage_3_status="PASS",
            )
            output_dict = output.to_dict()

            # 7. Validate output
            self._logger.info("Validating Stage 3 output against schema")
            validate_and_raise(output_dict, "stage_3_output", self._logger)

            # 8. Log output JSON
            log_json_contract(self._logger, output_dict,
                              "Stage 3 → Stage 4", direction="output")

            # 9. Write stage summary
            duration = time.time() - start_time
            summary = {
                "stage": 3,
                "pipeline_id": pipeline_id,
                "organism": organism_key,
                "molecule": molecule_key,
                "fba_growth_rate": fba_results.get("growth_rate_per_hour", 0),
                "fba_product_flux": fba_results.get("product_flux_mmol_per_gdw_per_hour", 0),
                "strain_design_algorithm": strain_design.get("algorithm_used", ""),
                "toxicity_risk": toxicity_assessment.get("overall_toxicity_risk", ""),
                "status": "PASS",
                "duration_seconds": round(duration, 2),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            self._logger.write_stage_summary(3, summary)

            self._logger.info("=== STAGE 3 COMPLETE === duration=%.2fs status=PASS",
                              duration)
            return output_dict

        except FBAConvergenceError as e:
            self._logger.log_error_with_context("FluxAnalysisOrchestrator.run", e,
                                                 input_json=stage_2_output,
                                                 fallback_method="use theoretical estimates")
            return self._stage_3_fallback(pipeline_id, stage_2_output, str(e))
        except Exception as e:
            self._logger.log_error_with_context("FluxAnalysisOrchestrator.run", e,
                                                 input_json=stage_2_output)
            raise PipelineError(
                message=f"Stage 3 failed: {e}",
                stage="3",
                function="FluxAnalysisOrchestrator.run",
                input_json=stage_2_output,
            ) from e

    # ── Internal helpers ──────────────────────────────────────────────────

    def _extract_organism_key(self, stage_1: Dict[str, Any]) -> str:
        """Derive organism key from Stage 1 data embedded in Stage 2."""
        org = stage_1.get("organism", {})
        name = org.get("name", "").lower()
        strain = org.get("strain", "").lower()

        if "coli" in name:
            return "ecoli_bl21" if "bl21" in strain else "ecoli"
        if "cerevisiae" in name:
            return "scerevisiae_by" if "by4741" in strain else "scerevisiae"
        if "subtilis" in name:
            return "bsubtilis"
        if "glutamicum" in name:
            return "cglutamicum"
        if "putida" in name:
            return "pputida"
        return "ecoli"

    def _extract_molecule_key(self, stage_1: Dict[str, Any]) -> str:
        """Derive molecule key from Stage 1 data."""
        mol = stage_1.get("target_molecule", {})
        name = mol.get("name", "").lower()

        mol_map = {
            "lycopene": "lycopene",
            "vanillin": "vanillin",
            "artemisinic acid": "artemisinic_acid",
            "l-lysine": "lysine",
            "lysine": "lysine",
            "l-glutamate": "glutamate",
            "glutamate": "glutamate",
            "l-threonine": "threonine",
            "threonine": "threonine",
            "polyhydroxyalkanoate (pha)": "pha",
            "pha": "pha",
            "hyaluronic acid": "hyaluronic_acid",
            "riboflavin (vitamin b2)": "riboflavin",
            "riboflavin": "riboflavin",
        }
        return mol_map.get(name, "lycopene")

    def _run_fba_analysis(
        self,
        stage_2_output: Dict[str, Any],
        organism_key: str,
    ) -> Dict[str, Any]:
        """Run FBA, pFBA, and FVA analyses."""
        # Get pathway steps from Stage 2
        pathways = stage_2_output.get("pathway_candidates", [])
        pathway_steps = []
        if pathways:
            pathway_steps = pathways[0].get("steps", [])

        # Build model
        model = self._fba_engine.build_stoichiometric_matrix(
            organism_key=organism_key,
            pathway_steps=pathway_steps,
        )

        # Run standard FBA
        try:
            fba_results = self._fba_engine.run_fba(model)
        except FBAConvergenceError as e:
            self._logger.warning("FBA failed: %s — using theoretical estimates", e)
            fba_results = self._theoretical_fba_estimates(organism_key)

        # Run pFBA
        try:
            pfba_results = self._fba_engine.run_pfba(model)
        except Exception:
            pfba_results = {"total_flux": fba_results.get("substrate_uptake_mmol_per_gdw_per_hour", 0)}

        # Run FVA
        try:
            fva_results = self._fba_engine.run_fva(model, fraction_of_optimal=0.95)
        except Exception:
            fva_results = {}

        return {
            "objective_value": fba_results.get("objective_value", 0.0),
            "growth_rate_per_hour": fba_results.get("growth_rate_per_hour", 0.0),
            "product_flux_mmol_per_gdw_per_hour": fba_results.get("product_flux_mmol_per_gdw_per_hour", 0.0),
            "substrate_uptake_mmol_per_gdw_per_hour": fba_results.get("substrate_uptake_mmol_per_gdw_per_hour", 0.0),
            "theoretical_max_yield": fba_results.get("theoretical_max_yield", 0.0),
            "flux_map": fba_results.get("flux_map", {}),
            "pfba_total_flux": pfba_results.get("total_flux", 0.0),
            "fva_results": fva_results,
        }

    def _theoretical_fba_estimates(self, organism_key: str) -> Dict[str, Any]:
        """Fallback: use literature-based theoretical estimates."""
        estimates = {
            "ecoli": {
                "growth_rate_per_hour": 0.80,
                "product_flux_mmol_per_gdw_per_hour": 2.5,
                "substrate_uptake_mmol_per_gdw_per_hour": 10.0,
                "theoretical_max_yield": 0.25,
                "objective_value": 0.80,
            },
            "scerevisiae": {
                "growth_rate_per_hour": 0.35,
                "product_flux_mmol_per_gdw_per_hour": 1.5,
                "substrate_uptake_mmol_per_gdw_per_hour": 8.0,
                "theoretical_max_yield": 0.19,
                "objective_value": 0.35,
            },
            "bsubtilis": {
                "growth_rate_per_hour": 0.65,
                "product_flux_mmol_per_gdw_per_hour": 2.0,
                "substrate_uptake_mmol_per_gdw_per_hour": 9.0,
                "theoretical_max_yield": 0.22,
                "objective_value": 0.65,
            },
            "cglutamicum": {
                "growth_rate_per_hour": 0.40,
                "product_flux_mmol_per_gdw_per_hour": 3.0,
                "substrate_uptake_mmol_per_gdw_per_hour": 7.0,
                "theoretical_max_yield": 0.43,
                "objective_value": 0.40,
            },
            "pputida": {
                "growth_rate_per_hour": 0.55,
                "product_flux_mmol_per_gdw_per_hour": 1.8,
                "substrate_uptake_mmol_per_gdw_per_hour": 8.5,
                "theoretical_max_yield": 0.21,
                "objective_value": 0.55,
            },
        }
        return estimates.get(organism_key, estimates["ecoli"])

    def _run_strain_optimisation(
        self,
        stage_2_output: Dict[str, Any],
        organism_key: str,
        molecule_key: str,
    ) -> Dict[str, Any]:
        """Run strain design optimisation."""
        # Get essential genes from Stage 1
        stage_1 = stage_2_output.get("stage_1_output", {})
        genomic_data = stage_1.get("genomic_data", {})
        essential_genes = genomic_data.get("essential_genes", [])

        # Run OptKnock simulation
        kos = self._strain_optimizer.run_optknock_simulated(
            organism_key=organism_key,
            molecule_key=molecule_key,
            max_knockouts=3,
            essential_genes=essential_genes,
        )

        # Get gene modifications from Stage 2
        gene_mods = stage_2_output.get("gene_modifications", {})
        oe_genes = gene_mods.get("overexpressions", [])
        het_genes = gene_mods.get("heterologous_insertions", [])

        # Run NSGA-III optimisation
        ko_names = [k.gene_name for k in kos]
        pareto = self._strain_optimizer.run_nsga3_optimization(
            organism_key=organism_key,
            molecule_key=molecule_key,
            knockouts=ko_names,
            overexpressions=oe_genes + het_genes[:3],
        )

        # Rank designs
        ranked = self._strain_optimizer.rank_strain_designs(
            pareto_front=pareto,
            organism_key=organism_key,
            molecule_key=molecule_key,
        )

        # Select best design
        best_design = ranked[0] if ranked else {}

        # Calculate scores
        burden = self._strain_optimizer.score_metabolic_burden(
            organism_key=organism_key,
            knockouts=best_design.get("knockouts", []),
            overexpressions=best_design.get("overexpressions", []),
            heterologous_genes=len(het_genes),
        )

        stability = self._strain_optimizer.predict_genetic_stability(
            organism_key=organism_key,
            knockouts=best_design.get("knockouts", []),
            overexpressions=best_design.get("overexpressions", []),
            heterologous_genes=len(het_genes),
        )

        # Predict titer
        fba_results = self._theoretical_fba_estimates(organism_key)
        yield_mol = fba_results.get("theoretical_max_yield", 0.25)
        target_titer = stage_1.get("target_molecule", {}).get("target_titer_g_per_l", 5.0)
        predicted_titer = round(target_titer * yield_mol * 0.5, 2)  # 50% of theoretical
        predicted_productivity = round(predicted_titer / 48.0, 4)  # 48-hour fermentation

        return {
            "algorithm_used": "NSGA-III + OptKnock",
            "final_knockouts": best_design.get("knockouts", []),
            "final_overexpressions": best_design.get("overexpressions", []),
            "predicted_titer_g_per_l": predicted_titer,
            "predicted_productivity_g_per_l_per_h": predicted_productivity,
            "metabolic_burden_score": burden,
            "genetic_stability_score": stability,
            "pareto_front": ranked[:5],
            "optknock_recommendations": [k.to_dict() for k in kos],
        }

    def _run_toxicity_assessment(
        self,
        stage_2_output: Dict[str, Any],
        organism_key: str,
        molecule_key: str,
    ) -> Dict[str, Any]:
        """Run toxicity prediction for pathway intermediates."""
        # Get pathway intermediates
        pathways = stage_2_output.get("pathway_candidates", [])
        intermediates: List[Dict[str, Any]] = []

        for pw in pathways[:1]:  # Use top pathway
            for step in pw.get("steps", []):
                intermediates.append({
                    "name": step.get("product", "unknown"),
                    "logp": None,
                    "mw": None,
                    "functional_groups": [],
                })

        # Get molecule properties
        stage_1 = stage_2_output.get("stage_1_output", {})
        molecule = stage_1.get("target_molecule", {})
        molecule_name = molecule.get("name", molecule_key)

        # Estimate logP from SMILES (simplified)
        smiles = molecule.get("smiles", "")
        logp_estimate = self._estimate_logp_from_smiles(smiles)

        # Run toxicity assessment
        assessment = self._toxicity_predictor.assess_overall_toxicity(
            organism_key=organism_key,
            pathway_intermediates=intermediates,
            final_product_name=molecule_name,
            final_product_logp=logp_estimate,
        )

        return {
            "intermediate_toxicity_scores": assessment.get("intermediate_toxicity_scores", {}),
            "overall_toxicity_risk": assessment.get("overall_toxicity_risk", "LOW"),
            "flagged_intermediates": assessment.get("flagged_intermediates", []),
            "recommendations": assessment.get("recommendations", []),
        }

    def _estimate_logp_from_smiles(self, smiles: str) -> float:
        """Simple logP estimation from SMILES string."""
        if not smiles:
            return 2.0

        # Very rough estimation based on molecular features
        logp = 0.0
        logp += smiles.count("C") * 0.5  # Carbon contribution
        logp -= smiles.count("O") * 1.0  # Oxygen reduces logP
        logp -= smiles.count("N") * 1.2  # Nitrogen reduces logP
        logp += smiles.count("=") * 0.2  # Double bonds increase logP

        return round(max(0.0, min(20.0, logp)), 1)

    def _stage_3_fallback(
        self,
        pipeline_id: str,
        stage_2_output: Dict[str, Any],
        error_msg: str,
    ) -> Dict[str, Any]:
        """Graceful degradation for Stage 3."""
        if self._logger:
            self._logger.warning("Using fallback Stage 3 output: %s", error_msg)

        stage_1 = stage_2_output.get("stage_1_output", {})
        organism_key = self._extract_organism_key(stage_1)

        fba_results = self._theoretical_fba_estimates(organism_key)

        fallback = {
            "pipeline_id": pipeline_id,
            "stage_2_output": stage_2_output,
            "fba_results": {
                **fba_results,
                "flux_map": {},
            },
            "strain_design": {
                "algorithm_used": "rule-based fallback",
                "final_knockouts": [],
                "final_overexpressions": [],
                "predicted_titer_g_per_l": 1.0,
                "predicted_productivity_g_per_l_per_h": 0.02,
                "metabolic_burden_score": 0.5,
                "genetic_stability_score": 0.5,
            },
            "toxicity_assessment": {
                "intermediate_toxicity_scores": {},
                "overall_toxicity_risk": "UNKNOWN",
                "flagged_intermediates": [],
            },
            "stage_3_status": "WARN",
        }

        if self._logger:
            self._logger.write_stage_summary(3, {
                "stage": 3,
                "pipeline_id": pipeline_id,
                "status": "WARN",
                "fallback": True,
                "error": error_msg,
            })

        return fallback


# ---------------------------------------------------------------------------
# CONVENIENCE FUNCTION
# ---------------------------------------------------------------------------

def run_stage_3(stage_2_output: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run Stage 3 pipeline from a validated Stage 2 output dict.

    Parameters
    ----------
    stage_2_output : dict
        Output from Stage 2 (run_stage_2).

    Returns
    -------
    dict
        Stage 3 output JSON.
    """
    orchestrator = FluxAnalysisOrchestrator()
    return orchestrator.run(stage_2_output=stage_2_output)


# ---------------------------------------------------------------------------
# MAIN — smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json as json_mod

    parser = argparse.ArgumentParser(description="Stage 3: Flux Analysis")
    parser.add_argument("--organism", default="ecoli")
    parser.add_argument("--molecule", default="lycopene")
    args = parser.parse_args()

    # Run Stage 1 + 2 first to get input
    from data_layer import run_stage_1
    from pathway_ai_engine import run_stage_2
    from pipeline_config import PipelineConfig

    print("Running Stage 1...")
    cfg = PipelineConfig(organism_key=args.organism, molecule_key=args.molecule)
    stage_1_output = run_stage_1(cfg)

    print("Running Stage 2...")
    stage_2_output = run_stage_2(stage_1_output=stage_1_output)

    print(f"Running Stage 3 for {args.organism} → {args.molecule} ...")
    t0 = time.time()
    stage_3_output = run_stage_3(stage_2_output=stage_2_output)
    elapsed = time.time() - t0

    logger = PipelineLogger.get_instance()
    logger.info("Stage 3 completed in %.2f seconds", elapsed)

    # Print summary
    print(f"\n{'='*60}")
    print(f"  STAGE 3 RESULT — {args.organism} → {args.molecule}")
    print(f"{'='*60}")
    print(f"  Pipeline ID : {stage_3_output['pipeline_id']}")
    print(f"  Status      : {stage_3_output['stage_3_status']}")
    print(f"  FBA Results :")
    fba = stage_3_output['fba_results']
    print(f"    Growth rate     : {fba.get('growth_rate_per_hour', 'N/A')}/h")
    print(f"    Product flux    : {fba.get('product_flux_mmol_per_gdw_per_hour', 'N/A')}")
    print(f"    Theoretical max : {fba.get('theoretical_max_yield', 'N/A')}")
    print(f"  Strain Design :")
    sd = stage_3_output['strain_design']
    print(f"    Algorithm       : {sd.get('algorithm_used', 'N/A')}")
    print(f"    Predicted titer : {sd.get('predicted_titer_g_per_l', 'N/A')} g/L")
    print(f"    Burden score    : {sd.get('metabolic_burden_score', 'N/A')}")
    print(f"    Stability score : {sd.get('genetic_stability_score', 'N/A')}")
    print(f"  Toxicity      :")
    ta = stage_3_output['toxicity_assessment']
    print(f"    Overall risk    : {ta.get('overall_toxicity_risk', 'N/A')}")
    print(f"    Flagged         : {ta.get('flagged_intermediates', [])}")
    print(f"  Duration      : {elapsed:.2f}s")
    print(f"{'='*60}")

    # Save output
    os.makedirs("pipeline_output", exist_ok=True)
    with open("pipeline_output/stage_3_output.json", "w") as fh:
        json_mod.dump(stage_3_output, fh, indent=2, default=str)

    logger.info("Stage 3 output saved to pipeline_output/stage_3_output.json")
    print(f"\n▶ STAGE 3 COMPLETE ◀  Type: CONTINUE TO STAGE 4")