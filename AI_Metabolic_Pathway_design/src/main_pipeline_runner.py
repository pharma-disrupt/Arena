"""
Master Pipeline Runner (Stage 5 Entry Point)

Orchestrates ALL 5 stages of the metabolic pathway design pipeline.
Handles JSON passing between stages, error catching, and report generation.

Usage:
    python main_pipeline_runner.py --organism ecoli --molecule lycopene --cycles 3
"""

from __future__ import annotations

import argparse
import json
import logging as py_logging
import math
import os
import sys
import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# ── Stage imports ────────────────────────────────────────────────────────
from pipeline_config import PipelineConfig, ORGANISM_CHOICES, MOLECULE_CHOICES
from logger_setup import PipelineLogger
from data_layer import run_stage_1
from pathway_ai_engine import run_stage_2
from flux_analysis_orchestrator import run_stage_3
from dbtl_orchestrator import run_stage_4
from scaleup_engine import ScaleUpEngine
from downstream_processor import DownstreamProcessor
from regulatory_checker import RegulatoryChecker
from pipeline_report_generator import ReportGenerator, PipelineReport
from exceptions import PipelineError


# ---------------------------------------------------------------------------
# PIPELINE RUNNER
# ---------------------------------------------------------------------------

class PipelineRunner:
    """
    Master orchestrator that runs all 5 pipeline stages sequentially,
    passing validated JSON between each stage.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.logger = PipelineLogger(config)
        self.errors: List[Dict[str, Any]] = []
        self.stage_outputs: Dict[str, Any] = {}
        self.start_time: float = 0.0

    def run_full_pipeline(self) -> PipelineReport:
        """
        Execute the complete 5-stage pipeline.

        Returns
        -------
        PipelineReport
            Comprehensive report with all stage outputs and recommendations.
        """
        self.start_time = time.time()
        self.logger.set_stage("1")
        self.logger.info("=" * 70)
        self.logger.info("  SYNTHETIC BIOLOGY METABOLIC PATHWAY PIPELINE")
        self.logger.info("  Organism : %s (%s)", self.config.organism_name,
                         self.config.organism_strain)
        self.logger.info("  Molecule : %s", self.config.molecule_name)
        self.logger.info("  DBTL     : %d cycles", self.config.dbtl_cycles)
        self.logger.info("=" * 70)

        # ── STAGE 1: Core Infrastructure + Data Layer ────────────────────
        self.logger.set_stage("1")
        try:
            self.logger.info(">>> EXECUTING STAGE 1: Data Layer")
            stage_1_output = run_stage_1(self.config)
            self.stage_outputs["stage_1"] = stage_1_output
            self.logger.info(">>> STAGE 1 COMPLETE: status=%s",
                             stage_1_output.get("stage_1_status"))
        except Exception as e:
            self._handle_error("1", "run_stage_1", e)
            stage_1_output = self._stage_1_fallback_output()
            self.stage_outputs["stage_1"] = stage_1_output

        # ── STAGE 2: Pathway Prediction + AI Engine ──────────────────────
        self.logger.set_stage("2")
        try:
            self.logger.info(">>> EXECUTING STAGE 2: Pathway Prediction")
            stage_2_output = run_stage_2(
                stage_1_output=stage_1_output,
                max_pathways=self.config.max_pathway_candidates,
                dbtl_cycles=self.config.dbtl_cycles,
            )
            self.stage_outputs["stage_2"] = stage_2_output
            self.logger.info(">>> STAGE 2 COMPLETE: status=%s",
                             stage_2_output.get("stage_2_status"))
        except Exception as e:
            self._handle_error("2", "run_stage_2", e)
            stage_2_output = self._stage_2_fallback_output(stage_1_output)
            self.stage_outputs["stage_2"] = stage_2_output

        # ── STAGE 3: Flux Analysis + Strain Optimization ─────────────────
        self.logger.set_stage("3")
        try:
            self.logger.info(">>> EXECUTING STAGE 3: Flux Analysis")
            stage_3_output = run_stage_3(stage_2_output=stage_2_output)
            self.stage_outputs["stage_3"] = stage_3_output
            self.logger.info(">>> STAGE 3 COMPLETE: status=%s",
                             stage_3_output.get("stage_3_status"))
        except Exception as e:
            self._handle_error("3", "run_stage_3", e)
            stage_3_output = self._stage_3_fallback_output(stage_2_output)
            self.stage_outputs["stage_3"] = stage_3_output

        # ── STAGE 4: DBTL Loop + Fermentation Simulation ─────────────────
        self.logger.set_stage("4")
        try:
            self.logger.info(">>> EXECUTING STAGE 4: DBTL + Fermentation")
            stage_4_output = run_stage_4(
                stage_3_output=stage_3_output,
                n_dbtl_cycles=self.config.dbtl_cycles,
            )
            self.stage_outputs["stage_4"] = stage_4_output
            self.logger.info(">>> STAGE 4 COMPLETE: status=%s",
                             stage_4_output.get("stage_4_status"))
        except Exception as e:
            self._handle_error("4", "run_stage_4", e)
            stage_4_output = self._stage_4_fallback_output(stage_3_output)
            self.stage_outputs["stage_4"] = stage_4_output

        # ── STAGE 5: Scale-up + Downstream + Regulatory ──────────────────
        self.logger.set_stage("5")
        try:
            self.logger.info(">>> EXECUTING STAGE 5: Scale-up + Downstream")
            stage_5_output = self.run_stage_5(stage_4_output)
            self.stage_outputs["stage_5"] = stage_5_output
            self.logger.info(">>> STAGE 5 COMPLETE: status=%s",
                             stage_5_output.get("stage_5_status", "PASS"))
        except Exception as e:
            self._handle_error("5", "run_stage_5", e)
            stage_5_output = self._stage_5_fallback_output(stage_4_output)
            self.stage_outputs["stage_5"] = stage_5_output

        # ── Compile final report ─────────────────────────────────────────
        total_duration = time.time() - self.start_time
        self.logger.info("=" * 70)
        self.logger.info("  PIPELINE COMPLETE")
        self.logger.info("  Total duration: %.2fs", total_duration)
        self.logger.info("  Errors: %d", len(self.errors))
        self.logger.info("=" * 70)

        # Generate and export reports
        report_generator = ReportGenerator()
        report_generator.set_logger(self.logger)

        report = report_generator.compile_all_stages(
            stage_1_output=self.stage_outputs.get("stage_1"),
            stage_2_output=self.stage_outputs.get("stage_2"),
            stage_3_output=self.stage_outputs.get("stage_3"),
            stage_4_output=self.stage_outputs.get("stage_4"),
            stage_5_output=self.stage_outputs.get("stage_5"),
            errors=self.errors,
        )
        report.total_duration_seconds = total_duration

        # Export reports
        output_dir = self.config.output_dir
        paths = report_generator.export_to_files(report, output_dir)

        self.logger.info("Reports exported:")
        for fmt, path in paths.items():
            self.logger.info("  %s: %s", fmt, path)

        # Print summary
        self._print_summary(report, total_duration)

        return report

    def run_stage_5(self, stage_4_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run Stage 5: Scale-up + Downstream + Regulatory.

        Parameters
        ----------
        stage_4_output : dict
            Validated JSON from Stage 4.

        Returns
        -------
        dict
            Stage 5 output JSON with scale-up, downstream, and regulatory info.
        """
        start_time = time.time()
        pipeline_id = stage_4_output.get("pipeline_id", str(uuid.uuid4()))

        self.logger.info("=== STAGE 5 START === pipeline_id=%s", pipeline_id)

        # Validate input
        from schema_validator import validate_and_raise
        validate_and_raise(stage_4_output, "stage_4_output", self.logger)

        # Extract organism and molecule info
        stage_1 = (stage_4_output.get("stage_3_output", {})
                   .get("stage_2_output", {})
                   .get("stage_1_output", {}))
        organism_info = stage_1.get("organism", {})
        molecule_info = stage_1.get("target_molecule", {})
        organism_key = self.config.organism_key
        molecule_key = self.config.molecule_key

        strain_design = (stage_4_output.get("stage_3_output", {})
                         .get("strain_design", {}))
        gene_mods = (stage_4_output.get("stage_2_output", {})
                     .get("gene_modifications", {}))

        # ── Scale-up ────────────────────────────────────────────────────
        scaleup_engine = ScaleUpEngine()
        scaleup_engine.set_logger(self.logger)

        product_logp = self.config.molecule_config.logp
        scale_results = scaleup_engine.run_scale_up_cascade(
            organism_key=organism_key,
            product_name=self.config.molecule_name,
            product_logp=product_logp,
            scale_levels=self.config.scale_levels,
            lab_agitation_rpm=stage_4_output.get(
                "optimal_fermentation_conditions", {}).get("agitation_rpm", 300),
            lab_aeration_vvm=stage_4_output.get(
                "optimal_fermentation_conditions", {}).get("aeration_vvm", 1.0),
        )

        # ── Downstream Processing ───────────────────────────────────────
        downstream = DownstreamProcessor()
        downstream.set_logger(self.logger)

        fermentation_titer = stage_4_output.get(
            "fermentation_simulation", {}).get("final_titer_g_per_l", 1.0)

        purification = downstream.simulate_purification_train(
            product_name=molecule_key,
            organism_key=organism_key,
            starting_titer_g_per_l=fermentation_titer,
            starting_volume_l=200.0,  # Pilot scale
        )

        harvest_strategy = downstream.organism_specific_harvest_strategy(
            organism_key, molecule_key
        )

        # ── Regulatory Assessment ───────────────────────────────────────
        checker = RegulatoryChecker()
        checker.set_logger(self.logger)

        heterologous_genes = gene_mods.get("heterologous_insertions", [])
        all_modifications = (
            gene_mods.get("knockouts", [])
            + gene_mods.get("overexpressions", [])
            + heterologous_genes
        )

        regulatory = checker.generate_compliance_report(
            organism_key=organism_key,
            strain=organism_info.get("strain", ""),
            product_name=self.config.molecule_name,
            product_smiles=molecule_info.get("smiles", ""),
            genetic_modifications=all_modifications,
            heterologous_genes=heterologous_genes,
            gene_modifications=gene_mods,
        )

        # ── Assemble Stage 5 Output ─────────────────────────────────────
        stage_5_output = {
            "pipeline_id": pipeline_id,
            "stage_4_output": stage_4_output,
            "scale_up": {
                "cascade": [s.to_dict() for s in scale_results],
                "total_yield_loss_percent": round(
                    sum(s.yield_loss_percent for s in scale_results), 2
                ),
                "max_scale_l": max(s.volume_l for s in scale_results),
            },
            "downstream_processing": {
                "purification_train": purification,
                "harvest_strategy": harvest_strategy,
                "estimated_cost_per_kg_usd": purification.get("cost_per_kg_usd", 0),
                "overall_recovery_percent": purification.get(
                    "overall_recovery_percent", 0),
            },
            "regulatory_assessment": regulatory.to_dict(),
            "stage_5_status": "PASS",
        }

        # Note: No schema validation for stage_5_output as it's the final output
        self.logger.info("Stage 5 output assembled successfully")

        # Write stage summary
        duration = time.time() - start_time
        self.logger.write_stage_summary(5, {
            "stage": 5,
            "pipeline_id": pipeline_id,
            "organism": organism_key,
            "molecule": molecule_key,
            "scale_levels": [s.volume_l for s in scale_results],
            "purification_recovery": purification.get("overall_recovery_percent", 0),
            "regulatory_status": regulatory.overall_assessment,
            "status": "PASS",
            "duration_seconds": round(duration, 2),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        self.logger.info("=== STAGE 5 COMPLETE === duration=%.2fs status=PASS", duration)

        return stage_5_output

    # ── Error handling ──────────────────────────────────────────────────────

    def _handle_error(self, stage: str, function: str,
                      exception: Exception) -> None:
        """Log and record an error."""
        tb = traceback.format_exc()
        error_record = {
            "stage": stage,
            "function": function,
            "exception_type": type(exception).__name__,
            "message": str(exception),
            "traceback": tb,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "fallback_attempted": True,
        }
        self.errors.append(error_record)
        self.logger.error("Exception in Stage %s, %s: %s: %s",
                          stage, function, type(exception).__name__, exception)
        self.logger.error("Traceback:\n%s", tb)
        self.logger.warning("Attempting fallback for Stage %s", stage)

    # ── Fallback outputs ───────────────────────────────────────────────────

    def _stage_1_fallback_output(self) -> Dict[str, Any]:
        return {
            "pipeline_id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "organism": self.config.organism_config.__dict__
            if hasattr(self.config, 'organism_config')
            else {"name": "unknown", "strain": "unknown"},
            "target_molecule": self.config.molecule_config.__dict__
            if hasattr(self.config, 'molecule_config')
            else {"name": "unknown", "smiles": ""},
            "genomic_data": {"total_genes": 0, "essential_genes": [],
                             "available_promoters": [], "codon_table_id": 11,
                             "gc_content_percent": 50.0},
            "data_quality_report": {"completeness_score": 0.0,
                                    "warnings": ["Stage 1 failed"], "errors": ["Stage 1 failed"]},
            "stage_1_status": "FAIL",
        }

    def _stage_2_fallback_output(self, stage_1: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "pipeline_id": stage_1.get("pipeline_id", ""),
            "stage_1_output": stage_1,
            "pathway_candidates": [],
            "gene_modifications": {"knockouts": [], "overexpressions": [],
                                   "heterologous_insertions": []},
            "codon_optimized_sequences": {},
            "stage_2_status": "FAIL",
        }

    def _stage_3_fallback_output(self, stage_2: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "pipeline_id": stage_2.get("pipeline_id", ""),
            "stage_2_output": stage_2,
            "fba_results": {"objective_value": 0, "growth_rate_per_hour": 0,
                            "product_flux_mmol_per_gdw_per_hour": 0,
                            "substrate_uptake_mmol_per_gdw_per_hour": 0,
                            "theoretical_max_yield": 0, "flux_map": {}},
            "strain_design": {"algorithm_used": "fallback", "final_knockouts": [],
                              "final_overexpressions": [], "predicted_titer_g_per_l": 0,
                              "predicted_productivity_g_per_l_per_h": 0,
                              "metabolic_burden_score": 0, "genetic_stability_score": 0},
            "toxicity_assessment": {"intermediate_toxicity_scores": {},
                                    "overall_toxicity_risk": "UNKNOWN",
                                    "flagged_intermediates": []},
            "stage_3_status": "FAIL",
        }

    def _stage_4_fallback_output(self, stage_3: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "pipeline_id": stage_3.get("pipeline_id", ""),
            "stage_3_output": stage_3,
            "dbtl_cycles": [],
            "fermentation_simulation": {"mode": "batch", "duration_hours": 0,
                                        "final_titer_g_per_l": 0,
                                        "final_yield_g_per_g": 0,
                                        "final_productivity_g_per_l_per_h": 0,
                                        "ode_convergence": False,
                                        "organism_specific_events": []},
            "optimal_fermentation_conditions": {"temperature_c": 37, "ph": 7,
                                                "do_percent_saturation": 30,
                                                "glucose_feed_g_per_l_per_h": 0.5,
                                                "agitation_rpm": 300,
                                                "aeration_vvm": 1},
            "stage_4_status": "FAIL",
        }

    def _stage_5_fallback_output(self, stage_4: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "pipeline_id": stage_4.get("pipeline_id", ""),
            "stage_4_output": stage_4,
            "scale_up": {"cascade": [], "total_yield_loss_percent": 0,
                         "max_scale_l": 0},
            "downstream_processing": {"purification_train": {},
                                      "harvest_strategy": {},
                                      "estimated_cost_per_kg_usd": 0,
                                      "overall_recovery_percent": 0},
            "regulatory_assessment": {},
            "stage_5_status": "FAIL",
        }

    # ── Summary printer ─────────────────────────────────────────────────────

    def _print_summary(self, report: PipelineReport, duration: float) -> None:
        """Print a human-readable pipeline summary."""
        print(f"\n{'=' * 70}")
        print(f"  PIPELINE SUMMARY")
        print(f"{'=' * 70}")
        print(f"  Organism    : {report.organism} ({report.strain})")
        print(f"  Molecule    : {report.target_molecule}")
        print(f"  Status      : {report.overall_status}")
        print(f"  Duration    : {duration:.2f}s")
        print(f"  Errors      : {len(self.errors)}")
        print()

        # Stage results
        print("  STAGE RESULTS:")
        for i in range(1, 6):
            key = f"stage_{i}"
            output = self.stage_outputs.get(key, {})
            status = output.get(f"{key}_status", "NOT RUN")
            print(f"    Stage {i}: {status}")

        # Key metrics
        s3 = self.stage_outputs.get("stage_3", {})
        s4 = self.stage_outputs.get("stage_4", {})
        s5 = self.stage_outputs.get("stage_5", {})

        if s3:
            fba = s3.get("fba_results", {})
            print(f"\n  FBA RESULTS:")
            print(f"    Growth rate : {fba.get('growth_rate_per_hour', 'N/A')}/h")
            print(f"    Product flux: {fba.get('product_flux_mmol_per_gdw_per_hour', 'N/A')}")
            print(f"    Max yield   : {fba.get('theoretical_max_yield', 'N/A')}")

        if s4:
            ferm = s4.get("fermentation_simulation", {})
            print(f"\n  FERMENTATION:")
            print(f"    Mode        : {ferm.get('mode', 'N/A')}")
            print(f"    Titer       : {ferm.get('final_titer_g_per_l', 'N/A')} g/L")
            print(f"    Productivity: {ferm.get('final_productivity_g_per_l_per_h', 'N/A')} g/L/h")

        if s5:
            scale = s5.get("scale_up", {})
            print(f"\n  SCALE-UP:")
            cascade = scale.get("cascade", [])
            for level in cascade:
                print(f"    {level.get('volume_l', 0):>8.0f}L | "
                      f"kLa={level.get('kla_h', 0):.1f}/h | "
                      f"loss={level.get('yield_loss_percent', 0):.1f}%")

            ds = s5.get("downstream_processing", {})
            print(f"\n  DOWNSTREAM:")
            print(f"    Recovery  : {ds.get('overall_recovery_percent', 'N/A')}%")
            print(f"    Cost      : ${ds.get('estimated_cost_per_kg_usd', 'N/A')}/kg")

            reg = s5.get("regulatory_assessment", {})
            if isinstance(reg, dict):
                print(f"\n  REGULATORY:")
                print(f"    BSL       : {reg.get('biosafety_level', 'N/A')}")
                print(f"    GRAS      : {'Yes' if reg.get('gras_status') else 'No'}")
                print(f"    Overall   : {reg.get('overall_assessment', 'N/A')}")

        print(f"\n  RECOMMENDATIONS:")
        for rec in report.recommendations:
            print(f"    → {rec}")

        print(f"\n{'=' * 70}")


# ---------------------------------------------------------------------------
# CLI ENTRY POINT
# ---------------------------------------------------------------------------

def main() -> None:
    """Command-line entry point for the full pipeline."""
    parser = argparse.ArgumentParser(
        description="Synthetic Biology Metabolic Pathway Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_pipeline_runner.py --organism ecoli --molecule lycopene
  python main_pipeline_runner.py --organism scerevisiae --molecule vanillin --cycles 5
  python main_pipeline_runner.py --organism cglutamicum --molecule lysine --output-dir ./results
        """,
    )
    parser.add_argument(
        "--organism", choices=ORGANISM_CHOICES, default="ecoli",
        help="Target organism (default: ecoli)",
    )
    parser.add_argument(
        "--molecule", choices=MOLECULE_CHOICES, default="lycopene",
        help="Target molecule (default: lycopene)",
    )
    parser.add_argument(
        "--cycles", type=int, default=3,
        help="Number of DBTL cycles (default: 3)",
    )
    parser.add_argument(
        "--output-dir", default="./pipeline_output",
        help="Output directory for reports (default: ./pipeline_output)",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    args = parser.parse_args()

    # Configure logging
    numeric_level = getattr(py_logging, args.log_level.upper(), py_logging.INFO)
    py_logging.basicConfig(level=numeric_level)

    # Create config
    config = PipelineConfig(
        organism_key=args.organism,
        molecule_key=args.molecule,
        dbtl_cycles=args.cycles,
        output_dir=args.output_dir,
    )

    # Run pipeline
    runner = PipelineRunner(config)
    report = runner.run_full_pipeline()

    # Final message
    print(f"\n{'=' * 70}")
    print(f"  ALL 5 STAGES COMPLETE.")
    print(f"  Run: python main_pipeline_runner.py --organism {args.organism} "
          f"--molecule {args.molecule} --cycles {args.cycles}")
    print(f"  Logs: ./logs/pipeline_*.log")
    print(f"  Output: {args.output_dir}/final_report.json")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
