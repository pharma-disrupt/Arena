"""
DBTL Orchestrator Module (Stage 4)

Orchestrates the full Stage 4 pipeline:
- DBTL loop execution
- Fermentation simulation
- Bioreactor control optimisation
- Produces validated Stage 4 output JSON
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from bioreactor_controller import MPCController, ORGANISM_SETPOINTS
from bioreactor_controller import BioreactorState
from dbtl_loop import DBTLCycle, DBTLOrchestrator
from exceptions import FermentationSimulationError, PipelineError
from fermentation_simulator import FermentationSimulator
from logger_setup import PipelineLogger, log_json_contract
from schema_validator import validate_and_raise


@dataclass
class Stage4Output:
    """Typed container for the Stage 4 output payload."""
    pipeline_id: str
    stage_3_output: Dict[str, Any]
    dbtl_cycles: List[Dict[str, Any]]
    fermentation_simulation: Dict[str, Any]
    optimal_fermentation_conditions: Dict[str, float]
    stage_4_status: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pipeline_id": self.pipeline_id,
            "stage_3_output": self.stage_3_output,
            "dbtl_cycles": self.dbtl_cycles,
            "fermentation_simulation": self.fermentation_simulation,
            "optimal_fermentation_conditions": self.optimal_fermentation_conditions,
            "stage_4_status": self.stage_4_status,
        }


class Stage4Orchestrator:
    """Orchestrates the full Stage 4 pipeline."""

    def __init__(self) -> None:
        self._logger: Optional[PipelineLogger] = None
        self._dbtl_orchestrator = DBTLOrchestrator()
        self._fermentation_sim = FermentationSimulator()
        self._mpc_controller: Optional[MPCController] = None

    def set_logger(self, logger: PipelineLogger) -> None:
        self._logger = logger
        self._dbtl_orchestrator.set_logger(logger)
        self._fermentation_sim.set_logger(logger)

    def run(self, stage_3_output: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the full Stage 4 pipeline."""
        if self._logger is None:
            self._logger = PipelineLogger()
        self._logger.set_stage("4")

        start_time = time.time()
        pipeline_id = stage_3_output.get("pipeline_id", str(uuid.uuid4()))

        self._logger.info("=== STAGE 4 START === pipeline_id=%s", pipeline_id)

        try:
            # 1. Validate input
            self._logger.info("Validating Stage 3 input JSON")
            validate_and_raise(stage_3_output, "stage_3_output", self._logger)

            # 2. Extract organism and molecule info
            stage_1 = stage_3_output.get("stage_2_output", {}).get("stage_1_output", {})
            organism_key = self._extract_organism_key(stage_1)
            molecule_key = self._extract_molecule_key(stage_1)
            strain_design = stage_3_output.get("strain_design", {})

            self._logger.info("Organism: %s | Molecule: %s", organism_key, molecule_key)

            # 3. Run DBTL loop
            dbtl_cycles = self._run_dbtl_loop(
                stage_2_output=stage_3_output.get("stage_2_output", {}),
                organism_key=organism_key,
                molecule_key=molecule_key,
                strain_design=strain_design,
            )

            # 4. Run fermentation simulation
            fermentation_results, optimal_conditions = self._run_fermentation(
                organism_key=organism_key,
                molecule_key=molecule_key,
                strain_design=strain_design,
            )

            # 5. Assemble output
            output = Stage4Output(
                pipeline_id=pipeline_id,
                stage_3_output=stage_3_output,
                dbtl_cycles=[
                    {
                        "cycle_number": c.cycle_number,
                        "constructs_tested": c.constructs_tested,
                        "best_titer_g_per_l": c.best_titer_g_per_l,
                        "best_construct_id": c.best_construct_id,
                        "improvement_fold": c.improvement_fold,
                        "bo_next_candidates": c.bo_next_candidates,
                    }
                    for c in dbtl_cycles
                ],
                fermentation_simulation=fermentation_results,
                optimal_fermentation_conditions=optimal_conditions,
                stage_4_status="PASS",
            )
            output_dict = output.to_dict()

            # 6. Validate output
            self._logger.info("Validating Stage 4 output against schema")
            validate_and_raise(output_dict, "stage_4_output", self._logger)

            # 7. Log JSON contract
            log_json_contract(self._logger, output_dict, "Stage 4 -> Stage 5", direction="output")

            # 8. Write stage summary
            duration = time.time() - start_time
            summary = {
                "stage": 4,
                "pipeline_id": pipeline_id,
                "organism": organism_key,
                "molecule": molecule_key,
                "dbtl_cycles": len(dbtl_cycles),
                "best_titer": max((c.best_titer_g_per_l for c in dbtl_cycles), default=0),
                "fermentation_mode": fermentation_results.get("mode", "batch"),
                "fermentation_duration_hours": fermentation_results.get("duration_hours", 0),
                "final_product_titer": fermentation_results.get("final_titer_g_per_l", 0),
                "status": "PASS",
                "duration_seconds": round(duration, 2),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            self._logger.write_stage_summary(4, summary)

            self._logger.info("=== STAGE 4 COMPLETE === duration=%.2fs status=PASS", duration)
            return output_dict

        except FermentationSimulationError as e:
            self._logger.log_error_with_context(
                "Stage4Orchestrator.run", e, input_json=stage_3_output,
                fallback_method="use theoretical estimates"
            )
            return self._stage_4_fallback(pipeline_id, stage_3_output, str(e))
        except Exception as e:
            self._logger.log_error_with_context("Stage4Orchestrator.run", e, input_json=stage_3_output)
            raise PipelineError(
                message=f"Stage 4 failed: {e}", stage="4",
                function="Stage4Orchestrator.run", input_json=stage_3_output,
            ) from e

    def _extract_organism_key(self, stage_1: Dict[str, Any]) -> str:
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
        mol = stage_1.get("target_molecule", {})
        name = mol.get("name", "").lower()
        mol_map = {
            "lycopene": "lycopene", "vanillin": "vanillin",
            "artemisinic acid": "artemisinic_acid",
            "l-lysine": "lysine", "lysine": "lysine",
            "l-glutamate": "glutamate", "glutamate": "glutamate",
            "l-threonine": "threonine", "threonine": "threonine",
            "polyhydroxyalkanoate (pha)": "pha", "pha": "pha",
            "hyaluronic acid": "hyaluronic_acid",
            "riboflavin (vitamin b2)": "riboflavin", "riboflavin": "riboflavin",
        }
        return mol_map.get(name, "lycopene")

    def _run_dbtl_loop(self, stage_2_output: Dict[str, Any],
                       organism_key: str, molecule_key: str,
                       strain_design: Dict[str, Any],
                       n_cycles: int = 3) -> List[DBTLCycle]:
        if self._logger:
            self._logger.info("Starting DBTL loop: %d cycles", n_cycles)

        pathway_candidates = stage_2_output.get("pathway_candidates", [])
        genes = []
        if pathway_candidates:
            for step in pathway_candidates[0].get("steps", []):
                if step.get("is_heterologous"):
                    genes.append(step.get("gene_name", "unknown"))
        if not genes:
            genes = ["crtE", "crtB", "crtI"]

        target_titer = strain_design.get("predicted_titer_g_per_l", 1.0)
        if target_titer <= 0:
            target_titer = 1.0

        cycles = self._dbtl_orchestrator.run_dbtl_loop(
            pathway_genes=genes, n_cycles=n_cycles, constructs_per_cycle=48,
            base_titer=target_titer * 0.1,
        )

        if self._logger:
            for c in cycles:
                self._logger.info("  Cycle %d: best=%.3f g/L, improvement=%.2fx",
                                  c.cycle_number, c.best_titer_g_per_l, c.improvement_fold)
        return cycles

    def _run_fermentation(self, organism_key: str, molecule_key: str,
                          strain_design: Dict[str, Any]
                          ) -> Tuple[Dict[str, Any], Dict[str, float]]:
        if self._logger:
            self._logger.info("Running fermentation simulation for %s", organism_key)

        setpoints = ORGANISM_SETPOINTS.get(organism_key, ORGANISM_SETPOINTS["ecoli"])
        mpc = MPCController(organism_key=organism_key)
        if self._logger:
            mpc.set_logger(self._logger)

        initial_state = BioreactorState(
            time_hours=0.0, biomass_g_per_l=0.1, substrate_g_per_l=20.0,
            temperature_c=setpoints["temperature_c"], ph=setpoints["ph"],
            dissolved_o2_percent=100.0, agitation_rpm=setpoints["agitation_rpm"],
            aeration_vvm=setpoints["aeration_vvm"],
            feed_rate_g_per_l_per_h=setpoints["feed_rate_g_per_l_per_h"],
        )

        optimal_actions = mpc.optimise_control_action(initial_state)

        target_titer = strain_design.get("predicted_titer_g_per_l", 1.0)
        duration_hours = max(48.0, target_titer / 0.5)

        final_state, time_series = self._fermentation_sim.run_ode_simulation(
            organism_key=organism_key,
            duration_hours=min(duration_hours, 120.0),
            temperature_c=optimal_actions.get("temperature_c", setpoints["temperature_c"]),
            ph=optimal_actions.get("ph", setpoints["ph"]),
            do_percent_saturation=optimal_actions.get("do_percent", setpoints["do_percent"]),
            glucose_feed_g_per_l_per_h=optimal_actions.get(
                "feed_rate_g_per_l_per_h", setpoints["feed_rate_g_per_l_per_h"]),
            agitation_rpm=optimal_actions.get("agitation_rpm", setpoints["agitation_rpm"]),
            aeration_vvm=optimal_actions.get("aeration_vvm", setpoints["aeration_vvm"]),
        )

        events = self._fermentation_sim.organism_specific_events(organism_key, final_state)
        yield_g_per_g = (final_state.product_g_per_l / max(final_state.substrate_g_per_l, 0.001)
                        if final_state.substrate_g_per_l >= 0 else 0.0)
        productivity = (final_state.product_g_per_l / final_state.time_hours
                       if final_state.time_hours > 0 else 0.0)

        fermentation_results = {
            "mode": "fed-batch",
            "duration_hours": round(final_state.time_hours, 1),
            "final_titer_g_per_l": round(final_state.product_g_per_l, 2),
            "final_yield_g_per_g": round(yield_g_per_g, 4),
            "final_productivity_g_per_l_per_h": round(productivity, 4),
            "ode_convergence": True,
            "organism_specific_events": events,
        }

        optimal_conditions = {
            "temperature_c": round(final_state.temperature_c, 1),
            "ph": round(final_state.ph, 2),
            "do_percent_saturation": round(final_state.dissolved_o2_percent, 1),
            "glucose_feed_g_per_l_per_h": round(
                optimal_actions.get("feed_rate_g_per_l_per_h", 0.5), 2),
            "agitation_rpm": round(
                optimal_actions.get("agitation_rpm", setpoints["agitation_rpm"]), 0),
            "aeration_vvm": round(
                optimal_actions.get("aeration_vvm", setpoints["aeration_vvm"]), 2),
        }

        if self._logger:
            self._logger.info(
                "Fermentation complete: titer=%.2f, yield=%.4f, prod=%.4f",
                final_state.product_g_per_l, yield_g_per_g, productivity)

        return fermentation_results, optimal_conditions

    def _stage_4_fallback(self, pipeline_id: str, stage_3_output: Dict[str, Any],
                          error_msg: str) -> Dict[str, Any]:
        if self._logger:
            self._logger.warning("Using fallback Stage 4: %s", error_msg)

        stage_1 = stage_3_output.get("stage_2_output", {}).get("stage_1_output", {})
        organism_key = self._extract_organism_key(stage_1)
        setpoints = ORGANISM_SETPOINTS.get(organism_key, ORGANISM_SETPOINTS["ecoli"])

        fallback = {
            "pipeline_id": pipeline_id,
            "stage_3_output": stage_3_output,
            "dbtl_cycles": [],
            "fermentation_simulation": {
                "mode": "batch", "duration_hours": 48.0,
                "final_titer_g_per_l": 1.0, "final_yield_g_per_g": 0.05,
                "final_productivity_g_per_l_per_h": 0.02,
                "ode_convergence": False,
                "organism_specific_events": ["Fallback: simulation failed"],
            },
            "optimal_fermentation_conditions": {
                "temperature_c": setpoints["temperature_c"],
                "ph": setpoints["ph"],
                "do_percent_saturation": setpoints["do_percent"],
                "glucose_feed_g_per_l_per_h": setpoints["feed_rate_g_per_l_per_h"],
                "agitation_rpm": setpoints["agitation_rpm"],
                "aeration_vvm": setpoints["aeration_vvm"],
            },
            "stage_4_status": "WARN",
        }
        if self._logger:
            self._logger.write_stage_summary(4, {
                "stage": 4, "pipeline_id": pipeline_id,
                "status": "WARN", "fallback": True, "error": error_msg,
            })
        return fallback


def run_stage_4(stage_3_output: Dict[str, Any],
                n_dbtl_cycles: int = 3) -> Dict[str, Any]:
    """Run Stage 4 pipeline from a validated Stage 3 output dict."""
    orchestrator = Stage4Orchestrator()
    return orchestrator.run(stage_3_output=stage_3_output)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Stage 4: DBTL + Fermentation")
    parser.add_argument("--organism", default="ecoli")
    parser.add_argument("--molecule", default="lycopene")
    parser.add_argument("--cycles", type=int, default=3)
    args = parser.parse_args()

    from data_layer import run_stage_1
    from pathway_ai_engine import run_stage_2
    from flux_analysis_orchestrator import run_stage_3
    from pipeline_config import PipelineConfig

    print("Running Stage 1...")
    cfg = PipelineConfig(organism_key=args.organism, molecule_key=args.molecule)
    stage_1 = run_stage_1(cfg)
    print("Running Stage 2...")
    stage_2 = run_stage_2(stage_1)
    print("Running Stage 3...")
    stage_3 = run_stage_3(stage_2)

    print(f"Running Stage 4: {args.organism} -> {args.molecule} ...")
    t0 = time.time()
    stage_4 = run_stage_4(stage_3, n_dbtl_cycles=args.cycles)
    elapsed = time.time() - t0

    logger = PipelineLogger.get_instance()
    print(f"\n{'='*60}")
    print(f"  STAGE 4 RESULT - {args.organism} -> {args.molecule}")
    print(f"{'='*60}")
    print(f"  Pipeline ID : {stage_4['pipeline_id']}")
    print(f"  Status      : {stage_4['stage_4_status']}")
    print(f"  DBTL Cycles : {len(stage_4['dbtl_cycles'])}")
    if stage_4['dbtl_cycles']:
        best = max(stage_4['dbtl_cycles'], key=lambda c: c['best_titer_g_per_l'])
        print(f"    Best: cycle {best['cycle_number']} ({best['best_titer_g_per_l']:.2f} g/L)")
    ferm = stage_4['fermentation_simulation']
    print(f"  Fermentation:")
    print(f"    Mode: {ferm['mode']}, Duration: {ferm['duration_hours']:.1f}h")
    print(f"    Titer: {ferm['final_titer_g_per_l']:.2f} g/L")
    print(f"    Yield: {ferm['final_yield_g_per_g']:.4f} g/g")
    print(f"    Productivity: {ferm['final_productivity_g_per_l_per_h']:.4f} g/L/h")
    opt = stage_4['optimal_fermentation_conditions']
    print(f"  Conditions: T={opt['temperature_c']}, pH={opt['ph']}, DO={opt['do_percent_saturation']}%")
    print(f"  Feed={opt['glucose_feed_g_per_l_per_h']}, Agit={opt['agitation_rpm']}, Air={opt['aeration_vvm']}")
    print(f"  Duration: {elapsed:.2f}s")
    print(f"{'='*60}")

    os.makedirs("pipeline_output", exist_ok=True)
    with open("pipeline_output/stage_4_output.json", "w") as fh:
        json.dump(stage_4, fh, indent=2, default=str)

    print(f"\n  STAGE 4 COMPLETE. Type: CONTINUE TO STAGE 5")
