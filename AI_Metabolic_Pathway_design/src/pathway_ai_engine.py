"""
Pathway AI Engine Module

Orchestrates all Stage 2 components (retrosynthesis, enzyme selection,
codon optimisation, promoter/RBS design) into a single pipeline step.
Validates Stage 1 input JSON and produces validated Stage 2 output JSON.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from codon_optimizer import CodonOptimizer
from enzyme_selector import EnzymeSelector
from exceptions import ModelInferenceError, PipelineError, SchemaValidationError
from logger_setup import PipelineLogger, log_json_contract
from promoter_rbs_designer import PromoterRBSDesigner
from retrosynthesis_engine import PathwayCandidate, RetrosynthesisEngine
from schema_validator import validate_and_raise, validate_stage_output


# ---------------------------------------------------------------------------
# STAGE 2 OUTPUT
# ---------------------------------------------------------------------------

@dataclass
class Stage2Output:
    """Typed container for the Stage 2 output payload."""

    pipeline_id: str
    stage_1_output: Dict[str, Any]
    pathway_candidates: List[Dict[str, Any]]
    gene_modifications: Dict[str, List[str]]
    codon_optimized_sequences: Dict[str, str]
    expression_cassettes: List[Dict[str, Any]]
    stage_2_status: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pipeline_id": self.pipeline_id,
            "stage_1_output": self.stage_1_output,
            "pathway_candidates": self.pathway_candidates,
            "gene_modifications": self.gene_modifications,
            "codon_optimized_sequences": self.codon_optimized_sequences,
            "expression_cassettes": self.expression_cassettes,
            "stage_2_status": self.stage_2_status,
        }


# ---------------------------------------------------------------------------
# PATHWAY AI ENGINE
# ---------------------------------------------------------------------------

class PathwayAIEngine:
    """
    Orchestrates the full Stage 2 pipeline:
      1. Retrosynthesis (MCTS pathway generation)
      2. Enzyme selection & compatibility scoring
      3. Codon optimisation for host organism
      4. Promoter / RBS cassette design

    Takes Stage 1 output JSON as input and returns Stage 2 output JSON.
    """

    def __init__(self) -> None:
        self._logger: Optional[PipelineLogger] = PipelineLogger.get_instance()
        self._retro_engine = RetrosynthesisEngine()
        self._enzyme_selector = EnzymeSelector()
        self._codon_optimizer: Optional[CodonOptimizer] = None
        self._promoter_designer: Optional[PromoterRBSDesigner] = None

    def set_logger(self, logger: PipelineLogger) -> None:
        self._logger = logger
        self._retro_engine.set_logger(logger)
        self._enzyme_selector.set_logger(logger)

    def run(self, stage_1_output: Dict[str, Any],
            max_pathways: int = 10,
            dbtl_cycles: int = 3) -> Dict[str, Any]:
        """
        Execute the full Stage 2 pipeline.

        Parameters
        ----------
        stage_1_output : dict
            Validated JSON from Stage 1.
        max_pathways : int
            Maximum number of pathway candidates to return.
        dbtl_cycles : int
            Number of DBTL cycles (used for planning gene modifications).

        Returns
        -------
        dict
            Stage 2 output JSON matching the schema contract.
        """
        if self._logger is None:
            self._logger = PipelineLogger()
        self._logger.set_stage("2")

        start_time = time.time()
        pipeline_id = stage_1_output.get("pipeline_id", str(uuid.uuid4()))

        self._logger.info("=== STAGE 2 START === pipeline_id=%s", pipeline_id)
        self._logger.debug("Input JSON received: %s",
                           {k: v for k, v in stage_1_output.items()
                            if k != "stage_1_output"})

        try:
            # 1. Validate input
            self._logger.info("Validating Stage 1 input JSON")
            validate_and_raise(stage_1_output, "stage_1_output", self._logger)

            # 2. Extract organism and molecule info
            organism_key = self._extract_organism_key(stage_1_output)
            molecule_config = self._extract_molecule_config(stage_1_output)

            self._logger.info("Organism: %s | Molecule: %s",
                              organism_key, molecule_config["name"])

            # 3. Initialise sub-engines
            self._codon_optimizer = CodonOptimizer(organism_key)
            self._codon_optimizer.set_logger(self._logger)
            self._promoter_designer = PromoterRBSDesigner(organism_key)
            self._promoter_designer.set_logger(self._logger)

            # 4. Run retrosynthesis
            self._logger.info("Running retrosynthesis for target='%s'",
                              molecule_config["name"])
            pathway_candidates = self._run_retrosynthesis(
                molecule_config, organism_key, max_pathways
            )

            if not pathway_candidates:
                self._logger.warning("No pathways found — using fallback")
                pathway_candidates = self._fallback_pathway(molecule_config, organism_key)

            # 5. Enzyme selection
            self._logger.info("Running enzyme selection for %d pathways",
                              len(pathway_candidates))
            enzyme_selections = self._run_enzyme_selection(
                pathway_candidates, organism_key
            )

            # 6. Codon optimisation
            self._logger.info("Running codon optimisation")
            codon_sequences = self._run_codon_optimisation(
                enzyme_selections, organism_key
            )

            # 7. Promoter/RBS design
            self._logger.info("Designing expression cassettes")
            cassettes = self._run_cassette_design(
                enzyme_selections, organism_key
            )

            # 8. Gene modifications planning
            gene_mods = self._plan_gene_modifications(
                pathway_candidates, enzyme_selections, organism_key, dbtl_cycles
            )

            # 9. Assemble output
            output = Stage2Output(
                pipeline_id=pipeline_id,
                stage_1_output=stage_1_output,
                pathway_candidates=pathway_candidates,
                gene_modifications=gene_mods,
                codon_optimized_sequences=codon_sequences,
                expression_cassettes=cassettes,
                stage_2_status="PASS",
            )
            output_dict = output.to_dict()

            # 10. Validate output
            self._logger.info("Validating Stage 2 output against schema")
            validate_and_raise(output_dict, "stage_2_output", self._logger)

            # 11. Log output JSON
            log_json_contract(self._logger, output_dict,
                              "Stage 2 → Stage 3", direction="output")

            # 12. Write stage summary
            duration = time.time() - start_time
            summary = {
                "stage": 2,
                "pipeline_id": pipeline_id,
                "organism": organism_key,
                "molecule": molecule_config["name"],
                "pathways_found": len(pathway_candidates),
                "enzymes_selected": len(enzyme_selections),
                "codon_optimised": len(codon_sequences),
                "cassettes_designed": len(cassettes),
                "gene_modifications": gene_mods,
                "status": "PASS",
                "duration_seconds": round(duration, 2),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            self._logger.write_stage_summary(2, summary)

            self._logger.info("=== STAGE 2 COMPLETE === duration=%.2fs status=PASS",
                              duration)
            return output_dict

        except SchemaValidationError as e:
            self._logger.log_error_with_context("PathwayAIEngine.run", e,
                                                 input_json=stage_1_output)
            return self._stage_2_fallback(pipeline_id, stage_1_output, str(e))
        except ModelInferenceError as e:
            self._logger.log_error_with_context("PathwayAIEngine.run", e,
                                                 input_json=stage_1_output,
                                                 fallback_method="rule-based pathway assembly")
            return self._stage_2_fallback(pipeline_id, stage_1_output, str(e))
        except Exception as e:
            self._logger.log_error_with_context("PathwayAIEngine.run", e,
                                                 input_json=stage_1_output)
            raise PipelineError(
                message=f"Stage 2 failed: {e}",
                stage="2",
                function="PathwayAIEngine.run",
                input_json=stage_1_output,
            ) from e

    # ── Internal helpers ──────────────────────────────────────────────────

    def _extract_organism_key(self, stage_1: Dict[str, Any]) -> str:
        """Derive organism key from Stage 1 output."""
        org = stage_1.get("organism", {})
        name = org.get("name", "").lower()
        strain = org.get("strain", "").lower()

        if "coli" in name:
            if "bl21" in strain:
                return "ecoli_bl21"
            return "ecoli"
        if "cerevisiae" in name:
            if "by4741" in strain:
                return "scerevisiae_by"
            return "scerevisiae"
        if "subtilis" in name:
            return "bsubtilis"
        if "glutamicum" in name:
            return "cglutamicum"
        if "putida" in name:
            return "pputida"

        # Default
        return "ecoli"

    def _extract_molecule_config(self, stage_1: Dict[str, Any]) -> Dict[str, Any]:
        """Extract molecule configuration from Stage 1 output."""
        return stage_1.get("target_molecule", {
            "name": "unknown",
            "smiles": "",
            "chebi_id": "",
            "target_titer_g_per_l": 1.0,
            "target_yield_mol_per_mol": 0.1,
        })

    def _run_retrosynthesis(self, molecule_config: Dict[str, Any],
                            organism_key: str,
                            max_pathways: int) -> List[Dict[str, Any]]:
        """Run the retrosynthesis engine and return pathway candidates."""
        target_name = molecule_config.get("name", "unknown")

        # Load reaction rules
        self._retro_engine.load_reaction_rules()

        # Run MCTS
        raw_pathways = self._retro_engine.run_mcts(
            target=target_name,
            max_iterations=300,
            max_depth=8,
            n_top=max_pathways,
        )

        # Rank pathways
        host_name = organism_key.replace("_", " ").title().replace("Ecoli", "E. coli")
        ranked = self._retro_engine.rank_pathways(
            target=target_name,
            pathways=raw_pathways,
            host_organism=host_name,
            n_top=max_pathways,
        )

        # Convert to dicts
        return [rc.to_dict() for rc in ranked]

    def _run_enzyme_selection(self, pathway_candidates: List[Dict[str, Any]],
                              organism_key: str) -> Dict[str, List[Any]]:
        """Select optimal enzymes for each pathway step."""
        all_enzymes: Dict[str, List[Any]] = {}

        for pc in pathway_candidates:
            pw_id = pc["pathway_id"]
            steps = pc.get("steps", [])

            if not steps:
                continue

            enzymes = self._enzyme_selector.select_best_enzymes(
                pathway_steps=steps,
                host_organism_key=organism_key,
                n_candidates_per_step=3,
            )
            all_enzymes[pw_id] = enzymes

        return all_enzymes

    def _run_codon_optimisation(self, enzyme_selections: Dict[str, List[Any]],
                                organism_key: str) -> Dict[str, str]:
        """Generate codon-optimised DNA sequences for each gene."""
        if self._codon_optimizer is None:
            return {}

        sequences: Dict[str, str] = {}
        seen_genes: set = set()

        for pw_id, enzymes in enzyme_selections.items():
            for step_idx, candidates in enzymes.items():
                for cand in candidates[:1]:  # Take best candidate only
                    if cand.gene_name in seen_genes:
                        continue
                    seen_genes.add(cand.gene_name)

                    # Generate a mock protein sequence (for demonstration)
                    random.seed(hash(f"{cand.gene_name}_{organism_key}"))
                    aa_length = random.randint(100, 500)
                    aa_chars = "ACDEFGHIKLMNPQRSTVWY"
                    protein_seq = "".join(random.choice(aa_chars) for _ in range(aa_length))

                    # Optimise
                    dna = self._codon_optimizer.optimize_sequence(
                        protein_seq,
                        remove_sites=["EcoRI", "BamHI", "HindIII"],
                    )
                    sequences[cand.gene_name] = dna

        return sequences

    def _run_cassette_design(self, enzyme_selections: Dict[str, List[Any]],
                             organism_key: str) -> List[Dict[str, Any]]:
        """Design expression cassettes for each selected gene."""
        if self._promoter_designer is None:
            return []

        cassettes: List[Dict[str, Any]] = []
        seen_genes: set = set()

        for pw_id, enzymes in enzyme_selections.items():
            for step_idx, candidates in enzymes.items():
                for cand in candidates[:1]:
                    if cand.gene_name in seen_genes:
                        continue
                    seen_genes.add(cand.gene_name)

                    cassette = self._promoter_designer.design_expression_cassette(
                        gene_name=cand.gene_name,
                        desired_expression_tpm=1000.0,
                    )
                    cassettes.append(cassette)

        return cassettes

    def _plan_gene_modifications(self,
                                  pathway_candidates: List[Dict[str, Any]],
                                  enzyme_selections: Dict[str, List[Any]],
                                  organism_key: str,
                                  dbtl_cycles: int) -> Dict[str, List[str]]:
        """
        Plan gene knockouts, overexpressions, and heterologous insertions.

        Uses pathway information to identify metabolic engineering targets.
        """
        knockouts: List[str] = []
        overexpressions: List[str] = []
        insertions: List[str] = []

        # Collect all heterologous genes
        for pw_id, enzymes in enzyme_selections.items():
            for step_idx, candidates in enzymes.items():
                for cand in candidates:
                    if cand.is_heterologous:
                        insertions.append(cand.gene_name)
                    else:
                        overexpressions.append(cand.gene_name)

        # Identify knockout targets (competing pathways)
        knockout_map = {
            "ecoli": ["ldhA", "adhE", "frdA", "poxB", "ackA"],
            "ecoli_bl21": ["ldhA", "adhE", "frdA", "poxB"],
            "scerevisiae": ["PDC1", "ADH1", "GPD1", "ALD6"],
            "scerevisiae_by": ["PDC1", "ADH1", "GPD1"],
            "bsubtilis": ["ldh", "pfl", "alsS", "alsD"],
            "cglutamicum": ["ldh", "pqo", "odx", "pck"],
            "pputida": ["edd", "eda", "pcaHG", "catBC"],
        }
        knockouts = knockout_map.get(organism_key, [])[:3]  # Top 3

        # Deduplicate
        insertions = list(set(insertions))
        overexpressions = list(set(overexpressions))[:5]

        return {
            "knockouts": knockouts,
            "overexpressions": overexpressions,
            "heterologous_insertions": insertions,
        }

    def _fallback_pathway(self, molecule_config: Dict[str, Any],
                           organism_key: str) -> List[Dict[str, Any]]:
        """Generate a minimal fallback pathway if retrosynthesis fails."""
        name = molecule_config.get("name", "unknown")
        self._logger.warning("Using fallback pathway for '%s'", name)

        # Create a synthetic single-step pathway
        fallback = {
            "pathway_id": f"pw_fallback_{name.lower().replace(' ', '_')}",
            "rank": 1,
            "pathway_name": f"Fallback {name} pathway",
            "steps": [
                {
                    "step_number": 1,
                    "reaction_id": "FALLBACK_001",
                    "enzyme_name": "FallbackEnzyme",
                    "gene_name": f"fallback_{name[:5].lower()}",
                    "ec_number": "1.1.1.-",
                    "substrate": "precursor_pool",
                    "product": name,
                    "delta_g_kj_per_mol": -10.0,
                    "kcat_per_sec": 1.0,
                    "km_mm": 1.0,
                    "is_heterologous": True,
                    "source_organism": "simulated",
                },
            ],
            "total_steps": 1,
            "predicted_yield_mol_per_mol": 0.1,
            "thermodynamic_feasibility_score": 0.5,
            "gnn_viability_score": 0.4,
            "host_compatibility_score": 0.5,
        }
        return [fallback]

    def _stage_2_fallback(self, pipeline_id: str,
                           stage_1_output: Dict[str, Any],
                           error_msg: str) -> Dict[str, Any]:
        """Graceful degradation for Stage 2 — return minimal valid output."""
        self._logger.warning("Using fallback Stage 2 output: %s", error_msg)

        molecule = stage_1_output.get("target_molecule", {})
        name = molecule.get("name", "unknown")

        fallback_pathway = self._fallback_pathway(molecule,
                                                    stage_1_output.get("organism", {}).get("name", "ecoli"))

        output = {
            "pipeline_id": pipeline_id,
            "stage_1_output": stage_1_output,
            "pathway_candidates": fallback_pathway,
            "gene_modifications": {
                "knockouts": [],
                "overexpressions": [],
                "heterologous_insertions": [f"fallback_{name[:5].lower()}"],
            },
            "codon_optimized_sequences": {},
            "expression_cassettes": [],
            "stage_2_status": "WARN",
        }

        self._logger.write_stage_summary(2, {
            "stage": 2,
            "pipeline_id": pipeline_id,
            "status": "WARN",
            "fallback": True,
            "error": error_msg,
        })
        return output


# ---------------------------------------------------------------------------
# CONVENIENCE FUNCTION
# ---------------------------------------------------------------------------

def run_stage_2(stage_1_output: Dict[str, Any],
                max_pathways: int = 10,
                dbtl_cycles: int = 3) -> Dict[str, Any]:
    """
    Run Stage 2 pipeline from a validated Stage 1 output dict.

    Parameters
    ----------
    stage_1_output : dict
        Output from Stage 1 (run_stage_1).
    max_pathways : int
        Maximum number of pathway candidates to generate.
    dbtl_cycles : int
        Number of DBTL cycles to plan for.

    Returns
    -------
    dict
        Stage 2 output JSON.
    """
    engine = PathwayAIEngine()
    return engine.run(
        stage_1_output=stage_1_output,
        max_pathways=max_pathways,
        dbtl_cycles=dbtl_cycles,
    )


# ---------------------------------------------------------------------------
# MAIN — smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json
    import sys
    import time

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    # Run Stage 1 first to get input JSON
    from data_layer import run_stage_1
    from pipeline_config import PipelineConfig

    parser = argparse.ArgumentParser(description="Pathway AI Engine (Stage 2)")
    parser.add_argument("--organism", default="ecoli")
    parser.add_argument("--molecule", default="lycopene")
    parser.add_argument("--max-pathways", type=int, default=5)
    args = parser.parse_args()

    print("Running Stage 1 to generate input JSON...")
    cfg = PipelineConfig(organism_key=args.organism, molecule_key=args.molecule)
    stage_1_output = run_stage_1(cfg)

    print(f"\nRunning Stage 2 for {args.organism} → {args.molecule} ...")
    t0 = time.time()
    stage_2_output = run_stage_2(
        stage_1_output=stage_1_output,
        max_pathways=args.max_pathways,
        dbtl_cycles=3,
    )
    elapsed = time.time() - t0

    logger = PipelineLogger.get_instance()
    logger.info("Stage 2 completed in %.2f seconds", elapsed)

    # Print summary
    print(f"\n{'='*60}")
    print(f"  STAGE 2 RESULT — {args.organism} → {args.molecule}")
    print(f"{'='*60}")
    print(f"  Pipeline ID : {stage_2_output['pipeline_id']}")
    print(f"  Status      : {stage_2_output['stage_2_status']}")
    print(f"  Pathways    : {len(stage_2_output['pathway_candidates'])}")
    print(f"  Gene mods   : {len(stage_2_output['gene_modifications']['knockouts'])} KOs, "
          f"{len(stage_2_output['gene_modifications']['overexpressions'])} OEs, "
          f"{len(stage_2_output['gene_modifications']['heterologous_insertions'])} inserts")
    print(f"  Codon seqs  : {len(stage_2_output['codon_optimized_sequences'])}")
    print(f"  Cassettes   : {len(stage_2_output['expression_cassettes'])}")
    print(f"  Duration    : {elapsed:.2f}s")
    print(f"{'='*60}")

    # Save output
    os.makedirs("pipeline_output", exist_ok=True)
    with open("pipeline_output/stage_2_output.json", "w") as fh:
        json.dump(stage_2_output, fh, indent=2, default=str)

    print(f"\n  Output saved : pipeline_output/stage_2_output.json")
    print(f"  Log file     : {logger.get_log_file_path()}")

    print("\n▶ STAGE 2 COMPLETE ◀  Type: CONTINUE TO STAGE 3")
