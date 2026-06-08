"""
Data Layer Module

Loads organism and molecule data, simulates genomic data retrieval,
performs data quality checks, and produces the Stage 1 output JSON.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from exceptions import DataIngestionError
from logger_setup import PipelineLogger, log_json_contract
from pipeline_config import (
    ORGANISM_DATABASE,
    MOLECULE_DATABASE,
    OrganismConfig,
    MoleculeConfig,
    PipelineConfig,
)
from schema_validator import (
    validate_and_raise,
    validate_stage_output,
    SCHEMA_REGISTRY,
)


# ──────────────────────────────────────────────────────────────────────────────
# ORGANISM DATABASE
# ──────────────────────────────────────────────────────────────────────────────

class OrganismDatabase:
    """
    Provides access to organism-specific configurations.

    Acts as the single source of truth for all target organisms.
    """

    def __init__(self, custom_organisms: Optional[Dict[str, OrganismConfig]] = None) -> None:
        self._organisms: Dict[str, OrganismConfig] = dict(ORGANISM_DATABASE)
        if custom_organisms:
            self._organisms.update(custom_organisms)

    def get(self, key: str) -> OrganismConfig:
        """Return the OrganismConfig for the given key."""
        if key not in self._organisms:
            raise DataIngestionError(
                f"Organism key '{key}' not found in database.",
                source="OrganismDatabase",
            )
        return self._organisms[key]

    def list_available(self) -> List[str]:
        """Return sorted list of organism keys."""
        return sorted(self._organisms.keys())

    def to_dict(self, key: str) -> Dict[str, Any]:
        """Convert an organism config to a JSON-serialisable dict."""
        org = self.get(key)
        return {
            "name": org.name,
            "strain": org.strain,
            "gem_model_id": org.gem_model_id,
            "doubling_time_min": org.doubling_time_min,
            "optimal_ph": org.optimal_ph,
            "optimal_temp_c": org.optimal_temp_c,
            "gram_stain": org.gram_stain,
        }

    @property
    def all_organisms(self) -> Dict[str, OrganismConfig]:
        return dict(self._organisms)


# ──────────────────────────────────────────────────────────────────────────────
# MOLECULE DATABASE
# ──────────────────────────────────────────────────────────────────────────────

class MoleculeDatabase:
    """Provides access to target molecule / product configurations."""

    def __init__(self, custom_molecules: Optional[Dict[str, MoleculeConfig]] = None) -> None:
        self._molecules: Dict[str, MoleculeConfig] = dict(MOLECULE_DATABASE)
        if custom_molecules:
            self._molecules.update(custom_molecules)

    def get(self, key: str) -> MoleculeConfig:
        if key not in self._molecules:
            raise DataIngestionError(
                f"Molecule key '{key}' not found in database.",
                source="MoleculeDatabase",
            )
        return self._molecules[key]

    def list_available(self) -> List[str]:
        return sorted(self._molecules.keys())

    def to_dict(self, key: str) -> Dict[str, Any]:
        mol = self.get(key)
        return {
            "name": mol.name,
            "smiles": mol.smiles,
            "chebi_id": mol.chebi_id,
            "target_titer_g_per_l": mol.target_titer_g_per_l,
            "target_yield_mol_per_mol": mol.target_yield_mol_per_mol,
        }

    @property
    def all_molecules(self) -> Dict[str, MoleculeConfig]:
        return dict(self._molecules)


# ──────────────────────────────────────────────────────────────────────────────
# GENOMIC DATA LOADER
# ──────────────────────────────────────────────────────────────────────────────

class GenomicDataLoader:
    """
    Simulates loading genomic metadata from external databases (KEGG, UniProt, NCBI).

    In a production pipeline this would query REST APIs; here we use curated mock data.
    """

    # Simulated genomic databases
    _KEGG_GENE_COUNTS: Dict[str, int] = {
        "ecoli": 4494,
        "ecoli_bl21": 4405,
        "scerevisiae": 6604,
        "scerevisiae_by": 6550,
        "bsubtilis": 4100,
        "cglutamicum": 3057,
        "pputida": 5452,
    }

    _KEGG_GI_MAP: Dict[str, List[str]] = {
        "ecoli": ["dnaA", "dnaN", "dnaG", "rpoB", "rpoC", "rpoA",
                  "gyrA", "gyrB", "ftsZ", "ftsA", "murA", "murB",
                  "tufA", "tufB", "rplA", "rplB", "rplC", "rplD"],
        "ecoli_bl21": ["dnaA", "dnaN", "rpoB", "rpoC", "gyrA", "ftsZ",
                       "tufA", "rplA", "rplB"],
        "scerevisiae": ["ACT1", "TUB1", "TUB2", "CDC28", "RPB1", "RPB2",
                        "RPB3", "NUP49", "NUP57", "SEC61"],
        "scerevisiae_by": ["ACT1", "TUB1", "TUB2", "CDC28", "RPB1", "RPB2"],
        "bsubtilis": ["dnaA", "dnaB", "dnaC", "rpoB", "rpoC", "ftsZ",
                      "murA", "accD", "fabH", "tuf"],
        "cglutamicum": ["dnaA", "rpoB", "rpoC", "ftsZ", "murA", "secA",
                        "tuf", "rplA", "rplB", "gyrA"],
        "pputida": ["dnaA", "rpoB", "rpoC", "ftsZ", "gyrA", "gyrB",
                    "tufA", "rplA", "rplB", "secA"],
    }

    _CODON_TABLE_MAP: Dict[str, int] = {
        "ecoli": 11,
        "ecoli_bl21": 11,
        "scerevisiae": 1,
        "scerevisiae_by": 1,
        "bsubtilis": 11,
        "cglutamicum": 11,
        "pputida": 11,
    }

    def __init__(self, organism_db: OrganismDatabase) -> None:
        self._organism_db = organism_db
        self.logger: Optional[Any] = None

    def set_logger(self, logger: Any) -> None:
        self.logger = logger

    def load_genomic_data(self, organism_key: str) -> Dict[str, Any]:
        """
        Simulate loading genomic metadata for an organism.

        Returns a dict matching the genomic_data portion of the Stage 1 output schema.
        """
        org: OrganismConfig = self._organism_db.get(organism_key)

        total_genes = self._KEGG_GENE_COUNTS.get(organism_key, org.total_genes)
        essential_genes = self._KEGG_GI_MAP.get(organism_key, org.essential_genes)
        codon_table_id = self._CODON_TABLE_MAP.get(organism_key, org.codon_table_id)

        if self.logger:
            self.logger.debug(
                "Genomic data loaded for %s: %d genes, %d essential, codon table %d, GC=%.1f%%",
                organism_key, total_genes, len(essential_genes), codon_table_id, org.gc_content_percent,
            )

        return {
            "total_genes": total_genes,
            "essential_genes": essential_genes,
            "available_promoters": org.available_promoters,
            "codon_table_id": codon_table_id,
            "gc_content_percent": org.gc_content_percent,
        }


# ──────────────────────────────────────────────────────────────────────────────
# DATA QUALITY CHECKER
# ──────────────────────────────────────────────────────────────────────────────

class DataQualityChecker:
    """
    Evaluates the completeness and quality of loaded data for a given
    organism–molecule pair.
    """

    def __init__(self) -> None:
        self.logger: Optional[Any] = None
        self._warnings: List[str] = []
        self._errors: List[str] = []

    def set_logger(self, logger: Any) -> None:
        self.logger = logger
        self._warnings = []
        self._errors = []

    def run_checks(
        self,
        organism: OrganismConfig,
        molecule: MoleculeConfig,
        genomic_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Run data quality checks and return a report dict.

        The report matches the data_quality_report portion of the Stage 1 schema.
        """
        score = 1.0

        # Check organism data completeness
        if not organism.essential_genes:
            self._warnings.append(f"No essential genes listed for {organism.name}")
            score -= 0.1

        if not organism.available_promoters:
            self._warnings.append(f"No promoter library available for {organism.name}")
            score -= 0.15

        # Check molecule data completeness
        if not molecule.smiles:
            self._errors.append(f"No SMILES string for molecule '{molecule.name}'")
            score -= 0.3

        if molecule.target_titer_g_per_l <= 0:
            self._warnings.append(f"Target titer <= 0 for '{molecule.name}'")
            score -= 0.05

        if molecule.target_yield_mol_per_mol <= 0:
            self._warnings.append(f"Target yield <= 0 for '{molecule.name}'")
            score -= 0.05

        # Check genomic data
        if genomic_data.get("total_genes", 0) == 0:
            self._errors.append("total_genes is 0 — genomic data missing")
            score -= 0.2

        if len(genomic_data.get("essential_genes", [])) == 0:
            self._warnings.append("No essential genes in genomic data")
            score -= 0.1

        if genomic_data.get("gc_content_percent", 0) < 20 or genomic_data.get("gc_content_percent", 100) > 80:
            self._warnings.append(f"GC content outside expected range: {genomic_data.get('gc_content_percent')}%")
            score -= 0.05

        score = max(0.0, min(1.0, score))

        report: Dict[str, Any] = {
            "completeness_score": round(score, 3),
            "warnings": self._warnings,
            "errors": self._errors,
        }

        if self.logger:
            self.logger.info(
                "Data quality report: score=%.3f  warnings=%d  errors=%d",
                score, len(self._warnings), len(self._errors),
            )
            for w in self._warnings:
                self.logger.warning("DQ warning: %s", w)
            for e in self._errors:
                self.logger.error("DQ error: %s", e)

        return report


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 1 OUTPUT
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Stage1Output:
    """Typed container for the Stage 1 output payload."""

    pipeline_id: str
    timestamp: str
    organism: Dict[str, Any]
    target_molecule: Dict[str, Any]
    genomic_data: Dict[str, Any]
    data_quality_report: Dict[str, Any]
    stage_1_status: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pipeline_id": self.pipeline_id,
            "timestamp": self.timestamp,
            "organism": self.organism,
            "target_molecule": self.target_molecule,
            "genomic_data": self.genomic_data,
            "data_quality_report": self.data_quality_report,
            "stage_1_status": self.stage_1_status,
        }


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 1 RUNNER
# ──────────────────────────────────────────────────────────────────────────────

def run_stage_1(config: PipelineConfig) -> Dict[str, Any]:
    """
    Execute Stage 1: Core Infrastructure + Data Layer.

    Loads organism and molecule data, simulates genomic data retrieval,
    runs quality checks, and returns a validated Stage 1 output dict.

    Parameters
    ----------
    config : PipelineConfig
        Fully initialised pipeline configuration.

    Returns
    -------
    dict
        JSON payload matching STAGE_1_OUTPUT_SCHEMA.
    """
    logger = PipelineLogger.get_instance()
    logger.set_stage("1")

    start_time = datetime.now(timezone.utc)
    pipeline_id = config.pipeline_id or str(uuid.uuid4())

    logger.info("=== STAGE 1 START === pipeline_id=%s organism=%s molecule=%s",
                pipeline_id, config.organism_key, config.molecule_key)

    try:
        # 1. Initialise databases
        org_db = OrganismDatabase()
        mol_db = MoleculeDatabase()
        geno_loader = GenomicDataLoader(org_db)
        geno_loader.set_logger(logger)
        dq_checker = DataQualityChecker()
        dq_checker.set_logger(logger)

        # 2. Load organism config
        logger.info("Loading organism config for key='%s'", config.organism_key)
        organism_cfg = org_db.get(config.organism_key)
        organism_dict = org_db.to_dict(config.organism_key)
        logger.debug("Organism loaded: %s", organism_dict)

        # 3. Load molecule config
        logger.info("Loading molecule config for key='%s'", config.molecule_key)
        molecule_cfg = mol_db.get(config.molecule_key)
        molecule_dict = mol_db.to_dict(config.molecule_key)
        logger.debug("Molecule loaded: %s", molecule_dict)

        # 4. Load genomic data
        logger.info("Loading genomic data for organism='%s'", config.organism_key)
        genomic_data = geno_loader.load_genomic_data(config.organism_key)
        logger.debug("Genomic data: %d genes, codon table %d, GC=%.1f%%",
                     genomic_data["total_genes"],
                     genomic_data["codon_table_id"],
                     genomic_data["gc_content_percent"])

        # 5. Run quality checks
        logger.info("Running data quality checks")
        dq_report = dq_checker.run_checks(organism_cfg, molecule_cfg, genomic_data)

        # 6. Determine stage status
        if dq_report["errors"]:
            stage_status = "FAIL"
        elif dq_report["warnings"]:
            stage_status = "WARN"
        else:
            stage_status = "PASS"

        # 7. Assemble output
        output = Stage1Output(
            pipeline_id=pipeline_id,
            timestamp=start_time.isoformat(),
            organism=organism_dict,
            target_molecule=molecule_dict,
            genomic_data=genomic_data,
            data_quality_report=dq_report,
            stage_1_status=stage_status,
        )
        output_dict = output.to_dict()

        # 8. Validate against schema
        logger.info("Validating Stage 1 output against schema")
        validate_and_raise(output_dict, "stage_1_output", logger)

        # 9. Log output JSON
        log_json_contract(logger, output_dict, "Stage 1 → Stage 2", direction="output")

        # 10. Write stage summary
        duration = (datetime.now(timezone.utc) - start_time).total_seconds()
        summary = {
            "stage": 1,
            "pipeline_id": pipeline_id,
            "organism": config.organism_key,
            "molecule": config.molecule_key,
            "status": stage_status,
            "data_quality_score": dq_report["completeness_score"],
            "duration_seconds": round(duration, 2),
            "timestamp": start_time.isoformat(),
        }
        logger.write_stage_summary(1, summary)

        logger.info("=== STAGE 1 COMPLETE === duration=%.2fs status=%s",
                    duration, stage_status)

        return output_dict

    except DataIngestionError as e:
        logger.log_error_with_context("run_stage_1", e, fallback_method="use defaults")
        # Graceful degradation: return a FAIL output with available data
        return _stage_1_fallback(config, pipeline_id, logger, str(e))
    except Exception as e:
        logger.log_error_with_context("run_stage_1", e)
        raise


def _stage_1_fallback(
    config: PipelineConfig,
    pipeline_id: str,
    logger: Any,
    error_msg: str,
) -> Dict[str, Any]:
    """
    Graceful degradation for Stage 1 — return a FAIL output with
    whatever data we managed to load.
    """
    logger.warning("Using fallback Stage 1 output due to error: %s", error_msg)
    org_db = OrganismDatabase()
    mol_db = MoleculeDatabase()

    try:
        org_dict = org_db.to_dict(config.organism_key)
    except Exception:
        org_dict = {"name": "unknown", "strain": "unknown", "gem_model_id": "",
                    "doubling_time_min": 30.0, "optimal_ph": 7.0,
                    "optimal_temp_c": 37.0, "gram_stain": "unknown"}

    try:
        mol_dict = mol_db.to_dict(config.molecule_key)
    except Exception:
        mol_dict = {"name": "unknown", "smiles": "", "chebi_id": "",
                    "target_titer_g_per_l": 1.0, "target_yield_mol_per_mol": 0.1}

    fallback = {
        "pipeline_id": pipeline_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "organism": org_dict,
        "target_molecule": mol_dict,
        "genomic_data": {
            "total_genes": 0,
            "essential_genes": [],
            "available_promoters": [],
            "codon_table_id": 11,
            "gc_content_percent": 50.0,
        },
        "data_quality_report": {
            "completeness_score": 0.0,
            "warnings": [error_msg],
            "errors": [error_msg],
        },
        "stage_1_status": "FAIL",
    }

    logger.write_stage_summary(1, {
        "stage": 1,
        "pipeline_id": pipeline_id,
        "status": "FAIL",
        "fallback": True,
        "error": error_msg,
    })
    return fallback


# ──────────────────────────────────────────────────────────────────────────────
# MAIN — Stage 1 smoke test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import json as json_mod
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser(description="Stage 1: Data Layer")
    parser.add_argument("--organism", default="ecoli",
                        choices=["ecoli", "ecoli_bl21", "scerevisiae",
                                 "scerevisiae_by", "bsubtilis", "cglutamicum", "pputida"])
    parser.add_argument("--molecule", default="lycopene",
                        choices=["lycopene", "vanillin", "artemisinic_acid",
                                 "lysine", "glutamate", "threonine", "pha",
                                 "hyaluronic_acid", "riboflavin"])
    args = parser.parse_args()

    cfg = PipelineConfig(organism_key=args.organism, molecule_key=args.molecule)

    # Run stage 1
    result = run_stage_1(cfg)

    # Print summary
    print(f"\n{'='*60}")
    print(f"  STAGE 1 RESULT — {args.organism} → {args.molecule}")
    print(f"{'='*60}")
    print(f"  Pipeline ID  : {result['pipeline_id']}")
    print(f"  Status       : {result['stage_1_status']}")
    print(f"  Organism     : {result['organism']['name']} {result['organism']['strain']}")
    print(f"  Molecule     : {result['target_molecule']['name']}")
    print(f"  Genes        : {result['genomic_data']['total_genes']}")
    print(f"  Quality Score: {result['data_quality_report']['completeness_score']:.3f}")
    print(f"  Warnings     : {len(result['data_quality_report']['warnings'])}")
    print(f"  Errors       : {len(result['data_quality_report']['errors'])}")
    print(f"  Log file     : {PipelineLogger.get_instance().get_log_file_path()}")
    print(f"{'='*60}")

    # Save output JSON
    os.makedirs("pipeline_output", exist_ok=True)
    out_path = os.path.join("pipeline_output", "stage_1_output.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json_mod.dump(result, fh, indent=2, default=str)
    print(f"  Output saved : {out_path}")

    print("\n▶ STAGE 1 COMPLETE ◀  Type: CONTINUE TO STAGE 2")
