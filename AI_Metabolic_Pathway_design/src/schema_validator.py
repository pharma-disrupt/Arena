"""
Schema Validator Module

Defines JSON schemas for inter-stage data contracts and provides
validation utilities using the `jsonschema` library.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:
    import jsonschema
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False
    jsonschema = None  # type: ignore

from exceptions import SchemaValidationError


# ──────────────────────────────────────────────────────────────────────────────
# JSON SCHEMAS — One per inter-stage handoff
# ──────────────────────────────────────────────────────────────────────────────

SCHEMA_VERSION = "1.0.0"

# ── Stage 1 → Stage 2 ────────────────────────────────────────────────────

STAGE_1_OUTPUT_SCHEMA: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "Stage 1 Output",
    "version": SCHEMA_VERSION,
    "type": "object",
    "required": [
        "pipeline_id", "timestamp", "organism", "target_molecule",
        "genomic_data", "data_quality_report", "stage_1_status",
    ],
    "properties": {
        "pipeline_id": {"type": "string", "format": "uuid"},
        "timestamp": {"type": "string"},
        "organism": {
            "type": "object",
            "required": ["name", "strain", "gem_model_id", "doubling_time_min",
                         "optimal_ph", "optimal_temp_c", "gram_stain"],
            "properties": {
                "name": {"type": "string"},
                "strain": {"type": "string"},
                "gem_model_id": {"type": "string"},
                "doubling_time_min": {"type": "number"},
                "optimal_ph": {"type": "number"},
                "optimal_temp_c": {"type": "number"},
                "gram_stain": {"type": "string"},
            },
        },
        "target_molecule": {
            "type": "object",
            "required": ["name", "smiles", "chebi_id", "target_titer_g_per_l",
                         "target_yield_mol_per_mol"],
            "properties": {
                "name": {"type": "string"},
                "smiles": {"type": "string"},
                "chebi_id": {"type": "string"},
                "target_titer_g_per_l": {"type": "number"},
                "target_yield_mol_per_mol": {"type": "number"},
            },
        },
        "genomic_data": {
            "type": "object",
            "required": ["total_genes", "essential_genes", "available_promoters",
                         "codon_table_id", "gc_content_percent"],
            "properties": {
                "total_genes": {"type": "integer"},
                "essential_genes": {"type": "array", "items": {"type": "string"}},
                "available_promoters": {"type": "array", "items": {"type": "string"}},
                "codon_table_id": {"type": "integer"},
                "gc_content_percent": {"type": "number"},
            },
        },
        "data_quality_report": {
            "type": "object",
            "required": ["completeness_score", "warnings", "errors"],
            "properties": {
                "completeness_score": {"type": "number"},
                "warnings": {"type": "array", "items": {"type": "string"}},
                "errors": {"type": "array", "items": {"type": "string"}},
            },
        },
        "stage_1_status": {"type": "string", "enum": ["PASS", "FAIL", "WARN"]},
    },
}

# ── Stage 2 → Stage 3 ────────────────────────────────────────────────────

STAGE_2_OUTPUT_SCHEMA: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "Stage 2 Output",
    "version": SCHEMA_VERSION,
    "type": "object",
    "required": [
        "pipeline_id", "stage_1_output", "pathway_candidates",
        "gene_modifications", "stage_2_status",
    ],
    "properties": {
        "pipeline_id": {"type": "string"},
        "stage_1_output": {"type": "object"},
        "pathway_candidates": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": [
                    "pathway_id", "rank", "pathway_name", "steps",
                    "total_steps", "predicted_yield_mol_per_mol",
                    "thermodynamic_feasibility_score",
                    "gnn_viability_score", "host_compatibility_score",
                ],
                "properties": {
                    "pathway_id": {"type": "string"},
                    "rank": {"type": "integer"},
                    "pathway_name": {"type": "string"},
                    "steps": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": [
                                "step_number", "reaction_id", "enzyme_name",
                                "gene_name", "ec_number", "substrate", "product",
                                "delta_g_kj_per_mol", "kcat_per_sec", "km_mm",
                                "is_heterologous", "source_organism",
                            ],
                            "properties": {
                                "step_number": {"type": "integer"},
                                "reaction_id": {"type": "string"},
                                "enzyme_name": {"type": "string"},
                                "gene_name": {"type": "string"},
                                "ec_number": {"type": "string"},
                                "substrate": {"type": "string"},
                                "product": {"type": "string"},
                                "delta_g_kj_per_mol": {"type": "number"},
                                "kcat_per_sec": {"type": "number"},
                                "km_mm": {"type": "number"},
                                "is_heterologous": {"type": "boolean"},
                                "source_organism": {"type": "string"},
                            },
                        },
                    },
                    "total_steps": {"type": "integer"},
                    "predicted_yield_mol_per_mol": {"type": "number"},
                    "thermodynamic_feasibility_score": {"type": "number"},
                    "gnn_viability_score": {"type": "number"},
                    "host_compatibility_score": {"type": "number"},
                },
            },
        },
        "gene_modifications": {
            "type": "object",
            "required": ["knockouts", "overexpressions", "heterologous_insertions"],
            "properties": {
                "knockouts": {"type": "array", "items": {"type": "string"}},
                "overexpressions": {"type": "array", "items": {"type": "string"}},
                "heterologous_insertions": {"type": "array", "items": {"type": "string"}},
            },
        },
        "codon_optimized_sequences": {"type": "object"},
        "stage_2_status": {"type": "string", "enum": ["PASS", "FAIL", "WARN"]},
    },
}

# ── Stage 3 → Stage 4 ────────────────────────────────────────────────────

STAGE_3_OUTPUT_SCHEMA: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "Stage 3 Output",
    "version": SCHEMA_VERSION,
    "type": "object",
    "required": [
        "pipeline_id", "stage_2_output", "fba_results",
        "strain_design", "toxicity_assessment", "stage_3_status",
    ],
    "properties": {
        "pipeline_id": {"type": "string"},
        "stage_2_output": {"type": "object"},
        "fba_results": {
            "type": "object",
            "required": [
                "objective_value", "growth_rate_per_hour",
                "product_flux_mmol_per_gdw_per_hour",
                "substrate_uptake_mmol_per_gdw_per_hour",
                "theoretical_max_yield", "flux_map",
            ],
            "properties": {
                "objective_value": {"type": "number"},
                "growth_rate_per_hour": {"type": "number"},
                "product_flux_mmol_per_gdw_per_h": {"type": "number"},
                "substrate_uptake_mmol_per_gdw_per_h": {"type": "number"},
                "theoretical_max_yield": {"type": "number"},
                "flux_map": {"type": "object"},
            },
        },
        "strain_design": {
            "type": "object",
            "required": [
                "algorithm_used", "final_knockouts", "final_overexpressions",
                "predicted_titer_g_per_l", "predicted_productivity_g_per_l_per_h",
                "metabolic_burden_score", "genetic_stability_score",
            ],
            "properties": {
                "algorithm_used": {"type": "string"},
                "final_knockouts": {"type": "array", "items": {"type": "string"}},
                "final_overexpressions": {"type": "array", "items": {"type": "string"}},
                "predicted_titer_g_per_l": {"type": "number"},
                "predicted_productivity_g_per_l_per_h": {"type": "number"},
                "metabolic_burden_score": {"type": "number"},
                "genetic_stability_score": {"type": "number"},
            },
        },
        "toxicity_assessment": {
            "type": "object",
            "required": [
                "intermediate_toxicity_scores", "overall_toxicity_risk",
                "flagged_intermediates",
            ],
            "properties": {
                "intermediate_toxicity_scores": {"type": "object"},
                "overall_toxicity_risk": {
                    "type": "string", "enum": ["LOW", "MEDIUM", "HIGH"],
                },
                "flagged_intermediates": {"type": "array", "items": {"type": "string"}},
            },
        },
        "stage_3_status": {"type": "string", "enum": ["PASS", "FAIL", "WARN"]},
    },
}

# ── Stage 4 → Stage 5 ────────────────────────────────────────────────────

STAGE_4_OUTPUT_SCHEMA: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "Stage 4 Output",
    "version": SCHEMA_VERSION,
    "type": "object",
    "required": [
        "pipeline_id", "stage_3_output", "dbtl_cycles",
        "fermentation_simulation", "optimal_fermentation_conditions",
        "stage_4_status",
    ],
    "properties": {
        "pipeline_id": {"type": "string"},
        "stage_3_output": {"type": "object"},
        "dbtl_cycles": {
            "type": "array",
            "items": {
                "type": "object",
                "required": [
                    "cycle_number", "constructs_tested", "best_titer_g_per_l",
                    "best_construct_id", "improvement_fold", "bo_next_candidates",
                ],
                "properties": {
                    "cycle_number": {"type": "integer"},
                    "constructs_tested": {"type": "integer"},
                    "best_titer_g_per_l": {"type": "number"},
                    "best_construct_id": {"type": "string"},
                    "improvement_fold": {"type": "number"},
                    "bo_next_candidates": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "fermentation_simulation": {
            "type": "object",
            "required": [
                "mode", "duration_hours", "final_titer_g_per_l",
                "final_yield_g_per_g", "final_productivity_g_per_l_per_h",
                "ode_convergence", "organism_specific_events",
            ],
            "properties": {
                "mode": {"type": "string"},
                "duration_hours": {"type": "number"},
                "final_titer_g_per_l": {"type": "number"},
                "final_yield_g_per_g": {"type": "number"},
                "final_productivity_g_per_l_per_h": {"type": "number"},
                "ode_convergence": {"type": "boolean"},
                "organism_specific_events": {"type": "array", "items": {"type": "string"}},
            },
        },
        "optimal_fermentation_conditions": {
            "type": "object",
            "required": [
                "temperature_c", "ph", "do_percent_saturation",
                "glucose_feed_g_per_l_per_h", "agitation_rpm", "aeration_vvm",
            ],
            "properties": {
                "temperature_c": {"type": "number"},
                "ph": {"type": "number"},
                "do_percent_saturation": {"type": "number"},
                "glucose_feed_g_per_l_per_h": {"type": "number"},
                "agitation_rpm": {"type": "number"},
                "aeration_vvm": {"type": "number"},
            },
        },
        "stage_4_status": {"type": "string", "enum": ["PASS", "FAIL", "WARN"]},
    },
}

# ── Schema registry ──────────────────────────────────────────────────────

SCHEMA_REGISTRY: Dict[str, Dict[str, Any]] = {
    "stage_1_output": STAGE_1_OUTPUT_SCHEMA,
    "stage_2_output": STAGE_2_OUTPUT_SCHEMA,
    "stage_3_output": STAGE_3_OUTPUT_SCHEMA,
    "stage_4_output": STAGE_4_OUTPUT_SCHEMA,
}

# Map from stage number → which schema to validate as *input*
STAGE_INPUT_SCHEMA_MAP: Dict[int, str] = {
    1: "",  # Stage 1 has no upstream JSON input
    2: "stage_1_output",
    3: "stage_2_output",
    4: "stage_3_output",
    5: "stage_4_output",
}

# Map from stage number → which schema to validate as *output*
STAGE_OUTPUT_SCHEMA_MAP: Dict[int, str] = {
    1: "stage_1_output",
    2: "stage_2_output",
    3: "stage_3_output",
    4: "stage_4_output",
    5: "stage_5_output",
}


# ──────────────────────────────────────────────────────────────────────────────
# VALIDATION FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    """Result of a schema validation operation."""

    passed: bool
    schema_name: str
    payload_summary: str
    errors: List[str] = field(default_factory=list)
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "schema_name": self.schema_name,
            "errors": self.errors,
            "payload_summary": self.payload_summary,
            "timestamp": self.timestamp,
        }


def _summarise_payload(payload: Dict[str, Any]) -> str:
    """Short summary of a JSON payload for logging."""
    parts: List[str] = []
    for key, val in payload.items():
        if isinstance(val, dict):
            parts.append(f"{key}={{...}}")
        elif isinstance(val, list):
            parts.append(f"{key}=[{len(val)}]")
        else:
            parts.append(f"{key}={val}")
    return "; ".join(parts)


def validate_stage_output(
    payload: Dict[str, Any],
    schema_name: str,
    logger: Optional[Any] = None,
) -> ValidationResult:
    """
    Validate a JSON payload against the named schema.

    Parameters
    ----------
    payload : dict
        The data to validate.
    schema_name : str
        Key in SCHEMA_REGISTRY, e.g. "stage_1_output".
    logger : optional
        PipelineLogger instance or any object with .warning() / .error() methods.

    Returns
    -------
    ValidationResult
    """
    if schema_name not in SCHEMA_REGISTRY:
        msg = f"Unknown schema_name='{schema_name}'. Available: {list(SCHEMA_REGISTRY.keys())}"
        if logger:
            logger.error(msg)
        return ValidationResult(passed=False, schema_name=schema_name,
                                payload_summary="", errors=[msg])

    schema = SCHEMA_REGISTRY[schema_name]

    if not HAS_JSONSCHEMA:
        warn_msg = (
            "jsonschema package not installed — performing only basic "
            "presence check. Install with: pip install jsonschema"
        )
        if logger:
            logger.warning(warn_msg)
        # Minimal check: ensure all top-level required keys exist
        required = schema.get("required", [])
        missing = [k for k in required if k not in payload]
        if missing:
            errs = [f"Missing required key: {k}" for k in missing]
            if logger:
                for e in errs:
                    logger.error(e)
            return ValidationResult(
                passed=False,
                schema_name=schema_name,
                payload_summary=_summarise_payload(payload),
                errors=errs,
            )
        if logger:
            logger.info("Schema validation (basic): PASSED for %s", schema_name)
        return ValidationResult(
            passed=True,
            schema_name=schema_name,
            payload_summary=_summarise_payload(payload),
        )

    try:
        jsonschema.validate(instance=payload, schema=schema)
        if logger:
            logger.info("Schema validation: PASSED for %s", schema_name)
        return ValidationResult(
            passed=True,
            schema_name=schema_name,
            payload_summary=_summarise_payload(payload),
        )
    except jsonschema.ValidationError as ve:
        if logger:
            logger.error("Schema validation FAILED for %s: %s", schema_name, ve.message)
        return ValidationResult(
            passed=False,
            schema_name=schema_name,
            payload_summary=_summarise_payload(payload),
            errors=[ve.message],
        )
    except jsonschema.SchemaError as se:
        if logger:
            logger.critical("Schema definition error for %s: %s", schema_name, se.message)
        return ValidationResult(
            passed=False,
            schema_name=schema_name,
            payload_summary=_summarise_payload(payload),
            errors=[str(se)],
        )


def validate_and_raise(
    payload: Dict[str, Any],
    schema_name: str,
    logger: Optional[Any] = None,
) -> None:
    """Validate payload and raise SchemaValidationError on failure."""
    result = validate_stage_output(payload, schema_name, logger)
    if not result.passed:
        raise SchemaValidationError(
            message=f"Validation failed for {schema_name}: {result.errors}",
            schema_name=schema_name,
            validation_errors=result.errors,
            payload=payload,
        )


def save_schema_files(output_dir: str = "schemas") -> None:
    """Persist all schemas as individual JSON files for external reference."""
    os.makedirs(output_dir, exist_ok=True)
    for name, schema in SCHEMA_REGISTRY.items():
        path = os.path.join(output_dir, f"{name}_schema.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(schema, fh, indent=2, default=str)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN — Validation smoke test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import logging as py_logging

    py_logging.basicConfig(level=py_logging.DEBUG)
    logger = py_logging.getLogger(__name__)

    # Test valid stage 1 output
    test_s1 = {
        "pipeline_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
        "timestamp": "2026-06-08T12:00:00+00:00",
        "organism": {
            "name": "Escherichia coli",
            "strain": "K-12 MG1655",
            "gem_model_id": "iML1515",
            "doubling_time_min": 20.0,
            "optimal_ph": 7.0,
            "optimal_temp_c": 37.0,
            "gram_stain": "negative",
        },
        "target_molecule": {
            "name": "Lycopene",
            "smiles": "CC(C)=CCCC(C)=C/C=C/C(C)=C/C=C/C(C)=C/C=C/C=C(C)/C=C/C=C(C)/C=C/C=C(C)C",
            "chebi_id": "CHEBI:15938",
            "target_titer_g_per_l": 5.0,
            "target_yield_mol_per_mol": 0.35,
        },
        "genomic_data": {
            "total_genes": 4494,
            "essential_genes": ["dnaA", "dnaN", "dnaG"],
            "available_promoters": ["Ptac", "Ptrc", "Plac"],
            "codon_table_id": 11,
            "gc_content_percent": 50.8,
        },
        "data_quality_report": {
            "completeness_score": 0.95,
            "warnings": [],
            "errors": [],
        },
        "stage_1_status": "PASS",
    }

    result = validate_stage_output(test_s1, "stage_1_output", logger)
    logger.info("Validation result: %s", result.to_dict())
    assert result.passed, f"Expected PASS, got {result.errors}"

    # Test invalid payload (missing key)
    bad_payload = {"pipeline_id": "xxx"}
    result2 = validate_stage_output(bad_payload, "stage_1_output", logger)
    logger.info("Validation result (bad): %s", result2.to_dict())
    assert not result2.passed, "Expected FAIL for incomplete payload"

    # Test stage 2 schema
    test_s2 = {
        "pipeline_id": "b1b2c3d4-e5f6-7890-abcd-ef1234567890",
        "stage_1_output": test_s1,
        "pathway_candidates": [
            {
                "pathway_id": "path_001",
                "rank": 1,
                "pathway_name": "MEP → lycopene",
                "steps": [
                    {
                        "step_number": 1,
                        "reaction_id": "R00001",
                        "enzyme_name": "Dxs",
                        "gene_name": "dxs",
                        "ec_number": "2.2.1.7",
                        "substrate": "pyruvate + G3P",
                        "product": "DOXP",
                        "delta_g_kj_per_mol": -3.5,
                        "kcat_per_sec": 12.5,
                        "km_mm": 0.45,
                        "is_heterologous": False,
                        "source_organism": "Escherichia coli",
                    },
                ],
                "total_steps": 1,
                "predicted_yield_mol_per_mol": 0.28,
                "thermodynamic_feasibility_score": 0.85,
                "gnn_viability_score": 0.72,
                "host_compatibility_score": 0.90,
            },
        ],
        "gene_modifications": {
            "knockouts": ["ldhA"],
            "overexpressions": ["dxs", "idi"],
            "heterologous_insertions": ["crtE", "crtB", "crtI"],
        },
        "codon_optimized_sequences": {},
        "stage_2_status": "PASS",
    }

    result3 = validate_stage_output(test_s2, "stage_2_output", logger)
    logger.info("Stage 2 validation result: %s", result3.to_dict())
    assert result3.passed, f"Expected PASS, got {result3.errors}"

    # Save schemas to disk
    save_schema_files()
    logger.info("Schema files saved to ./schemas/")

    logger.info("All schema validation tests passed.")
