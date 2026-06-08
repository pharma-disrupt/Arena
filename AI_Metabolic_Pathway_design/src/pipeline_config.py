"""
Pipeline Configuration Module

Defines all configurable parameters for the metabolic pathway design pipeline
using dataclasses. Contains organism-specific configs, molecule targets,
and logging settings.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ──────────────────────────────────────────────────────────────────────────────
# ORGANISM CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class OrganismConfig:
    """Immutable configuration for a single industrial microorganism."""

    name: str
    strain: str
    gem_model_id: str
    doubling_time_min: float
    optimal_ph: float
    optimal_temp_c: float
    gram_stain: str
    genome_size_bp: int = 0
    total_genes: int = 0
    essential_genes: List[str] = field(default_factory=list)
    available_promoters: List[str] = field(default_factory=list)
    codon_table_id: int = 11  # Standard bacterial/archaeal table
    gc_content_percent: float = 50.0
    oxygen_requirement: str = "aerobic"
    biosafety_level: int = 1


ORGANISM_DATABASE: Dict[str, OrganismConfig] = {
    "ecoli": OrganismConfig(
        name="Escherichia coli",
        strain="K-12 MG1655",
        gem_model_id="iML1515",
        doubling_time_min=20.0,
        optimal_ph=7.0,
        optimal_temp_c=37.0,
        gram_stain="negative",
        genome_size_bp=4_641_652,
        total_genes=4_494,
        essential_genes=["dnaA", "dnaN", "dnaG", "rpoB", "rpoC", "rpoA",
                         "gyrA", "gyrB", "ftsZ", "ftsA", "murA", "murB",
                         "tufA", "tufB", "rplA", "rplB", "rplC", "rplD"],
        available_promoters=["Ptac", "Ptrc", "Plac", "ParaBAD", "PT7",
                             "PlacUV5", "Pbad", "PrhaBAD"],
        codon_table_id=11,
        gc_content_percent=50.8,
        oxygen_requirement="facultative anaerobe",
        biosafety_level=1,
    ),
    "ecoli_bl21": OrganismConfig(
        name="Escherichia coli",
        strain="BL21(DE3)",
        gem_model_id="iML1515",
        doubling_time_min=22.0,
        optimal_ph=7.0,
        optimal_temp_c=37.0,
        gram_stain="negative",
        genome_size_bp=4_609_491,
        total_genes=4_405,
        essential_genes=["dnaA", "dnaN", "rpoB", "rpoC", "gyrA", "ftsZ",
                         "tufA", "rplA", "rplB"],
        available_promoters=["PT7", "Ptac", "Ptrc", "PlacUV5"],
        codon_table_id=11,
        gc_content_percent=50.6,
        oxygen_requirement="facultative anaerobe",
        biosafety_level=1,
    ),
    "scerevisiae": OrganismConfig(
        name="Saccharomyces cerevisiae",
        strain="S288C",
        gem_model_id="yeast8",
        doubling_time_min=90.0,
        optimal_ph=5.5,
        optimal_temp_c=30.0,
        gram_stain="positive",
        genome_size_bp=12_157_105,
        total_genes=6_604,
        essential_genes=["ACT1", "TUB1", "TUB2", "CDC28", "RPB1", "RPB2",
                         "RPB3", "NUP49", "NUP57", "SEC61"],
        available_promoters=["PGAL1", "PGAL10", "PTEF1", "PGPD", "PADC1",
                             "PCYC1", "PTPI1", "PMET25"],
        codon_table_id=1,  # Standard eukaryotic
        gc_content_percent=38.3,
        oxygen_requirement="facultative anaerobe",
        biosafety_level=1,
    ),
    "scerevisiae_by": OrganismConfig(
        name="Saccharomyces cerevisiae",
        strain="BY4741",
        gem_model_id="yeast8",
        doubling_time_min=95.0,
        optimal_ph=5.5,
        optimal_temp_c=30.0,
        gram_stain="positive",
        genome_size_bp=12_150_000,
        total_genes=6_550,
        essential_genes=["ACT1", "TUB1", "TUB2", "CDC28", "RPB1", "RPB2"],
        available_promoters=["PGAL1", "PGAL10", "PTEF1", "PGPD", "PADC1"],
        codon_table_id=1,
        gc_content_percent=38.3,
        oxygen_requirement="facultative anaerobe",
        biosafety_level=1,
    ),
    "bsubtilis": OrganismConfig(
        name="Bacillus subtilis",
        strain="168",
        gem_model_id="iYO844",
        doubling_time_min=25.0,
        optimal_ph=7.0,
        optimal_temp_c=37.0,
        gram_stain="positive",
        genome_size_bp=4_214_810,
        total_genes=4_100,
        essential_genes=["dnaA", "dnaB", "dnaC", "rpoB", "rpoC", "ftsZ",
                         "murA", "accD", "fabH", "tuf"],
        available_promoters=["Pveg", "P43", "Phyperspank", "Pgrac",
                             "PspoVG", "PaprE", "PsacB"],
        codon_table_id=11,
        gc_content_percent=43.5,
        oxygen_requirement="aerobic",
        biosafety_level=1,
    ),
    "cglutamicum": OrganismConfig(
        name="Corynebacterium glutamicum",
        strain="ATCC 13032",
        gem_model_id="iCW773",
        doubling_time_min=60.0,
        optimal_ph=7.2,
        optimal_temp_c=30.0,
        gram_stain="positive",
        genome_size_bp=3_282_708,
        total_genes=3_057,
        essential_genes=["dnaA", "rpoB", "rpoC", "ftsZ", "murA", "secA",
                         "tuf", "rplA", "rplB", "gyrA"],
        available_promoters=["Ptac", "Psod", "PgapA", "PfdhF", "Ptuf",
                             "Pcg0929", "Pncgl0049"],
        codon_table_id=11,
        gc_content_percent=53.8,
        oxygen_requirement="aerobic",
        biosafety_level=1,
    ),
    "pputida": OrganismConfig(
        name="Pseudomonas putida",
        strain="KT2440",
        gem_model_id="iJN746",
        doubling_time_min=40.0,
        optimal_ph=7.0,
        optimal_temp_c=30.0,
        gram_stain="negative",
        genome_size_bp=6_181_863,
        total_genes=5_452,
        essential_genes=["dnaA", "rpoB", "rpoC", "ftsZ", "gyrA", "gyrB",
                         "tufA", "rplA", "rplB", "secA"],
        available_promoters=["Ptac", "Pbad", "Plac", "Pm", "Pu", "Pm/XylS",
                             "PrpoD"],
        codon_table_id=11,
        gc_content_percent=61.6,
        oxygen_requirement="aerobic",
        biosafety_level=1,
    ),
}


# ──────────────────────────────────────────────────────────────────────────────
# MOLECULE / TARGET PRODUCT CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MoleculeConfig:
    """Immutable configuration for a target product molecule."""

    name: str
    smiles: str
    chebi_id: str
    target_titer_g_per_l: float
    target_yield_mol_per_mol: float
    molecular_weight_g_per_mol: float = 0.0
    product_category: str = "small_molecule"  # small_molecule, amino_acid, biopolymer, vitamin
    logp: float = 0.0  # Octanol-water partition coefficient
    water_solubility_mg_per_l: float = 0.0
    toxicity_class: str = "low"


MOLECULE_DATABASE: Dict[str, MoleculeConfig] = {
    "lycopene": MoleculeConfig(
        name="Lycopene",
        smiles="CC(C)=CCCC(C)=C/C=C/C(C)=C/C=C/C(C)=C/C=C/C=C(C)/C=C/C=C(C)/C=C/C=C(C)C",
        chebi_id="CHEBI:15938",
        target_titer_g_per_l=5.0,
        target_yield_mol_per_mol=0.35,
        molecular_weight_g_per_mol=536.85,
        product_category="small_molecule",
        logp=17.6,
        water_solubility_mg_per_l=0.003,
        toxicity_class="low",
    ),
    "vanillin": MoleculeConfig(
        name="Vanillin",
        smiles="COc1cc(C=O)ccc1O",
        chebi_id="CHEBI:27430",
        target_titer_g_per_l=15.0,
        target_yield_mol_per_mol=0.45,
        molecular_weight_g_per_mol=152.15,
        product_category="small_molecule",
        logp=1.21,
        water_solubility_mg_per_l=10000.0,
        toxicity_class="low",
    ),
    "artemisinic_acid": MoleculeConfig(
        name="Artemisinic Acid",
        smiles="CC1=C/C(=C/C=C1/C(=O)O)C2CC(=C)CCC2(C)C",
        chebi_id="CHEBI:57392",
        target_titer_g_per_l=25.0,
        target_yield_mol_per_mol=0.30,
        molecular_weight_g_per_mol=234.33,
        product_category="small_molecule",
        logp=3.8,
        water_solubility_mg_per_l=50.0,
        toxicity_class="low",
    ),
    "lysine": MoleculeConfig(
        name="L-Lysine",
        smiles="NCCCC[C@H](N)C(=O)O",
        chebi_id="CHEBI:18033",
        target_titer_g_per_l=120.0,
        target_yield_mol_per_mol=0.65,
        molecular_weight_g_per_mol=146.19,
        product_category="amino_acid",
        logp=-3.0,
        water_solubility_mg_per_l=1500000.0,
        toxicity_class="low",
    ),
    "glutamate": MoleculeConfig(
        name="L-Glutamate",
        smiles="C(CC(=O)O)[C@@H](C(=O)O)N",
        chebi_id="CHEBI:29987",
        target_titer_g_per_l=100.0,
        target_yield_mol_per_mol=0.60,
        molecular_weight_g_per_mol=147.13,
        product_category="amino_acid",
        logp=-3.0,
        water_solubility_mg_per_l=860000.0,
        toxicity_class="low",
    ),
    "threonine": MoleculeConfig(
        name="L-Threonine",
        smiles="C[C@H]([C@@H](C(=O)O)N)O",
        chebi_id="CHEBI:16847",
        target_titer_g_per_l=80.0,
        target_yield_mol_per_mol=0.55,
        molecular_weight_g_per_mol=119.12,
        product_category="amino_acid",
        logp=-3.1,
        water_solubility_mg_per_l=250000.0,
        toxicity_class="low",
    ),
    "pha": MoleculeConfig(
        name="Polyhydroxyalkanoate (PHA)",
        smiles="CC(C)CC(=O)OCC(C)(C)C",  # Simplified PHB monomer
        chebi_id="CHEBI:26457",
        target_titer_g_per_l=80.0,
        target_yield_mol_per_mol=0.50,
        molecular_weight_g_per_mol=86.09,
        product_category="biopolymer",
        logp=1.5,
        water_solubility_mg_per_l=13000.0,
        toxicity_class="low",
    ),
    "hyaluronic_acid": MoleculeConfig(
        name="Hyaluronic Acid",
        smiles="CC(=O)N[C@H]1[C@@H](O)C[C@@H](O)[C@H](O)[C@H]1O",  # Simplified disaccharide
        chebi_id="CHEBI:28468",
        target_titer_g_per_l=6.0,
        target_yield_mol_per_mol=0.25,
        molecular_weight_g_per_mol=776.65,
        product_category="biopolymer",
        logp=-5.0,
        water_solubility_mg_per_l=1000000.0,
        toxicity_class="low",
    ),
    "riboflavin": MoleculeConfig(
        name="Riboflavin (Vitamin B2)",
        smiles="CC1=CC(=O)c2c(N)ncnc2N1C[C@H](O)[C@@H](O)[C@@H](O)CCO",
        chebi_id="CHEBI:17807",
        target_titer_g_per_l=10.0,
        target_yield_mol_per_mol=0.40,
        molecular_weight_g_per_mol=376.36,
        product_category="vitamin",
        logp=-1.4,
        water_solubility_mg_per_l=100.0,
        toxicity_class="low",
    ),
}


# ──────────────────────────────────────────────────────────────────────────────
# LOGGING CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class LoggingConfig:
    """Configuration for pipeline logging behaviour."""

    log_dir: str = "logs"
    log_file_prefix: str = "pipeline"
    console_level: int = 20        # INFO
    file_level: int = 10           # DEBUG
    log_format: str = (
        "%(asctime)s | %(levelname)-8s | STAGE:%(stage)s | "
        "%(funcName)s | %(message)s"
    )
    date_format: str = "%Y-%m-%d %H:%M:%S"
    max_file_size_mb: int = 50
    backup_count: int = 5


# ──────────────────────────────────────────────────────────────────────────────
# MASTER PIPELINE CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    """Top-level configurable parameters for the entire pipeline."""

    # Identification
    pipeline_id: str = ""  # Auto-generated UUID if empty
    output_dir: str = "./pipeline_output"

    # Target organism key — must match ORGANISM_DATABASE
    organism_key: str = "ecoli"

    # Target molecule key — must match MOLECULE_DATABASE
    molecule_key: str = "lycopene"

    # Pipeline behaviour
    dbtl_cycles: int = 3
    max_pathway_candidates: int = 10
    fba_solver: str = "highs"  # or "simplex"
    use_ml_fallback: bool = True  # Graceful degradation toggle

    # Scale-up
    scale_levels: List[float] = field(default_factory=lambda: [2.0, 200.0, 20000.0])

    # Sub-configs
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def __post_init__(self) -> None:
        """Derive organism and molecule configs after initialisation."""
        if self.organism_key not in ORGANISM_DATABASE:
            raise ValueError(
                f"Unknown organism_key='{self.organism_key}'. "
                f"Choose from: {list(ORGANISM_DATABASE.keys())}"
            )
        if self.molecule_key not in MOLECULE_DATABASE:
            raise ValueError(
                f"Unknown molecule_key='{self.molecule_key}'. "
                f"Choose from: {list(MOLECULE_DATABASE.keys())}"
            )
        self.organism_config: OrganismConfig = ORGANISM_DATABASE[self.organism_key]
        self.molecule_config: MoleculeConfig = MOLECULE_DATABASE[self.molecule_key]

    @property
    def organism_name(self) -> str:
        return self.organism_config.name

    @property
    def organism_strain(self) -> str:
        return self.organism_config.strain

    @property
    def molecule_name(self) -> str:
        return self.molecule_config.name


# ──────────────────────────────────────────────────────────────────────────────
# CLI-friendly helpers
# ──────────────────────────────────────────────────────────────────────────────

ORGANISM_CHOICES = list(ORGANISM_DATABASE.keys())
MOLECULE_CHOICES = list(MOLECULE_DATABASE.keys())


if __name__ == "__main__":
    import json
    import logging as py_logging

    # Quick sanity check
    py_logging.basicConfig(level=py_logging.DEBUG)
    logger = py_logging.getLogger(__name__)

    for key, org in ORGANISM_DATABASE.items():
        logger.info(
            "Organism [%s]: %s %s  |  DT=%.0f min  |  GC=%.1f%%  |  %d genes",
            key, org.name, org.strain, org.doubling_time_min,
            org.gc_content_percent, org.total_genes,
        )

    for key, mol in MOLECULE_DATABASE.items():
        logger.info(
            "Molecule [%s]: %s  |  titer=%.1f g/L  |  yield=%.2f mol/mol",
            key, mol.name, mol.target_titer_g_per_l, mol.target_yield_mol_per_mol,
        )

    cfg = PipelineConfig(organism_key="ecoli", molecule_key="lycopene")
    logger.info("PipelineConfig created: organism=%s, molecule=%s",
                cfg.organism_name, cfg.molecule_name)
    logger.info("All configs valid.")
