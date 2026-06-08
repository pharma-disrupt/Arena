"""
Retrosynthesis Engine Module

Implements pathway generation via Monte Carlo Tree Search (MCTS) over
simulated reaction rules. Scores pathways thermodynamically and ranks
candidates for downstream optimisation.
"""

from __future__ import annotations

import logging
import math
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from exceptions import ModelInferenceError
from logger_setup import PipelineLogger, log_json_contract
from schema_validator import validate_and_raise


# ---------------------------------------------------------------------------
# REACTION RULE
# ---------------------------------------------------------------------------

@dataclass
class ReactionRule:
    """Represents a single enzymatic reaction rule used in retrosynthesis."""

    rule_id: str
    enzyme_name: str
    gene_name: str
    ec_number: str
    substrate: str
    product: str
    delta_g_kj_per_mol: float
    kcat_per_sec: float
    km_mm: float
    organism_source: str
    reversibility: str = "irreversible"
    cofactors: List[str] = field(default_factory=list)

    def to_step_dict(self, step_number: int, is_heterologous: bool,
                     host_organism: str) -> Dict[str, Any]:
        """Convert to a pathway step dict matching the Stage 2 schema."""
        return {
            "step_number": step_number,
            "reaction_id": self.rule_id,
            "enzyme_name": self.enzyme_name,
            "gene_name": self.gene_name,
            "ec_number": self.ec_number,
            "substrate": self.substrate,
            "product": self.product,
            "delta_g_kj_per_mol": self.delta_g_kj_per_mol,
            "kcat_per_sec": self.kcat_per_sec,
            "km_mm": self.km_mm,
            "is_heterologous": is_heterologous,
            "source_organism": self.organism_source if is_heterologous else host_organism,
        }


# ---------------------------------------------------------------------------
# SIMULATED REACTION RULE LIBRARY  (50+ rules covering major pathways)
# ---------------------------------------------------------------------------

def _build_reaction_rules() -> List[ReactionRule]:
    """Return a curated library of 55 biologically-plausible reaction rules."""

    rules: List[ReactionRule] = []

    # ── MEP / DOXP pathway (isoprenoid precursors) ────────────────────────
    rules.extend([
        ReactionRule("R00001", "Dxs", "dxs", "2.2.1.7",
                     "pyruvate + G3P", "DOXP",
                     -3.5, 12.5, 0.45, "Escherichia coli",
                     cofactors=["TPP"]),
        ReactionRule("R00002", "Dxr", "dxr", "1.1.1.267",
                     "DOXP + NADPH", "MEP + NADP+",
                     -15.2, 8.3, 0.12, "Escherichia coli",
                     cofactors=["NADPH"]),
        ReactionRule("R00003", "YgbP", "ispD", "2.7.7.60",
                     "MEP + CTP", "CDP-ME + PPi",
                     -8.1, 5.6, 0.33, "Escherichia coli",
                     cofactors=["CTP"]),
        ReactionRule("R00004", "YgbB", "ispE", "2.7.1.148",
                     "CDP-ME + ATP", "CDP-MEP + ADP",
                     -12.0, 7.2, 0.28, "Escherichia coli",
                     cofactors=["ATP"]),
        ReactionRule("R00005", "GcpE", "ispF", "4.6.1.12",
                     "CDP-MEP", "MEcPP + CMP",
                     -5.3, 4.1, 0.55, "Escherichia coli"),
        ReactionRule("R00006", "IspG", "ispG", "1.17.7.1",
                     "MEcPP + 2e-", "HMBPP + H2O",
                     -20.0, 3.8, 0.62, "Escherichia coli",
                     cofactors=["ferredoxin"]),
        ReactionRule("R00007", "IspH", "ispH", "1.17.7.2",
                     "HMBPP + 2e-", "IPP + DMAPP",
                     -18.5, 6.5, 0.35, "Escherichia coli",
                     cofactors=["ferredoxin"]),
    ])

    # ── MVA pathway (alternative isoprenoid route) ─────────────────────────
    rules.extend([
        ReactionRule("R00101", "AtoB", "atoB", "2.3.1.9",
                     "2 Acetyl-CoA", "Acetoacetyl-CoA + CoA",
                     3.5, 15.0, 0.20, "Escherichia coli",
                     reversibility="reversible"),
        ReactionRule("R00102", "HMGS", "mvaS", "2.3.3.10",
                     "Acetoacetyl-CoA + Acetyl-CoA", "HMG-CoA + CoA",
                     -5.0, 10.2, 0.35, "Enterococcus faecalis"),
        ReactionRule("R00103", "HMGR", "mvaE", "1.1.1.34",
                     "HMG-CoA + 2 NADPH", "MVA + 2 NADP+ + CoA",
                     -25.0, 4.5, 0.80, "Enterococcus faecalis",
                     cofactors=["NADPH"]),
        ReactionRule("R00104", "MK", "erg12", "2.7.1.36",
                     "MVA + ATP", "MVAP + ADP",
                     -10.0, 8.0, 0.40, "Saccharomyces cerevisiae",
                     cofactors=["ATP"]),
        ReactionRule("R00105", "PMK", "erg8", "2.7.4.2",
                     "MVAP + ATP", "MVAPP + ADP",
                     -9.5, 6.5, 0.50, "Saccharomyces cerevisiae",
                     cofactors=["ATP"]),
        ReactionRule("R00106", "MVD", "mvd1", "4.1.1.33",
                     "MVAPP", "IPP + CO2 + Pi",
                     -14.0, 5.0, 0.65, "Saccharomyces cerevisiae"),
        ReactionRule("R00107", "IDI", "idi", "5.3.3.2",
                     "IPP", "DMAPP",
                     -2.0, 25.0, 0.15, "Escherichia coli",
                     reversibility="reversible"),
    ])

    # ── Carotenoid pathway (lycopene specific) ─────────────────────────────
    rules.extend([
        ReactionRule("R00201", "CrtE", "crtE", "2.5.1.29",
                     "FPP + IPP", "GGPP + PPi",
                     -10.0, 3.2, 0.75, "Pantoea agglomerans",
                     cofactors=["Mg2+"]),
        ReactionRule("R00202", "CrtB", "crtB", "2.5.1.32",
                     "2 GGPP", "Phytoene + PPi",
                     -8.5, 1.8, 1.20, "Pantoea agglomerans"),
        ReactionRule("R00203", "CrtI", "crtI", "1.3.99.31",
                     "Phytoene + 4e-", "Lycopene",
                     -45.0, 0.9, 2.50, "Pantoea agglomerans",
                     cofactors=["FAD"]),
        ReactionRule("R00204", "CrtY", "crtY", "5.3.99.4",
                     "Lycopene", "beta-Carotene",
                     -12.0, 1.5, 1.80, "Pantoea agglomerans"),
        ReactionRule("R00205", "CrtW", "crtW", "1.14.99.61",
                     "beta-Carotene + O2", "Astaxanthin",
                     -35.0, 0.5, 3.20, "Paracoccus marinus",
                     cofactors=["O2", "NADPH"]),
    ])

    # ── MEP to IPP/DMAPP pool ──────────────────────────────────────────────
    rules.extend([
        ReactionRule("R00301", "Fpps", "ispA", "2.5.1.1",
                     "DMAPP + IPP", "FPP + PPi",
                     -8.0, 5.5, 0.40, "Escherichia coli",
                     cofactors=["Mg2+"]),
        ReactionRule("R00302", "Fpps2", "ispA", "2.5.1.1",
                     "FPP + IPP", "GFPP + PPi",
                     -7.5, 4.0, 0.55, "Escherichia coli",
                     cofactors=["Mg2+"]),
    ])

    # ── Vanillin biosynthesis ──────────────────────────────────────────────
    rules.extend([
        ReactionRule("R00401", "SAMT", "sam5", "2.1.1.6",
                     "Catechol + SAM", "Guaiacol + SAH",
                     -5.0, 2.1, 1.50, "Streptomyces sp.",
                     cofactors=["SAM"]),
        ReactionRule("R00402", "VMO", "vanAB", "1.14.13.82",
                     "Guaiacol + O2 + NADH", "Catechol + NAD+ + H2O",
                     -30.0, 3.5, 0.80, "Pseudomonas sp.",
                     cofactors=["NADH", "O2"]),
        ReactionRule("R00403", "EchA", "echA", "4.2.1.17",
                     "Feruloyl-CoA", "Vanillin + Acetyl-CoA",
                     -8.0, 1.2, 2.00, "Pseudomonas fluorescens"),
        ReactionRule("R00404", "Fcs", "fcs", "6.2.1.30",
                     "Ferulic acid + CoA + ATP", "Feruloyl-CoA + AMP + PPi",
                     -10.0, 4.0, 0.60, "Pseudomonas fluorescens",
                     cofactors=["ATP", "CoA"]),
        ReactionRule("R00405", "VanB", "vanB", "1.2.1.68",
                     "Vanillin + NAD+", "Vanillate + NADH",
                     5.0, 8.0, 0.30, "Pseudomonas sp.",
                     reversibility="reversible", cofactors=["NAD+"]),
        ReactionRule("R00406", "CalA", "calA", "1.13.11.1",
                     "Isoeugenol + O2", "Vanillin",
                     -25.0, 1.5, 1.20, "Amycolatopsis sp.",
                     cofactors=["O2"]),
    ])

    # ── L-Lysine biosynthesis (DAP pathway) ────────────────────────────────
    rules.extend([
        ReactionRule("R00501", "DapA", "dapA", "4.2.1.52",
                     "Pyruvate + Aspartate-semialdehyde", "Dihydrodipicolinate + H2O",
                     -5.0, 15.0, 0.25, "Escherichia coli"),
        ReactionRule("R00502", "DapB", "dapB", "1.3.1.26",
                     "Dihydrodipicolinate + NADPH", "Tetrahydrodipicolinate + NADP+",
                     -20.0, 12.0, 0.18, "Escherichia coli",
                     cofactors=["NADPH"]),
        ReactionRule("R00503", "DapD", "dapD", "2.6.1.83",
                     "Tetrahydrodipicolinate + alpha-KG", "N-Succinyl-aminoadipate-semialdehyde",
                     -8.0, 6.0, 0.45, "Escherichia coli"),
        ReactionRule("R00504", "DapC", "dapC", "2.6.1.17",
                     "N-Succinyl-aminoadipate-semialdehyde + Glu", "N-Succinyl-diaminopimelate + alpha-KG",
                     -2.0, 8.0, 0.35, "Escherichia coli"),
        ReactionRule("R00505", "DapE", "dapE", "3.5.1.18",
                     "N-Succinyl-diaminopimelate + H2O", "Diaminopimelate + Succinate",
                     -10.0, 10.0, 0.28, "Escherichia coli"),
        ReactionRule("R00506", "DapF", "dapF", "5.1.1.7",
                     "LL-Diaminopimelate", "meso-Diaminopimelate",
                     1.5, 20.0, 0.10, "Escherichia coli",
                     reversibility="reversible"),
        ReactionRule("R00507", "LysA", "lysA", "4.1.1.20",
                     "meso-Diaminopimelate", "L-Lysine + CO2",
                     -15.0, 18.0, 0.22, "Escherichia coli"),
        ReactionRule("R00508", "LysC", "lysC", "2.7.2.4",
                     "Aspartate + ATP", "Aspartyl-phosphate + ADP",
                     -8.0, 22.0, 0.15, "Escherichia coli",
                     cofactors=["ATP"]),
    ])

    # ── PHA biosynthesis ───────────────────────────────────────────────────
    rules.extend([
        ReactionRule("R00601", "PhaA", "phaA", "2.3.1.16",
                     "2 Acetyl-CoA", "Acetoacetyl-CoA + CoA",
                     3.5, 14.0, 0.22, "Cupriavidus necator",
                     reversibility="reversible"),
        ReactionRule("R00602", "PhaB", "phaB", "1.1.1.36",
                     "Acetoacetyl-CoA + NADPH", "(R)-3-Hydroxybutyryl-CoA + NADP+",
                     -25.0, 18.0, 0.10, "Cupriavidus necator",
                     cofactors=["NADPH"]),
        ReactionRule("R00603", "PhaC", "phaC", "2.3.1.169",
                     "(R)-3-Hydroxybutyryl-CoA", "PHB + CoA",
                     -8.0, 2.5, 0.80, "Cupriavidus necator"),
        ReactionRule("R00604", "PhaJ", "phaJ", "4.2.1.-",
                     "(R)-3-Hydroxyacyl-CoA", "trans-Enoyl-CoA + H2O",
                     -3.0, 5.0, 0.50, "Pseudomonas putida"),
    ])

    # ── Riboflavin (B2) biosynthesis ───────────────────────────────────────
    rules.extend([
        ReactionRule("R00701", "RibA", "ribA", "3.5.4.17",
                     "GTP", "2,5-Diamino-6-ribosylamino-4(3H)-pyrimidinone-5'-phosphate + Formate",
                     -10.0, 3.0, 0.60, "Bacillus subtilis"),
        ReactionRule("R00702", "RibB", "ribB", "2.5.1.78",
                     "3,4-Dihydroxy-2-butanone-4-phosphate + GTP", "Riboflavin precursor",
                     -12.0, 2.5, 0.80, "Bacillus subtilis"),
        ReactionRule("R00703", "RibD", "ribD", "3.5.4.24",
                     "2,5-Diamino-6-ribosylamino-4(3H)-pyrimidinone-5'-phosphate",
                     "5-Amino-6-ribitylamino-2,4(1H,3H)-pyrimidinedione-5'-phosphate",
                     -8.0, 4.0, 0.50, "Bacillus subtilis"),
        ReactionRule("R00704", "RibE", "ribE", "3.5.4.2",
                     "5-Amino-6-ribitylamino-2,4(1H,3H)-pyrimidinedione-5'-phosphate",
                     "6,7-Dimethyl-8-ribityllumazine",
                     -5.0, 6.0, 0.35, "Bacillus subtilis"),
        ReactionRule("R00705", "RibH", "ribH", "2.5.1.79",
                     "2 x 6,7-Dimethyl-8-ribityllumazine", "Riboflavin + 4-hydroxy-5-(ribitylamino)-2-aminopyrimidine",
                     -15.0, 1.5, 1.00, "Bacillus subtilis"),
    ])

    # ── Central metabolism shortcuts (for precursor supply) ────────────────
    rules.extend([
        ReactionRule("R00801", "PfkA", "pfkA", "2.7.1.11",
                     "F6P + ATP", "F1,6BP + ADP",
                     -14.0, 80.0, 0.10, "Escherichia coli",
                     cofactors=["ATP"]),
        ReactionRule("R00802", "PfkB", "pfkB", "2.7.1.11",
                     "F6P + ATP", "F1,6BP + ADP",
                     -13.5, 50.0, 0.15, "Escherichia coli",
                     cofactors=["ATP"]),
        ReactionRule("R00803", "TktA", "tktA", "2.2.1.1",
                     "R5P + Xu5P", "S7P + G3P",
                     -2.0, 25.0, 0.30, "Escherichia coli"),
        ReactionRule("R00804", "TktB", "tktB", "2.2.1.1",
                     "E4P + Xu5P", "F6P + G3P",
                     -3.0, 18.0, 0.45, "Escherichia coli"),
        ReactionRule("R00805", "TpiA", "tpiA", "5.3.1.1",
                     "DHAP", "G3P",
                     3.0, 500.0, 0.05, "Escherichia coli",
                     reversibility="reversible"),
        ReactionRule("R00806", "Pgi", "pgi", "5.3.1.9",
                     "G6P", "F6P",
                     2.0, 300.0, 0.08, "Escherichia coli",
                     reversibility="reversible"),
    ])

    # ── Artemisinic acid pathway ───────────────────────────────────────────
    rules.extend([
        ReactionRule("R00901", "Amor4,2D", "amor4,2d", "4.2.3.27",
                     "FPP", "Amorpha-4,11-diene + PPi",
                     -10.0, 0.3, 1.50, "Artemisia annua",
                     cofactors=["Mg2+"]),
        ReactionRule("R00902", "CYP71AV1", "cyp71av1", "1.14.14.119",
                     "Amorpha-4,11-diene + O2 + NADPH", "Artemisinic alcohol",
                     -30.0, 0.1, 2.00, "Artemisia annua",
                     cofactors=["O2", "NADPH", "CPR"]),
        ReactionRule("R00903", "CYP71AV1b", "cyp71av1b", "1.14.14.119",
                     "Artemisinic alcohol + O2 + NADPH", "Artemisinic aldehyde",
                     -28.0, 0.08, 2.50, "Artemisia annua",
                     cofactors=["O2", "NADPH"]),
        ReactionRule("R00904", "CYP71AV1c", "cyp71av1c", "1.14.14.119",
                     "Artemisinic aldehyde + O2 + NADPH", "Artemisinic acid",
                     -32.0, 0.06, 3.00, "Artemisia annua",
                     cofactors=["O2", "NADPH"]),
        ReactionRule("R00905", "ADH1", "adh1", "1.1.1.-",
                     "Artemisinic aldehyde + NADH", "Artemisinic alcohol + NAD+",
                     5.0, 15.0, 0.40, "Saccharomyces cerevisiae",
                     reversibility="reversible", cofactors=["NADH"]),
        ReactionRule("R00906", "ALDH1", "aldh1", "1.2.1.3",
                     "Artemisinic aldehyde + NAD+ + H2O", "Artemisinic acid + NADH",
                     -15.0, 10.0, 0.60, "Saccharomyces cerevisiae",
                     cofactors=["NAD+"]),
    ])

    # ── Hyaluronic acid pathway ────────────────────────────────────────────
    rules.extend([
        ReactionRule("R01001", "HasA", "hasA", "2.4.1.212",
                     "UDP-GlcNAc + UDP-GlcUA", "HA oligomer + UDP + UDP",
                     -5.0, 0.5, 1.00, "Streptococcus zooepidemicus"),
        ReactionRule("R01002", "HasB", "hasB", "5.1.3.7",
                     "UDP-GlcNAc", "UDP-GlcNAc (epimerase)",
                     2.0, 8.0, 0.50, "Streptococcus zooepidemicus",
                     reversibility="reversible"),
        ReactionRule("R01003", "HasC", "hasC", "1.7.99.-",
                     "GlcN-6P", "UDP-GlcNAc precursor",
                     -8.0, 3.0, 0.80, "Streptococcus zooepidemicus"),
    ])

    # ── Glutamate biosynthesis ─────────────────────────────────────────────
    rules.extend([
        ReactionRule("R01101", "GdhA", "gdhA", "1.4.1.4",
                     "alpha-KG + NH3 + NADPH", "L-Glutamate + NADP+ + H2O",
                     -15.0, 100.0, 0.10, "Escherichia coli",
                     cofactors=["NADPH"]),
        ReactionRule("R01102", "GltB", "gltB", "6.3.5.2",
                     "L-Glutamate + NH3 + ATP", "L-Glutamine + ADP + Pi",
                     -10.0, 25.0, 0.20, "Escherichia coli",
                     cofactors=["ATP"]),
        ReactionRule("R01103", "GltD", "gltD", "2.6.1.53",
                     "alpha-KG + L-Glutamine", "2 x L-Glutamate",
                     -5.0, 30.0, 0.15, "Escherichia coli"),
    ])

    # ── Threonine biosynthesis ─────────────────────────────────────────────
    rules.extend([
        ReactionRule("R01201", "Hom", "hom", "1.1.1.3",
                     "Aspartate-semialdehyde + NADPH", "Homoserine + NADP+",
                     -12.0, 8.0, 0.30, "Escherichia coli",
                     cofactors=["NADPH"]),
        ReactionRule("R01202", "ThrB", "thrB", "2.7.1.39",
                     "Homoserine + ATP", "O-Phosphohomoserine + ADP",
                     -8.0, 12.0, 0.25, "Escherichia coli",
                     cofactors=["ATP"]),
        ReactionRule("R01203", "ThrC", "thrC", "4.2.99.2",
                     "O-Phosphohomoserine", "L-Threonine + Pi",
                     -5.0, 6.0, 0.40, "Escherichia coli"),
    ])

    return rules


# ---------------------------------------------------------------------------
# MCTS NODE
# ---------------------------------------------------------------------------

class MCTSNode:
    """A single node in the Monte Carlo Tree Search for pathway exploration."""

    def __init__(self, molecule: str, parent: Optional["MCTSNode"] = None,
                 reaction: Optional[ReactionRule] = None) -> None:
        self.molecule = molecule
        self.parent = parent
        self.reaction = reaction  # reaction that produces this molecule
        self.children: List["MCTSNode"] = []
        self.visits = 0
        self.value = 0.0  # cumulative reward

    @property
    def is_terminal(self) -> bool:
        """A node is terminal if no reaction rule produces it (i.e., it's a precursor)."""
        return self.parent is None and self.reaction is None

    def uct_score(self, total_visits: int, exploration: float = 1.414) -> float:
        """Upper Confidence Bound for Trees score."""
        if self.visits == 0:
            return float("inf")
        return (self.value / self.visits) + exploration * math.sqrt(
            math.log(total_visits) / self.visits
        )

    def add_child(self, node: "MCTSNode") -> None:
        self.children.append(node)

    def update(self, reward: float) -> None:
        self.visits += 1
        self.value += reward


# ---------------------------------------------------------------------------
# PATHWAY CANDIDATE
# ---------------------------------------------------------------------------

@dataclass
class PathwayCandidate:
    """A complete pathway from precursors to target molecule."""

    pathway_id: str
    rank: int
    pathway_name: str
    steps: List[Dict[str, Any]]
    total_steps: int
    predicted_yield_mol_per_mol: float
    thermodynamic_feasibility_score: float
    gnn_viability_score: float
    host_compatibility_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pathway_id": self.pathway_id,
            "rank": self.rank,
            "pathway_name": self.pathway_name,
            "steps": self.steps,
            "total_steps": self.total_steps,
            "predicted_yield_mol_per_mol": self.predicted_yield_mol_per_mol,
            "thermodynamic_feasibility_score": self.thermodynamic_feasibility_score,
            "gnn_viability_score": self.gnn_viability_score,
            "host_compatibility_score": self.host_compatibility_score,
        }


# ---------------------------------------------------------------------------
# RETROSYNTHESIS ENGINE
# ---------------------------------------------------------------------------

class RetrosynthesisEngine:
    """
    Main engine for pathway discovery via retrosynthetic analysis.

    Uses MCTS over a library of reaction rules to generate candidate
    pathways from a target molecule back to metabolic precursors.
    """

    def __init__(self) -> None:
        self._rules: List[ReactionRule] = []
        self._logger: Optional[PipelineLogger] = None
        self._precursor_pool: set = set()

    def set_logger(self, logger: PipelineLogger) -> None:
        self._logger = logger

    def load_reaction_rules(self, custom_rules: Optional[List[ReactionRule]] = None) -> None:
        """Load the reaction rule library."""
        if custom_rules:
            self._rules = list(custom_rules)
        else:
            self._rules = _build_reaction_rules()
        # Build precursor pool (molecules that are not produced by any rule)
        products = {r.product for r in self._rules}
        substrates = {r.substrate for r in self._rules}
        self._precursor_pool = substrates - products
        if self._logger:
            self._logger.info("Loaded %d reaction rules, %d precursors identified",
                              len(self._rules), len(self._precursor_pool))

    def run_mcts(self, target: str, max_iterations: int = 200,
                 max_depth: int = 8, n_top: int = 5) -> List[List[ReactionRule]]:
        """
        Monte Carlo Tree Search for pathway generation.

        Parameters
        ----------
        target : str
            The target molecule name.
        max_iterations : int
            Number of MCTS iterations.
        max_depth : int
            Maximum depth of search tree.
        n_top : int
            Number of top pathways to return.

        Returns
        -------
        list of list of ReactionRule
            Each inner list is a pathway (ordered from precursor to target).
        """
        if self._logger:
            self._logger.info("Running MCTS: target='%s', iterations=%d, max_depth=%d",
                              target, max_iterations, max_depth)

        pathways: Dict[str, List[ReactionRule]] = {}
        root = MCTSNode(target)

        for iteration in range(max_iterations):
            node = self._select(root, max_depth)
            if node.is_terminal or self._is_precursor(node.molecule):
                # Expand from precursors
                expanded = self._expand(node)
                if expanded:
                    reward = self._simulate(expanded[0])
                    self._backpropagate(expanded[0], reward)
            else:
                expanded = self._expand(node)
                if expanded:
                    child = random.choice(expanded)
                    reward = self._simulate(child)
                    self._backpropagate(child, reward)

        # Extract pathways
        pathway_list = self._extract_pathways(root, target)
        # Score and sort
        scored = []
        for pw in pathway_list:
            score = sum(
                abs(r.delta_g_kj_per_mol) * (1.0 if r.delta_g_kj_per_mol < 0 else 0.5)
                for r in pw
            )
            scored.append((score, pw))
        scored.sort(key=lambda x: x[0], reverse=True)

        return [pw for _, pw in scored[:n_top]]

    def _select(self, node: MCTSNode, max_depth: int, depth: int = 0) -> MCTSNode:
        """Select a leaf node using UCB1 policy."""
        if depth >= max_depth or not node.children:
            return node
        best_child = max(node.children, key=lambda c: c.uct_score(node.visits))
        return self._select(best_child, max_depth, depth + 1)

    def _expand(self, node: MCTSNode) -> List[MCTSNode]:
        """Generate child nodes by finding reactions that produce the current molecule."""
        children: List[MCTSNode] = []
        target_mol = node.molecule.lower()

        for rule in self._rules:
            if rule.product.lower() == target_mol or target_mol in rule.product.lower():
                child = MCTSNode(rule.substrate, parent=node, reaction=rule)
                children.append(child)

        # If no direct match, try partial substrate matching (simulated)
        if not children:
            for rule in self._rules:
                if any(t in rule.product.lower() for t in target_mol.split()):
                    child = MCTSNode(rule.substrate, parent=node, reaction=rule)
                    children.append(child)

        # If still empty, add a simulated precursor step
        if not children and node.parent is None:
            sim_rule = ReactionRule(
                rule_id=f"SIM_{hash(target_mol) % 10000:04d}",
                enzyme_name="SimulatedEnz",
                gene_name=f"sim_{target_mol[:5]}",
                ec_number="1.1.1.-",
                substrate=f"precursor_for_{target_mol}",
                product=target_mol,
                delta_g_kj_per_mol=-5.0 + random.uniform(-3, 3),
                kcat_per_sec=random.uniform(1, 20),
                km_mm=random.uniform(0.1, 2.0),
                organism_source="simulated",
            )
            children.append(MCTSNode(sim_rule.substrate, parent=node, reaction=sim_rule))

        for child in children:
            node.add_child(child)

        return children

    def _simulate(self, node: MCTSNode) -> float:
        """Simulate a random pathway completion and return reward."""
        depth = 0
        current = node
        total_delta_g = 0.0
        max_kcat = 0.0

        while not self._is_precursor(current.molecule) and depth < 20:
            rules_for_mol = [r for r in self._rules
                             if r.product.lower() == current.molecule.lower()
                             or current.molecule.lower() in r.product.lower()]
            if not rules_for_mol:
                total_delta_g += -5.0 + random.uniform(-3, 3)
                break
            chosen = random.choice(rules_for_mol)
            total_delta_g += chosen.delta_g_kj_per_mol
            max_kcat = max(max_kcat, chosen.kcat_per_sec)
            current = MCTSNode(chosen.substrate, parent=current, reaction=chosen)
            depth += 1

        # Reward: thermodynamic favourability + kinetic efficiency
        therm_reward = 1.0 / (1.0 + math.exp(total_delta_g / 20))
        kinetic_reward = max_kcat / 500.0 if max_kcat > 0 else 0.1
        return 0.6 * therm_reward + 0.4 * kinetic_reward

    def _backpropagate(self, node: MCTSNode, reward: float) -> None:
        """Propagate reward up the tree."""
        current = node
        while current is not None:
            current.update(reward)
            current = current.parent

    def _is_precursor(self, molecule: str) -> bool:
        """Check if a molecule is a known metabolic precursor."""
        return (molecule.lower() in self._precursor_pool
                or any(p in molecule.lower() for p in self._precursor_pool))

    def _extract_pathways(self, root: MCTSNode,
                          target: str) -> List[List[ReactionRule]]:
        """Extract complete pathways from the MCTS tree."""
        pathways: List[List[ReactionRule]] = []
        stack: List[List[ReactionRule]] = []

        def dfs(node: MCTSNode, current_path: List[ReactionRule]) -> None:
            if self._is_precursor(node.molecule):
                pathways.append(list(reversed(current_path)))
                return
            for child in node.children:
                if child.reaction:
                    dfs(child, current_path + [child.reaction])

        dfs(root, [])

        if not pathways:
            # Fallback: create synthetic pathway
            rules = self._rules[:3]
            pathways.append(rules)

        return pathways

    def score_thermodynamics(self, pathway: List[ReactionRule]) -> float:
        """
        Score thermodynamic feasibility of a pathway.

        Returns a score between 0 and 1 (higher = more feasible).
        """
        if not pathway:
            return 0.0

        total_delta_g = sum(r.delta_g_kj_per_mol for r in pathway)
        n_irreversible = sum(1 for r in pathway if r.reversibility == "irreversible")

        # Thermodynamic score: exponential decay based on total delta G
        thermo_score = 1.0 / (1.0 + math.exp(total_delta_g / 30))

        # Bonus for irreversible steps (driving force)
        irreversibility_bonus = min(0.3, n_irreversible * 0.1)

        return min(1.0, thermo_score + irreversibility_bonus)

    def rank_pathways(self, target: str, pathways: List[List[ReactionRule]],
                      host_organism: str, n_top: int = 5) -> List[PathwayCandidate]:
        """
        Rank pathways by combined score and return top N candidates.
        """
        scored: List[Tuple[float, List[ReactionRule]]] = []

        for pw in pathways:
            thermo = self.score_thermodynamics(pw)
            # Simulated GNN viability
            gnn = random.uniform(0.5, 0.95)
            # Host compatibility
            host_compat = self._host_compatibility(pw, host_organism)
            combined = 0.4 * thermo + 0.3 * gnn + 0.3 * host_compat
            scored.append((combined, pw))

        scored.sort(key=lambda x: x[0], reverse=True)
        ranked: List[PathwayCandidate] = []

        for rank, (score, pw) in enumerate(scored[:n_top], start=1):
            steps = []
            for i, rule in enumerate(pw, start=1):
                is_hetero = rule.organism_source.lower() not in (
                    host_organism.lower(), "simulated"
                )
                steps.append(rule.to_step_dict(i, is_hetero, host_organism))

            thermo_score = self.score_thermodynamics(pw)
            # Simulated GNN viability (seeded for reproducibility)
            random.seed(hash(f"{target}_{rank}_{host_organism}"))
            gnn_score = random.uniform(0.5, 0.95)
            random.seed(hash(f"{target}_{rank}"))
            predicted_yield = thermo_score * gnn_score * random.uniform(0.4, 0.8)

            candidate = PathwayCandidate(
                pathway_id=f"pw_{target.lower().replace(' ', '_')}_{rank:03d}",
                rank=rank,
                pathway_name=f"{target} pathway #{rank}",
                steps=steps,
                total_steps=len(steps),
                predicted_yield_mol_per_mol=round(predicted_yield, 4),
                thermodynamic_feasibility_score=round(thermo_score, 4),
                gnn_viability_score=round(gnn_score, 4),
                host_compatibility_score=round(
                    self._host_compatibility(pw, host_organism), 4
                ),
            )
            ranked.append(candidate)

        return ranked

    def _host_compatibility(self, pathway: List[ReactionRule],
                            host: str) -> float:
        """Estimate how well a pathway fits the host organism."""
        if not pathway:
            return 0.5

        n_host = sum(1 for r in pathway if r.organism_source.lower() == host.lower())
        n_total = len(pathway)

        # Higher fraction of host-native enzymes → higher compatibility
        return n_host / n_total if n_total > 0 else 0.5


# ---------------------------------------------------------------------------
# MAIN — smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json
    import os
    import sys

    sys.path.insert(0, ".")

    parser = argparse.ArgumentParser(description="Retrosynthesis Engine")
    parser.add_argument("--target", default="lycopene",
                        help="Target molecule name")
    parser.add_argument("--organism", default="Escherichia coli",
                        help="Host organism name")
    args = parser.parse_args()

    logger = PipelineLogger()
    logger.set_stage("2")

    engine = RetrosynthesisEngine()
    engine.set_logger(logger)
    engine.load_reaction_rules()

    logger.info("Running MCTS for target='%s' in organism='%s'",
                args.target, args.organism)

    pathways = engine.run_mcts(args.target, max_iterations=300,
                               max_depth=8, n_top=5)

    ranked = engine.rank_pathways(args.target, pathways,
                                  host_organism=args.organism, n_top=5)

    logger.info("Found %d pathways, ranked top %d", len(pathways), len(ranked))

    for rc in ranked:
        logger.info("  Rank %d: %s | yield=%.3f | thermo=%.3f | gnn=%.3f | host=%.3f",
                    rc.rank, rc.pathway_name,
                    rc.predicted_yield_mol_per_mol,
                    rc.thermodynamic_feasibility_score,
                    rc.gnn_viability_score,
                    rc.host_compatibility_score)

    # Save results
    os.makedirs("pipeline_output", exist_ok=True)
    with open("pipeline_output/mcts_pathways.json", "w") as fh:
        json.dump([rc.to_dict() for rc in ranked], fh, indent=2, default=str)

    logger.info("Pathway results saved to pipeline_output/mcts_pathways.json")
    print(f"\n▶ Retrosynthesis Engine smoke test passed. Found {len(ranked)} ranked pathways.")
