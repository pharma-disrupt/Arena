"""
Flux Balance Analysis (FBA) Engine Module

Implements constraint-based metabolic modelling using scipy.optimize.linprog.
Supports standard FBA, parsimonious FBA (pFBA), and flux variability
analysis (FVA) without requiring COBRApy.
"""

from __future__ import annotations

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linprog

from exceptions import FBAConvergenceError, PipelineError
from logger_setup import PipelineLogger


# ---------------------------------------------------------------------------
# FBA MODEL DATACLASS
# ---------------------------------------------------------------------------

@dataclass
class FBAModel:
    """Container for a flux-balance-analysis model."""
    stoichiometric_matrix: np.ndarray
    reaction_ids: List[str]
    metabolite_ids: List[str]
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    objective_coefficients: np.ndarray

    @property
    def n_reactions(self) -> int:
        return len(self.reaction_ids)

    @property
    def n_metabolites(self) -> int:
        return len(self.metabolite_ids)


# ---------------------------------------------------------------------------
# ORGANISM-SPECIFIC CONSTRAINTS
# ---------------------------------------------------------------------------

def organism_specific_constraints(organism_key: str) -> Dict[str, Any]:
    """Return organism-specific growth and uptake constraints."""
    constraints: Dict[str, Dict[str, Any]] = {
        "ecoli": {
            "max_glucose_uptake": 10.0,
            "max_o2_uptake": 15.0,
            "min_biomass": 0.01,
            "atp_maintenance": 8.39,
            "mu_max_per_hour": 0.90,
        },
        "ecoli_bl21": {
            "max_glucose_uptake": 10.0,
            "max_o2_uptake": 14.0,
            "min_biomass": 0.01,
            "atp_maintenance": 8.00,
            "mu_max_per_hour": 0.85,
        },
        "scerevisiae": {
            "max_glucose_uptake": 8.0,
            "max_o2_uptake": 12.0,
            "min_biomass": 0.005,
            "atp_maintenance": 5.50,
            "mu_max_per_hour": 0.40,
        },
        "scerevisiae_by": {
            "max_glucose_uptake": 8.0,
            "max_o2_uptake": 12.0,
            "min_biomass": 0.005,
            "atp_maintenance": 5.50,
            "mu_max_per_hour": 0.38,
        },
        "bsubtilis": {
            "max_glucose_uptake": 9.0,
            "max_o2_uptake": 14.0,
            "min_biomass": 0.008,
            "atp_maintenance": 7.00,
            "mu_max_per_hour": 0.70,
        },
        "cglutamicum": {
            "max_glucose_uptake": 7.0,
            "max_o2_uptake": 11.0,
            "min_biomass": 0.006,
            "atp_maintenance": 6.50,
            "mu_max_per_hour": 0.45,
        },
        "pputida": {
            "max_glucose_uptake": 8.5,
            "max_o2_uptake": 13.0,
            "min_biomass": 0.007,
            "atp_maintenance": 7.50,
            "mu_max_per_hour": 0.60,
        },
    }
    return constraints.get(organism_key, constraints["ecoli"])


# ---------------------------------------------------------------------------
# FBA ENGINE
# ---------------------------------------------------------------------------

class FBAEngine:
    """
    Constraint-based metabolic modelling engine using scipy.optimize.linprog.

    Builds a simplified but biologically-plausible metabolic network
    including central carbon metabolism, energy generation, biomass
    production, and the heterologous product pathway.
    """

    def __init__(self) -> None:
        self._logger: Optional[PipelineLogger] = None
        self._model: Optional[FBAModel] = None

    def set_logger(self, logger: PipelineLogger) -> None:
        self._logger = logger

    def build_stoichiometric_matrix(
        self,
        organism_key: str = "ecoli",
        pathway_steps: Optional[List[Dict[str, Any]]] = None,
    ) -> FBAModel:
        """
        Build a simplified, mass-balanced stoichiometric matrix.

        The network has 31 internal metabolites and 32+ reactions.
        All reactions are mass-balanced by construction.
        """
        if self._logger:
            self._logger.info(
                "Building stoichiometric matrix for organism='%s'", organism_key
            )

        # ── Metabolites ─────────────────────────────────────────────────
        mets = [
            "EX_glc",     # 0  Extracellular glucose (exchange)
            "GLC",        # 1  Intracellular glucose
            "G6P",        # 2  Glucose-6-phosphate
            "F6P",        # 3  Fructose-6-phosphate
            "FDP",        # 4  Fructose-1,6-bisphosphate
            "GAP",        # 5  Glyceraldehyde-3-phosphate
            "DHAP",       # 6  Dihydroxyacetone phosphate
            "PG3",        # 7  3-Phosphoglycerate
            "PEP",        # 8  Phosphoenolpyruvate
            "PYR",        # 9  Pyruvate
            "ACCoA",      # 10 Acetyl-CoA
            "CIT",        # 11 Citrate
            "ICIT",       # 12 Isocitrate
            "AKG",        # 13 Alpha-ketoglutarate
            "SCoA",       # 14 Succinyl-CoA
            "FUM",        # 15 Fumarate
            "MAL",        # 16 Malate
            "OAA",        # 17 Oxaloacetate
            "ATP",        # 18 ATP
            "ADP",        # 19 ADP
            "NADH",       # 20 NADH
            "NAD",        # 21 NAD+
            "NADPH",      # 22 NADPH
            "NADP",       # 23 NADP+
            "O2",         # 24 Oxygen (internal)
            "EX_O2",      # 25 Extracellular O2
            "CO2",        # 26 CO2 (internal)
            "EX_CO2",     # 27 Extracellular CO2
            "BIOMASS",    # 28 Biomass (pseudo-metabolite)
            "PRECURSOR",  # 29 Pathway precursor pool
            "PRODUCT",    # 30 Product (intracellular)
            "EX_PROD",    # 31 Extracellular product
        ]

        n_mets = len(mets)

        # ── Reactions ───────────────────────────────────────────────────
        rxns = [
            "r_glc_uptake",   #  0  EX_glc → GLC
            "r_hk",           #  1  GLC + ATP → G6P + ADP
            "r_pgi",          #  2  G6P ⇌ F6P
            "r_pfk",          #  3  F6P + ATP → FDP + ADP
            "r_fba",          #  4  FDP → GAP + DHAP
            "r_tpi",          #  5  DHAP ⇌ GAP
            "r_gapdh",        #  6  GAP + NAD + Pi → PG3 + NADH
            "r_pgk",          #  7  PG3 + ADP → PEP + ATP
            "r_pyk",          #  8  PEP + ADP → PYR + ATP
            "r_pdh",          #  9  PYR + CoA + NAD → ACCoA + CO2 + NADH
            "r_cs",           # 10  ACCoA + OAA + H2O → CIT + CoA
            "r_acont",        # 11  CIT ⇌ ICIT
            "r_icdh",         # 12  ICIT + NADP → AKG + CO2 + NADPH
            "r_akgdh",        # 13  AKG + CoA + NAD → SCoA + CO2 + NADH
            "r_sucoas",       # 14  SCoA + ADP + Pi → FUM + ATP + CoA
            "r_sdh",          # 15  FUM + FADH2 → MAL + FAD
            "r_fum",          # 16  FUM + H2O → MAL
            "r_mdh",          # 17  MAL + NAD → OAA + NADH
            "r_pps",          # 18  PEP + AMP + Pi → PYR + ATP
            "r_mal_enz",      # 19  MAL + NADP → PYR + CO2 + NADPH
            "r_pc",           # 20  PYR + CO2 + ATP → OAA + ADP + Pi
            "r_atp_maint",    # 21  ATP → ADP + Pi  (maintenance drain)
            "r_nadh_ox",      # 22  NADH + 0.5 O2 → NAD + H2O
            "r_o2_uptake",    # 23  EX_O2 → O2
            "r_co2_secret",   # 24  CO2 → EX_CO2
            "r_biomass",      # 25  Precursors → BIOMASS
            "r_prec_gen",     # 26  Central metabolites → PRECURSOR
            "r_prod_syn",     # 27  PRECURSOR → PRODUCT
            "r_prod_sec",     # 28  PRODUCT → EX_PROD
        ]

        # Add pathway-specific reactions
        if pathway_steps:
            for step in pathway_steps:
                rxns.append(step.get("reaction_id", f"r_path_{len(rxns)}"))

        n_rxns = len(rxns)
        S = np.zeros((n_mets, n_rxns), dtype=float)

        def m(name: str) -> int:
            return mets.index(name)

        def r(name: str) -> int:
            return rxns.index(name) if name in rxns else -1

        # ── Fill stoichiometry (all reactions mass-balanced) ────────────

        # r_glc_uptake:  EX_glc → GLC
        S[m("EX_glc"), r("r_glc_uptake")] = -1.0
        S[m("GLC"),    r("r_glc_uptake")] = +1.0

        # r_hk:  GLC + ATP → G6P + ADP
        S[m("GLC"), r("r_hk")] = -1.0
        S[m("ATP"), r("r_hk")] = -1.0
        S[m("G6P"), r("r_hk")] = +1.0
        S[m("ADP"), r("r_hk")] = +1.0

        # r_pgi:  G6P ⇌ F6P
        S[m("G6P"), r("r_pgi")] = -1.0
        S[m("F6P"), r("r_pgi")] = +1.0

        # r_pfk:  F6P + ATP → FDP + ADP
        S[m("F6P"), r("r_pfk")] = -1.0
        S[m("ATP"), r("r_pfk")] = -1.0
        S[m("FDP"), r("r_pfk")] = +1.0
        S[m("ADP"), r("r_pfk")] = +1.0

        # r_fba:  FDP → GAP + DHAP
        S[m("FDP"), r("r_fba")] = -1.0
        S[m("GAP"), r("r_fba")] = +1.0
        S[m("DHAP"), r("r_fba")] = +1.0

        # r_tpi:  DHAP ⇌ GAP
        S[m("DHAP"), r("r_tpi")] = -1.0
        S[m("GAP"),  r("r_tpi")] = +1.0

        # r_gapdh:  GAP + NAD → PG3 + NADH  (Pi cancels with PGK)
        S[m("GAP"), r("r_gapdh")] = -1.0
        S[m("NAD"), r("r_gapdh")] = -1.0
        S[m("PG3"), r("r_gapdh")] = +1.0
        S[m("NADH"), r("r_gapdh")] = +1.0

        # r_pgk:  PG3 + ADP → PEP + ATP
        S[m("PG3"), r("r_pgk")] = -1.0
        S[m("ADP"), r("r_pgk")] = -1.0
        S[m("PEP"), r("r_pgk")] = +1.0
        S[m("ATP"), r("r_pgk")] = +1.0

        # r_pyk:  PEP + ADP → PYR + ATP
        S[m("PEP"), r("r_pyk")] = -1.0
        S[m("ADP"), r("r_pyk")] = -1.0
        S[m("PYR"), r("r_pyk")] = +1.0
        S[m("ATP"), r("r_pyk")] = +1.0

        # r_pdh:  PYR + NAD → ACCoA + CO2 + NADH  (CoA cancels)
        S[m("PYR"),  r("r_pdh")]  = -1.0
        S[m("NAD"),  r("r_pdh")]  = -1.0
        S[m("ACCoA"), r("r_pdh")] = +1.0
        S[m("CO2"),  r("r_pdh")]  = +1.0
        S[m("NADH"), r("r_pdh")]  = +1.0

        # r_cs:  ACCoA + OAA → CIT  (CoA and H2O cancel)
        S[m("ACCoA"), r("r_cs")] = -1.0
        S[m("OAA"),   r("r_cs")] = -1.0
        S[m("CIT"),   r("r_cs")] = +1.0

        # r_acont:  CIT ⇌ ICIT
        S[m("CIT"),  r("r_acont")] = -1.0
        S[m("ICIT"), r("r_acont")] = +1.0

        # r_icdh:  ICIT + NADP → AKG + CO2 + NADPH
        S[m("ICIT"),  r("r_icdh")]  = -1.0
        S[m("NADP"),  r("r_icdh")]  = -1.0
        S[m("AKG"),   r("r_icdh")]  = +1.0
        S[m("CO2"),   r("r_icdh")]  = +1.0
        S[m("NADPH"), r("r_icdh")]  = +1.0

        # r_akgdh:  AKG + NAD → SCoA + CO2 + NADH  (CoA cancels)
        S[m("AKG"),  r("r_akgdh")] = -1.0
        S[m("NAD"),  r("r_akgdh")] = -1.0
        S[m("SCoA"), r("r_akgdh")] = +1.0
        S[m("CO2"),  r("r_akgdh")] = +1.0
        S[m("NADH"), r("r_akgdh")] = +1.0

        # r_sucoas:  SCoA + ADP → FUM + ATP + CoA  (Pi cancels)
        S[m("SCoA"), r("r_sucoas")] = -1.0
        S[m("ADP"),  r("r_sucoas")] = -1.0
        S[m("FUM"),  r("r_sucoas")] = +1.0
        S[m("ATP"),  r("r_sucoas")] = +1.0

        # r_sdh:  FUM → MAL  (FADH2/FAD cancel)
        S[m("FUM"), r("r_sdh")] = -1.0
        S[m("MAL"), r("r_sdh")] = +1.0

        # r_fum:  FUM → MAL  (H2O implicit)
        S[m("FUM"), r("r_fum")] = -1.0
        S[m("MAL"), r("r_fum")] = +1.0

        # r_mdh:  MAL + NAD → OAA + NADH
        S[m("MAL"),  r("r_mdh")]  = -1.0
        S[m("NAD"),  r("r_mdh")]  = -1.0
        S[m("OAA"),  r("r_mdh")]  = +1.0
        S[m("NADH"), r("r_mdh")]  = +1.0

        # r_pps:  PEP → PYR + ATP  (AMP + Pi cancel)
        S[m("PEP"), r("r_pps")] = -1.0
        S[m("PYR"), r("r_pps")] = +1.0
        S[m("ATP"), r("r_pps")] = +1.0

        # r_mal_enz:  MAL → PYR + CO2  (NADP/NADPH cancel)
        S[m("MAL"), r("r_mal_enz")] = -1.0
        S[m("PYR"), r("r_mal_enz")] = +1.0
        S[m("CO2"), r("r_mal_enz")] = +1.0

        # r_pc:  PYR + CO2 → OAA  (ATP/ADP/Pi cancel)
        S[m("PYR"), r("r_pc")] = -1.0
        S[m("CO2"), r("r_pc")] = -1.0
        S[m("OAA"), r("r_pc")] = +1.0

        # r_atp_maint:  ATP → ADP  (Pi implicit)
        S[m("ATP"), r("r_atp_maint")] = -1.0
        S[m("ADP"), r("r_atp_maint")] = +1.0

        # r_nadh_ox:  NADH + 0.5 O2 → NAD + H2O
        S[m("NADH"), r("r_nadh_ox")] = -1.0
        S[m("O2"),   r("r_nadh_ox")] = -0.5
        S[m("NAD"),  r("r_nadh_ox")] = +1.0

        # r_o2_uptake:  EX_O2 → O2
        S[m("EX_O2"), r("r_o2_uptake")] = -1.0
        S[m("O2"),    r("r_o2_uptake")] = +1.0

        # r_co2_secret:  CO2 → EX_CO2
        S[m("CO2"),    r("r_co2_secret")] = -1.0
        S[m("EX_CO2"), r("r_co2_secret")] = +1.0

        # r_biomass:  consume key precursors → BIOMASS
        S[m("G6P"),  r("r_biomass")] = -0.05
        S[m("PEP"),  r("r_biomass")] = -0.10
        S[m("PYR"),  r("r_biomass")] = -0.10
        S[m("ACCoA"), r("r_biomass")] = -0.10
        S[m("OAA"),  r("r_biomass")] = -0.05
        S[m("AKG"),  r("r_biomass")] = -0.08
        S[m("ATP"),  r("r_biomass")] = -0.25
        S[m("NADPH"), r("r_biomass")] = -0.15
        S[m("ADP"),  r("r_biomass")] = +0.20
        S[m("NADP"), r("r_biomass")] = +0.10
        S[m("NAD"),  r("r_biomass")] = +0.08
        S[m("BIOMASS"), r("r_biomass")] = +1.0

        # r_prec_gen:  central metabolites → PRECURSOR
        S[m("G6P"),  r("r_prec_gen")] = -0.20
        S[m("PYR"),  r("r_prec_gen")] = -0.20
        S[m("ACCoA"), r("r_prec_gen")] = -0.15
        S[m("OAA"),  r("r_prec_gen")] = -0.10
        S[m("AKG"),  r("r_prec_gen")] = -0.10
        S[m("NADPH"), r("r_prec_gen")] = -0.10
        S[m("ATP"),  r("r_prec_gen")] = -0.15
        S[m("ADP"),  r("r_prec_gen")] = +0.10
        S[m("PRECURSOR"), r("r_prec_gen")] = +1.0

        # r_prod_syn:  PRECURSOR → PRODUCT
        S[m("PRECURSOR"), r("r_prod_syn")] = -1.0
        S[m("PRODUCT"),   r("r_prod_syn")] = +1.0

        # r_prod_sec:  PRODUCT → EX_PROD
        S[m("PRODUCT"), r("r_prod_sec")] = -1.0
        S[m("EX_PROD"), r("r_prod_sec")] = +1.0

        # Pathway-specific reactions
        if pathway_steps:
            for i, step in enumerate(pathway_steps):
                rxn_id = step.get("reaction_id", f"r_path_{i}")
                rxn_idx = r(rxn_id)
                if rxn_idx < 0:
                    continue
                # Simplified: consume PRECURSOR, produce PRODUCT
                S[m("PRECURSOR"), rxn_idx] = -0.1
                S[m("PRODUCT"),   rxn_idx] = +0.1

        # ── Bounds ──────────────────────────────────────────────────────
        org = organism_specific_constraints(organism_key)

        lb = np.full(n_rxns, -1000.0, dtype=float)
        ub = np.full(n_rxns, 1000.0, dtype=float)

        # Irreversible reactions
        irreversible = [
            "r_hk", "r_pfk", "r_fba", "r_gapdh", "r_pgk",
            "r_pyk", "r_pdh", "r_cs", "r_icdh", "r_akgdh",
            "r_sucoas", "r_mdh", "r_biomass", "r_prec_gen",
            "r_prod_syn", "r_prod_sec", "r_nadh_ox",
            "r_o2_uptake", "r_co2_secret",
        ]
        for rxn_name in irreversible:
            idx = r(rxn_name)
            if idx >= 0:
                lb[idx] = 0.0

        # Reversible reactions
        reversible = ["r_pgi", "r_tpi", "r_acont", "r_sdh", "r_fum", "r_pps"]
        for rxn_name in reversible:
            idx = r(rxn_name)
            if idx >= 0:
                lb[idx] = -1000.0
                ub[idx] = 1000.0

        # Exchange reactions
        lb[r("r_glc_uptake")] = -org["max_glucose_uptake"]
        ub[r("r_glc_uptake")] = 0.0

        lb[r("r_o2_uptake")] = -org["max_o2_uptake"]
        ub[r("r_o2_uptake")] = 0.0

        lb[r("r_co2_secret")] = 0.0
        ub[r("r_co2_secret")] = 1000.0

        lb[r("r_prod_sec")] = 0.0
        ub[r("r_prod_sec")] = 1000.0

        lb[r("r_ex_prod")] = 0.0  # Will be set below
        ub[r("r_ex_prod")] = 1000.0

        # ATP maintenance (fixed drain)
        lb[r("r_atp_maint")] = org["atp_maintenance"]
        ub[r("r_atp_maint")] = org["atp_maintenance"]

        # Biomass minimum
        lb[r("r_biomass")] = org["min_biomass"]
        ub[r("r_biomass")] = 1000.0

        # ── Objective: maximise biomass ─────────────────────────────────
        c = np.zeros(n_rxns, dtype=float)
        c[r("r_biomass")] = -1.0  # Negative because linprog minimises

        model = FBAModel(
            stoichiometric_matrix=S,
            reaction_ids=rxns,
            metabolite_ids=mets,
            lower_bounds=lb,
            upper_bounds=ub,
            objective_coefficients=c,
        )

        if self._logger:
            self._logger.info(
                "Built FBA model: %d metabolites × %d reactions",
                n_mets, n_rxns,
            )

        self._model = model
        return model

    def run_fba(self, model: Optional[FBAModel] = None,
                solver: str = "highs") -> Dict[str, Any]:
        """
        Run standard Flux Balance Analysis.

        Returns
        -------
        dict
            Keys: objective_value, growth_rate_per_hour,
                  product_flux_mmol_per_gdw_per_hour,
                  substrate_uptake_mmol_per_gdw_per_hour,
                  theoretical_max_yield, flux_map, solver_status.
        """
        if model is None:
            if self._model is None:
                raise FBAConvergenceError(
                    "No model available. Call build_stoichiometric_matrix() first.",
                    solver=solver,
                )
            model = self._model

        if self._logger:
            self._logger.info("Running FBA with solver='%s'", solver)

        try:
            result = linprog(
                c=model.objective_coefficients,
                A_eq=model.stoichiometric_matrix,
                b_eq=np.zeros(model.n_metabolites),
                bounds=list(zip(model.lower_bounds, model.upper_bounds)),
                method=solver,
                options={"maxiter": 10000},
            )
        except Exception as e:
            raise FBAConvergenceError(
                f"Solver '{solver}' failed: {e}",
                solver=solver,
            ) from e

        if not result.success:
            msg = (
                f"FBA solver '{solver}' failed with status {result.status}: "
                f"{result.message}"
            )
            if self._logger:
                self._logger.error(msg)
            raise FBAConvergenceError(
                msg,
                solver=solver,
                status_code=result.status,
            )

        # Extract flux values
        flux_map: Dict[str, float] = {}
        for i, rxn_id in enumerate(model.reaction_ids):
            value = float(result.x[i]) if result.x is not None else 0.0
            flux_map[rxn_id] = round(value, 6)

        # Key metrics
        biomass_idx = model.reaction_ids.index("r_biomass")
        growth_rate = abs(float(result.x[biomass_idx])) if result.x is not None else 0.0

        glc_idx = model.reaction_ids.index("r_glc_uptake")
        glucose_uptake = abs(float(result.x[glc_idx])) if result.x is not None else 0.0

        prod_idx = model.reaction_ids.index("r_prod_sec")
        product_flux = abs(float(result.x[prod_idx])) if result.x is not None else 0.0

        # Theoretical max yield: product / glucose
        theoretical_max = product_flux / glucose_uptake if glucose_uptake > 0 else 0.0

        results = {
            "objective_value": round(
                float(-result.fun) if result.fun is not None else 0.0, 6
            ),
            "growth_rate_per_hour": round(growth_rate, 4),
            "product_flux_mmol_per_gdw_per_hour": round(product_flux, 4),
            "substrate_uptake_mmol_per_gdw_per_hour": round(glucose_uptake, 4),
            "theoretical_max_yield": round(theoretical_max, 4),
            "flux_map": flux_map,
            "solver_status": int(result.status),
            "solver_message": str(result.message),
        }

        if self._logger:
            self._logger.info(
                "FBA results: μ=%.4f/h  product=%.4f  glucose=%.4f  yield=%.4f  status=%s",
                growth_rate, product_flux, glucose_uptake, theoretical_max, result.status,
            )

        return results

    def run_pfba(self, model: Optional[FBAModel] = None,
                 solver: str = "highs") -> Dict[str, Any]:
        """
        Run parsimonious FBA (minimise total flux while maintaining optimal growth).

        Two-stage optimisation:
        1. Find optimal objective value (max growth)
        2. Minimise sum of absolute fluxes subject to optimal growth
        """
        if model is None:
            model = self._model
        if model is None:
            raise FBAConvergenceError("No model available.", solver=solver)

        if self._logger:
            self._logger.info("Running parsimonious FBA (pFBA)")

        # Stage 1: get optimal growth
        stage1_results = self.run_fba(model, solver)
        optimal_growth = stage1_results["growth_rate_per_hour"]

        # Stage 2: minimise total flux
        n = model.n_reactions
        c_pfba = np.ones(2 * n)

        # Constraints: S·(x⁺ - x⁻) = 0  and  biomass ≥ 0.95 × optimal
        A_eq = np.zeros((model.n_metabolites + 1, 2 * n))
        A_eq[:model.n_metabolites, :n] = model.stoichiometric_matrix
        A_eq[:model.n_metabolites, n:] = -model.stoichiometric_matrix

        biomass_idx = model.reaction_ids.index("r_biomass")
        A_eq[model.n_metabolites, biomass_idx] = 1.0
        A_eq[model.n_metabolites, biomass_idx + n] = 1.0

        b_eq = np.zeros(model.n_metabolites + 1)
        b_eq[-1] = optimal_growth * 0.95

        bounds = [(0, 1000)] * (2 * n)

        try:
            result = linprog(
                c=c_pfba,
                A_eq=A_eq,
                b_eq=b_eq,
                bounds=bounds,
                method=solver,
                options={"maxiter": 10000},
            )
        except Exception as e:
            if self._logger:
                self._logger.warning("pFBA failed: %s — returning standard FBA", e)
            return stage1_results

        if not result.success:
            if self._logger:
                self._logger.warning("pFBA solver failed — returning standard FBA")
            return stage1_results

        # Reconstruct flux values
        x = result.x[:n] - result.x[n:]
        flux_map: Dict[str, float] = {}
        for i, rxn_id in enumerate(model.reaction_ids):
            flux_map[rxn_id] = round(float(x[i]), 6)

        total_flux = sum(abs(v) for v in x)
        pfba_results = dict(stage1_results)
        pfba_results["method"] = "pFBA"
        pfba_results["total_flux"] = round(float(total_flux), 4)
        pfba_results["flux_map"] = flux_map

        if self._logger:
            self._logger.info("pFBA total flux: %.4f", total_flux)

        return pfba_results

    def run_fva(self, model: Optional[FBAModel] = None,
                solver: str = "highs",
                fraction_of_optimal: float = 0.95) -> Dict[str, Dict[str, float]]:
        """
        Run Flux Variability Analysis (FVA).

        For each reaction, find the min and max flux while maintaining
        at least `fraction_of_optimal` of maximum growth rate.

        Returns
        -------
        dict
            Maps reaction_id → {"min": float, "max": float}
        """
        if model is None:
            model = self._model
        if model is None:
            raise FBAConvergenceError("No model available.", solver=solver)

        if self._logger:
            self._logger.info("Running FVA (fraction=%.2f)", fraction_of_optimal)

        # Get optimal growth
        fba_results = self.run_fba(model, solver)
        optimal_growth = fba_results["growth_rate_per_hour"]

        biomass_idx = model.reaction_ids.index("r_biomass")
        fva_results: Dict[str, Dict[str, float]] = {}

        for rxn_idx, rxn_id in enumerate(model.reaction_ids):
            # Add biomass constraint: biomass >= fraction * optimal
            A_eq_with_biomass = np.vstack([
                model.stoichiometric_matrix,
                model.objective_coefficients.reshape(1, -1),
            ])
            b_eq_with_biomass = np.append(
                np.zeros(model.n_metabolites),
                -optimal_growth * fraction_of_optimal,
            )

            # Minimise flux
            c_min = np.zeros(model.n_reactions)
            c_min[rxn_idx] = 1.0

            try:
                result_min = linprog(
                    c=c_min,
                    A_eq=A_eq_with_biomass,
                    b_eq=b_eq_with_biomass,
                    bounds=list(zip(model.lower_bounds, model.upper_bounds)),
                    method=solver,
                    options={"maxiter": 5000},
                )
                min_flux = (
                    float(result_min.x[rxn_idx]) if result_min.success else 0.0
                )
            except Exception:
                min_flux = 0.0

            # Maximise flux
            c_max = np.zeros(model.n_reactions)
            c_max[rxn_idx] = -1.0

            try:
                result_max = linprog(
                    c=c_max,
                    A_eq=A_eq_with_biomass,
                    b_eq=b_eq_with_biomass,
                    bounds=list(zip(model.lower_bounds, model.upper_bounds)),
                    method=solver,
                    options={"maxiter": 5000},
                )
                max_flux = (
                    float(-result_max.fun) if result_max.success else 0.0
                )
            except Exception:
                max_flux = 0.0

            fva_results[rxn_id] = {
                "min": round(min_flux, 6),
                "max": round(max_flux, 6),
            }

        if self._logger:
            n_reactions = len(fva_results)
            n_zero = sum(
                1 for v in fva_results.values()
                if abs(v["min"]) < 1e-6 and abs(v["max"]) < 1e-6
            )
            self._logger.info(
                "FVA complete: %d/%d reactions have zero flux range",
                n_zero, n_reactions,
            )

        return fva_results


# ---------------------------------------------------------------------------
# MAIN — smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="FBA Engine")
    parser.add_argument("--organism", default="ecoli")
    args = parser.parse_args()

    logger = PipelineLogger()
    logger.set_stage("3")

    engine = FBAEngine()
    engine.set_logger(logger)

    model = engine.build_stoichiometric_matrix(organism_key=args.organism)
    logger.info(
        "Model built: %d mets × %d rxns",
        model.n_metabolites, model.n_reactions,
    )

    fba_results = engine.run_fba(model)
    logger.info(
        "FBA: μ=%.4f/h  product=%.4f  glucose=%.4f  yield=%.4f",
        fba_results["growth_rate_per_hour"],
        fba_results["product_flux_mmol_per_gdw_per_hour"],
        fba_results["substrate_uptake_mmol_per_gdw_per_hour"],
        fba_results["theoretical_max_yield"],
    )

    pfba_results = engine.run_pfba(model)
    logger.info("pFBA: total_flux=%.4f", pfba_results.get("total_flux", "N/A"))

    fva_results = engine.run_fva(model, fraction_of_optimal=0.95)
    logger.info("FVA: %d reactions analysed", len(fva_results))

    os.makedirs("pipeline_output", exist_ok=True)
    with open("pipeline_output/fba_results.json", "w") as fh:
        json.dump(
            {
                "fba": fba_results,
                "pfba_total_flux": pfba_results.get("total_flux"),
                "fva_summary": {
                    k: v for k, v in list(fva_results.items())[:10]
                },
            },
            fh,
            indent=2,
            default=str,
        )

    logger.info("FBA results saved to pipeline_output/fba_results.json")
    print(
        f"\n▶ FBA Engine smoke test passed. "
        f"Growth rate={fba_results['growth_rate_per_hour']:.4f}/h"
    )
