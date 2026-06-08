"""
Fermentation Simulator Module

Implements ODE-based fermentation kinetics with organism-specific
models including Monod growth, substrate consumption, product
formation, and bioreactor control.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from dbtl_loop import DBTLCycle, DBTLOrchestrator
from exceptions import FermentationSimulationError
from logger_setup import PipelineLogger


# ---------------------------------------------------------------------------
# DATACLASSES
# ---------------------------------------------------------------------------

@dataclass
class FermentationState:
    """Current state of the fermentation process."""
    time_hours: float = 0.0
    biomass_g_per_l: float = 0.1
    substrate_g_per_l: float = 20.0
    product_g_per_l: float = 0.0
    dissolved_o2_percent: float = 100.0
    ph: float = 7.0
    temperature_c: float = 37.0
    volume_l: float = 2.0
    acetate_g_per_l: float = 0.0  # E. coli overflow metabolite
    ethanol_g_per_l: float = 0.0  # S. cerevisiae overflow
    spore_fraction: float = 0.0   # B. subtilis sporulation
    pha_g_per_l: float = 0.0      # P. putida PHA accumulation
    glutamate_g_per_l: float = 0.0  # C. glutamicum secretion


# ---------------------------------------------------------------------------
# ORGANISM-SPECIFIC KINETIC PARAMETERS
# ---------------------------------------------------------------------------

ORGANISM_KINETICS: Dict[str, Dict[str, float]] = {
    "ecoli": {
        "mu_max": 0.90,        # h⁻¹
        "Ks": 0.1,             # g/L (glucose half-saturation)
        "Ki_glucose": 50.0,    # g/L (substrate inhibition)
        "Ki_acetate": 5.0,     # g/L (acetate inhibition)
        "Yx_s": 0.45,          # g biomass / g glucose
        "Yp_s": 0.15,          # g product / g glucose
        "Yp_x": 0.08,          # g product / g biomass
        "m_s": 0.02,           # g glucose / g biomass / h (maintenance)
        "qO2_max": 15.0,       # mmol O2 / g biomass / h
        "critical_do": 10.0,   # % saturation below which growth stops
        "optimal_temp": 37.0,
        "optimal_ph": 7.0,
    },
    "ecoli_bl21": {
        "mu_max": 0.85,
        "Ks": 0.12,
        "Ki_glucose": 45.0,
        "Ki_acetate": 4.5,
        "Yx_s": 0.42,
        "Yp_s": 0.18,
        "Yp_x": 0.10,
        "m_s": 0.025,
        "qO2_max": 14.0,
        "critical_do": 10.0,
        "optimal_temp": 37.0,
        "optimal_ph": 7.0,
    },
    "scerevisiae": {
        "mu_max": 0.40,
        "Ks": 0.15,
        "Ki_glucose": 100.0,
        "Ki_acetate": 1.0,
        "Yx_s": 0.10,
        "Yp_s": 0.48,  # Ethanol yield
        "Yp_x": 2.0,
        "m_s": 0.015,
        "qO2_max": 10.0,
        "critical_do": 5.0,
        "optimal_temp": 30.0,
        "optimal_ph": 5.5,
        "crabtree_threshold": 0.1,  # g/L glucose above which Crabtree effect kicks in
    },
    "scerevisiae_by": {
        "mu_max": 0.38,
        "Ks": 0.15,
        "Ki_glucose": 95.0,
        "Ki_acetate": 1.0,
        "Yx_s": 0.09,
        "Yp_s": 0.45,
        "Yp_x": 1.8,
        "m_s": 0.018,
        "qO2_max": 9.5,
        "critical_do": 5.0,
        "optimal_temp": 30.0,
        "optimal_ph": 5.5,
        "crabtree_threshold": 0.1,
    },
    "bsubtilis": {
        "mu_max": 0.70,
        "Ks": 0.08,
        "Ki_glucose": 30.0,
        "Ki_acetate": 3.0,
        "Yx_s": 0.40,
        "Yp_s": 0.12,
        "Yp_x": 0.06,
        "m_s": 0.02,
        "qO2_max": 12.0,
        "critical_do": 15.0,
        "optimal_temp": 37.0,
        "optimal_ph": 7.0,
        "sporulation_threshold": 0.01,  # g/L substrate below which sporulation starts
    },
    "cglutamicum": {
        "mu_max": 0.45,
        "Ks": 0.05,
        "Ki_glucose": 80.0,
        "Ki_acetate": 2.0,
        "Yx_s": 0.35,
        "Yp_s": 0.50,  # Glutamate secretion
        "Yp_x": 1.5,
        "m_s": 0.01,
        "qO2_max": 8.0,
        "critical_do": 20.0,
        "optimal_temp": 30.0,
        "optimal_ph": 7.2,
    },
    "pputida": {
        "mu_max": 0.60,
        "Ks": 0.06,
        "Ki_glucose": 40.0,
        "Ki_acetate": 2.5,
        "Yx_s": 0.38,
        "Yp_s": 0.20,  # PHA yield
        "Yp_x": 0.50,
        "m_s": 0.015,
        "qO2_max": 13.0,
        "critical_do": 10.0,
        "optimal_temp": 30.0,
        "optimal_ph": 7.0,
        "pha_trigger_biomass": 5.0,  # g/L biomass above which PHA accumulates
    },
}


# ---------------------------------------------------------------------------
# FERMENTATION SIMULATOR
# ---------------------------------------------------------------------------

class FermentationSimulator:
    """
    ODE-based fermentation simulator with organism-specific kinetics.

    Simulates:
    - Monod growth kinetics
    - Substrate consumption
    - Product formation
    - Organism-specific phenomena (acetate overflow, Crabtree effect, etc.)
    - Fed-batch feeding strategies
    """

    def __init__(self) -> None:
        self._logger: Optional[PipelineLogger] = None
        self._state: Optional[FermentationState] = None
        self._time_series: Optional[pd.DataFrame] = None

    def set_logger(self, logger: PipelineLogger) -> None:
        self._logger = logger

    def monod_kinetics(
        self,
        substrate: float,
        mu_max: float,
        Ks: float,
        Ki: Optional[float] = None,
    ) -> float:
        """
        Calculate specific growth rate using Monod kinetics with optional
        substrate inhibition (Haldane modification).

        μ = μ_max × S / (Ks + S)           (standard Monod)
        μ = μ_max × S / (Ks + S + S²/Ki)   (Haldane inhibition)
        """
        if Ki is not None and Ki > 0:
            # Haldane substrate inhibition
            mu = mu_max * substrate / (Ks + substrate + substrate**2 / Ki)
        else:
            # Standard Monod
            mu = mu_max * substrate / (Ks + substrate)

        return max(0.0, mu)

    def _ode_system(
        self,
        t: float,
        y: np.ndarray,
        params: Dict[str, Any],
    ) -> List[float]:
        """
        ODE system for fermentation dynamics.

        State vector y = [biomass, substrate, product, acetate, dissolved_O2]

        Parameters
        ----------
        t : float
            Current time (hours).
        y : np.ndarray
            Current state vector.
        params : dict
            Kinetic parameters and operating conditions.

        Returns
        -------
        list of float
            Derivatives of state variables.
        """
        X, S, P, A, DO = y

        organism = params.get("organism", "ecoli")
        kinetics = ORGANISM_KINETICS.get(organism, ORGANISM_KINETICS["ecoli"])

        mu_max = kinetics["mu_max"]
        Ks = kinetics["Ks"]
        Ki_glc = kinetics.get("Ki_glucose")
        Ki_ace = kinetics.get("Ki_acetate", 10.0)
        Yx_s = kinetics["Yx_s"]
        Yp_s = kinetics["Yp_s"]
        Yp_x = kinetics["Yp_x"]
        m_s = kinetics["m_s"]
        qO2_max = kinetics["qO2_max"]
        critical_do = kinetics["critical_do"]
        optimal_temp = kinetics["optimal_temp"]
        optimal_ph = kinetics["optimal_ph"]

        # Operating conditions
        temp = params.get("temperature", optimal_temp)
        ph = params.get("ph", optimal_ph)
        glucose_feed = params.get("glucose_feed", 0.0)  # g/L/h

        # Temperature and pH correction factors
        temp_factor = math.exp(-((temp - optimal_temp) / 10.0) ** 2)
        ph_factor = math.exp(-((ph - optimal_ph) / 1.0) ** 2)

        # Specific growth rate (Monod with Haldane inhibition)
        mu = self.monod_kinetics(S, mu_max, Ks, Ki_glc)
        mu *= temp_factor * ph_factor

        # DO limitation
        if DO < critical_do:
            mu *= max(0.0, DO / critical_do)

        # Organism-specific effects
        if organism in ("ecoli", "ecoli_bl21"):
            # Acetate overflow metabolism
            overflow_rate = self.acetate_overflow_term(S, X, A, kinetics, params)
            A_dot = overflow_rate - 0.1 * A  # Acetate re-assimilation
            mu *= max(0.0, 1.0 - A / Ki_ace)
        elif organism in ("scerevisiae", "scerevisiae_by"):
            # Crabtree effect (ethanol production at high glucose)
            crabtree_rate = self.crabtree_effect_term(S, X, kinetics, params)
            A = crabtree_rate  # Reuse A variable for ethanol
            A_dot = crabtree_rate - 0.05 * A
            mu *= max(0.0, 1.0 - A / Ki_ace)
        elif organism == "bsubtilis":
            # Sporulation trigger at low substrate
            sporulation = self.sporulation_trigger_term(S, kinetics, params)
            A_dot = -0.1 * A  # No acetate in B. subtilis
            mu *= max(0.0, 1.0 - sporulation)
        elif organism == "cglutamicum":
            # Amino acid secretion (glutamate)
            aa_secretion = self.amino_acid_secretion_term(X, S, kinetics, params)
            A_dot = aa_secretion - 0.02 * A
            mu *= max(0.0, 1.0 - A / Ki_ace)
        elif organism == "pputida":
            # PHA accumulation
            pha_rate = self.pha_accumulation_term(X, S, kinetics, params)
            A_dot = pha_rate - 0.01 * A
            mu *= max(0.0, 1.0 - A / Ki_ace)
        else:
            A_dot = 0.0

        # Mass balance equations
        dX_dt = mu * X  # Biomass growth

        # Substrate consumption: growth + maintenance + feed (prevent going below 0.001)
        substrate_consumption = mu * X / Yx_s + m_s * X
        dS_dt = -substrate_consumption + glucose_feed
        # Clamp to prevent numerical issues near zero
        if S < 0.01 and dS_dt < 0:
            dS_dt = 0.0

        # Product formation: growth-associated + non-growth-associated
        dP_dt = Yp_s * substrate_consumption + Yp_x * mu * X

        # Dissolved oxygen: consumption + transfer
        oxygen_consumption = qO2_max * X * (S / (Ks + S))
        kla = params.get("kla", 200.0)  # h⁻¹
        DO_sat = params.get("do_sat", 100.0)
        dDO_dt = -oxygen_consumption + kla * (DO_sat - DO)

        # Ensure non-negative values
        dS_dt = max(dS_dt, -S) if dS_dt < 0 else dS_dt
        dDO_dt = max(dDO_dt, -DO) if dDO_dt < 0 else dDO_dt

        return [dX_dt, dS_dt, dP_dt, A_dot, dDO_dt]

    def acetate_overflow_term(
        self, S: float, X: float, A: float,
        kinetics: Dict[str, float], params: Dict[str, Any],
    ) -> float:
        """E. coli acetate overflow at high glucose uptake rates."""
        threshold = kinetics.get("Ki_glucose", 50.0) * 0.3
        if S > threshold:
            overflow = 0.15 * X * (S - threshold) / (S + threshold)
        else:
            overflow = 0.0
        return overflow

    def crabtree_effect_term(
        self, S: float, X: float,
        kinetics: Dict[str, float], params: Dict[str, Any],
    ) -> float:
        """S. cerevisiae Crabtree effect (ethanol production at high glucose)."""
        threshold = kinetics.get("crabtree_threshold", 0.1)
        if S > threshold:
            ethanol_rate = 0.25 * X * (S - threshold) / (S + threshold)
        else:
            ethanol_rate = 0.0
        return ethanol_rate

    def sporulation_trigger_term(
        self, S: float,
        kinetics: Dict[str, float], params: Dict[str, Any],
    ) -> float:
        """B. subtilis sporulation trigger at low substrate."""
        threshold = kinetics.get("sporulation_threshold", 0.01)
        if S < threshold:
            return 0.5  # 50% growth reduction
        return 0.0

    def amino_acid_secretion_term(
        self, X: float, S: float,
        kinetics: Dict[str, float], params: Dict[str, Any],
    ) -> float:
        """C. glutamicum amino acid (glutamate) secretion."""
        if S > 0.5:
            return 0.3 * X * S / (S + 0.5)
        return 0.0

    def pha_accumulation_term(
        self, X: float, S: float,
        kinetics: Dict[str, float], params: Dict[str, Any],
    ) -> float:
        """P. putida PHA accumulation."""
        threshold = kinetics.get("pha_trigger_biomass", 5.0)
        if X > threshold:
            return 0.2 * (X - threshold) * S / (S + 0.1)
        return 0.0

    def run_ode_simulation(
        self,
        organism_key: str = "ecoli",
        duration_hours: float = 72.0,
        initial_state: Optional[FermentationState] = None,
        glucose_feed_g_per_l_per_h: float = 0.5,
        temperature_c: Optional[float] = None,
        ph: Optional[float] = None,
        do_percent_saturation: Optional[float] = None,
        agitation_rpm: float = 300.0,
        aeration_vvm: float = 1.0,
        save_time_series: bool = True,
    ) -> Tuple[FermentationState, Optional[pd.DataFrame]]:
        """
        Run the full ODE simulation for a fermentation process.

        Parameters
        ----------
        organism_key : str
            Target organism.
        duration_hours : float
            Total fermentation duration.
        initial_state : FermentationState, optional
            Starting conditions.
        glucose_feed_g_per_l_per_h : float
            Continuous glucose feed rate.
        temperature_c : float, optional
            Fermentation temperature (uses organism optimal if None).
        ph : float, optional
            Fermentation pH (uses organism optimal if None).
        do_percent_saturation : float, optional
            Target DO% (uses organism critical if None).
        agitation_rpm : float
            Agitator speed (RPM).
        aeration_vvm : float
            Aeration rate (vessel volumes per minute).
        save_time_series : bool
            Whether to return the full time series.

        Returns
        -------
        tuple of (FermentationState, DataFrame or None)
        """
        if self._logger:
            self._logger.info(
                "Running ODE simulation: organism=%s, duration=%.1f h, "
                "feed=%.2f g/L/h",
                organism_key, duration_hours, glucose_feed_g_per_l_per_h,
            )

        kinetics = ORGANISM_KINETICS.get(organism_key, ORGANISM_KINETICS["ecoli"])

        # Initial conditions
        if initial_state is None:
            state = FermentationState(
                temperature_c=kinetics["optimal_temp"],
                ph=kinetics["optimal_ph"],
            )
        else:
            state = initial_state

        # ODE parameters
        params = {
            "organism": organism_key,
            "temperature": temperature_c or kinetics["optimal_temp"],
            "ph": ph or kinetics["optimal_ph"],
            "glucose_feed": glucose_feed_g_per_l_per_h,
            "kla": 200.0 + agitation_rpm * 0.5 + aeration_vvm * 50.0,  # Simplified kLa
            "do_sat": do_percent_saturation or 100.0,
        }

        # Initial state vector
        y0 = [
            state.biomass_g_per_l,
            state.substrate_g_per_l,
            state.product_g_per_l,
            state.acetate_g_per_l if organism_key in ("ecoli", "ecoli_bl21") else 0.0,
            state.dissolved_o2_percent,
        ]

        # Time span
        t_span = (0.0, duration_hours)
        t_eval = np.linspace(0.0, duration_hours, int(duration_hours * 4) + 1)

        try:
            sol = solve_ivp(
                fun=lambda t, y: self._ode_system(t, y, params),
                t_span=t_span,
                y0=y0,
                t_eval=t_eval,
                method="LSODA",
                rtol=1e-6,
                atol=1e-9,
                max_step=0.5,
            )
        except Exception as e:
            raise FermentationSimulationError(
                f"ODE integration failed: {e}",
                simulation_mode="batch",
                time_point=0.0,
            ) from e

        if not sol.success:
            msg = f"ODE solver failed: {sol.message}"
            if self._logger:
                self._logger.error(msg)
            raise FermentationSimulationError(
                msg,
                simulation_mode="batch",
                time_point=float(sol.t[-1]) if len(sol.t) > 0 else 0.0,
            )

        # Extract final state
        final = sol.y[:, -1]
        final_state = FermentationState(
            time_hours=duration_hours,
            biomass_g_per_l=round(float(final[0]), 4),
            substrate_g_per_l=round(float(final[1]), 4),
            product_g_per_l=round(float(final[2]), 4),
            dissolved_o2_percent=round(float(final[4]), 2),
            temperature_c=params["temperature"],
            ph=params["ph"],
            acetate_g_per_l=round(float(final[3]), 4) if organism_key in ("ecoli", "ecoli_bl21") else 0.0,
            ethanol_g_per_l=round(float(final[3]), 4) if organism_key in ("scerevisiae", "scerevisiae_by") else 0.0,
            glutamate_g_per_l=round(float(final[3]), 4) if organism_key == "cglutamicum" else 0.0,
            pha_g_per_l=round(float(final[3]), 4) if organism_key == "pputida" else 0.0,
        )

        # Time series
        time_series = None
        if save_time_series:
            time_series = pd.DataFrame({
                "time_hours": sol.t,
                "biomass_g_per_l": sol.y[0],
                "substrate_g_per_l": sol.y[1],
                "product_g_per_l": sol.y[2],
                "byproduct_g_per_l": sol.y[3],
                "do_percent": sol.y[4],
            })
            self._time_series = time_series

        self._state = final_state

        if self._logger:
            self._logger.info(
                "Simulation complete: biomass=%.2f g/L, product=%.2f g/L, "
                "substrate=%.2f g/L, DO=%.1f%%",
                final_state.biomass_g_per_l,
                final_state.product_g_per_l,
                final_state.substrate_g_per_l,
                final_state.dissolved_o2_percent,
            )

        return final_state, time_series

    def fed_batch_feeding_strategy(
        self,
        target_biomass: float = 50.0,
        initial_substrate: float = 20.0,
        yield_coeff: float = 0.45,
    ) -> List[Dict[str, float]]:
        """
        Calculate glucose feed profile for fed-batch operation.

        Returns a list of time points with feed rates.
        """
        if self._logger:
            self._logger.info(
                "Calculating fed-batch feed profile: target=%.1f g/L",
                target_biomass,
            )

        # Simple exponential feed strategy
        feed_profile = []
        total_time = 72.0  # hours
        dt = 1.0  # 1-hour intervals

        for t in np.arange(0, total_time, dt):
            # Exponential feed to maintain constant specific growth rate
            mu = 0.15  # h⁻¹ (moderate growth rate)
            feed_rate = (mu * target_biomass / yield_coeff) * math.exp(mu * t / 24.0)
            feed_profile.append({
                "time_hours": round(t, 1),
                "feed_rate_g_per_l_per_h": round(min(feed_rate, 5.0), 3),
                "cumulative_glucose_g_per_l": round(initial_substrate + sum(
                    fp["feed_rate_g_per_l_per_h"] for fp in feed_profile
                ) * dt, 2),
            })

        if self._logger:
            self._logger.info(
                "Feed profile: %d time points, final cumulative=%.1f g/L",
                len(feed_profile), feed_profile[-1]["cumulative_glucose_g_per_l"],
            )

        return feed_profile

    def organism_specific_events(
        self,
        organism_key: str,
        state: FermentationState,
    ) -> List[str]:
        """
        Identify organism-specific events during fermentation.
        """
        events: List[str] = []

        if organism_key in ("ecoli", "ecoli_bl21"):
            if state.acetate_g_per_l > 2.0:
                events.append(f"Acetate accumulation detected: {state.acetate_g_per_l:.2f} g/L")
            if state.substrate_g_per_l > 10.0:
                events.append("High glucose: risk of overflow metabolism")
            if state.dissolved_o2_percent < 20.0:
                events.append("Low DO: potential oxygen limitation")

        elif organism_key in ("scerevisiae", "scerevisiae_by"):
            if state.ethanol_g_per_l > 5.0:
                events.append(f"Crabtree effect: ethanol at {state.ethanol_g_per_l:.2f} g/L")
            if state.ph < 4.5:
                events.append("Low pH: potential acid stress")

        elif organism_key == "bsubtilis":
            if state.substrate_g_per_l < 0.5:
                events.append("Low substrate: sporulation may be triggered")
            if state.biomass_g_per_l > 30.0:
                events.append("High biomass: risk of protease secretion")

        elif organism_key == "cglutamicum":
            if state.glutamate_g_per_l > 10.0:
                events.append(f"Glutamate secretion: {state.glutamate_g_per_l:.2f} g/L")
            if state.ph < 6.5:
                events.append("Low pH: may affect glutamate production")

        elif organism_key == "pputida":
            if state.pha_g_per_l > 5.0:
                events.append(f"PHA accumulation: {state.pha_g_per_l:.2f} g/L")
            if state.biomass_g_per_l > 40.0:
                events.append("High biomass: PHA production phase")

        return events


# ---------------------------------------------------------------------------
# MAIN — smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fermentation Simulator")
    parser.add_argument("--organism", default="ecoli")
    parser.add_argument("--duration", type=float, default=48.0)
    parser.add_argument("--feed", type=float, default=0.5)
    args = parser.parse_args()

    logger = PipelineLogger()
    logger.set_stage("4")

    simulator = FermentationSimulator()
    simulator.set_logger(logger)

    state, ts = simulator.run_ode_simulation(
        organism_key=args.organism,
        duration_hours=args.duration,
        glucose_feed_g_per_l_per_h=args.feed,
    )

    events = simulator.organism_specific_events(args.organism, state)

    logger.info("Final state: biomass=%.2f, product=%.2f, substrate=%.2f",
                state.biomass_g_per_l, state.product_g_per_l, state.substrate_g_per_l)
    if events:
        logger.info("Events: %s", events)

    feed_profile = simulator.fed_batch_feeding_strategy(target_biomass=50.0)

    os.makedirs("pipeline_output", exist_ok=True)
    with open("pipeline_output/fermentation_results.json", "w") as fh:
        json.dump({
            "final_state": {
                "biomass_g_per_l": state.biomass_g_per_l,
                "product_g_per_l": state.product_g_per_l,
                "substrate_g_per_l": state.substrate_g_per_l,
                "do_percent": state.dissolved_o2_percent,
                "time_hours": state.time_hours,
            },
            "events": events,
            "feed_profile_summary": feed_profile[-1] if feed_profile else {},
        }, fh, indent=2, default=str)

    if ts is not None:
        ts.to_csv("pipeline_output/fermentation_timeseries.csv", index=False)
        logger.info("Time series saved to pipeline_output/fermentation_timeseries.csv")

    logger.info("Fermentation results saved to pipeline_output/fermentation_results.json")
    print(
        f"\n▶ Fermentation Simulator smoke test passed. "
        f"Product: {state.product_g_per_l:.2f} g/L"
    )
