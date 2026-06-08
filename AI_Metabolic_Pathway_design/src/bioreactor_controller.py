"""
Bioreactor Controller Module

Implements simplified Model Predictive Control (MPC) for bioreactor
operation, with PID fallback for when MPC fails.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from logger_setup import PipelineLogger


# ---------------------------------------------------------------------------
# DATACLASSES
# ---------------------------------------------------------------------------

@dataclass
class BioreactorState:
    """Current state of the bioreactor."""
    time_hours: float = 0.0
    biomass_g_per_l: float = 0.1
    substrate_g_per_l: float = 20.0
    product_g_per_l: float = 0.0
    dissolved_o2_percent: float = 100.0
    ph: float = 7.0
    temperature_c: float = 37.0
    agitation_rpm: float = 300.0
    aeration_vvm: float = 1.0
    feed_rate_g_per_l_per_h: float = 0.0
    volume_l: float = 2.0


# ---------------------------------------------------------------------------
# ORGANISM SETPOINTS
# ---------------------------------------------------------------------------

ORGANISM_SETPOINTS: Dict[str, Dict[str, Any]] = {
    "ecoli": {
        "temperature_c": 37.0,
        "ph": 7.0,
        "do_percent": 30.0,
        "agitation_rpm": 400.0,
        "aeration_vvm": 1.0,
        "feed_rate_g_per_l_per_h": 0.5,
        "max_biomass_g_per_l": 50.0,
    },
    "ecoli_bl21": {
        "temperature_c": 37.0,
        "ph": 7.0,
        "do_percent": 30.0,
        "agitation_rpm": 400.0,
        "aeration_vvm": 1.0,
        "feed_rate_g_per_l_per_h": 0.5,
        "max_biomass_g_per_l": 50.0,
    },
    "scerevisiae": {
        "temperature_c": 30.0,
        "ph": 5.5,
        "do_percent": 20.0,
        "agitation_rpm": 250.0,
        "aeration_vvm": 0.8,
        "feed_rate_g_per_l_per_h": 0.3,
        "max_biomass_g_per_l": 80.0,
    },
    "scerevisiae_by": {
        "temperature_c": 30.0,
        "ph": 5.5,
        "do_percent": 20.0,
        "agitation_rpm": 250.0,
        "aeration_vvm": 0.8,
        "feed_rate_g_per_l_per_h": 0.3,
        "max_biomass_g_per_l": 80.0,
    },
    "bsubtilis": {
        "temperature_c": 37.0,
        "ph": 7.0,
        "do_percent": 30.0,
        "agitation_rpm": 350.0,
        "aeration_vvm": 1.2,
        "feed_rate_g_per_l_per_h": 0.4,
        "max_biomass_g_per_l": 40.0,
    },
    "cglutamicum": {
        "temperature_c": 30.0,
        "ph": 7.2,
        "do_percent": 25.0,
        "agitation_rpm": 300.0,
        "aeration_vvm": 0.8,
        "feed_rate_g_per_l_per_h": 0.3,
        "max_biomass_g_per_l": 60.0,
    },
    "pputida": {
        "temperature_c": 30.0,
        "ph": 7.0,
        "do_percent": 35.0,
        "agitation_rpm": 350.0,
        "aeration_vvm": 1.0,
        "feed_rate_g_per_l_per_h": 0.4,
        "max_biomass_g_per_l": 45.0,
    },
}


# ---------------------------------------------------------------------------
# PID CONTROLLER
# ---------------------------------------------------------------------------

class PIDController:
    """Standard PID controller for single-variable control."""

    def __init__(
        self,
        kp: float = 1.0,
        ki: float = 0.1,
        kd: float = 0.05,
        setpoint: float = 0.0,
        output_limits: Tuple[float, float] = (0.0, 1.0),
    ) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.output_limits = output_limits
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time: Optional[float] = None

    def update(self, measurement: float, current_time: float) -> float:
        """Compute control output."""
        error = self.setpoint - measurement

        if self._prev_time is not None:
            dt = current_time - self._prev_time
            if dt > 0:
                # Proportional
                p_out = self.kp * error

                # Integral
                self._integral += error * dt
                i_out = self.ki * self._integral

                # Derivative
                d_out = self.kd * (error - self._prev_error) / dt

                output = p_out + i_out + d_out
                output = max(self.output_limits[0], min(self.output_limits[1], output))

                self._prev_error = error
                self._prev_time = current_time

                return output

        self._prev_error = error
        self._prev_time = current_time
        return 0.0

    def reset(self) -> None:
        """Reset controller state."""
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = None


# ---------------------------------------------------------------------------
# MPC CONTROLLER
# ---------------------------------------------------------------------------

class MPCController:
    """
    Simplified Model Predictive Controller for bioreactor operation.

    Predicts state 4 hours ahead and optimises control actions
    to minimise deviation from setpoints.
    """

    def __init__(self, organism_key: str = "ecoli",
                 prediction_horizon_hours: float = 4.0,
                 control_interval_hours: float = 1.0) -> None:
        self._organism_key = organism_key
        self._prediction_horizon = prediction_horizon_hours
        self._control_interval = control_interval_hours
        self._setpoints = ORGANISM_SETPOINTS.get(
            organism_key, ORGANISM_SETPOINTS["ecoli"]
        )
        self._logger: Optional[PipelineLogger] = None
        self._pid_fallbacks: Dict[str, PIDController] = {}

    def set_logger(self, logger: PipelineLogger) -> None:
        self._logger = logger

    def predict_state(
        self,
        current_state: BioreactorState,
        control_actions: Dict[str, float],
    ) -> BioreactorState:
        """
        Predict bioreactor state at prediction horizon.

        Uses simplified kinetic model for prediction.
        """
        if self._logger:
            self._logger.debug(
                "Predicting state %.1f h ahead from t=%.1f",
                self._prediction_horizon, current_state.time_hours,
            )

        # Simplified prediction: exponential growth with substrate limitation
        dt = self._prediction_horizon
        mu_max = 0.8  # h⁻¹ (typical E. coli)
        Ks = 0.1  # g/L

        # Current specific growth rate
        mu = mu_max * current_state.substrate_g_per_l / (
            Ks + current_state.substrate_g_per_l
        )

        # Adjust for control actions
        temp_factor = 1.0 - 0.02 * abs(
            control_actions.get("temperature_c", current_state.temperature_c)
            - self._setpoints["temperature_c"]
        )
        ph_factor = 1.0 - 0.05 * abs(
            control_actions.get("ph", current_state.ph)
            - self._setpoints["ph"]
        )
        do_factor = min(1.0, control_actions.get("do_percent", 100.0) / 30.0)
        feed_factor = min(1.0, control_actions.get("feed_rate", 0.5) / 0.5)

        mu *= temp_factor * ph_factor * do_factor * feed_factor

        # Predict future state
        future = BioreactorState(
            time_hours=current_state.time_hours + dt,
            biomass_g_per_l=round(
                current_state.biomass_g_per_l * math.exp(mu * dt), 4
            ),
            substrate_g_per_l=round(
                max(0.0, current_state.substrate_g_per_l - mu * dt * 0.5
                    + control_actions.get("feed_rate", 0.0) * dt), 4
            ),
            product_g_per_l=round(
                current_state.product_g_per_l + mu * dt * 0.15, 4
            ),
            dissolved_o2_percent=round(
                max(0.0, current_state.dissolved_o2_percent
                    - mu * dt * 2.0
                    + control_actions.get("aeration_vvm", 1.0) * 5.0), 2
            ),
            temperature_c=control_actions.get(
                "temperature_c", current_state.temperature_c
            ),
            ph=control_actions.get("ph", current_state.ph),
            agitation_rpm=control_actions.get(
                "agitation_rpm", current_state.agitation_rpm
            ),
            aeration_vvm=control_actions.get(
                "aeration_vvm", current_state.aeration_vvm
            ),
            feed_rate_g_per_l_per_h=control_actions.get(
                "feed_rate", current_state.feed_rate_g_per_l_per_h
            ),
            volume_l=round(
                current_state.volume_l + control_actions.get("feed_rate", 0.0) * dt * 0.01, 2
            ),
        )

        return future

    def optimise_control_action(
        self,
        current_state: BioreactorState,
    ) -> Dict[str, float]:
        """
        Optimise control actions to minimise deviation from setpoints.

        Uses gradient-free optimisation (Nelder-Mead via scipy) to find
        the best control action for the next interval.

        Returns
        -------
        dict
            Optimised control actions.
        """
        if self._logger:
            self._logger.info("Optimising control actions at t=%.1f h",
                              current_state.time_hours)

        def cost_function(actions: np.ndarray) -> float:
            """Cost: sum of squared deviations from setpoints."""
            control = {
                "temperature_c": float(actions[0]),
                "ph": float(actions[1]),
                "do_percent": float(actions[2]),
                "feed_rate": float(actions[3]),
                "agitation_rpm": float(actions[4]),
                "aeration_vvm": float(actions[5]),
            }

            future = self.predict_state(current_state, control)

            # Deviation costs
            temp_cost = (future.temperature_c - self._setpoints["temperature_c"]) ** 2
            ph_cost = (future.ph - self._setpoints["ph"]) ** 2
            do_cost = max(0, self._setpoints["do_percent"] - future.dissolved_o2_percent) ** 2
            feed_cost = (future.feed_rate_g_per_l_per_h - self._setpoints["feed_rate_g_per_l_per_h"]) ** 2

            # Biomass target cost
            biomass_target = min(
                self._setpoints["max_biomass_g_per_l"],
                future.biomass_g_per_l,
            )
            biomass_cost = (biomass_target - future.biomass_g_per_l) ** 2 * 0.1

            return temp_cost + ph_cost * 10 + do_cost + feed_cost + biomass_cost

        # Initial guess: current setpoints
        x0 = np.array([
            current_state.temperature_c,
            current_state.ph,
            current_state.dissolved_o2_percent,
            current_state.feed_rate_g_per_l_per_h,
            current_state.agitation_rpm,
            current_state.aeration_vvm,
        ])

        # Bounds
        bounds = [
            (25.0, 42.0),   # temperature
            (5.0, 8.0),     # pH
            (5.0, 100.0),   # DO%
            (0.0, 5.0),     # feed rate
            (100.0, 1000.0),# agitation
            (0.1, 2.0),     # aeration
        ]

        try:
            from scipy.optimize import differential_evolution
            result = differential_evolution(
                cost_function,
                bounds=bounds,
                maxiter=100,
                seed=42,
            )

            if result.success or result.fun < cost_function(x0):
                optimal = result.x
                return {
                    "temperature_c": round(float(optimal[0]), 1),
                    "ph": round(float(optimal[1]), 2),
                    "do_percent": round(float(optimal[2]), 1),
                    "feed_rate_g_per_l_per_h": round(float(optimal[3]), 2),
                    "agitation_rpm": round(float(optimal[4]), 0),
                    "aeration_vvm": round(float(optimal[5]), 2),
                }
        except Exception as e:
            if self._logger:
                self._logger.warning("MPC optimisation failed: %s — using PID fallback", e)

        # Fallback to PID
        return self._pid_control(current_state)

    def _pid_control(self, current_state: BioreactorState) -> Dict[str, float]:
        """PID fallback control for bioreactor variables."""
        if not self._pid_fallbacks:
            # Initialise PID controllers
            self._pid_fallbacks["temperature"] = PIDController(
                kp=0.5, ki=0.05, kd=0.1,
                setpoint=self._setpoints["temperature_c"],
                output_limits=(25.0, 42.0),
            )
            self._pid_fallbacks["ph"] = PIDController(
                kp=1.0, ki=0.1, kd=0.05,
                setpoint=self._setpoints["ph"],
                output_limits=(5.0, 8.0),
            )
            self._pid_fallbacks["do"] = PIDController(
                kp=0.8, ki=0.08, kd=0.02,
                setpoint=self._setpoints["do_percent"],
                output_limits=(5.0, 100.0),
            )
            self._pid_fallbacks["feed"] = PIDController(
                kp=0.3, ki=0.03, kd=0.01,
                setpoint=self._setpoints["feed_rate_g_per_l_per_h"],
                output_limits=(0.0, 5.0),
            )

        t = current_state.time_hours

        actions = {
            "temperature_c": self._pid_fallbacks["temperature"].update(
                current_state.temperature_c, t
            ),
            "ph": self._pid_fallbacks["ph"].update(
                current_state.ph, t
            ),
            "do_percent": self._pid_fallbacks["do"].update(
                current_state.dissolved_o2_percent, t
            ),
            "feed_rate_g_per_l_per_h": self._pid_fallbacks["feed"].update(
                current_state.feed_rate_g_per_l_per_h, t
            ),
            "agitation_rpm": self._setpoints["agitation_rpm"],
            "aeration_vvm": self._setpoints["aeration_vvm"],
        }

        return actions


# ---------------------------------------------------------------------------
# MAIN — smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Bioreactor Controller")
    parser.add_argument("--organism", default="ecoli")
    args = parser.parse_args()

    logger = PipelineLogger()
    logger.set_stage("4")

    controller = MPCController(organism_key=args.organism)
    controller.set_logger(logger)

    state = BioreactorState(
        time_hours=0.0,
        biomass_g_per_l=0.1,
        substrate_g_per_l=20.0,
        temperature_c=37.0,
        ph=7.0,
        dissolved_o2_percent=100.0,
    )

    actions = controller.optimise_control_action(state)
    logger.info("Optimal control actions: %s", actions)

    future = controller.predict_state(state, actions)
    logger.info("Predicted state at t=4h: biomass=%.2f g/L, product=%.2f g/L",
                future.biomass_g_per_l, future.product_g_per_l)

    os.makedirs("pipeline_output", exist_ok=True)
    with open("pipeline_output/controller_results.json", "w") as fh:
        json.dump({
            "current_state": {
                "time_hours": state.time_hours,
                "biomass_g_per_l": state.biomass_g_per_l,
                "substrate_g_per_l": state.substrate_g_per_l,
            },
            "control_actions": actions,
            "predicted_state": {
                "time_hours": future.time_hours,
                "biomass_g_per_l": future.biomass_g_per_l,
                "product_g_per_l": future.product_g_per_l,
                "substrate_g_per_l": future.substrate_g_per_l,
            },
        }, fh, indent=2, default=str)

    print(f"\n▶ Bioreactor Controller smoke test passed. Actions: {actions}")
