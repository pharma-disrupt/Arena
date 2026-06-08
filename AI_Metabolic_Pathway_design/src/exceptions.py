"""
Custom Exceptions Module

Defines all pipeline-specific exception classes for consistent error handling
across all stages.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class PipelineError(Exception):
    """
    Base exception for all pipeline errors.

    Attributes
    ----------
    message : str
        Human-readable error description.
    stage : str
        Pipeline stage where the error occurred (e.g. "1", "2", ...).
    function : str
        Name of the function that raised the error.
    input_json : dict, optional
        The JSON payload that was being processed when the error occurred.
    """

    def __init__(
        self,
        message: str,
        stage: str = "UNKNOWN",
        function: str = "unknown",
        input_json: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.stage = stage
        self.function = function
        self.input_json = input_json or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "exception_type": self.__class__.__name__,
            "message": self.message,
            "stage": self.stage,
            "function": self.function,
            "input_json": self.input_json,
        }


class DataIngestionError(PipelineError):
    """Raised when organism/molecule data cannot be loaded or is malformed."""

    def __init__(
        self,
        message: str,
        source: str = "unknown",
        input_json: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            stage="1",
            function=f"data_ingestion:{source}",
            input_json=input_json,
        )
        self.source = source


class SchemaValidationError(PipelineError):
    """Raised when a JSON payload fails schema validation."""

    def __init__(
        self,
        message: str,
        schema_name: str = "unknown",
        validation_errors: Optional[list] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            stage="VALIDATION",
            function=f"validate:{schema_name}",
            input_json=payload,
        )
        self.schema_name = schema_name
        self.validation_errors = validation_errors or []

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["schema_name"] = self.schema_name
        d["validation_errors"] = [str(e) for e in self.validation_errors]
        return d


class ModelInferenceError(PipelineError):
    """Raised when an ML/AI model fails during inference."""

    def __init__(
        self,
        message: str,
        model_name: str = "unknown",
        fallback_method: Optional[str] = None,
        input_json: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            stage="2",
            function=f"model_inference:{model_name}",
            input_json=input_json,
        )
        self.model_name = model_name
        self.fallback_method = fallback_method


class FBAConvergenceError(PipelineError):
    """Raised when the flux balance analysis solver fails to converge."""

    def __init__(
        self,
        message: str,
        solver: str = "unknown",
        status_code: Optional[int] = None,
        input_json: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            stage="3",
            function=f"fba_solve:{solver}",
            input_json=input_json,
        )
        self.solver = solver
        self.status_code = status_code


class FermentationSimulationError(PipelineError):
    """Raised when the ODE-based fermentation simulation fails."""

    def __init__(
        self,
        message: str,
        simulation_mode: str = "unknown",
        time_point: Optional[float] = None,
        input_json: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            stage="4",
            function=f"fermentation_sim:{simulation_mode}",
            input_json=input_json,
        )
        self.simulation_mode = simulation_mode
        self.time_point = time_point


class ScaleUpError(PipelineError):
    """Raised when scale-up cascade calculations fail."""

    def __init__(
        self,
        message: str,
        scale_level: Optional[float] = None,
        input_json: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            stage="5",
            function=f"scaleup:{scale_level}L",
            input_json=input_json,
        )
        self.scale_level = scale_level


class DownstreamError(PipelineError):
    """Raised when downstream processing simulation fails."""

    def __init__(
        self,
        message: str,
        step: str = "unknown",
        input_json: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            stage="5",
            function=f"downstream:{step}",
            input_json=input_json,
        )
        self.step = step


class RegulatoryError(PipelineError):
    """Raised when regulatory assessment fails."""

    def __init__(
        self,
        message: str,
        check_type: str = "unknown",
        input_json: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            stage="5",
            function=f"regulatory:{check_type}",
            input_json=input_json,
        )
        self.check_type = check_type


# ──────────────────────────────────────────────────────────────────────────────
# MAIN — Quick exception test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    import logging

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    # Test each exception
    try:
        raise PipelineError("Generic pipeline failure", stage="1", function="test")
    except PipelineError as e:
        logger.info("PipelineError: %s", e.to_dict())

    try:
        raise DataIngestionError("File not found", source="KEGG")
    except DataIngestionError as e:
        logger.info("DataIngestionError: %s", e.to_dict())

    try:
        raise SchemaValidationError("Missing field", schema_name="stage_1_output",
                                    validation_errors=["'organism' is required"])
    except SchemaValidationError as e:
        logger.info("SchemaValidationError: %s", e.to_dict())

    try:
        raise ModelInferenceError("ESM-2 model timeout", model_name="ESM-2",
                                  fallback_method="rule-based")
    except ModelInferenceError as e:
        logger.info("ModelInferenceError: %s", e.to_dict())

    try:
        raise FBAConvergenceError("Solver returned status 3", solver="scipy-linprog",
                                  status_code=3)
    except FBAConvergenceError as e:
        logger.info("FBAConvergenceError: %s", e.to_dict())

    try:
        raise FermentationSimulationError("ODE integration failed at t=48h",
                                          simulation_mode="fed-batch",
                                          time_point=48.0)
    except FermentationSimulationError as e:
        logger.info("FermentationSimulationError: %s", e.to_dict())

    logger.info("All exception tests passed.")
