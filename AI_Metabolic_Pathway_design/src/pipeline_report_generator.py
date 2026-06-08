"""
Pipeline Report Generator Module

Compiles all stage outputs into comprehensive reports:
- Human-readable summary
- Machine-readable JSON
- Error report
- HTML export
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from logger_setup import PipelineLogger


# ---------------------------------------------------------------------------
# PIPELINE REPORT
# ---------------------------------------------------------------------------

@dataclass
class PipelineReport:
    """Complete pipeline report merging all stage outputs."""
    pipeline_id: str
    run_timestamp: str
    organism: str
    strain: str
    target_molecule: str
    stage_outputs: Dict[str, Any]
    stage_summaries: Dict[str, Any]
    errors: List[Dict[str, Any]]
    overall_status: str
    total_duration_seconds: float
    recommendations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pipeline_id": self.pipeline_id,
            "run_timestamp": self.run_timestamp,
            "organism": self.organism,
            "strain": self.strain,
            "target_molecule": self.target_molecule,
            "stage_outputs": self.stage_outputs,
            "stage_summaries": self.stage_summaries,
            "errors": self.errors,
            "overall_status": self.overall_status,
            "total_duration_seconds": round(self.total_duration_seconds, 2),
            "recommendations": self.recommendations,
        }


# ---------------------------------------------------------------------------
# REPORT GENERATOR
# ---------------------------------------------------------------------------

class ReportGenerator:
    """Generates comprehensive pipeline reports."""

    def __init__(self) -> None:
        self._logger: Optional[PipelineLogger] = None

    def set_logger(self, logger: PipelineLogger) -> None:
        self._logger = logger

    def compile_all_stages(self,
                           stage_1_output: Optional[Dict[str, Any]] = None,
                           stage_2_output: Optional[Dict[str, Any]] = None,
                           stage_3_output: Optional[Dict[str, Any]] = None,
                           stage_4_output: Optional[Dict[str, Any]] = None,
                           stage_5_output: Optional[Dict[str, Any]] = None,
                           errors: Optional[List[Dict[str, Any]]] = None,
                           ) -> PipelineReport:
        """
        Merge all stage outputs into a single PipelineReport.
        """
        if self._logger:
            self._logger.info("Compiling all stage outputs into final report")

        # Extract organism and molecule info from Stage 1
        organism = "unknown"
        strain = "unknown"
        molecule = "unknown"
        pipeline_id = ""
        start_time = ""

        if stage_1_output:
            org = stage_1_output.get("organism", {})
            organism = org.get("name", "unknown")
            strain = org.get("strain", "unknown")
            mol = stage_1_output.get("target_molecule", {})
            molecule = mol.get("name", "unknown")
            pipeline_id = stage_1_output.get("pipeline_id", "")
            start_time = stage_1_output.get("timestamp", "")

        # Collect all stage outputs
        stage_outputs: Dict[str, Any] = {}
        stage_summaries: Dict[str, str] = {}
        for i, output in enumerate(
            [stage_1_output, stage_2_output, stage_3_output,
             stage_4_output, stage_5_output],
            start=1,
        ):
            if output:
                stage_outputs[f"stage_{i}"] = output
                # Extract status — try various keys
                status = "UNKNOWN"
                for key in [f"stage_{i}_status", "stage_5_status", "status"]:
                    if key in output:
                        val = output[key]
                        if isinstance(val, str):
                            status = val
                        elif isinstance(val, dict):
                            status = val.get("status", "UNKNOWN")
                        break
                stage_summaries[f"stage_{i}"] = status

        # Determine overall status
        statuses = list(stage_summaries.values())
        if "FAIL" in statuses:
            overall_status = "FAIL"
        elif "WARN" in statuses:
            overall_status = "WARN"
        else:
            overall_status = "PASS"

        # Generate recommendations
        recommendations = self._generate_recommendations(
            stage_outputs, stage_summaries, errors or []
        )

        # Calculate duration
        total_duration = 0.0
        for i in range(1, 6):
            summary_path = os.path.join("logs", f"stage_{i}_summary.json")
            if os.path.exists(summary_path):
                try:
                    with open(summary_path, "r") as fh:
                        summary_data = json.load(fh)
                        total_duration += summary_data.get("duration_seconds", 0)
                except Exception:
                    pass

        report = PipelineReport(
            pipeline_id=pipeline_id,
            run_timestamp=start_time or datetime.now(timezone.utc).isoformat(),
            organism=organism,
            strain=strain,
            target_molecule=molecule,
            stage_outputs=stage_outputs,
            stage_summaries=stage_summaries,
            errors=errors or [],
            overall_status=overall_status,
            total_duration_seconds=total_duration,
            recommendations=recommendations,
        )

        if self._logger:
            self._logger.info(
                "Report compiled: pipeline_id=%s, organism=%s, "
                "molecule=%s, status=%s",
                pipeline_id, organism, molecule, overall_status,
            )

        return report

    def generate_summary_report(self, report: PipelineReport) -> str:
        """Generate a human-readable text summary."""
        lines = [
            "=" * 70,
            "  SYNTHETIC BIOLOGY METABOLIC PATHWAY PIPELINE — FINAL REPORT",
            "=" * 70,
            "",
            f"  Pipeline ID     : {report.pipeline_id}",
            f"  Timestamp       : {report.run_timestamp}",
            f"  Organism        : {report.organism} ({report.strain})",
            f"  Target Molecule : {report.target_molecule}",
            f"  Overall Status  : {report.overall_status}",
            f"  Duration        : {report.total_duration_seconds:.2f}s",
            "",
            "  STAGE RESULTS:",
            "  " + "-" * 40,
        ]

        for stage_name, status in report.stage_summaries.items():
            lines.append(f"    {stage_name}: {status}")

        lines.extend([
            "",
            "  RECOMMENDATIONS:",
            "  " + "-" * 40,
        ])
        for rec in report.recommendations:
            lines.append(f"    → {rec}")

        if report.errors:
            lines.extend([
                "",
                f"  ERRORS ({len(report.errors)}):",
                "  " + "-" * 40,
            ])
            for err in report.errors:
                lines.append(
                    f"    [{err.get('stage', '?')}] {err.get('function', '?')}: "
                    f"{err.get('message', '')}"
                )

        lines.extend(["", "=" * 70])
        return "\n".join(lines)

    def generate_json_report(self, report: PipelineReport) -> str:
        """Generate machine-readable JSON report."""
        return json.dumps(report.to_dict(), indent=2, default=str)

    def generate_error_report(self, report: PipelineReport) -> str:
        """Generate error-focused report."""
        if not report.errors:
            return "No errors recorded during pipeline execution."

        lines = [
            "=" * 70,
            "  PIPELINE ERROR REPORT",
            "=" * 70,
            f"  Pipeline ID : {report.pipeline_id}",
            f"  Total Errors: {len(report.errors)}",
            "",
        ]

        for i, err in enumerate(report.errors, start=1):
            lines.extend([
                f"  Error #{i}:",
                f"    Stage    : {err.get('stage', 'unknown')}",
                f"    Function : {err.get('function', 'unknown')}",
                f"    Type     : {err.get('exception_type', 'unknown')}",
                f"    Message  : {err.get('message', '')}",
                f"    Fallback : {err.get('fallback_attempted', False)}",
                f"    Time     : {err.get('timestamp', 'unknown')}",
                "",
            ])

        lines.append("=" * 70)
        return "\n".join(lines)

    def generate_html_report(self, report: PipelineReport) -> str:
        """Generate an HTML report for web viewing."""
        status_color = {
            "PASS": "#28a745",
            "WARN": "#ffc107",
            "FAIL": "#dc3545",
        }.get(report.overall_status, "#6c757d")

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Pipeline Report - {report.pipeline_id}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         max-width: 900px; margin: 40px auto; padding: 0 20px;
         background: #f8f9fa; color: #333; }}
  h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
  h2 {{ color: #34495e; margin-top: 30px; }}
  .card {{ background: white; border-radius: 8px; padding: 20px; margin: 15px 0;
           box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
  .status-badge {{ display: inline-block; padding: 4px 12px; border-radius: 20px;
                   color: white; font-weight: bold; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
           gap: 15px; margin: 15px 0; }}
  .grid-item {{ background: #f8f9fa; padding: 15px; border-radius: 8px;
                border-left: 4px solid #3498db; }}
  .grid-item label {{ font-size: 0.85em; color: #7f8c8d; display: block; margin-bottom: 5px; }}
  .grid-item span {{ font-size: 1.1em; font-weight: bold; }}
  table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
  th, td {{ padding: 10px 15px; text-align: left; border-bottom: 1px solid #eee; }}
  th {{ background: #3498db; color: white; }}
  .recommendation {{ padding: 10px 15px; margin: 8px 0; background: #e8f5e9;
                     border-left: 4px solid #28a745; border-radius: 4px; }}
  .error-item {{ padding: 10px 15px; margin: 8px 0; background: #fce4ec;
                 border-left: 4px solid #dc3545; border-radius: 4px; }}
</style>
</head>
<body>
<h1>Synthetic Biology Pipeline Report</h1>

<div class="card">
  <div class="grid">
    <div class="grid-item">
      <label>Pipeline ID</label>
      <span>{report.pipeline_id}</span>
    </div>
    <div class="grid-item">
      <label>Organism</label>
      <span>{report.organism} ({report.strain})</span>
    </div>
    <div class="grid-item">
      <label>Target Molecule</label>
      <span>{report.target_molecule}</span>
    </div>
    <div class="grid-item">
      <label>Overall Status</label>
      <span class="status-badge" style="background:{status_color}">
        {report.overall_status}
      </span>
    </div>
    <div class="grid-item">
      <label>Duration</label>
      <span>{report.total_duration_seconds:.2f}s</span>
    </div>
    <div class="grid-item">
      <label>Timestamp</label>
      <span>{report.run_timestamp}</span>
    </div>
  </div>
</div>

<h2>Stage Results</h2>
<div class="card">
  <table>
    <tr><th>Stage</th><th>Status</th></tr>
"""
        for stage_name, status_val in report.stage_summaries.items():
            if isinstance(status_val, dict):
                status = status_val.get("status", "UNKNOWN")
            else:
                status = str(status_val)
            # status_color_map is the dict; status_color is the overall color string
            status_color_map = {"PASS": "#28a745", "WARN": "#ffc107", "FAIL": "#dc3545"}
            color = status_color_map.get(status, "#6c757d")
            html += (
                f"    <tr><td>{stage_name}</td>"
                f'<td><span class="status-badge" style="background:{color}">'
                f"{status}</span></td></tr>\n"
            )

        html += """  </table>
</div>

<h2>Recommendations</h2>
<div class="card">
"""
        for rec in report.recommendations:
            html += f'  <div class="recommendation">{rec}</div>\n'

        html += "</div>\n"

        if report.errors:
            html += f"""
<h2>Errors ({len(report.errors)})</h2>
<div class="card">
"""
            for err in report.errors:
                html += (
                    f'  <div class="error-item">'
                    f"<strong>[{err.get('stage', '?')}] {err.get('function', '?')}:</strong> "
                    f"{err.get('message', '')}</div>\n"
                )
            html += "</div>\n"

        html += f"""
<div style="text-align:center; color:#999; margin-top:30px;">
  Generated by Synthetic Biology Metabolic Pathway Pipeline<br>
  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}
</div>
</body>
</html>
"""
        return html

    def export_to_files(self, report: PipelineReport,
                         output_dir: str = "./pipeline_output") -> Dict[str, str]:
        """
        Export report to multiple file formats.

        Returns
        -------
        dict
            Maps format → file path.
        """
        os.makedirs(output_dir, exist_ok=True)

        paths: Dict[str, str] = {}

        # JSON report
        json_path = os.path.join(output_dir, "final_report.json")
        with open(json_path, "w", encoding="utf-8") as fh:
            fh.write(self.generate_json_report(report))
        paths["json"] = json_path

        # Text summary
        txt_path = os.path.join(output_dir, "final_report.txt")
        with open(txt_path, "w", encoding="utf-8") as fh:
            fh.write(self.generate_summary_report(report))
        paths["text"] = txt_path

        # HTML report
        html_path = os.path.join(output_dir, "final_report.html")
        with open(html_path, "w", encoding="utf-8") as fh:
            fh.write(self.generate_html_report(report))
        paths["html"] = html_path

        # Error report
        err_path = os.path.join(output_dir, "error_report.txt")
        with open(err_path, "w", encoding="utf-8") as fh:
            fh.write(self.generate_error_report(report))
        paths["errors"] = err_path

        if self._logger:
            self._logger.info("Reports exported to: %s", output_dir)
            for fmt, path in paths.items():
                self._logger.debug("  %s: %s", fmt, path)

        return paths

    def _generate_recommendations(
        self,
        stage_outputs: Dict[str, Any],
        stage_summaries: Dict[str, Any],
        errors: List[Dict[str, Any]],
    ) -> List[str]:
        """Generate actionable recommendations based on pipeline results."""
        recommendations: List[str] = []

        # Check each stage
        stage_3 = stage_outputs.get("stage_3", {})
        if stage_3:
            tox = stage_3.get("toxicity_assessment", {})
            if tox.get("overall_toxicity_risk") in ("MEDIUM", "HIGH"):
                recommendations.append(
                    f"Toxicity risk is {tox['overall_toxicity_risk']}. "
                    "Consider toxicity mitigation strategies."
                )
            strain = stage_3.get("strain_design", {})
            if strain.get("metabolic_burden_score", 0) > 0.7:
                recommendations.append(
                    "High metabolic burden detected. Consider reducing "
                    "heterologous gene expression levels."
                )
            if strain.get("genetic_stability_score", 1.0) < 0.7:
                recommendations.append(
                    "Low genetic stability predicted. Review modification strategy."
                )

        stage_4 = stage_outputs.get("stage_4", {})
        if stage_4:
            ferm = stage_4.get("fermentation_simulation", {})
            if not ferm.get("ode_convergence", True):
                recommendations.append(
                    "Fermentation simulation did not converge. "
                    "Review kinetic parameters."
                )

        stage_5 = stage_outputs.get("stage_5", {})
        if stage_5:
            reg = stage_5.get("regulatory_assessment", {})
            if isinstance(reg, dict):
                if reg.get("overall_assessment") == "REQUIRES_REVIEW":
                    recommendations.append(
                        "Regulatory review required before scale-up."
                    )
                if not reg.get("antibiotic_markers_removed", True):
                    recommendations.append(
                        "Remove antibiotic markers before industrial use."
                    )

        # Error-based recommendations
        if errors:
            recommendations.append(
                f"{len(errors)} errors encountered during pipeline execution. "
                "Review error_report.txt for details."
            )

        if not recommendations:
            recommendations.append(
                "Pipeline completed successfully. Proceed with experimental validation."
            )

        return recommendations


# ---------------------------------------------------------------------------
# MAIN — smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger = PipelineLogger()
    logger.set_stage("5")

    generator = ReportGenerator()
    generator.set_logger(logger)

    # Create a mock report
    report = PipelineReport(
        pipeline_id="test-uuid-12345",
        run_timestamp=datetime.now(timezone.utc).isoformat(),
        organism="Escherichia coli",
        strain="K-12 MG1655",
        target_molecule="Lycopene",
        stage_outputs={"stage_1": {"stage_1_status": "PASS"}},
        stage_summaries={"stage_1": "PASS"},
        errors=[],
        overall_status="PASS",
        total_duration_seconds=1.5,
        recommendations=["Test recommendation"],
    )

    # Generate reports
    print(generator.generate_summary_report(report))

    paths = generator.export_to_files(report, "pipeline_output")
    print(f"\nReports exported:")
    for fmt, path in paths.items():
        print(f"  {fmt}: {path}")

    print("\n▶ Report Generator smoke test passed.")
