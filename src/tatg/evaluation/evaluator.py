from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from tatg.evaluation.coverage import CoverageStats, parse_jacoco_csv, parse_pytest_coverage_json
from tatg.evaluation.mutation import MutationStats, parse_pitest_json, parse_pitest_log, parse_pitest_xml
from tatg.build_system import BuildSystem, detect_build_system


@dataclass(slots=True)
class EvaluationReport:
    test_count: int = 0
    coverage: CoverageStats | None = None
    mutation: MutationStats | None = None
    metadata: dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_count": self.test_count,
            "coverage": self.coverage.to_dict() if self.coverage else None,
            "mutation": self.mutation.to_dict() if self.mutation else None,
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=True)


class Evaluator:
    def __init__(self):
        pass

    def _run_command(self, command: str, cwd: Path) -> tuple[int, str, str]:
        proc = subprocess.run(
            command,
            cwd=str(cwd),
            shell=True,
            text=True,
            capture_output=True,
            check=False,
        )
        return proc.returncode, proc.stdout, proc.stderr

    def _run_native_coverage(self, project_root: Path, build_system: BuildSystem, coverage_type: str) -> dict[str, Any]:
        if coverage_type == "pytest" or build_system == BuildSystem.PYTHON:
            rc, out, err = self._run_command("pytest -q --cov=. --cov-report=json", project_root)
            return {"rc": rc, "stdout": out, "stderr": err}

        if build_system == BuildSystem.MAVEN:
            rc, out, err = self._run_command("mvn -q -DskipTests=false test jacoco:report", project_root)
            return {"rc": rc, "stdout": out, "stderr": err}

        if build_system == BuildSystem.GRADLE:
            rc, out, err = self._run_command("gradle test jacocoTestReport", project_root)
            return {"rc": rc, "stdout": out, "stderr": err}

        return {"rc": 1, "stdout": "", "stderr": f"Unsupported build system for coverage: {build_system.value}"}

    def _run_native_mutation(self, project_root: Path, build_system: BuildSystem) -> dict[str, Any]:
        if build_system == BuildSystem.MAVEN:
            rc, out, err = self._run_command("mvn -q -DskipTests=false org.pitest:pitest-maven:mutationCoverage", project_root)
            return {"rc": rc, "stdout": out, "stderr": err}

        if build_system == BuildSystem.GRADLE:
            rc, out, err = self._run_command("gradle pitest", project_root)
            return {"rc": rc, "stdout": out, "stderr": err}

        return {"rc": 1, "stdout": "", "stderr": f"Unsupported build system for mutation: {build_system.value}"}

    def evaluate_coverage(
        self,
        project_root: Path,
        coverage_report_path: str | Path | None = None,
        coverage_type: str = "jacoco",
        run_if_missing: bool = False,
    ) -> CoverageStats:
        """
        Evaluate test coverage by parsing coverage reports.
        
        Args:
            project_root: Root of the project
            coverage_report_path: Path to coverage report (auto-inferred if None)
            coverage_type: Type of coverage report ("jacoco" or "pytest")
        
        Returns:
            CoverageStats object
        """
        if coverage_report_path is None:
            if coverage_type == "jacoco":
                coverage_report_path = project_root / "target" / "jacoco" / "jacoco.csv"
            elif coverage_type == "pytest":
                coverage_report_path = project_root / ".coverage.json"
            else:
                coverage_report_path = project_root / "coverage.json"

        coverage_report_path = Path(coverage_report_path)

        if coverage_type == "jacoco":
            csv_path = Path(coverage_report_path)
            if not csv_path.exists() and run_if_missing:
                build_system = detect_build_system(project_root)
                self._run_native_coverage(project_root, build_system, coverage_type)
                candidates = [
                    project_root / "target" / "site" / "jacoco" / "jacoco.csv",
                    project_root / "target" / "jacoco" / "jacoco.csv",
                ]
                for c in candidates:
                    if c.exists():
                        csv_path = c
                        break
            return parse_jacoco_csv(csv_path)
        elif coverage_type == "pytest":
            json_path = Path(coverage_report_path)
            if not json_path.exists() and run_if_missing:
                build_system = detect_build_system(project_root)
                self._run_native_coverage(project_root, build_system, coverage_type)
                candidates = [
                    project_root / "coverage.json",
                    project_root / ".coverage.json",
                ]
                for c in candidates:
                    if c.exists():
                        json_path = c
                        break
            return parse_pytest_coverage_json(json_path)
        else:
            return CoverageStats()

    def evaluate_mutation(
        self,
        project_root: Path,
        mutation_json_path: str | Path | None = None,
        mutation_log_path: str | Path | None = None,
        run_if_missing: bool = False,
    ) -> MutationStats:
        """
        Evaluate mutation testing results.
        
        Args:
            project_root: Root of the project
            mutation_json_path: Path to PITest mutations.json
            mutation_log_path: Path to PITest log file (fallback)
        
        Returns:
            MutationStats object
        """
        if mutation_json_path is None:
            mutation_json_path = project_root / "target" / "pit-reports" / "mutations.json"

        mutation_json_path = Path(mutation_json_path)

        if mutation_json_path.exists():
            return parse_pitest_json(mutation_json_path)

        xml_candidates = [
            project_root / "target" / "pit-reports" / "mutations.xml",
        ]
        for xml_path in xml_candidates:
            if xml_path.exists():
                return parse_pitest_xml(xml_path)

        if run_if_missing:
            build_system = detect_build_system(project_root)
            self._run_native_mutation(project_root, build_system)

            if mutation_json_path.exists():
                return parse_pitest_json(mutation_json_path)

            pit_reports = project_root / "target" / "pit-reports"
            if pit_reports.exists():
                xml_files = sorted(pit_reports.rglob("mutations.xml"))
                if xml_files:
                    return parse_pitest_xml(xml_files[-1])

        if mutation_log_path is None:
            mutation_log_path = project_root / "pit.log"

        if Path(mutation_log_path).exists():
            return parse_pitest_log(Path(mutation_log_path))

        return MutationStats()

    def evaluate(
        self,
        project_root: Path,
        coverage_report_path: str | Path | None = None,
        coverage_type: str = "jacoco",
        mutation_json_path: str | Path | None = None,
        mutation_log_path: str | Path | None = None,
        test_count: int = 0,
        run_coverage_if_missing: bool = False,
        run_mutation_if_missing: bool = False,
    ) -> EvaluationReport:
        """
        Comprehensive evaluation combining coverage and mutation metrics.
        """
        coverage = self.evaluate_coverage(
            project_root,
            coverage_report_path,
            coverage_type,
            run_if_missing=run_coverage_if_missing,
        )
        mutation = self.evaluate_mutation(
            project_root,
            mutation_json_path,
            mutation_log_path,
            run_if_missing=run_mutation_if_missing,
        )

        return EvaluationReport(
            test_count=test_count,
            coverage=coverage,
            mutation=mutation,
            metadata={
                "project_root": str(project_root),
                "coverage_type": coverage_type,
            },
        )


def write_evaluation_report(report: EvaluationReport, output_path: Path) -> None:
    """Write evaluation report to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report.to_json(), encoding="utf-8")
