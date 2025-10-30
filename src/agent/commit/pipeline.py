"""
Batch processing pipeline for repository analysis.
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict, Any
from tqdm import tqdm
from datetime import datetime

from ..config import SystemConfig
from ..tools.implementations import GitRepositoryTool, CommitData
from .orchestrator import CommitDefectOrchestrator, CommitDefectResult


class RepositoryAnalysisPipeline:
    """
    Pipeline for analyzing Git repositories for defects.

    Extracts commits, analyzes developer intent, and predicts defects.
    """

    def __init__(
        self,
        config: Optional[SystemConfig] = None,
        verbose: bool = True
    ):
        """
        Initialize pipeline.

        Args:
            config: System configuration
            verbose: Print progress information
        """
        self.config = config or SystemConfig.default()
        self.verbose = verbose

        # Initialize components
        self.git_tool = GitRepositoryTool()
        self.orchestrator = CommitDefectOrchestrator(config, verbose=False)  # Suppress per-commit verbosity

    def analyze_repository(
        self,
        repo_path: str,
        max_commits: int = 100,
        branch: Optional[str] = None,
        file_extensions: Optional[List[str]] = None
    ) -> List[CommitDefectResult]:
        """
        Analyze a Git repository for defects.

        Args:
            repo_path: Path to cloned repository
            max_commits: Maximum commits to analyze
            branch: Branch to analyze (default: current)
            file_extensions: Filter by file extensions (e.g., ['.py'])

        Returns:
            List of CommitDefectResult objects
        """
        if self.verbose:
            print(f"\n{'='*70}")
            print("Repository Defect Analysis Pipeline")
            print(f"{'='*70}")
            print(f"Repository: {repo_path}")
            print(f"Max commits: {max_commits}")
            if branch:
                print(f"Branch: {branch}")
            if file_extensions:
                print(f"File extensions: {file_extensions}")
            print(f"{'='*70}\n")

        # Step 1: Extract commits
        if self.verbose:
            print("[Step 1/2] Extracting commits from repository...")

        commits = self.git_tool.execute(
            repo_path=repo_path,
            max_commits=max_commits,
            branch=branch,
            file_extensions=file_extensions
        )

        if self.verbose:
            print(f"  Extracted {len(commits)} commits\n")

        if not commits:
            print("No commits found matching criteria")
            return []

        # Step 2: Analyze each commit
        if self.verbose:
            print("[Step 2/2] Analyzing commits for defects...")

        results = []
        for commit in tqdm(commits, desc="Analyzing commits", disable=not self.verbose):
            result = self.orchestrator.predict_commit(commit)
            results.append(result)

        # Summary
        if self.verbose:
            defective_count = sum(1 for r in results if r.is_defective)
            print(f"\n{'='*70}")
            print("Analysis Complete")
            print(f"{'='*70}")
            print(f"Total commits analyzed: {len(results)}")
            print(f"Defective commits: {defective_count} ({defective_count/len(results)*100:.1f}%)")
            print(f"Clean commits: {len(results) - defective_count} ({(len(results)-defective_count)/len(results)*100:.1f}%)")
            print(f"{'='*70}\n")

        return results

    def save_results(
        self,
        results: List[CommitDefectResult],
        output_path: str,
        format: str = "json"
    ):
        """
        Save analysis results to file.

        Args:
            results: Analysis results
            output_path: Output file path
            format: Output format ('json', 'csv', or 'html')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            self._save_json(results, output_path)
        elif format == "csv":
            self._save_csv(results, output_path)
        elif format == "html":
            self._save_html(results, output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")

        if self.verbose:
            print(f"Results saved to: {output_path}")

    def _save_json(self, results: List[CommitDefectResult], output_path: Path):
        """Save results as JSON."""
        data = {
            "timestamp": datetime.now().isoformat(),
            "total_commits": len(results),
            "defective_commits": sum(1 for r in results if r.is_defective),
            "results": [r.to_dict() for r in results]
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _save_csv(self, results: List[CommitDefectResult], output_path: Path):
        """Save results as CSV."""
        data = []
        for result in results:
            data.append({
                "commit_hash": result.commit_hash,
                "author": result.commit_data.author,
                "date": result.commit_data.date.isoformat(),
                "message": result.commit_data.message,
                "files_changed": len(result.commit_data.changed_files),
                "lines_added": result.commit_data.stats["total_insertions"],
                "lines_removed": result.commit_data.stats["total_deletions"],
                "intent_type": result.intent.intent_type,
                "risk_level": result.intent.risk_level,
                "prediction": result.prediction.prediction,
                "confidence": result.prediction.confidence,
                "explanation": result.prediction.explanation,
                "defect_types": ', '.join(result.prediction.defect_types) if result.prediction.defect_types else ""
            })

        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)

    def _save_html(self, results: List[CommitDefectResult], output_path: Path):
        """Save results as HTML report."""
        html = self._generate_html_report(results)

        with open(output_path, 'w') as f:
            f.write(html)

    def _generate_html_report(self, results: List[CommitDefectResult]) -> str:
        """Generate HTML report."""
        defective_count = sum(1 for r in results if r.is_defective)
        clean_count = len(results) - defective_count

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Repository Defect Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        .summary {{ background: #f0f0f0; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .commit {{ border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 5px; }}
        .defective {{ border-left: 4px solid #f44336; }}
        .clean {{ border-left: 4px solid #4CAF50; }}
        .commit-hash {{ font-family: monospace; color: #666; }}
        .prediction {{ font-weight: bold; }}
        .defective-pred {{ color: #f44336; }}
        .clean-pred {{ color: #4CAF50; }}
    </style>
</head>
<body>
    <h1>Repository Defect Analysis Report</h1>
    <div class="summary">
        <h2>Summary</h2>
        <p><strong>Total Commits:</strong> {len(results)}</p>
        <p><strong>Defective:</strong> {defective_count} ({defective_count/len(results)*100:.1f}%)</p>
        <p><strong>Clean:</strong> {clean_count} ({clean_count/len(results)*100:.1f}%)</p>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>

    <h2>Commits</h2>
"""

        for result in results:
            status_class = "defective" if result.is_defective else "clean"
            pred_class = "defective-pred" if result.is_defective else "clean-pred"
            status_text = "DEFECTIVE" if result.is_defective else "CLEAN"

            html += f"""
    <div class="commit {status_class}">
        <p><span class="commit-hash">{result.commit_hash[:8]}</span> - <strong>{result.commit_data.author}</strong> - {result.commit_data.date.strftime('%Y-%m-%d')}</p>
        <p><strong>Message:</strong> {result.commit_data.message}</p>
        <p><strong>Files Changed:</strong> {len(result.commit_data.changed_files)}</p>
        <p><strong>Intent:</strong> {result.intent.intent_type} (Risk: {result.intent.risk_level})</p>
        <p class="prediction {pred_class}">Prediction: {status_text} (Confidence: {result.prediction.confidence:.2f})</p>
        <p><strong>Explanation:</strong> {result.prediction.explanation}</p>
    </div>
"""

        html += """
</body>
</html>
"""
        return html

    def get_statistics(self, results: List[CommitDefectResult]) -> Dict[str, Any]:
        """
        Get statistics from analysis results.

        Args:
            results: Analysis results

        Returns:
            Dictionary with statistics
        """
        defective_commits = [r for r in results if r.is_defective]
        clean_commits = [r for r in results if not r.is_defective]

        # Intent type distribution
        intent_types = {}
        for result in results:
            intent_type = result.intent.intent_type
            if intent_type not in intent_types:
                intent_types[intent_type] = {"total": 0, "defective": 0}
            intent_types[intent_type]["total"] += 1
            if result.is_defective:
                intent_types[intent_type]["defective"] += 1

        # Risk level distribution
        risk_levels = {}
        for result in results:
            risk_level = result.intent.risk_level
            if risk_level not in risk_levels:
                risk_levels[risk_level] = {"total": 0, "defective": 0}
            risk_levels[risk_level]["total"] += 1
            if result.is_defective:
                risk_levels[risk_level]["defective"] += 1

        return {
            "total_commits": len(results),
            "defective_commits": len(defective_commits),
            "clean_commits": len(clean_commits),
            "defect_rate": len(defective_commits) / len(results) if results else 0,
            "average_confidence": sum(r.prediction.confidence for r in results) / len(results) if results else 0,
            "intent_distribution": intent_types,
            "risk_distribution": risk_levels
        }
