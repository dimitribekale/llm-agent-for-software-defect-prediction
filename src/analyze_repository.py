"""
Repository analysis script for commit-based defect prediction.

This script analyzes Git repositories to predict which commits introduce defects.
"""

import argparse
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import create_repository_analyzer


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Git repository for defective commits",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze last 100 commits
  python analyze_repository.py /path/to/repo

  # Analyze last 50 commits on specific branch
  python analyze_repository.py /path/to/repo --max-commits 50 --branch main

  # Filter by Python files only
  python analyze_repository.py /path/to/repo --extensions .py

  # Save results
  python analyze_repository.py /path/to/repo --output results.json

  # Generate HTML report
  python analyze_repository.py /path/to/repo --output report.html --format html

  # Use different model
  python analyze_repository.py /path/to/repo --model deepseek-r1:7b
"""
    )

    parser.add_argument(
        "repo_path",
        type=str,
        help="Path to Git repository"
    )
    parser.add_argument(
        "--max-commits",
        type=int,
        default=100,
        help="Maximum number of commits to analyze (default: 100)"
    )
    parser.add_argument(
        "--branch",
        type=str,
        default=None,
        help="Branch to analyze (default: current branch)"
    )
    parser.add_argument(
        "--extensions",
        type=str,
        nargs="+",
        default=None,
        help="Filter by file extensions (e.g., .py .java .cpp)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="codegemma:7b",
        help="LLM model to use (default: codegemma:7b)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: print to console)"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["json", "csv", "html"],
        default="json",
        help="Output format (default: json)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )

    args = parser.parse_args()

    # Validate repository path
    repo_path = Path(args.repo_path)
    if not repo_path.exists():
        print(f"Error: Repository path does not exist: {args.repo_path}", file=sys.stderr)
        return 1

    if not (repo_path / ".git").exists():
        print(f"Error: Not a Git repository: {args.repo_path}", file=sys.stderr)
        return 1

    # Create analyzer
    print(f"Initializing analyzer with model: {args.model}")
    analyzer = create_repository_analyzer(
        model_name=args.model,
        verbose=not args.quiet
    )

    # Analyze repository
    try:
        results = analyzer.analyze_repository(
            repo_path=str(repo_path),
            max_commits=args.max_commits,
            branch=args.branch,
            file_extensions=args.extensions
        )

        if not results:
            print("No commits found matching criteria")
            return 0

        # Save or print results
        if args.output:
            analyzer.save_results(results, args.output, format=args.format)
            print(f"\n✓ Results saved to: {args.output}")
        else:
            # Print summary to console
            print_summary(results)

        # Print statistics
        stats = analyzer.get_statistics(results)
        print_statistics(stats)

        return 0

    except Exception as e:
        print(f"Error during analysis: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


def print_summary(results):
    """Print results summary to console."""
    print(f"\n{'='*70}")
    print("Defective Commits Summary")
    print(f"{'='*70}")

    defective_commits = [r for r in results if r.is_defective]

    if defective_commits:
        for result in defective_commits:
            print(f"\n{result.commit_hash[:8]} - {result.commit_data.author}")
            print(f"  Date: {result.commit_data.date.strftime('%Y-%m-%d %H:%M')}")
            print(f"  Message: {result.commit_data.message[:80]}")
            print(f"  Intent: {result.intent.intent_type} (Risk: {result.intent.risk_level})")
            print(f"  Confidence: {result.prediction.confidence:.2f}")
            print(f"  Reason: {result.prediction.explanation[:100]}")
    else:
        print("\nNo defective commits found!")


def print_statistics(stats):
    """Print statistics."""
    print(f"\n{'='*70}")
    print("Analysis Statistics")
    print(f"{'='*70}")
    print(f"Total Commits: {stats['total_commits']}")
    print(f"Defective: {stats['defective_commits']} ({stats['defect_rate']:.1%})")
    print(f"Clean: {stats['clean_commits']} ({1-stats['defect_rate']:.1%})")
    print(f"Average Confidence: {stats['average_confidence']:.2f}")

    print(f"\nIntent Type Distribution:")
    for intent_type, data in stats['intent_distribution'].items():
        defect_rate = data['defective'] / data['total'] if data['total'] > 0 else 0
        print(f"  {intent_type:15s}: {data['total']:3d} commits ({defect_rate:.1%} defective)")

    print(f"\nRisk Level Distribution:")
    for risk_level, data in stats['risk_distribution'].items():
        defect_rate = data['defective'] / data['total'] if data['total'] > 0 else 0
        print(f"  {risk_level:15s}: {data['total']:3d} commits ({defect_rate:.1%} defective)")


if __name__ == "__main__":
    sys.exit(main())
