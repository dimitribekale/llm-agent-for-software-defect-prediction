"""
Repository analysis examples.

Demonstrates how to use the commit-based defect prediction system.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from agent import create_repository_analyzer


def example_1_basic_analysis():
    """Example 1: Basic repository analysis."""
    print("\n" + "="*70)
    print("Example 1: Basic Repository Analysis")
    print("="*70 + "\n")

    # NOTE: Replace with path to an actual Git repository
    repo_path = "/Users/dang-geun/Documents/Documents/GitHub/ansible"

    # Check if path exists
    if not os.path.exists(repo_path):
        print(f"⚠️  Please update repo_path to point to a real Git repository")
        print(f"Current path: {repo_path}")
        return

    # Create analyzer
    analyzer = create_repository_analyzer(
        model_name="codegemma:7b",
        verbose=True
    )

    # Analyze repository (last 10 commits for quick demo)
    results = analyzer.analyze_repository(
        repo_path=repo_path,
        max_commits=10
    )

    # Print results
    print(f"\n{'='*70}")
    print("Results:")
    print(f"{'='*70}")
    for result in results:
        status = "DEFECTIVE" if result.is_defective else "CLEAN"
        print(f"\n{result.commit_hash[:8]} - {status}")
        print(f"  Message: {result.commit_data.message[:60]}")
        print(f"  Confidence: {result.prediction.confidence:.2f}")


def example_2_filter_by_language():
    """Example 2: Filter by file extensions."""
    print("\n" + "="*70)
    print("Example 2: Filter by Language (Python files only)")
    print("="*70 + "\n")

    repo_path = "/Users/dang-geun/Documents/Documents/GitHub/ansible"

    if not os.path.exists(repo_path):
        print(f"⚠️  Please update repo_path")
        return

    analyzer = create_repository_analyzer(verbose=False)

    # Analyze only commits that changed Python files
    results = analyzer.analyze_repository(
        repo_path=repo_path,
        max_commits=20,
        file_extensions=['.py']
    )

    # Count by intent type
    intent_counts = {}
    for result in results:
        intent = result.intent.intent_type
        if intent not in intent_counts:
            intent_counts[intent] = {"total": 0, "defective": 0}
        intent_counts[intent]["total"] += 1
        if result.is_defective:
            intent_counts[intent]["defective"] += 1

    print("\nIntent Type Analysis:")
    for intent, counts in intent_counts.items():
        defect_rate = counts["defective"] / counts["total"] if counts["total"] > 0 else 0
        print(f"  {intent:12s}: {counts['total']:2d} commits, {counts['defective']:2d} defective ({defect_rate:.1%})")


def example_3_save_results():
    """Example 3: Save results to files."""
    print("\n" + "="*70)
    print("Example 3: Save Results to Multiple Formats")
    print("="*70 + "\n")

    repo_path = "/Users/dang-geun/Documents/Documents/GitHub/ansible"

    if not os.path.exists(repo_path):
        print(f"⚠️  Please update repo_path")
        return

    analyzer = create_repository_analyzer(verbose=True)

    results = analyzer.analyze_repository(
        repo_path=repo_path,
        max_commits=15
    )

    # Save in different formats
    analyzer.save_results(results, "results.json", format="json")
    print("✓ Saved to results.json")

    analyzer.save_results(results, "results.csv", format="csv")
    print("✓ Saved to results.csv")

    analyzer.save_results(results, "report.html", format="html")
    print("✓ Saved to report.html (open in browser to view)")


def example_4_statistics():
    """Example 4: Get detailed statistics."""
    print("\n" + "="*70)
    print("Example 4: Detailed Statistics")
    print("="*70 + "\n")

    repo_path = "/Users/dang-geun/Documents/Documents/GitHub/ansible"

    if not os.path.exists(repo_path):
        print(f"⚠️  Please update repo_path")
        return

    analyzer = create_repository_analyzer(verbose=False)

    results = analyzer.analyze_repository(
        repo_path=repo_path,
        max_commits=50
    )

    # Get statistics
    stats = analyzer.get_statistics(results)

    print(f"Total Commits: {stats['total_commits']}")
    print(f"Defective: {stats['defective_commits']} ({stats['defect_rate']:.1%})")
    print(f"Average Confidence: {stats['average_confidence']:.2f}")

    print(f"\nRisk Level Distribution:")
    for risk_level, data in stats['risk_distribution'].items():
        defect_rate = data['defective'] / data['total'] if data['total'] > 0 else 0
        print(f"  {risk_level:8s}: {data['total']:3d} commits ({defect_rate:.0%} defective)")


def example_5_specific_branch():
    """Example 5: Analyze specific branch."""
    print("\n" + "="*70)
    print("Example 5: Analyze Specific Branch")
    print("="*70 + "\n")

    repo_path = "/Users/dang-geun/Documents/Documents/GitHub/ansible"

    if not os.path.exists(repo_path):
        print(f"⚠️  Please update repo_path")
        return

    analyzer = create_repository_analyzer(verbose=True)

    # Analyze specific branch
    results = analyzer.analyze_repository(
        repo_path=repo_path,
        max_commits=25,
        branch="main"  # or "develop", "feature/xyz", etc.
    )

    print(f"\nAnalyzed {len(results)} commits from 'main' branch")


def example_6_custom_model():
    """Example 6: Use different LLM model."""
    print("\n" + "="*70)
    print("Example 6: Custom LLM Model")
    print("="*70 + "\n")

    repo_path = "/Users/dang-geun/Documents/Documents/GitHub/ansible"

    if not os.path.exists(repo_path):
        print(f"⚠️  Please update repo_path")
        return

    # Use a different model
    analyzer = create_repository_analyzer(
        model_name="deepseek-r1:7b",  # or "codellama:7b"
        verbose=True
    )

    results = analyzer.analyze_repository(
        repo_path=repo_path,
        max_commits=10
    )

    print(f"\nAnalyzed {len(results)} commits with deepseek-r1:7b model")
    defective_count = sum(1 for r in results if r.is_defective)
    print(f"Found {defective_count} potentially defective commits")


def example_7_detailed_output():
    """Example 7: Detailed output showing all analysis steps."""
    print("\n" + "="*70)
    print("Example 7: Detailed Output - All Analysis Steps")
    print("="*70 + "\n")

    repo_path = "/Users/dang-geun/Documents/Documents/GitHub/ansible"

    if not os.path.exists(repo_path):
        print(f"⚠️  Please update repo_path to point to a real Git repository")
        print(f"Current path: {repo_path}")
        return

    # Create analyzer with tool support
    analyzer = create_repository_analyzer(
        model_name="codegemma:7b",
        enable_web_search=True,      # Enable web search for verification
        enable_doc_search=True,       # Enable documentation search
        verbose=True  # Show progress bars and tool usage during analysis
    )

    # Analyze repository (just 5 commits for detailed view)
    print("Analyzing last 5 commits...\n")
    results = analyzer.analyze_repository(
        repo_path=repo_path,
        max_commits=5
    )

    # Print detailed results for each commit
    for i, result in enumerate(results, 1):
        print(f"\n{'='*70}")
        print(f"COMMIT {i}/{len(results)}")
        print(f"{'='*70}")

        # Commit metadata
        print(f"\n📋 COMMIT INFORMATION:")
        print(f"  Hash:    {result.commit_hash}")
        print(f"  Author:  {result.commit_data.author}")
        print(f"  Date:    {result.commit_data.date.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Message: {result.commit_data.message}")

        # Files changed
        print(f"\n📁 FILES CHANGED ({len(result.commit_data.changed_files)}):")
        for file in result.commit_data.changed_files[:10]:  # Show first 10 files
            print(f"  - {file}")
        if len(result.commit_data.changed_files) > 10:
            print(f"  ... and {len(result.commit_data.changed_files) - 10} more files")

        # Statistics
        print(f"\n📊 STATISTICS:")
        print(f"  Lines added:   {result.commit_data.stats['total_insertions']:>5}")
        print(f"  Lines removed: {result.commit_data.stats['total_deletions']:>5}")
        print(f"  Net change:    {result.commit_data.stats['net_lines']:>5}")
        print(f"  Files changed: {result.commit_data.stats['files_changed']:>5}")

        # Code changes - show sample
        print(f"\n🔍 CODE CHANGES (sample):")
        print(f"  Added lines ({len(result.commit_data.added_lines)} total):")
        for line in result.commit_data.added_lines[:5]:  # Show first 5 added lines
            print(f"    + {line[:80]}")  # Truncate long lines
        if len(result.commit_data.added_lines) > 5:
            print(f"    ... and {len(result.commit_data.added_lines) - 5} more added lines")

        print(f"\n  Removed lines ({len(result.commit_data.removed_lines)} total):")
        for line in result.commit_data.removed_lines[:5]:  # Show first 5 removed lines
            print(f"    - {line[:80]}")  # Truncate long lines
        if len(result.commit_data.removed_lines) > 5:
            print(f"    ... and {len(result.commit_data.removed_lines) - 5} more removed lines")

        # Developer intent analysis
        print(f"\n🎯 DEVELOPER INTENT ANALYSIS:")
        print(f"  Intent Type:  {result.intent.intent_type}")
        print(f"  Risk Level:   {result.intent.risk_level}")
        print(f"  Confidence:   {result.intent.confidence:.2f}")
        print(f"  Description:  {result.intent.description}")

        # Defect prediction
        print(f"\n🔮 DEFECT PREDICTION:")
        status = "⚠️  DEFECTIVE" if result.is_defective else "✅ CLEAN"
        print(f"  Status:       {status}")
        print(f"  Confidence:   {result.prediction.confidence:.2f}")
        print(f"  Explanation:  {result.prediction.explanation}")

        print(f"\n{'-'*70}")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    stats = analyzer.get_statistics(results)
    print(f"\nTotal Commits Analyzed: {stats['total_commits']}")
    print(f"Defective Commits:      {stats['defective_commits']} ({stats['defect_rate']:.1%})")
    print(f"Clean Commits:          {stats['clean_commits']} ({1-stats['defect_rate']:.1%})")
    print(f"Average Confidence:     {stats['average_confidence']:.2f}")

    print(f"\nIntent Type Distribution:")
    for intent_type, data in stats['intent_distribution'].items():
        defect_rate = data['defective'] / data['total'] if data['total'] > 0 else 0
        print(f"  {intent_type:15s}: {data['total']:2d} commits ({defect_rate:5.1%} defective)")

    print(f"\nRisk Level Distribution:")
    for risk_level, data in stats['risk_distribution'].items():
        defect_rate = data['defective'] / data['total'] if data['total'] > 0 else 0
        print(f"  {risk_level:15s}: {data['total']:2d} commits ({defect_rate:5.1%} defective)")


def main():
    """Run examples."""
    print("\n" + "="*70)
    print("Repository Analysis Examples")
    print("="*70)

    print("\nAvailable examples:")
    print("  1. Basic repository analysis")
    print("  2. Filter by language (Python only)")
    print("  3. Save results to multiple formats")
    print("  4. Detailed statistics")
    print("  5. Analyze specific branch")
    print("  6. Use custom LLM model")
    print("  7. Detailed output (all analysis steps) ⭐ RECOMMENDED")

    print("\n⚠️  Note: These examples require a Git repository path.")
    print("   Please edit the examples and update 'repo_path' before running.\n")

    # Uncomment to run specific examples:
    # example_1_basic_analysis()
    # example_2_filter_by_language()
    # example_3_save_results()
    # example_4_statistics()
    # example_5_specific_branch()
    # example_6_custom_model()
    example_7_detailed_output()  # Shows all analysis details

    print("\n" + "="*70)
    print("Edit this file and uncomment examples to run them")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
