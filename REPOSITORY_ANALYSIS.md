

# Repository-Based Defect Prediction

Analyze Git repositories to predict which commits introduce defects by examining code changes, commit messages, and developer intent.

## Overview

This feature extends the agent to analyze **actual code changes in context** rather than isolated snippets, making predictions more practical and actionable for real-world software development.

### Workflow

```
Git Repository
    ↓
[Step 1] Extract Commits
    → Get last N commits
    → Extract diffs (added/removed lines)
    → Parse commit metadata
    ↓
[Step 2] Analyze Developer Intent
    → LLM analyzes commit message
    → Extracts developer intent
    → Classifies change type
    ↓
[Step 3] Predict Defects
    → Analyze added code
    → Analyze removed code
    → Consider commit message
    → Consider developer intent
    ↓
Report: Defective commits with explanations
```

## Quick Start

### Installation

```bash
pip install GitPython  # Or use requirements.txt
```

### Basic Usage

```python
from agent import create_repository_analyzer

# Create analyzer
analyzer = create_repository_analyzer(
    model_name="codegemma:7b",
    verbose=True
)

# Analyze repository
results = analyzer.analyze_repository(
    repo_path="/path/to/your/repo",
    max_commits=100
)

# Check results
for result in results:
    if result.is_defective:
        print(f"Defective commit: {result.commit_hash[:8]}")
        print(f"Confidence: {result.prediction.confidence:.2f}")
        print(f"Reason: {result.prediction.explanation}")
```

### Command Line

```bash
# Analyze repository
python src/analyze_repository.py /path/to/repo

# Analyze last 50 commits
python src/analyze_repository.py /path/to/repo --max-commits 50

# Filter by Python files
python src/analyze_repository.py /path/to/repo --extensions .py

# Save results
python src/analyze_repository.py /path/to/repo --output results.json

# Generate HTML report
python src/analyze_repository.py /path/to/repo --output report.html --format html

# Use different model
python src/analyze_repository.py /path/to/repo --model deepseek-r1:7b
```

## Features

### 1. Git Integration

**GitRepositoryTool** extracts commits with full context:
- Commit hash, author, date
- Commit message
- Added lines (+)
- Removed lines (-)
- Changed files
- Statistics (insertions, deletions, net change)

```python
from agent.tools.implementations import GitRepositoryTool

git_tool = GitRepositoryTool()
commits = git_tool.execute(
    repo_path="/path/to/repo",
    max_commits=100,
    branch="main",
    file_extensions=['.py']  # Optional filter
)

for commit in commits:
    print(f"{commit.commit_hash[:8]}: {len(commit.added_lines)} additions")
```

### 2. Intent Analysis

**CommitIntentAnalyzer** uses LLM to understand developer intent:
- What was the developer trying to do?
- Change type (bugfix, feature, refactor, etc.)
- Risk level (low, medium, high)

```python
from agent.commit import CommitIntentAnalyzer
from agent.llm import LLMClientFactory, LLMConfig

config = LLMConfig()
client = LLMClientFactory.create("ollama", config)
analyzer = CommitIntentAnalyzer(client)

intent = analyzer.analyze("Fix null pointer exception in UserService")

print(f"Intent: {intent.intent_type}")  # "bugfix"
print(f"Risk: {intent.risk_level}")     # "high"
print(f"Description: {intent.description}")
```

### 3. Commit-Specific Defect Prediction

**CommitDefectOrchestrator** predicts defects considering:
- Code changes (added + removed)
- Commit message
- Developer intent
- Change patterns

```python
from agent.commit import CommitDefectOrchestrator

orchestrator = CommitDefectOrchestrator()
result = orchestrator.predict_commit(commit_data)

print(f"Defective: {result.is_defective}")
print(f"Confidence: {result.prediction.confidence}")
print(f"Explanation: {result.prediction.explanation}")
```

### 4. Batch Processing

**RepositoryAnalysisPipeline** processes multiple commits:

```python
from agent import create_repository_analyzer

analyzer = create_repository_analyzer()

# Analyze repository
results = analyzer.analyze_repository(
    repo_path="/path/to/repo",
    max_commits=100,
    branch="main",
    file_extensions=['.py', '.java']
)

# Save results
analyzer.save_results(results, "results.json", format="json")
analyzer.save_results(results, "results.csv", format="csv")
analyzer.save_results(results, "report.html", format="html")

# Get statistics
stats = analyzer.get_statistics(results)
print(f"Defect rate: {stats['defect_rate']:.1%}")
```

## Output Formats

### JSON

```json
{
  "timestamp": "2025-10-31T01:30:00",
  "total_commits": 100,
  "defective_commits": 15,
  "results": [
    {
      "commit_hash": "abc123...",
      "author": "John Doe",
      "date": "2025-10-30T10:00:00",
      "message": "Fix bug in authentication",
      "files_changed": ["src/auth.py", "tests/test_auth.py"],
      "stats": {
        "total_insertions": 45,
        "total_deletions": 12,
        "files_changed": 2
      },
      "intent": {
        "intent_type": "bugfix",
        "risk_level": "medium",
        "confidence": 0.85
      },
      "prediction": {
        "prediction": 1,
        "confidence": 0.78,
        "explanation": "Incomplete error handling in added code"
      }
    }
  ]
}
```

### CSV

| commit_hash | author | date | message | files_changed | lines_added | lines_removed | intent_type | risk_level | prediction | confidence | explanation |
|------------|--------|------|---------|--------------|-------------|---------------|-------------|-----------|------------|------------|-------------|
| abc123 | John Doe | 2025-10-30 | Fix auth bug | 2 | 45 | 12 | bugfix | medium | 1 | 0.78 | Incomplete error handling |

### HTML

Interactive HTML report with:
- Summary statistics
- Defect rate by intent type
- Defect rate by risk level
- Individual commit details
- Color-coded predictions

## Configuration

### Model Selection

```python
# Use different models
analyzer = create_repository_analyzer(
    model_name="codegemma:7b"      # Fast, good for code
    # model_name="deepseek-r1:7b"  # More accurate
    # model_name="codellama:7b"    # Alternative
)
```

### Filter Options

```python
results = analyzer.analyze_repository(
    repo_path="/path/to/repo",
    max_commits=100,              # Limit commits
    branch="main",                # Specific branch
    file_extensions=['.py', '.java']  # Filter by language
)
```

### Verbosity

```python
# Detailed output
analyzer = create_repository_analyzer(verbose=True)

# Quiet mode
analyzer = create_repository_analyzer(verbose=False)
```

## Use Cases

### 1. Pre-Release Review

Analyze recent commits before release:

```python
analyzer = create_repository_analyzer()

results = analyzer.analyze_repository(
    repo_path="/path/to/repo",
    max_commits=50,  # Last 50 commits
    branch="release/v2.0"
)

defective = [r for r in results if r.is_defective]
if defective:
    print(f"⚠️  Found {len(defective)} potentially defective commits")
    for commit in defective:
        print(f"  - {commit.commit_hash[:8]}: {commit.commit_data.message}")
```

### 2. Code Review Prioritization

Identify high-risk commits for review:

```python
results = analyzer.analyze_repository(repo_path="/path/to/repo")

# Sort by risk
high_risk = sorted(
    [r for r in results if r.intent.risk_level == "high"],
    key=lambda r: r.prediction.confidence,
    reverse=True
)

print("High-risk commits to review:")
for commit in high_risk[:10]:
    print(f"{commit.commit_hash[:8]} - {commit.commit_data.message}")
```

### 3. Developer Analysis

Analyze commits by developer:

```python
from collections import defaultdict

results = analyzer.analyze_repository(repo_path="/path/to/repo")

by_author = defaultdict(list)
for result in results:
    by_author[result.commit_data.author].append(result)

for author, commits in by_author.items():
    defective = sum(1 for c in commits if c.is_defective)
    rate = defective / len(commits) if commits else 0
    print(f"{author}: {defective}/{len(commits)} defective ({rate:.1%})")
```

### 4. Historical Analysis

Track defect rates over time:

```python
results = analyzer.analyze_repository(
    repo_path="/path/to/repo",
    max_commits=500
)

# Group by month
from collections import defaultdict
from datetime import datetime

by_month = defaultdict(lambda: {"total": 0, "defective": 0})

for result in results:
    month = result.commit_data.date.strftime("%Y-%m")
    by_month[month]["total"] += 1
    if result.is_defective:
        by_month[month]["defective"] += 1

for month in sorted(by_month.keys()):
    data = by_month[month]
    rate = data["defective"] / data["total"] if data["total"] > 0 else 0
    print(f"{month}: {rate:.1%} defect rate")
```

## Advanced Usage

### Custom Pipeline

```python
from agent.commit import RepositoryAnalysisPipeline
from agent.config import SystemConfig

# Custom configuration
config = SystemConfig()
config.llm.model_name = "deepseek-r1:7b"
config.llm.temperature = 0.2
config.agent.max_iterations = 3

# Create pipeline
pipeline = RepositoryAnalysisPipeline(config, verbose=True)

# Analyze
results = pipeline.analyze_repository("/path/to/repo")

# Custom statistics
stats = pipeline.get_statistics(results)
```

### Direct Tool Usage

```python
from agent.tools.implementations import GitRepositoryTool
from agent.commit import CommitDefectOrchestrator

# Extract commits
git_tool = GitRepositoryTool()
commits = git_tool.execute(repo_path="/path/to/repo", max_commits=50)

# Analyze each commit
orchestrator = CommitDefectOrchestrator()

for commit in commits:
    result = orchestrator.predict_commit(commit)
    if result.is_defective:
        print(f"Defective: {commit.commit_hash[:8]}")
```

## Performance Tips

1. **Filter by file extensions** - Only analyze relevant files
   ```python
   file_extensions=['.py', '.java']  # Skip docs, configs, etc.
   ```

2. **Limit commits** - Start with recent commits
   ```python
   max_commits=50  # Start small, increase if needed
   ```

3. **Use faster models** - codegemma:7b is fastest
   ```python
   model_name="codegemma:7b"
   ```

4. **Batch processing** - Process in chunks for large repos
   ```python
   for branch in ['main', 'develop', 'feature/xyz']:
       results = analyzer.analyze_repository(repo_path, branch=branch)
   ```

## Troubleshooting

### "Not a Git repository"
```bash
# Ensure .git directory exists
ls -la /path/to/repo/.git
```

### "No commits found"
```bash
# Check if repository has commits
cd /path/to/repo && git log -10
```

### "GitPython not found"
```bash
pip install GitPython
```

### Slow analysis
```bash
# Use fewer commits or filter by extension
python src/analyze_repository.py /path/to/repo --max-commits 20 --extensions .py
```

## Examples

See `examples/repository_analysis.py` for complete examples:
- Basic analysis
- Filtering by language
- Saving results
- Statistics
- Branch analysis
- Custom models

## API Reference

### RepositoryAnalysisPipeline

```python
pipeline = RepositoryAnalysisPipeline(config, verbose=True)

# Main method
results = pipeline.analyze_repository(
    repo_path: str,
    max_commits: int = 100,
    branch: Optional[str] = None,
    file_extensions: Optional[List[str]] = None
) -> List[CommitDefectResult]

# Save results
pipeline.save_results(
    results: List[CommitDefectResult],
    output_path: str,
    format: str = "json"  # "json", "csv", or "html"
)

# Get statistics
stats = pipeline.get_statistics(results) -> Dict[str, Any]
```

### CommitDefectResult

```python
result.commit_hash: str
result.commit_data: CommitData
result.intent: CommitIntent
result.prediction: DefectPrediction
result.is_defective: bool

result.to_dict() -> dict
```

## Comparison: Snippet vs Repository Analysis

| Aspect | Snippet Analysis | Repository Analysis |
|--------|-----------------|---------------------|
| Input | Isolated code | Code changes (diff) |
| Context | None | Commit message + intent |
| Scope | Single snippet | Full commit |
| Use Case | Educational | Real-world development |
| Output | Prediction | Prediction + context |

## Next Steps

1. **Try the examples**: `python examples/repository_analysis.py`
2. **Analyze your repo**: `python src/analyze_repository.py /path/to/your/repo`
3. **Generate reports**: Use HTML format for visual analysis
4. **Integrate with CI/CD**: Run analysis on pull requests

---

**Version**: 2.0.0
**Documentation**: See `src/agent/commit/` for implementation details
