# Software Defect Prediction Agent

An intelligent AI agent for predicting software defects using Large Language Models (LLMs). The agent analyzes both isolated code snippets and Git repository commits to identify potential bugs, security vulnerabilities, and code quality issues before deployment.

## Overview

The Software Defect Prediction Agent leverages state-of-the-art LLMs to provide context-aware defect prediction with human-readable explanations. It supports two analysis modes:

1. **Snippet-based Analysis**: Analyze isolated code snippets for defects
2. **Repository-based Analysis**: Analyze Git commits with full context (code changes, commit messages, developer intent)

### Key Features

- **Intelligent Defect Detection**: Identifies bugs, security vulnerabilities, and code quality issues
- **Context-Aware Analysis**: Understands code changes in the context of commit messages and developer intent
- **Tool-Augmented Verification**: Uses web search and documentation search to verify uncertain predictions
- **Explainable Predictions**: Provides confidence scores and detailed explanations for all predictions
- **Multi-Language Support**: Works with Python, Java, JavaScript, C++, and other languages
- **Batch Processing**: Analyze entire repositories or multiple code samples efficiently
- **Flexible Output**: JSON, CSV, and HTML report formats

### What Makes It Smart?

The agent doesn't just analyze code blindly. When uncertain, it:
- Checks documentation to verify if code uses updated package APIs
- Searches the web for common defect patterns and best practices
- Distinguishes between actual bugs and legitimate API changes
- Provides transparent decision-making with logged queries and results

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Defect Prediction Agent                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────┐         ┌──────────────────┐        │
│  │  Snippet Analysis│         │Repository Analysis│        │
│  │    Workflow      │         │    Workflow       │        │
│  └────────┬─────────┘         └────────┬──────────┘        │
│           │                            │                   │
│           ├────────────┬───────────────┤                   │
│                        ▼                                   │
│           ┌────────────────────────┐                       │
│           │  LLM Client (Ollama)   │                       │
│           └────────┬───────────────┘                       │
│                    │                                       │
│           ┌────────┴────────────────────────┐             │
│           │      Tool Registry              │             │
│           ├─────────────────────────────────┤             │
│           │ • Git Repository Tool           │             │
│           │ • Web Search Tool               │             │
│           │ • Documentation Search Tool     │             │
│           │ • Commit Intent Analyzer        │             │
│           └─────────────────────────────────┘             │
│                                                             │
│  Output: Predictions + Confidence + Explanations           │
└─────────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites

- Python 3.8 or higher
- [Ollama](https://ollama.ai/) installed and running
- Git (for repository analysis)

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/llm-agent-for-software-defect-prediction.git
cd llm-agent-for-software-defect-prediction
```

2. **Create a virtual environment:**
```bash
python -m venv sdp
source sdp/bin/activate  # On Windows: sdp\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Install and start Ollama:**
```bash
# Install Ollama from https://ollama.ai/
# Then pull the model:
ollama pull codegemma:7b
```

### Verify Installation

```bash
python -c "from agent import create_agent; print('✓ Installation successful!')"
```

## Quick Start

### 1. Snippet-Based Analysis

Analyze isolated code snippets for defects:

```python
from agent import create_agent

# Create agent
agent = create_agent(
    model_name="codegemma:7b",
    verbose=True
)

# Analyze code
code = """
def calculate_average(numbers):
    total = sum(numbers)
    return total / len(numbers)

result = calculate_average([])  # Potential division by zero
"""

prediction, result = agent.predict(code, language="python")

print(f"Prediction: {'Defective' if prediction == 1 else 'Clean'}")
print(f"Confidence: {result.confidence:.2f}")
print(f"Explanation: {result.explanation}")
```

**Output:**
```
Prediction: Defective
Confidence: 0.92
Explanation: Division by zero error when empty list is passed to calculate_average
```

### 2. Repository-Based Analysis

Analyze Git commits for defects:

```python
from agent import create_repository_analyzer

# Create analyzer with tool support
analyzer = create_repository_analyzer(
    model_name="codegemma:7b",
    enable_web_search=True,
    enable_doc_search=True,
    verbose=True
)

# Analyze repository
results = analyzer.analyze_repository(
    repo_path="/path/to/your/repository",
    max_commits=50,
    file_extensions=['.py']  # Optional: filter by language
)

# Print results
for result in results:
    if result.is_defective:
        print(f"⚠️  {result.commit_hash[:8]}: {result.commit_data.message}")
        print(f"   Confidence: {result.prediction.confidence:.2f}")
        print(f"   Reason: {result.prediction.explanation}\n")
```

**Output:**
```
⚠️  a3f5c21d: Fix user authentication bug
   Confidence: 0.78
   Reason: Incomplete error handling in authentication flow

   Verified with external tools:
   - Documentation check for flask: Flask 2.3.0 requires proper exception handling...
```

### 3. Detailed Analysis Output

For comprehensive analysis with all details, use example 7:

```python
from agent import create_repository_analyzer

analyzer = create_repository_analyzer(
    model_name="codegemma:7b",
    enable_web_search=True,
    enable_doc_search=True,
    verbose=True
)

results = analyzer.analyze_repository(
    repo_path="/path/to/repo",
    max_commits=5
)
```

**Detailed Output Example:**

```
======================================================================
COMMIT 1/5
======================================================================

📋 COMMIT INFORMATION:
  Hash:    13496b9e115964130b55bbc64a98ce27f491e276
  Author:  Matt Clay
  Date:    2025-10-30 10:21:53
  Message: ansible-test - Fix integration test code coverage (#86100)

📁 FILES CHANGED (8):
  - changelogs/fragments/ansible-test-integration-coverage.yml
  - test/integration/targets/ansible-test-coverage/ansible_collections/ns/col/plugins/module_utils/test_util.py
  - test/integration/targets/ansible-test-coverage/ansible_collections/ns/col/tests/integration/targets/hello/aliases
  - test/integration/targets/ansible-test-coverage/ansible_collections/ns/col/tests/integration/targets/hello/runme.sh
  - test/integration/targets/ansible-test-coverage/ansible_collections/ns/col/tests/integration/targets/hello/world.py
  - test/integration/targets/ansible-test-coverage/ansible_collections/ns/col/tests/unit/test_something.py
  - test/integration/targets/ansible-test-coverage/runme.sh
  - test/lib/ansible_test/_internal/coverage_util.py

📊 STATISTICS:
  Lines added:      44
  Lines removed:     8
  Net change:       36
  Files changed:     8

🔍 CODE CHANGES (sample):
  Added lines (44 total):
    + bugfixes:
    + - ansible-test - Restore code coverage reporting for Python code residing in int
    + A_CONSTANT = True
    + context/controller
    + #!/usr/bin/env bash
    ... and 39 more added lines

  Removed lines (8 total):
    - def test_coverage():
    - pass
    - # run a lightweight test that generates code coverge output
    - ansible-test sanity --test import "${common[@]}" --coverage
    - ansible-test coverage report "${common[@]}"
    ... and 3 more removed lines

🎯 DEVELOPER INTENT ANALYSIS:
  Intent Type:  test
  Risk Level:   low
  Confidence:   0.90
  Description:  Fix integration test code coverage

🔮 DEFECT PREDICTION:
  Status:       ✅ CLEAN
  Confidence:   0.80
  Explanation:  The commit fixes integration test code coverage and adds new code
                to report coverage. The code changes do not introduce bugs or
                vulnerabilities, break existing functionality, or have incomplete
                changes. The commit message and developer intent match the code
                changes, indicating alignment between intent and execution.

----------------------------------------------------------------------

SUMMARY
======================================================================
Total Commits Analyzed: 5
Defective Commits:      0 (0.0%)
Clean Commits:          5 (100.0%)
Average Confidence:     0.84
```

This detailed output shows every step of the analysis:
- **Complete commit metadata** (hash, author, date, message)
- **All files changed** in the commit
- **Statistics** (lines added/removed, net change)
- **Code changes** (sample of added and removed lines)
- **Developer intent** (what they were trying to do, risk level)
- **Defect prediction** (clean or defective, confidence, explanation)
- **Summary statistics** for all analyzed commits

## Usage Guide

### Command Line Interface

**Analyze a repository:**
```bash
python src/analyze_repository.py /path/to/repo --max-commits 100 --output results.json
```

**Filter by file type:**
```bash
python src/analyze_repository.py /path/to/repo --extensions .py .java
```

**Generate HTML report:**
```bash
python src/analyze_repository.py /path/to/repo --output report.html --format html
```

**Use different model:**
```bash
python src/analyze_repository.py /path/to/repo --model deepseek-r1:7b
```

### Python API

#### Basic Configuration

```python
from agent import create_repository_analyzer

# Default configuration
analyzer = create_repository_analyzer(
    model_name="codegemma:7b",
    enable_web_search=True,
    enable_doc_search=True,
    verbose=True
)
```

#### Advanced Configuration

```python
from agent import RepositoryAnalysisPipeline
from agent.config import SystemConfig

# Custom configuration
config = SystemConfig()
config.llm.model_name = "deepseek-r1:7b"
config.llm.temperature = 0.2
config.llm.max_tokens = 2048
config.agent.max_iterations = 3

# Create pipeline
pipeline = RepositoryAnalysisPipeline(config, verbose=True)

# Analyze
results = pipeline.analyze_repository("/path/to/repo")
```

#### Save Results

```python
# Save as JSON
analyzer.save_results(results, "results.json", format="json")

# Save as CSV
analyzer.save_results(results, "results.csv", format="csv")

# Save as HTML report
analyzer.save_results(results, "report.html", format="html")
```

#### Get Statistics

```python
stats = analyzer.get_statistics(results)

print(f"Total commits: {stats['total_commits']}")
print(f"Defective: {stats['defective_commits']} ({stats['defect_rate']:.1%})")
print(f"Average confidence: {stats['average_confidence']:.2f}")

# Intent distribution
for intent_type, data in stats['intent_distribution'].items():
    print(f"{intent_type}: {data['total']} commits")
```

### Tool-Augmented Analysis

The agent automatically uses external tools when uncertain:

**When tools are used:**
- Confidence < 0.75 (uncertain prediction)
- Code contains package imports
- Very low confidence < 0.6 (triggers web search)

**Example output:**
```
[Step 3/3] Predicting defects...

  🔍 Low confidence (0.68) or package usage detected
  Decision: Using external tools to verify prediction...

  📚 Documentation Search:
     Query: "requests API documentation latest version"
     Summary: Requests 2.31.0 is the latest version...

  🌐 Web Search:
     Query: "security vulnerability common causes"
     Summary: Common vulnerabilities include SQL injection...

PREDICTION: DEFECTIVE
Verified with external tools:
- Documentation check for requests: Latest API uses context managers...
- Web search insights: Proper error handling required for HTTP requests...
```

## Use Cases

### 1. Pre-Release Review
Analyze recent commits before release:
```python
results = analyzer.analyze_repository(
    repo_path="/path/to/repo",
    max_commits=50,
    branch="release/v2.0"
)

defective = [r for r in results if r.is_defective]
print(f"⚠️  Found {len(defective)} potentially defective commits")
```

### 2. Code Review Prioritization
Identify high-risk commits:
```python
high_risk = [r for r in results if r.intent.risk_level == "high"]
high_risk.sort(key=lambda r: r.prediction.confidence, reverse=True)

print("High-risk commits to review:")
for commit in high_risk[:10]:
    print(f"  {commit.commit_hash[:8]}: {commit.commit_data.message}")
```

### 3. Developer Analysis
Track defect rates by developer:
```python
from collections import defaultdict

by_author = defaultdict(list)
for result in results:
    by_author[result.commit_data.author].append(result)

for author, commits in by_author.items():
    defective = sum(1 for c in commits if c.is_defective)
    rate = defective / len(commits)
    print(f"{author}: {defective}/{len(commits)} ({rate:.1%})")
```

### 4. Continuous Integration
Integrate into CI/CD pipeline:
```bash
# In your CI script
python src/analyze_repository.py $REPO_PATH --max-commits 10 --output ci_results.json

# Check results
python scripts/check_ci_results.py ci_results.json
```

## Configuration

### LLM Models

Supported models via Ollama:
- `codegemma:7b` (fast, good for code)
- `deepseek-r1:7b` (more accurate)
- `codellama:7b` (alternative)

```bash
# Pull additional models
ollama pull deepseek-r1:7b
ollama pull codellama:7b
```

### Environment Variables

```bash
export OLLAMA_HOST="http://localhost:11434"  # Ollama server
export LLM_TIMEOUT=120  # Request timeout in seconds
```

### Configuration File

Create a config file `config.yaml`:
```yaml
llm:
  model_name: "codegemma:7b"
  temperature: 0.1
  max_tokens: 2048

agent:
  max_iterations: 5
  confidence_threshold: 0.7
  enable_web_search: true
  enable_doc_search: true
```

## Examples

Explore comprehensive examples in the `examples/` directory:

- **`basic_usage.py`**: Snippet-based analysis examples
- **`repository_analysis.py`**: Repository analysis examples with 7 scenarios

Run examples:
```bash
python examples/repository_analysis.py
```

## Performance Tips

1. **Filter by file extensions**: Only analyze relevant files
   ```python
   file_extensions=['.py', '.java']
   ```

2. **Limit commits**: Start with recent commits
   ```python
   max_commits=50
   ```

3. **Use faster models**: codegemma:7b is fastest
   ```python
   model_name="codegemma:7b"
   ```

4. **Disable verbose mode**: Reduce output overhead
   ```python
   verbose=False
   ```

## Troubleshooting

**"Connection refused" error:**
```bash
# Make sure Ollama is running
ollama serve
```

**"Model not found" error:**
```bash
# Pull the model
ollama pull codegemma:7b
```

**"Not a Git repository" error:**
```bash
# Ensure .git directory exists
ls /path/to/repo/.git
```

**Slow analysis:**
```bash
# Reduce commits or filter by extension
python src/analyze_repository.py /path/to/repo --max-commits 20 --extensions .py
```

## Project Structure

```
.
├── src/
│   ├── agent/
│   │   ├── __init__.py          # Main exports
│   │   ├── config.py            # Configuration classes
│   │   ├── core/                # Core orchestrator
│   │   ├── llm/                 # LLM client
│   │   ├── tools/               # Tool implementations
│   │   │   ├── implementations/
│   │   │   │   ├── git_repository.py
│   │   │   │   ├── web_search.py
│   │   │   │   └── documentation_search.py
│   │   └── commit/              # Repository analysis
│   │       ├── pipeline.py      # Analysis pipeline
│   │       ├── orchestrator.py  # Commit defect predictor
│   │       └── intent_analyzer.py
│   └── analyze_repository.py    # CLI tool
├── examples/
│   ├── basic_usage.py           # Snippet examples
│   └── repository_analysis.py   # Repository examples
├── requirements.txt
└── README.md
```

## Requirements

### Core Dependencies
- Python >= 3.8
- GitPython >= 3.1.0
- requests >= 2.28.0
- tqdm >= 4.65.0
- pandas >= 1.5.0

### Development Dependencies
- pytest >= 7.0.0
- black >= 22.0.0
- flake8 >= 5.0.0

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{sdp_llm_agent,
  title={Software Defect Prediction Agent with Large Language Models},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/llm-agent-for-software-defect-prediction}
}
```

## Acknowledgments

- Built with [Ollama](https://ollama.ai/) for local LLM inference
- Inspired by recent advances in LLM-based code analysis
- Thanks to all contributors and users

## Support

- **Issues**: Report bugs or request features via [GitHub Issues](https://github.com/yourusername/llm-agent-for-software-defect-prediction/issues)
- **Discussions**: Join discussions in [GitHub Discussions](https://github.com/yourusername/llm-agent-for-software-defect-prediction/discussions)

---

**Version**: 2.0.0
**Last Updated**: 2025-01-01
