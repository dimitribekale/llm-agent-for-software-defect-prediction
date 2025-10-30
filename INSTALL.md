# Installation Guide

This guide provides detailed installation instructions for the SDP-LLM-Agent project.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation Methods](#installation-methods)
3. [Verifying Installation](#verifying-installation)
4. [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

- **Python**: 3.8 or higher (3.10+ recommended)
- **Operating System**: Linux, macOS, or Windows
- **RAM**: Minimum 8GB (16GB+ recommended for running LLMs)
- **Disk Space**: ~10GB for Ollama models

### Required Software

#### 1. Python

Check if Python is installed:
```bash
python --version
# or
python3 --version
```

If not installed, download from [python.org](https://www.python.org/downloads/)

#### 2. Ollama

Ollama is required for running local LLMs.

**macOS/Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Windows:**
Download from [ollama.ai](https://ollama.ai/download)

**Verify installation:**
```bash
ollama --version
```

**Start Ollama service:**
```bash
ollama serve
```

**Pull required models:**
```bash
# Recommended models
ollama pull codegemma:7b
ollama pull deepseek-r1:7b
ollama pull codellama:7b

# List installed models
ollama list
```

## Installation Methods

### Method 1: Quick Install (Recommended)

For most users, this is the fastest way to get started:

```bash
# Clone the repository
git clone https://github.com/yourusername/SDP-LLM-Agent.git
cd SDP-LLM-Agent

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install minimal requirements
pip install -r requirements-minimal.txt
```

### Method 2: Full Installation

Install all dependencies including optional tools:

```bash
# After creating and activating virtual environment
pip install -r requirements.txt
```

### Method 3: Development Installation

For contributors and developers:

```bash
# Install with development tools
pip install -r requirements-dev.txt

# Install package in editable mode
pip install -e .

# Set up pre-commit hooks
pre-commit install
```

### Method 4: Using setup.py

Install as a package:

```bash
# Basic installation
pip install .

# With all optional dependencies
pip install .[full]

# For development
pip install -e .[dev]
```

## Verifying Installation

### 1. Check Python Packages

```bash
pip list | grep -E "requests|pandas|scikit-learn|duckduckgo"
```

### 2. Test Ollama Connection

```bash
# In a Python shell
python << EOF
from agent.llm import LLMClientFactory, LLMConfig

config = LLMConfig()
client = LLMClientFactory.create("ollama", config)
print("Ollama available:", client.is_available())
print("Models:", client.list_models())
EOF
```

### 3. Run Quick Test

```bash
# Test agent creation
python << EOF
from agent import create_agent

agent = create_agent(model_name="codegemma:7b", verbose=False)
print("Agent created successfully!")
EOF
```

### 4. Run Example

```bash
python examples/basic_usage.py
```

### 5. Run Evaluation Test

```bash
cd src
python evaluate_agent_v2.py --test
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# LLM Configuration
AGENT_MODEL_NAME=codegemma:7b
OLLAMA_API_URL=http://localhost:11434/api/generate
AGENT_TEMPERATURE=0.1
AGENT_MAX_TOKENS=1024
AGENT_TIMEOUT=120

# Agent Configuration
AGENT_MAX_ITERATIONS=5
AGENT_CONFIDENCE_THRESHOLD=0.7
AGENT_ENABLE_WEB_SEARCH=true
AGENT_ENABLE_DOC_SEARCH=true
AGENT_VERBOSE=true

# Memory Configuration
AGENT_MAX_HISTORY=20
AGENT_MAX_CONTEXT_TOKENS=4096
AGENT_PRUNE_STRATEGY=oldest
```

Load environment variables:
```bash
export $(cat .env | xargs)
```

Or use python-dotenv:
```bash
pip install python-dotenv
```

## Troubleshooting

### Common Issues

#### 1. Ollama Connection Failed

**Error:** `LLMConnectionException: Failed to connect to http://localhost:11434`

**Solution:**
```bash
# Start Ollama service
ollama serve

# Check if running
curl http://localhost:11434/api/tags
```

#### 2. Model Not Found

**Error:** `Model 'codegemma:7b' not found`

**Solution:**
```bash
# Pull the model
ollama pull codegemma:7b

# Verify
ollama list
```

#### 3. Import Errors

**Error:** `ModuleNotFoundError: No module named 'agent'`

**Solution:**
```bash
# Make sure you're in the project root
cd SDP-LLM-Agent

# Add src to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:${PWD}/src"

# Or install in editable mode
pip install -e .
```

#### 4. SSL Certificate Errors

**Error:** `SSL: CERTIFICATE_VERIFY_FAILED`

**Solution:**
```bash
# Install certificates (macOS)
/Applications/Python*/Install\ Certificates.command

# Or disable SSL verification (not recommended for production)
export CURL_CA_BUNDLE=""
export REQUESTS_CA_BUNDLE=""
```

#### 5. Memory Issues

**Error:** `MemoryError` or system freeze

**Solution:**
- Use smaller models (3B instead of 7B)
- Reduce `max_iterations` in config
- Close other applications
- Increase system swap space

#### 6. DuckDuckGo Search Errors

**Error:** `duckduckgo_search failed`

**Solution:**
```bash
# Update to latest version
pip install --upgrade duckduckgo-search

# Or disable web search
python -c "from agent import create_agent; agent = create_agent(enable_web_search=False)"
```

### Platform-Specific Issues

#### macOS

- If using M1/M2 Mac, ensure you're using ARM-compatible Python
- Install Xcode Command Line Tools: `xcode-select --install`

#### Windows

- Use PowerShell or Git Bash
- May need to install Microsoft C++ Build Tools
- Use `python` instead of `python3`

#### Linux

- Install required system packages:
  ```bash
  sudo apt-get update
  sudo apt-get install python3-dev python3-pip build-essential
  ```

### Getting Help

If you encounter issues not covered here:

1. Check existing [GitHub Issues](https://github.com/yourusername/SDP-LLM-Agent/issues)
2. Review the [Documentation](src/agent/README.md)
3. Open a new issue with:
   - Error message
   - Python version (`python --version`)
   - OS version
   - Steps to reproduce

## Next Steps

After successful installation:

1. **Read Documentation:** `src/agent/README.md`
2. **Run Examples:** `python examples/basic_usage.py`
3. **Try Evaluation:** `python src/evaluate_agent_v2.py --help`
4. **Explore Architecture:** `ARCHITECTURE_V2.md`

---

**Installation successful?** You're ready to start predicting software defects!

```bash
python << EOF
from agent import create_agent

code = '''
def divide(a, b):
    return a / b
result = divide(10, 0)
'''

agent = create_agent()
prediction, result = agent.predict(code)
print(f"Prediction: {prediction}, Confidence: {result.confidence:.2f}")
EOF
```
