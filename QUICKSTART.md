# Quick Start Guide

Get up and running with the SDP-LLM-Agent in 5 minutes!

## 🚀 Installation (3 minutes)

### Step 1: Install Ollama

```bash
# macOS/Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Windows: Download from https://ollama.ai/download
```

### Step 2: Pull a Model

```bash
ollama pull codegemma:7b
```

### Step 3: Install Python Dependencies

```bash
# Clone the repository
git clone https://github.com/yourusername/SDP-LLM-Agent.git
cd SDP-LLM-Agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements-minimal.txt
```

## 🎯 First Prediction (2 minutes)

### Option 1: Command Line

Start Ollama:
```bash
ollama serve
```

In a new terminal:
```bash
python << EOF
from agent import create_agent

# Create agent
agent = create_agent(model_name="codegemma:7b")

# Code with a defect
code = """
def divide(a, b):
    return a / b

result = divide(10, 0)
"""

# Predict
prediction, result = agent.predict(code)
print(f"Prediction: {prediction} (1=defective, 0=clean)")
print(f"Confidence: {result.confidence:.2f}")
print(f"Explanation: {result.explanation}")
EOF
```

### Option 2: Run Examples

```bash
python examples/basic_usage.py
```

### Option 3: Interactive Python

```python
from agent import create_agent

# Create agent (tools disabled for speed)
agent = create_agent(
    model_name="codegemma:7b",
    enable_web_search=False,
    enable_doc_search=False,
    verbose=True  # See what the agent is thinking
)

# Test code samples
samples = {
    "division_by_zero": "result = 10 / 0",
    "list_index_error": "my_list = [1, 2, 3]\nvalue = my_list[10]",
    "safe_code": "x = 5\ny = 10\nz = x + y",
}

for name, code in samples.items():
    prediction, result = agent.predict(code)
    print(f"\n{name}: {prediction} (confidence: {result.confidence:.2f})")
```

## 📊 Run Evaluation

Test the agent on a dataset:

```bash
# Quick test (10 samples)
cd src
python evaluate_agent_v2.py \
    --dataset transformed_data.csv \
    --max-samples 10 \
    --verbose

# Full evaluation (200 samples)
python evaluate_agent_v2.py \
    --dataset transformed_data.csv \
    --max-samples 200

# With external tools (slower but more accurate)
python evaluate_agent_v2.py \
    --dataset transformed_data.csv \
    --max-samples 50 \
    --web-search \
    --doc-search
```

## 🔧 Configuration

### Quick Config

```python
from agent import SystemConfig, DefectPredictionOrchestrator

# Create custom config
config = SystemConfig()
config.llm.model_name = "deepseek-r1:7b"
config.llm.temperature = 0.2
config.agent.max_iterations = 3
config.agent.verbose = False

# Create agent with config
agent = DefectPredictionOrchestrator(config)
```

### Environment Variables

Create `.env` file:
```bash
AGENT_MODEL_NAME=codegemma:7b
AGENT_MAX_ITERATIONS=5
AGENT_VERBOSE=true
```

## 🎓 Next Steps

### Learn More
- **Architecture:** Read `src/agent/README.md`
- **Examples:** Explore `examples/basic_usage.py`
- **Full Docs:** See `ARCHITECTURE_V2.md`

### Try Advanced Features

**1. Add Custom Tools:**
```python
from agent.tools import BaseTool, ToolMetadata

class MyTool(BaseTool):
    def get_metadata(self):
        return ToolMetadata(
            name="my_tool",
            description="My custom tool",
            parameters={},
            returns="Result"
        )

    def execute(self, **kwargs):
        return "Custom result"

agent.register_tool(MyTool())
```

**2. Inspect Execution:**
```python
prediction, result = agent.predict(code)

# Get execution details
context = agent.get_context()
stats = agent.get_statistics()

print(f"Iterations: {stats['iterations']}")
print(f"Execution time: {stats['execution_time']:.2f}s")
print(f"Observations: {len(context.observations)}")
```

**3. Batch Processing:**
```python
import pandas as pd

df = pd.read_csv("transformed_data.csv")
predictions = []

for code in df['code_samples'].head(10):
    pred, _ = agent.predict(code)
    predictions.append(pred)

df['predictions'] = predictions
```

## ❓ Common Issues

### Ollama not responding
```bash
# Make sure Ollama is running
ollama serve

# Test connection
curl http://localhost:11434/api/tags
```

### Model not found
```bash
# List available models
ollama list

# Pull missing model
ollama pull codegemma:7b
```

### Import errors
```bash
# Add src to path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Or install package
pip install -e .
```

## 🐛 Troubleshooting

For detailed troubleshooting, see `INSTALL.md`

## 📚 Documentation

- **Installation:** `INSTALL.md`
- **Architecture:** `ARCHITECTURE_V2.md`
- **Agent Docs:** `src/agent/README.md`
- **Examples:** `examples/basic_usage.py`

## 💬 Getting Help

- Open an issue on GitHub
- Check existing issues and documentation
- Review example code

---

**Ready to go?** Start predicting defects! 🎉

```bash
python examples/basic_usage.py
```
