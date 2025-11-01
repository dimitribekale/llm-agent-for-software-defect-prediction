"""
Basic usage examples for the Defect Prediction Agent V2.

This script demonstrates common use cases and features.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from agent import create_agent, SystemConfig
from agent.tools.implementations import WebSearchTool, DocumentationSearchTool


def example_1_quick_start():
    """Example 1: Quick start with default settings."""
    print("\n" + "="*70)
    print("Example 1: Quick Start")
    print("="*70 + "\n")

    # Create agent with defaults
    agent = create_agent(
        model_name="codegemma:7b",
        verbose=False
    )

    # Code with division by zero
    code = """
def calculate_average(numbers):
    total = sum(numbers)
    return total / len(numbers)

# Potential issue: empty list
result = calculate_average([])
print(result)
"""

    print("Code to analyze:")
    print(code)
    print("\nAnalyzing...")

    # Predict
    prediction, result = agent.predict(code, language="python")

    print(f"\nResults:")
    print(f"  Prediction: {prediction} ({'Defective' if prediction == 1 else 'Clean'})")
    print(f"  Confidence: {result.confidence:.2f}")
    print(f"  Explanation: {result.explanation}")


def example_2_with_tools():
    """Example 2: Using tools for enhanced analysis."""
    print("\n" + "="*70)
    print("Example 2: With External Tools")
    print("="*70 + "\n")

    # Create agent with tools
    agent = create_agent(
        model_name="codegemma:7b",
        enable_web_search=True,
        enable_doc_search=True,
        verbose=True
    )

    # Code with potential SQL injection
    code = """
def get_user(username):
    query = f"SELECT * FROM users WHERE username = '{username}'"
    cursor.execute(query)
    return cursor.fetchone()

user = get_user(request.args['username'])
"""

    print("Code to analyze:")
    print(code)
    print("\nAnalyzing with tools...")

    prediction, result = agent.predict(code, language="python")

    print(f"\n{'='*70}")
    print("Results:")
    print(f"{'='*70}")
    print(f"Prediction: {prediction}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Explanation: {result.explanation}")

    # Show statistics
    stats = agent.get_statistics()
    print(f"\nStatistics:")
    print(f"  Iterations: {stats['iterations']}")
    print(f"  Execution time: {stats['execution_time']:.2f}s")
    print(f"  Observations: {stats['observations']}")


def example_3_custom_config():
    """Example 3: Custom configuration."""
    print("\n" + "="*70)
    print("Example 3: Custom Configuration")
    print("="*70 + "\n")

    # Create custom config
    config = SystemConfig()

    # LLM settings
    config.llm.model_name = "deepseek-r1:7b"
    config.llm.temperature = 0.2
    config.llm.max_tokens = 2048

    # Agent settings
    config.agent.max_iterations = 3
    config.agent.confidence_threshold = 0.8
    config.agent.verbose = False

    # Memory settings
    config.memory.max_history_size = 15

    from agent.core import DefectPredictionOrchestrator

    agent = DefectPredictionOrchestrator(config)

    code = """
def unsafe_file_read(filename):
    # Potential path traversal vulnerability
    with open(f"/var/data/{filename}", 'r') as f:
        return f.read()

content = unsafe_file_read(user_input)
"""

    print("Configuration:")
    print(f"  Model: {config.llm.model_name}")
    print(f"  Temperature: {config.llm.temperature}")
    print(f"  Max iterations: {config.agent.max_iterations}")
    print(f"  Confidence threshold: {config.agent.confidence_threshold}")

    print("\nCode to analyze:")
    print(code)

    prediction, result = agent.predict(code, language="python")

    print(f"\nResults:")
    print(f"  Prediction: {prediction}")
    print(f"  Confidence: {result.confidence:.2f}")
    print(f"  Explanation: {result.explanation}")


def example_4_batch_processing():
    """Example 4: Process multiple code samples."""
    print("\n" + "="*70)
    print("Example 4: Batch Processing")
    print("="*70 + "\n")

    agent = create_agent(verbose=False)

    # Multiple code samples
    samples = [
        {
            "name": "Division by zero",
            "code": "result = 10 / 0",
            "expected": 1
        },
        {
            "name": "List index error",
            "code": "my_list = [1, 2, 3]\nvalue = my_list[10]",
            "expected": 1
        },
        {
            "name": "Safe code",
            "code": "x = 5\ny = 10\nz = x + y\nprint(z)",
            "expected": 0
        },
        {
            "name": "Null pointer",
            "code": "obj = None\nobj.method()",
            "expected": 1
        }
    ]

    print(f"Processing {len(samples)} samples...\n")

    correct = 0
    for i, sample in enumerate(samples, 1):
        prediction, result = agent.predict(sample["code"], language="python")

        status = "✓" if prediction == sample["expected"] else "✗"
        print(f"{status} Sample {i}: {sample['name']}")
        print(f"   Predicted: {prediction}, Expected: {sample['expected']}")
        print(f"   Confidence: {result.confidence:.2f}")

        if prediction == sample["expected"]:
            correct += 1

    accuracy = correct / len(samples)
    print(f"\nAccuracy: {accuracy:.2%} ({correct}/{len(samples)})")


def example_5_error_handling():
    """Example 5: Error handling and recovery."""
    print("\n" + "="*70)
    print("Example 5: Error Handling")
    print("="*70 + "\n")

    from agent.llm import LLMTimeoutException, LLMConnectionException

    agent = create_agent(verbose=False)

    code = "def test(): pass"

    try:
        print("Attempting prediction...")
        prediction, result = agent.predict(code)
        print(f"Success: {prediction}")

    except LLMTimeoutException as e:
        print(f"[ERROR] Timeout: {e}")
        print("Consider increasing timeout or using a smaller model")

    except LLMConnectionException as e:
        print(f"[ERROR] Connection failed: {e}")
        print("Ensure Ollama is running: ollama serve")

    except Exception as e:
        print(f"[ERROR] Unexpected: {e}")
        print("Check logs for details")


def example_6_context_inspection():
    """Example 6: Inspect agent context and observations."""
    print("\n" + "="*70)
    print("Example 6: Context Inspection")
    print("="*70 + "\n")

    agent = create_agent(verbose=False)

    code = """
def process_data(data):
    if data:
        return data.upper()
    return None

result = process_data(None)
print(result.strip())  # Potential AttributeError
"""

    print("Code to analyze:")
    print(code)

    prediction, result = agent.predict(code)

    # Get execution context
    context = agent.get_context()

    print(f"\nExecution Context:")
    print(f"  State: {context.current_state.name}")
    print(f"  Iterations: {context.iteration + 1}")
    print(f"  Observations: {len(context.observations)}")

    # Inspect observations
    print(f"\nObservation Details:")
    for i, obs in enumerate(context.observations, 1):
        print(f"\n  Iteration {i}:")
        print(f"    Confidence: {obs.confidence_score:.2f}")
        print(f"    Thoughts: {obs.thoughts[:100]}...")
        print(f"    Tool results: {len(obs.tool_results)}")

    # Final prediction
    print(f"\nFinal Prediction:")
    print(f"  Result: {result.prediction}")
    print(f"  Confidence: {result.confidence:.2f}")
    print(f"  Explanation: {result.explanation}")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("Defect Prediction Agent V2 - Examples")
    print("="*70)

    examples = [
        ("Quick Start", example_1_quick_start),
        ("With Tools", example_2_with_tools),
        ("Custom Config", example_3_custom_config),
        ("Batch Processing", example_4_batch_processing),
        ("Error Handling", example_5_error_handling),
        ("Context Inspection", example_6_context_inspection),
    ]

    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")

    print("\nRunning examples...\n")

    # Run specific examples (modify as needed)
    example_1_quick_start()
    # example_2_with_tools()  # Uncomment to run
    # example_3_custom_config()  # Uncomment to run
    example_4_batch_processing()
    example_5_error_handling()
    # example_6_context_inspection()  # Uncomment to run

    print("\n" + "="*70)
    print("Examples completed!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
