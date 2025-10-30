"""
Evaluation script for the new agent architecture (v2).

This script evaluates the refactored agent on a defect prediction dataset.
"""

import pandas as pd
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    f1_score,
    classification_report,
    recall_score
)
from typing import Dict
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import create_agent, SystemConfig


def evaluate_agent_v2(
    dataset_path: str,
    model_name: str = "codegemma:7b",
    max_samples: int = 200,
    enable_web_search: bool = False,  # Disabled by default for speed
    enable_doc_search: bool = False,  # Disabled by default for speed
    verbose: bool = False
) -> Dict[str, float]:
    """
    Evaluate the new agent architecture.

    Args:
        dataset_path: Path to CSV dataset
        model_name: LLM model to use
        max_samples: Maximum samples to evaluate (for testing)
        enable_web_search: Enable web search tool
        enable_doc_search: Enable documentation search tool
        verbose: Print detailed execution info

    Returns:
        Dictionary of metrics
    """
    print(f"\n{'='*70}")
    print(f"Evaluating Agent V2 with {model_name}")
    print(f"{'='*70}")
    print(f"Configuration:")
    print(f"  - Max samples: {max_samples}")
    print(f"  - Web search: {enable_web_search}")
    print(f"  - Doc search: {enable_doc_search}")
    print(f"  - Verbose: {verbose}")
    print(f"{'='*70}\n")

    # Load dataset
    df = pd.read_csv(dataset_path)
    print(f"Loaded dataset: {len(df)} total samples")

    # Limit samples for testing
    df = df.head(max_samples)
    y_true = df['target'].values
    print(f"Evaluating on: {len(df)} samples\n")

    # Create agent
    agent = create_agent(
        model_name=model_name,
        enable_web_search=enable_web_search,
        enable_doc_search=enable_doc_search,
        verbose=verbose
    )

    # Evaluate
    y_pred = []
    failed_count = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        code = row['code_samples']

        try:
            prediction, full_result = agent.predict(code, language="python")
            y_pred.append(prediction)

            if verbose:
                print(f"\nSample {idx}:")
                print(f"  True: {row['target']}, Predicted: {prediction}")
                print(f"  Confidence: {full_result.confidence:.2f}")
                print(f"  Explanation: {full_result.explanation[:100]}...")

        except Exception as e:
            print(f"\n[ERROR] Sample {idx} failed: {str(e)}")
            y_pred.append(0)  # Default to clean
            failed_count += 1

    # Calculate metrics
    print(f"\n{'='*70}")
    print("Evaluation Results")
    print(f"{'='*70}")
    print(f"Failed predictions: {failed_count}/{len(df)}")

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    print(f"\nMetrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")

    print(f"\n{'-'*70}")
    print("Classification Report:")
    print(f"{'-'*70}")
    report = classification_report(
        y_true,
        y_pred,
        target_names=['Clean', 'Defective'],
        zero_division=0
    )
    print(report)
    print(f"{'='*70}\n")

    return metrics


def compare_models(
    dataset_path: str,
    models: list = None,
    max_samples: int = 50
) -> pd.DataFrame:
    """
    Compare multiple models.

    Args:
        dataset_path: Path to dataset
        models: List of model names to compare
        max_samples: Maximum samples per model

    Returns:
        DataFrame with comparison results
    """
    if models is None:
        models = ["codegemma:7b", "deepseek-r1:7b", "codellama:7b"]

    print(f"\n{'='*70}")
    print(f"Model Comparison (on {max_samples} samples)")
    print(f"{'='*70}\n")

    results = []

    for model in models:
        print(f"\nEvaluating {model}...")
        try:
            metrics = evaluate_agent_v2(
                dataset_path=dataset_path,
                model_name=model,
                max_samples=max_samples,
                verbose=False
            )
            metrics["model"] = model
            metrics["status"] = "success"
            results.append(metrics)
        except Exception as e:
            print(f"[ERROR] {model} failed: {str(e)}")
            results.append({
                "model": model,
                "status": "failed",
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0
            })

    # Create DataFrame
    df = pd.DataFrame(results)
    df = df.set_index("model")

    # Sort by F1 score
    df = df.sort_values("f1", ascending=False)

    print(f"\n{'='*70}")
    print("Model Comparison Results")
    print(f"{'='*70}")
    print(df[["accuracy", "precision", "recall", "f1", "status"]].to_string())
    print(f"{'='*70}\n")

    return df


def test_single_sample():
    """Test agent on a single code sample."""
    print(f"\n{'='*70}")
    print("Single Sample Test")
    print(f"{'='*70}\n")

    # Test code with a potential defect
    test_code = """
def divide_numbers(a, b):
    return a / b

result = divide_numbers(10, 0)
print(result)
"""

    print("Test Code:")
    print(test_code)
    print(f"\n{'-'*70}\n")

    # Create agent with tools
    agent = create_agent(
        model_name="codegemma:7b",
        enable_web_search=True,
        enable_doc_search=True,
        verbose=True
    )

    # Predict
    prediction, result = agent.predict(test_code, language="python")

    print(f"\n{'='*70}")
    print("Prediction Results")
    print(f"{'='*70}")
    print(f"Prediction: {prediction} ({'Defective' if prediction == 1 else 'Clean'})")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Explanation: {result.explanation}")

    # Show statistics
    stats = agent.get_statistics()
    print(f"\nExecution Statistics:")
    print(f"  Iterations: {stats['iterations']}")
    print(f"  Execution time: {stats['execution_time']:.2f}s")
    print(f"  Observations: {stats['observations']}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Agent V2")
    parser.add_argument(
        "--dataset",
        type=str,
        default="transformed_data.csv",
        help="Path to dataset CSV"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="codegemma:7b",
        help="Model to use"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=200,
        help="Maximum samples to evaluate"
    )
    parser.add_argument(
        "--web-search",
        action="store_true",
        help="Enable web search tool"
    )
    parser.add_argument(
        "--doc-search",
        action="store_true",
        help="Enable documentation search tool"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare multiple models"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run single sample test"
    )

    args = parser.parse_args()

    if args.test:
        # Run single sample test
        test_single_sample()
    elif args.compare:
        # Compare models
        compare_models(
            dataset_path=args.dataset,
            max_samples=args.max_samples
        )
    else:
        # Single model evaluation
        evaluate_agent_v2(
            dataset_path=args.dataset,
            model_name=args.model,
            max_samples=args.max_samples,
            enable_web_search=args.web_search,
            enable_doc_search=args.doc_search,
            verbose=args.verbose
        )
