
import requests
import pandas as pd
from tqdm import tqdm
from prompts import AGENT_ONLY_EVAL_SYSTEM_PROMPT
from typing import Any, Dict, List
from agent.agent import DefectPredictionAgent
from sklearn.metrics import (accuracy_score,
                             precision_score,
                             f1_score,
                             classification_report,
                             recall_score)


OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODELS_TO_EVALUATE = ["deepseek-r1:7b"]


def query_ollama(prompt: str, model: str = None, temperature: float = 0.1, max_tokens: int = 1024):
    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False
    }
    response = requests.post(OLLAMA_API_URL, json=payload)
    response.raise_for_status()
    return response.json()["response"]

def evaluate_agent(
    agent: DefectPredictionAgent,
    dataset_path: str
) -> Dict[str, float]:
    """Runs evaluation on full dataset"""
    df = pd.read_csv(dataset_path)
    #y_true = df['target'].values
    y_true = df['target'].values[:200]  # Limit to first 200 samples for testing
    y_pred = []

    #for _, row in tqdm(df.iterrows(), total=len(df), desc="Samples", leave=False):
    for _, row in tqdm(df[:200].iterrows(), total=200, desc="Samples", leave=False):
        code = row['code_samples']
        pred, _ = agent.loop(code_snippet=code)
        y_pred.append(pred)
        print(f"Predictions: {y_pred}")

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "report": classification_report(y_true, y_pred, target_names=['Clean', 'Defective'], zero_division=0)
    }

def evaluate_models(dataset_path: str) -> pd.DataFrame:
    """Evaluate all models and return comparison DataFrame"""
    results = []
    
    for model in tqdm(MODELS_TO_EVALUATE, desc="Evaluating Models"):
        # Initialize agent with current model
        agent = DefectPredictionAgent(
            system_prompt= AGENT_ONLY_EVAL_SYSTEM_PROMPT,
            model_name=model  # Add this parameter to your __init__
        )
        
        # Run evaluation
        metrics = evaluate_agent(agent, dataset_path)
        metrics["model"] = model
        results.append(metrics)
    
    # Create formatted DataFrame
    df = pd.DataFrame(results).set_index("model")
    column_order = ["accuracy", "precision", "recall", "f1"]
    return df[column_order].sort_values("f1", ascending=False)


# === Example Usage ===

if __name__ == "__main__":

    comparison_df = evaluate_models("transformed_data.csv")
    print("\nModel Comparison Results:")
    print(comparison_df.to_markdown(floatfmt=".2f"))


    """agent = DefectPredictionAgent(system_prompt)
    # Example code snippet (Python)
    metrics = evaluate_agent(agent, "transformed_data.csv")

    print(f"Defect Detection Metrics:\n"
      f"Accuracy: {metrics['accuracy']:.2f}\n"
      f"F1-Score: {metrics['f1']:.2f}\n"
      f"Precision: {metrics['precision']:.2f}\n"
      f"Recall: {metrics['recall']:.2f}\n"
      f"Classification Report:\n{metrics['report']}\n")"""