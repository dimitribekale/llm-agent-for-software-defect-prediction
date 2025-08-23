import pandas as pd
from tqdm import tqdm



def transform_dataset(input_path: str, output_path: str = "transformed_data.csv") -> pd.DataFrame:
    """
    Transforms JSON dataset with faulty/fixed code pairs into binary classification format
    
    Args:
        input_path: Path to input JSON file
        output_path: Output path for transformed CSV (optional)

    Returns:
        Transformed DataFrame with code_samples and target columns
    """
    # Read original dataset
    df = pd.read_json(input_path, lines=True)
    
    # Select only relevant columns
    code_pairs = df[['faulty_code', 'fixed_code']].copy()
    
    # Melt to create long format
    melted_df = code_pairs.melt(
        value_name='code_samples',
        var_name='code_type',
        value_vars=['faulty_code', 'fixed_code']
    )
    
    # Create binary labels
    melted_df['target'] = melted_df['code_type'].apply(
        lambda x: 1 if x == 'faulty_code' else 0
    )
    
    # Clean up and shuffle
    final_df = melted_df.drop_duplicates(subset=['code_samples']) \
    .sample(frac=1, random_state=42) \
    .reset_index(drop=True)
    
    # Save to CSV if output path specified
    if output_path:
        final_df.to_csv(output_path, index=False)
    
    return final_df

# Usage example

if __name__ == "__main__":

    final_df = pd.read_json(r"Datasets\faulty_vs_fixed_dataset.jsonl", lines=True)
    final_df = pd.DataFrame(final_df)

    print("Transformed DataFrame:")
    print(final_df.head())