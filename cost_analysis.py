import pandas as pd
import tiktoken
from tqdm import tqdm
import os
import json

# Load DataFrame
plan_file_path = ""
if not os.path.exists(plan_file_path):
    raise FileNotFoundError(f"Error: File {plan_file_path} not found.")

df = pd.read_excel(plan_file_path)

# Define Model Pricing Data
model_data = {
    "gpt-4o-mini-2024-07-18": {"encoding": "gpt-4o-mini-2024-07-18", "price_input": 0.15, "price_output": 0.075},
    "o1-mini-2024-09-12": {"encoding": "o1-mini-2024-09-12", "price_input": 1.10, "price_output": 0.55},
    "o3-mini-2025-01-31": {"encoding": "o3-mini-2025-01-31", "price_input": 1.10, "price_output": 0.55},
    "deepseek-reasoner": {"encoding": None, "price_input": 0.14, "price_output": 2.19},
    "gemini-2.0-flash-thinking-exp-01-21": {"encoding": None, "price_input": 0.0, "price_output": 0.0}
}

# Token Count Function
def count_tokens(text, model_name):
    """Counts tokens based on the selected model."""
    model_info = model_data.get(model_name)
    if not model_info:
        return 0

    if model_name == "deepseek-reasoner":
        return len(text) * 0.3 if isinstance(text, str) else 0

    encoding_name = model_info["encoding"]
    if not encoding_name:
        return 0

    try:
        encoding = tiktoken.encoding_for_model(encoding_name)
        return len(encoding.encode(text)) if isinstance(text, str) else 0
    except Exception:
        return 0

# Token Count
tqdm.pandas(desc="Calculating tokens")
df['augmented_prompt_tokens'] = df.progress_apply(lambda row: count_tokens(row['augmented_prompt'], row['model']), axis=1)
df['results_tokens'] = df['augmented_prompt_tokens'] * 3

# Aggregate tokens by model
model_tokens = df.groupby('model').agg({
    'augmented_prompt_tokens': 'sum',
    'results_tokens': 'sum'
})

# Calculate total tokens
total_augmented_prompt_tokens = model_tokens['augmented_prompt_tokens'].sum()
total_results_tokens = model_tokens['results_tokens'].sum()

print("\n🔹 Tokens per Model:")
for model, row in model_tokens.iterrows():
    print(f"Model: {model}")
    print(f"  - Augmented Prompt Tokens: {row['augmented_prompt_tokens']:,}")
    print(f"  - Results Tokens: {row['results_tokens']:,}")

print("\n🔹 Total Tokens:")
print(f"  - Total Augmented Prompt Tokens: {total_augmented_prompt_tokens:,}")
print(f"  - Total Results Tokens: {total_results_tokens:,}")
