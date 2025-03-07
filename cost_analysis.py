import pandas as pd
import tiktoken
from tqdm import tqdm
import os

# Load DataFrame
plan_file_path = "augmented_prompt.csv"
if not os.path.exists(plan_file_path):
    raise FileNotFoundError(f"Error: File {plan_file_path} not found.")

df = pd.read_csv(plan_file_path)

# Define Model Pricing Data (Prices in dollars per million tokens)
model_data = {
    "gpt-4o-mini-2024-07-18": {"encoding": "gpt-4o-mini-2024-07-18", "price_input": 0.15, "price_output": 0.075},
    "o1-mini-2024-09-12": {"encoding": "o1-mini-2024-09-12", "price_input": 1.10, "price_output": 0.55},
    "Mistral-7B-Instruct-v0.3": {"encoding": "Mistral-7B-Instruct-v0.3", "price_input": 0.0, "price_output": 0.0},
    "deepseek-reasoner": {"encoding": None, "price_input": 0.14, "price_output": 2.19},
    "gemini-2.0-flash": {"encoding": "gemini-2.0-flash", "price_input": 0.1, "price_output": 0.4},
    "TeenyTinyLlama-160m-NCM-ft": {"encoding": None, "price_input": 0.0, "price_output": 0.0}
}

# Token Count Function
def count_tokens(text, model_name):
    """Counts tokens based on the selected model."""
    model_info = model_data.get(model_name)
    if not model_info:
        return 0

    if model_name == "deepseek-reasoner" or model_name == "gemini-2.0-flash":
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
df['results_tokens'] = df.progress_apply(lambda row: count_tokens(row['results'], row['model']), axis=1)

# Aggregate tokens by model
model_tokens = df.groupby('model').agg({
    'augmented_prompt_tokens': 'sum',
    'results_tokens': 'sum'
})

# Calculate total tokens in millions
total_augmented_prompt_tokens = model_tokens['augmented_prompt_tokens'].sum() / 1_000_000
total_results_tokens = model_tokens['results_tokens'].sum() / 1_000_000

# Prepare data for DataFrame
data = []
for model, row in model_tokens.iterrows():
    augmented_prompt_tokens_millions = row['augmented_prompt_tokens'] / 1_000_000
    results_tokens_millions = row['results_tokens'] / 1_000_000
    
    # Get model pricing
    if model in model_data:
        model_price_input = model_data[model]["price_input"]
        model_price_output = model_data[model]["price_output"]
    else:
        print(f"Warning: Model '{model}' not found in model_data. Skipping cost calculation for this model.")
        continue
    
    # Calculate cost
    augmented_prompt_cost = augmented_prompt_tokens_millions * model_price_input
    results_cost = results_tokens_millions * model_price_output
    total_cost = augmented_prompt_cost + results_cost
    
    data.append([
        model,
        augmented_prompt_tokens_millions,
        augmented_prompt_cost,
        results_tokens_millions,
        results_cost,
        total_cost
    ])

# Create DataFrame
results_df = pd.DataFrame(data, columns=[
    "Model",
    "Tokens Augmented Prompt (M)",
    "Cost Augmented Prompt ($)",
    "Tokens Results (M)",
    "Cost Results ($)",
    "Total Cost ($)"
])

# Print DataFrame
print("\n🔹 Costs (Dollars) and Tokens (Millions) by Model:")
print(results_df.to_string(index=False))

print("\n🔹 Totals:")
print(f"  - Total Tokens Augmented Prompt: {total_augmented_prompt_tokens:.3f}M")
print(f"  - Total Tokens Results: {total_results_tokens:.3f}M")

# Calculate total cost
total_cost_all_models = results_df["Total Cost ($)"].sum()

print(f"  - Total Cost (All Models): ${total_cost_all_models:.2f}")
