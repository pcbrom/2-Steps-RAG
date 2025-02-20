import pandas as pd
import matplotlib.pyplot as plt
import tiktoken
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Load chromadd and ncm-all-data collection
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection(name="ncm-all-data")

# Load a pre-trained Portuguese sentence transformer model
# https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
model_sentence = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# File paths
plan_file_path = "experimental_design_plan_prompt_baseline.xlsx"
df = pd.read_excel(plan_file_path)

# Define a dictionary to map models to their corresponding encodings and prices
# https://platform.openai.com/docs/pricing
# https://api-docs.deepseek.com/quick_start/pricing and https://api-docs.deepseek.com/quick_start/token_usage?
# https://ai.google.dev/gemini-api/docs/pricing?hl=pt-br

model_data = {
    "gpt-4o-mini-2024-07-18": {"encoding": "gpt-4o-mini-2024-07-18", "price_input": 0.15, "price_output": 0.075},
    "o1-mini-2024-09-12": {"encoding": "o1-mini-2024-09-12", "price_input": 1.10, "price_output": 0.55},
    "o3-mini-2025-01-31": {"encoding": "o3-mini-2025-01-31", "price_input": 1.10, "price_output": 0.55},
    "deepseek-reasoner": {"encoding": None, "price_input": 0.14, "price_output": 2.19},
    "gemini-2.0-flash-thinking-exp-01-21": {"encoding": None, "price_input": 0.0, "price_output": 0.0}  # No encoding available
}

def count_tokens(text, model_name):
    """Counts the number of tokens in a given text based on the model."""
    try:
        model_info = model_data.get(model_name)
        if model_info is None:
            return 0

        if model_name == "deepseek-reasoner":
            # 1 English character ≈ 0.3 token
            return len(text) * 0.3 if isinstance(text, str) else 0

        encoding_name = model_info["encoding"]
        if encoding_name is None:
            return 0

        try:
            encoding = tiktoken.encoding_for_model(encoding_name)
        except Exception as e:
            print(f"Warning: Could not automatically map {encoding_name} to a tokenizer. Skipping token counting for this model.")
            return 0

        if isinstance(text, str):
            return len(encoding.encode(text))
        else:
            return 0
    except Exception as e:
        # Suppress the error message
        return 0

def create_augmented_prompt(prompt):
    """Creates an augmented prompt using RAG."""
    prompt_embeddings = model_sentence.encode(prompt)
    results = collection.query(
        query_embeddings=[prompt_embeddings],
        n_results=10,
        include=["documents", "embeddings", "distances"]
    )

    context = "\n".join(results['documents'][0])

    augmented_prompt = f"""
    Você é um assistente especializado em responder de forma objetiva e clara às perguntas com base em informações relevantes extraídas de uma base de conhecimento. 
    
    Abaixo está um contexto relevante recuperado da base de dados:

    Contexto:
    {context}

    Com base nessas informações, responda à seguinte pergunta de forma clara e objetiva:

    Pergunta:
    {prompt}

    Se o contexto não contiver informações suficientes para uma resposta precisa, indique que não há dados suficientes para responder com segurança.
    """
    return augmented_prompt

# Calculate augmented prompts and their tokens
tqdm.pandas(desc="Calculating augmented prompts")
df['augmented_prompt'] = df['prompt'].progress_apply(create_augmented_prompt)

# Calculate token counts
tqdm.pandas(desc="Calculating prompt tokens")
df['prompt_tokens'] = df.progress_apply(lambda row: count_tokens(row['prompt'], row['model']), axis=1)

tqdm.pandas(desc="Calculating augmented prompt tokens")
df['augmented_prompt_tokens'] = df.progress_apply(lambda row: count_tokens(row['augmented_prompt'], row['model']), axis=1)

# Assuming results tokens are double the prompt tokens.  This might need adjustment.
df['results_tokens'] = df['augmented_prompt_tokens'] * 3

# Calculate costs
def calculate_cost(row, token_type):
    model_name = row['model']
    model_info = model_data.get(model_name)
    if not model_info:
        return 0

    if token_type == 'prompt':
        price_per_1m_tokens = model_info['price_input']
        tokens = row['augmented_prompt_tokens']
    else:
        price_per_1m_tokens = model_info['price_output']
        tokens = row['results_tokens']

    return (tokens / 1000000) * price_per_1m_tokens

tqdm.pandas(desc="Calculating costs")
df['prompt_cost'] = df.progress_apply(lambda row: calculate_cost(row, 'prompt'), axis=1)
df['results_cost'] = df.progress_apply(lambda row: calculate_cost(row, 'results'), axis=1)
df['total_cost'] = df['prompt_cost'] + df['results_cost']

# Aggregate costs by model
model_costs = df.groupby('model').agg({
    'prompt_tokens': 'sum',
    'augmented_prompt_tokens': 'sum',
    'results_tokens': 'sum',
    'prompt_cost': 'sum',
    'results_cost': 'sum',
    'total_cost': 'sum'
})

# Print costs for each model
print("Costs per model:")
for model, row in model_costs.iterrows():
    print(f"Model: {model}, Total Cost: ${row['total_cost']:.2f}, Augmented Prompt Tokens: {row['augmented_prompt_tokens']}, Results Tokens: {row['results_tokens']}")

# Total experiment cost
total_experiment_cost = df['total_cost'].sum()
print(f"\nTotal cost of the experiment: ${total_experiment_cost:.2f}")

# Save the DataFrame to a CSV file
df.to_csv("cost_analysis_results.csv", index=False)
print("Results saved to cost_analysis_results.csv")
