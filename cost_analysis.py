import pandas as pd
import tiktoken
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import openai
from dotenv import load_dotenv
import os
import json

# Load API Key from .env file
dotenv_path = "/mnt/4d4f90e5-f220-481e-8701-f0a546491c35/arquivos/projetos/.env"
load_dotenv(dotenv_path=dotenv_path)

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("Error: OPENAI_API_KEY not found. Check your .env file.")

openai.api_key = openai_api_key
model = 'gpt-4o-mini-2024-07-18'

# Connect to ChromaDB
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection(name="ncm-all-data")

# Load Sentence Transformer model
model_sentence = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# Load DataFrame
plan_file_path = "experimental_design_plan_prompt_baseline.xlsx"
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

# Function to Generate Augmented Prompt Using RAG
def create_augmented_prompt(prompt):
    """Generates an augmented prompt using RAG (Retrieval-Augmented Generation)."""
    filters = None
    try:
        # Extract metadata using OpenAI
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": f"Tente identificar e extrair os possíveis metadados NCM, rótulo, Item e Produto da seguinte pergunta. Retorne um JSON. Se não encontrar a informação retorne null. Importante rótulo é igual ao produto, se encontrar um deles preenher automaticamente o outro com o conteúdo. Pergunta: {prompt}"}
            ],
            temperature=0.0
        )

        metadata_str = response.choices[0].message.content.strip()
        if metadata_str.startswith("```json") and metadata_str.endswith("```"):
            metadata_str = metadata_str.strip("```").strip()
        if metadata_str.startswith("json"):
            metadata_str = metadata_str[4:].lstrip()
        metadata = json.loads(metadata_str)

        filters = {}
        if isinstance(metadata, dict):
            filters = {k: v for k, v in metadata.items() if v is not None}


    except (openai.OpenAIError, json.JSONDecodeError) as e:
        print(f"Error extracting metadata: {e}")
    
    # Retrieve context from ChromaDB
    prompt_embeddings = model_sentence.encode([prompt], convert_to_tensor=True).tolist()
    results = None
    results_no_filters = collection.query(
            query_embeddings=prompt_embeddings,
            n_results=5,
            include=["documents", "embeddings", "distances"]
        )
    
    if "documents" not in results_no_filters or not results_no_filters["documents"]:
        context = "\nNo relevant information found."
    else:
        context = "\n".join(results_no_filters["documents"][0])

    if filters:
        # Handle multiple filters by iterating through them
        for key, value in filters.items():
            temp_results = collection.query(
                query_embeddings=prompt_embeddings,
                n_results=5,
                where={key: value},  # Apply filter for each key-value pair
                include=["documents", "embeddings", "distances"]
            )
            if results is None:
                results = temp_results
            else:
                # Combine results (you might need a more sophisticated combination logic)
                if "documents" in temp_results and temp_results["documents"]:
                    results["documents"][0].extend(temp_results["documents"][0])
    
    if "documents" not in results or not results["documents"]:
        context += "\nNo relevant information found."
    else:
        context += "\n".join(results["documents"][0])

    augmented_prompt = f"""
    Você é um assistente especializado em responder de forma objetiva e clara às perguntas com base em informações relevantes extraídas de uma base de conhecimento. 

    Contexto relevante recuperado:

    {context}

    Pergunta:
    {prompt}

    Se o contexto não contiver informações suficientes, indique que não há dados suficientes para responder com segurança.
    """
    return augmented_prompt

tqdm.pandas(desc="Generating augmented prompts")
tmp = df['prompt'].head(196).progress_apply(create_augmented_prompt)

# Apply Augmented Prompt to DataFrame
df['augmented_prompt'] = tmp

# Token Count and Cost Calculation
tqdm.pandas(desc="Calculating tokens")
df['prompt_tokens'] = df.progress_apply(lambda row: count_tokens(row['prompt'], row['model']), axis=1)
df['augmented_prompt_tokens'] = df.progress_apply(lambda row: count_tokens(row['augmented_prompt'], row['model']), axis=1)

# Assuming output tokens are 3x the augmented prompt tokens
df['results_tokens'] = df['augmented_prompt_tokens'] * 3

# Cost Calculation
def calculate_cost(row, token_type):
    model_info = model_data.get(row['model'])
    if not model_info:
        return 0

    tokens = row['augmented_prompt_tokens'] if token_type == 'prompt' else row['results_tokens']
    price_per_1m_tokens = model_info['price_input'] if token_type == 'prompt' else model_info['price_output']

    return (tokens / 1_000_000) * price_per_1m_tokens

tqdm.pandas(desc="Calculating costs")
df['prompt_cost'] = df.progress_apply(lambda row: calculate_cost(row, 'prompt'), axis=1)
df['results_cost'] = df.progress_apply(lambda row: calculate_cost(row, 'results'), axis=1)
df['total_cost'] = df['prompt_cost'] + df['results_cost']

# Aggregate Costs by Model
model_costs = df.groupby('model').agg({
    'prompt_tokens': 'sum',
    'augmented_prompt_tokens': 'sum',
    'results_tokens': 'sum',
    'prompt_cost': 'sum',
    'results_cost': 'sum',
    'total_cost': 'sum'
})

print("\n🔹 Cost per Model:")
for model, row in model_costs.iterrows():
    print(f"Model: {model}, Total Cost: ${row['total_cost']:.2f}")

# Save Results
df.to_csv("cost_analysis_results.csv", index=False)
print("Results saved in cost_analysis_results.csv")