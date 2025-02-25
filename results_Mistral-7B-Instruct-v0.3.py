import os
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
# huggingface-cli download mistralai/Mistral-7B-Instruct-v0.3 --local-dir /mnt/4d4f90e5-f220-481e-8701-f0a546491c35/arquivos/hf_models/mistralai/Mistral-7B-Instruct-v0.3 --local-dir-use-symlinks False

# Specify the path to your .env file
dotenv_path = "/mnt/4d4f90e5-f220-481e-8701-f0a546491c35/arquivos/projetos/.env"

# Load the .env file
load_dotenv(dotenv_path=dotenv_path)

# Define the local model path
model_path = "/mnt/4d4f90e5-f220-481e-8701-f0a546491c35/arquivos/hf_models/mistralai/Mistral-7B-Instruct-v0.3"
model_name = "Mistral-7B-Instruct-v0.3"

# Check if the model directory exists
if not os.path.exists(model_path):
    raise OSError(f"Model directory does not exist: {model_path}. Download the model first.")

# Load the CSV file
csv_file = "augmented_prompt_2step_rag.csv"
df = pd.read_csv(csv_file, decimal='.', sep=',', encoding='utf-8')
df = df[df['model'] == model_name]
cols_to_fill = ['results', 'score']
df[cols_to_fill] = df[cols_to_fill].fillna('')
print(df)

# Load the local model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Define the text generation pipeline
llm_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0
)

# Iterate over DataFrame and generate responses
for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    if index % 100 == 0 and index != 0:
        print("min. pause...")
        time.sleep(60)

    try:
        augmented_prompt = row['augmented_prompt']

        response = llm_pipeline(
            augmented_prompt,
            do_sample=True,
            temperature=float(row['temperature']),
            top_p=float(row['top_p']),
            max_new_tokens=200
        )[0]['generated_text']

        df.loc[index, 'results'] = response

    except Exception as e:
        print(f"Unexpected error at row {index}: {e}")
        df.loc[index, 'results'] = f"Error: {e}"

# Save the updated DataFrame
output_filename = f"results_{model_name}.csv"
df.to_csv(output_filename, index=False)

print(f"Completed. Results saved in {output_filename}")
