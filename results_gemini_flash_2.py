import os
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm
import time
from google import genai  # Novo import
from google.genai import types  # Novo import

# Specify the path to your .env file
dotenv_path = "/mnt/4d4f90e5-f220-481e-8701-f0a546491c35/arquivos/projetos/.env"

# Load the .env file
load_dotenv(dotenv_path=dotenv_path)

# Access and store the environment variable
google_api_key = os.getenv("GOOGLE_API_KEY")
model = 'gemini-2.0-flash'

# Config client
client = genai.Client(api_key=google_api_key)

# Import model
csv_file = "backup/augmented_prompt_common_rag.csv"
df = pd.read_csv(csv_file, decimal='.', sep=',', encoding='utf-8')
df = df[df['model'] == model]
cols_to_fill = ['results', 'score']
df[cols_to_fill] = df[cols_to_fill].fillna('')
print(df)

# Iterate over each row and make API call
output_filename = f"experimental_design_results_{model}.csv"
for index, row in tqdm(df.iterrows(), total=len(df), desc="Processando"):
    if index % 100 == 0 and index != 0:
        print("min. pause...")
        time.sleep(60)
    try:
        augmented_prompt = row['augmented_prompt']

        generation_config = types.GenerateContentConfig(
            temperature=float(row['temperature']),
            top_p=float(row['top_p'])
        )

        response = client.models.generate_content(
            model=model,
            contents=augmented_prompt,
            config=generation_config
        )
        
        # Extract and store the generated text
        generated_text = response.text if hasattr(response, 'text') else "Nenhuma resposta gerada"
        df.loc[index, 'results'] = generated_text

    except Exception as e:
        print(f"Erro ao processar a linha {index}: {e}")
        df.loc[index, 'results'] = f"Erro: {e}" # Store the error message

# Save the updated DataFrame (optional)
df.to_csv(output_filename, index=False)
