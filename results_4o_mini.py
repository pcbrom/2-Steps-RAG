import os
from dotenv import load_dotenv
import pandas as pd
import openai
from sentence_transformers import SentenceTransformer
import chromadb

# Specify the path to your .env file
dotenv_path = "/mnt/4d4f90e5-f220-481e-8701-f0a546491c35/arquivos/projetos/.env"

# Load the .env file
load_dotenv(dotenv_path=dotenv_path)

# Access and store the environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")

# Import model
excel_file = "experimental_design_plan_prompt_baseline.xlsx"
df = pd.read_excel(excel_file, decimal='.')
df = df[df['model'] == 'gpt-4o-mini-2024-07-18']
cols_to_fill = ['prompt', 'results', 'score']
df[cols_to_fill] = df[cols_to_fill].fillna('')
print(df)

# Load chromadd and ncm-all-data collection
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection(name="ncm-all-data")

# Load a pre-trained Portuguese sentence transformer model
# https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
model_sentence = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# Iterate over each row and make API call
model = 'gpt-4o-mini-2024-07-18'
for index, row in df.iterrows():
    try:
        prompt=row['prompt']
        
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

        response = openai.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": augmented_prompt}],
            temperature=float(row['temperature']),
            top_p=float(row['top_p'])
        )

        # Extract and store the generated text
        generated_text = response.choices[0].message.content
        df.loc[index, 'results'] = generated_text

        print(generated_text)

    except openai.OpenAIError as e:
        print(f"Error processing row {index}: {e}")
        df.loc[index, 'results'] = f"Error: {e}"  # Store the error message
    except Exception as e:
        print(f"Unexpected error processing row {index}: {e}")
        df.loc[index, 'results'] = f"Error: {e}"

# Save the updated DataFrame (optional)
df.to_csv(f"experimental_design_results_{df['model'].iloc[0]}.csv", index=False)
