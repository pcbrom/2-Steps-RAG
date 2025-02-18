import os
from dotenv import load_dotenv
import pandas as pd
import openai

# Specify the path to your .env file
dotenv_path = "/mnt/4d4f90e5-f220-481e-8701-f0a546491c35/arquivos/projetos/.env"

# Load the .env file
load_dotenv(dotenv_path=dotenv_path)

# Access and store the environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")

# Import model
excel_file = "experimental_design_plan.xlsx"
df = pd.read_excel(excel_file)
df = df[df['model'] == 'GPT 4o-mini']
print(df)

# Iterate over each row and make OpenAI API call
model = 'gpt-4o-mini-2024-07-18'
for index, row in df.iterrows():
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": row['prompt']}],
            temperature=row['temperature'],
            top_p=row['top_p'],
            top_k=row['top_k']
        )

        # Extract and store the generated text
        generated_text = response.choices[0].message.content
        df.loc[index, 'results'] = generated_text

    except openai.error.OpenAIError as e:
        print(f"Error processing row {index}: {e}")
        df.loc[index, 'results'] = f"Error: {e}"  # Store the error message
    except Exception as e:
        print(f"Unexpected error processing row {index}: {e}")
        df.loc[index, 'results'] = f"Error: {e}"

# Save the updated DataFrame (optional)
df.to_csv("experimental_design_results_4o_mini.csv", index=False)
