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
df = pd.read_excel(excel_file, decimal='.')
df = df[df['model'] == 'GPT 4o-mini']
cols_to_fill = ['prompt', 'results', 'score']
df[cols_to_fill] = df[cols_to_fill].fillna('')
print(df)

# Iterate over each row and make API call
model = 'gpt-4o-mini-2024-07-18'
for index, row in df.iterrows():
    try:
        prompt=row['prompt']
        response = openai.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=float(row['temperature']),
            top_p=float(row['top_p'])
        )

        # Extract and store the generated text
        generated_text = response.choices[0].message.content
        print(generated_text)
        df.loc[index, 'results'] = generated_text

    except openai.OpenAIError as e:
        print(f"Error processing row {index}: {e}")
        df.loc[index, 'results'] = f"Error: {e}"  # Store the error message
    except Exception as e:
        print(f"Unexpected error processing row {index}: {e}")
        df.loc[index, 'results'] = f"Error: {e}"

# Save the updated DataFrame (optional)
df.to_csv(f"experimental_design_results_{df['model'].iloc[0]}.csv", index=False)
