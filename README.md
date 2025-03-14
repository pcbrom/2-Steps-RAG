# Two-Step Retrieval-Augmented Generation for Metadata Filtering and Large Language Model Evaluation with Bootstrap Multivariate Linear Mixed Model with Random Intercepts for Prompts

## Experimental Design

This repository contains a Python script (`experimental_design.py`) that generates an experimental design plan for experiments with large language models (LLMs).  The script defines control variables such as `temperature`, `top_p`, and `model`, creating all possible combinations of these parameters using the `itertools.product` function. This design is then saved to an Excel file (`experimental_design_plan.xlsx`).

The Excel file includes columns for each control parameter, along with additional columns for prompts, results, and a score, which are initially left empty to be filled in during the experiment.  Furthermore, the script performs a power analysis using `statsmodels.stats.power.FTestAnovaPower` to determine the necessary number of replicates per condition, ensuring sufficient statistical power for the planned analysis. This facilitates the organization and execution of systematic experiments with LLMs, allowing for a more complete and statistically robust analysis of the results.

## Sample Size Determination

### 1. Defining the Minimal Effect Size of Interest (Cohen's f)

Cohen's f is defined as:

\[
f = \sqrt{\frac{\sum_{i=1}^{k} (\mu_i - \mu)^2}{k\,\sigma^2}},
\]

where:

- \( k \) is the number of levels of the factor,
- \(\mu_i\) are the means of each group,
- \(\mu\) is the overall mean, and
- \(\sigma^2\) is the within-group variance.

**Conventional Guidelines (Cohen, 1988):**

- **Small effect:** \( f \approx 0.10 \)
- **Medium effect:** \( f \approx 0.25 \)
- **Large effect:** \( f \approx 0.40 \)

Define the smallest effect size that you consider relevant for your experiment.

---

### 2. Setting the Significance Level (α) and Desired Power (1-β)

- **Significance Level (α):** Typically, \( \alpha = 0.05 \) is used.
- **Desired Power (1-β):** Common choices are 80% (\(1-\beta = 0.80\)) or 90% (\(1-\beta = 0.90\)).

These parameters determine:

- The probability of committing a Type I error (false positive) (α), and
- The probability of detecting the effect when it exists (1-β).

---

### 3. Using the Noncentrality Parameter Relationship

For F-tests (ANOVA), the noncentrality parameter (\(\lambda\)) is given by:

\[
\lambda = n \times f^2,
\]

where:

- \( n \) is the number of replicates per condition, and
- \( f \) is the effect size (Cohen's f).

To achieve the desired power, the following condition must be met:

\[
P\Big(F(v_1,\,v_2,\,\lambda) > F_{\alpha}(v_1,\,v_2)\Big) \geq 1-\beta,
\]

where:

- \( F(v_1,\,v_2,\,\lambda) \) is the noncentral F-distribution with degrees of freedom \( v_1 \) and \( v_2 \) and noncentrality parameter \(\lambda\),
- \( F_{\alpha}(v_1,\,v_2) \) is the critical F value at the significance level \(\alpha\).

This equation is usually solved numerically using specialized software.

---

### 4. Python Script Example

Below is an example Python script that uses the `statsmodels` package to calculate the required sample size per group for an ANOVA with one factor. This example assumes you are testing a factor (e.g., "model") with 5 levels.

```python
import numpy as np
from statsmodels.stats.power import FTestAnovaPower

# Experiment parameters:
effect_size = 0.25  # Medium effect size (Cohen's f)
alpha = 0.05        # Significance level
power = 0.80        # Desired power (80%)
k_groups = 5        # Number of groups (e.g., for the 'model' factor with 5 levels)

# Initialize the power analysis object for ANOVA
analysis = FTestAnovaPower()

# Calculate the sample size per group (replicates per condition)
n = analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power, k_groups=k_groups)

print(f"Number of replicates per condition: {np.ceil(n)}")
# Number of replicates per condition: 196.0
```

## Populating ChromaDB with `populate_chromadb.py`

This script performs the following actions:

1. **Data Import:** Reads NCM (Nomenclatura Comum do Mercosul) data from a CSV file (`data/BaseDESC_NCM.csv`) using the `polars` library. This file originates from https://github.com/vinidiol/descmerc/blob/main/BaseDESC_NCM.zip.
2. **Data Transformation:** Concatenates relevant information from each row (NCM, Rótulo, Item, Produto) into a single text string. This is done in parallel using `multiprocessing` to increase efficiency.
3. **Sentence Embedding:** Utilizes a pre-trained Portuguese sentence transformer model (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`) to generate embeddings for each text string. These embeddings represent the semantic meaning of the text.
4. **ChromaDB Setup:**

   * **Removes Existing Database (if any):** Deletes the `chroma_db` directory if it exists, to ensure a clean start.
   * **Creates Persistent Client:** Initializes a persistent ChromaDB client, storing the database in the `chroma_db` directory.
   * **Creates Collection:** Creates or retrieves a collection named `ncm-all-data` within the ChromaDB database.
5. **Data Population:**

   * **Batches Data:** Divides the documents and embeddings into batches to optimize the insertion process.
   * **Adds to ChromaDB:** Adds the documents, their corresponding embeddings, and unique IDs to the ChromaDB collection. This allows for efficient semantic search and retrieval of NCM data.

## Augmenting Prompts with Context

This process uses the methodology described in the `augmented_prompt.py` script to enrich prompts with relevant contextual information, improving the quality of responses generated by language models. The `augmented_prompt.py` script performs the following steps:

1. **Metadata Extraction:** Uses the OpenAI API (any free Hugging Face model can be used) to identify and extract relevant metadata (NCM, Label, Item, Product) from the original question. This metadata is used to refine the search in ChromaDB, acting as filters. The extraction is done through a call to the OpenAI API, which returns a JSON containing the identified metadata. If no metadata is found, the corresponding value will be `null`.
2. **Context Retrieval:** Queries ChromaDB, a vector database, to retrieve relevant contextual information. The retrieval occurs in two phases:

   * **Initial Search (Without Filters):** An initial search is performed using the original prompt, without applying any filters. This ensures that relevant information is retrieved even if the metadata is not present or is inaccurate. The 5 most relevant documents are retrieved.
   * **Search with Filters:** After extracting the metadata, additional searches are performed, applying the extracted metadata as filters. Each key-value pair in the metadata is used as a filter in the query to ChromaDB. The results of these filtered searches are combined with the results of the initial search. If there are multiple filters, the results of each filter are added, allowing for a combination of specific contextual information.

   The `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` model is used to generate embeddings (vector representations) of the prompt. These embeddings allow ChromaDB to perform a semantic search, finding documents that are semantically similar to the prompt, even if they do not contain the same words.
3. **Augmented Prompt Generation:** Combines the original prompt with the context retrieved from ChromaDB to create an augmented prompt. This augmented prompt provides the language model with additional information that helps it generate more accurate and relevant responses.

The augmented prompt has the following format:

```
Você é um assistente especializado em responder de forma objetiva e clara às perguntas com base em informações relevantes extraídas de uma base de conhecimento.

Contexto relevante recuperado:

{contexto}

Pergunta:
{prompt}

Se o contexto não contiver informações suficientes, indique que não há dados suficientes para responder com segurança.
```

Where:

* `contexto` is the information retrieved from ChromaDB.
* `prompt` is the original question.

## Cost Analysis (`cost_analysis.py`)

This section describes the `cost_analysis.py` script, which performs a cost analysis for different language models. The script performs the following steps:

1. **Loads Data:** Reads data from a CSV file (`augmented_prompt.csv`) containing prompts, model responses, and model configurations. This file is generated by the `augmented_prompt.py` script.
2. **Configures Model Information:** Sets up information about the models, including pricing per token and tokenization methods. Model prices are defined in the `model_data` dictionary.
3. **Calculates Token Usage:** Calculates the number of tokens used by augmented prompts (`augmented_prompt`) and model responses (`results`). It uses the `count_tokens` function to perform the counting, which adapts to different tokenization methods depending on the model.
4. **Calculates Costs:** Determines the total cost of using each model based on token usage and the defined prices. The cost is calculated separately for the tokens of the augmented prompts and for the tokens of the responses, and then summed to obtain the total cost per model.
5. **Aggregates Costs:** Groups the costs by model to provide a summary of expenses. Calculates the total cost for all models.
6. **Prints Results:** Displays the cost analysis results for each model, including the number of tokens used, the cost per token type (prompt and response), and the total cost. It also displays the totals of tokens and costs for all models.

For example, a sample output might look like this:

```
Costs (Dollars) and Tokens (Millions) by Model:
                     Model  Tokens Augmented Prompt (M)  Cost Augmented Prompt ($)  Tokens Results (M)  Cost Results ($)  Total Cost ($)
  Mistral-7B-Instruct-v0.3                     0.000000                   0.000000                 0.0               0.0        0.000000
TeenyTinyLlama-160m-NCM-ft                     0.000000                   0.000000                 0.0               0.0        0.000000
             deepseek-chat                     0.651154                   0.091162                 0.0               0.0        0.091162
          gemini-2.0-flash                     0.651154                   0.065115                 0.0               0.0        0.065115
    gpt-4o-mini-2024-07-18                     0.634707                   0.095206                 0.0               0.0        0.095206
        o1-mini-2024-09-12                     0.634707                   0.698178                 0.0               0.0        0.698178

🔹 Totals:
  - Total Tokens Augmented Prompt: 2.572M
  - Total Tokens Results: 0.000M
  - Total Cost (All Models) = (Input) $0.95 + (Output, max 3 times Input) $0.95 \leq $3.80
```

## About `results_4o_mini.py` and other `results_[model].py` scripts

To ensure all experimental conditions are evaluated and results are captured, the `results_4o_mini.py` script (and its similar `results_[model].py` scripts) should be updated to iterate through all rows of the experimental design plan. Additionally, error handling can be improved, and the script can be made more robust. Below is an outline of the necessary modifications:

1. **Remove Model Filtering:** Eliminate the line that filters the DataFrame to include only a specific model (e.g., `df = df[df['model'] == 'gpt-4o-mini-2024-07-18']`). This ensures that the script processes all models defined in your experimental design. The script should process all models defined in your experimental design file.
2. **Comprehensive Error Handling:** Implement more detailed error handling to catch potential issues during API calls. This includes logging errors and providing informative messages, including the model name, prompt, and error details. Consider using try-except blocks to catch `openai.OpenAIError` and other potential exceptions. Store the error messages in the 'results' column of the DataFrame.
3. **Model Iteration:** The script should be able to iterate over the models specified in the experimental design file. The `model` variable in the loop should dynamically pick up the model name from the DataFrame row.
4. **List of Supported Models:** The script should be compatible with the following models (but not limited to):
   * Mistral-7B-Instruct-v0.3: Locally generated results. 
   * TeenyTinyLlama-160m-NCM-ft: Locally generated results. 
   * deepseek-chat: API generated results.
   * gemini-2.0-flash: API generated results.
   * gpt-4o-mini-2024-07-18: API generated results.
   * o1-mini-2024-09-12: API generated results. Default values are fixed - Temperature and top_p cannot be changed: `temperature = 1.0` and `top_p = 1.0`)
5. **Saving All Results:** The script should save the results for all models into a single CSV file. The CSV filename should indicate that it contains results from multiple models (e.g., `experimental_design_results_all_models.csv`). Ensure the `index=False` argument is used when saving the CSV to avoid including the DataFrame index as a column.
6. **Retrieval-Augmented Generation Prompt Augmentation:** The script uses prompt augmentation with context retrieved from ChromaDB. Ensure that the ChromaDB collection name (`ncm-all-data`) is correctly specified and that the `populate_chromadb.py` script has been run to populate the database.
7. **Temperature and Top_p:** The script reads `temperature` and `top_p` values from the experimental design file. Ensure these columns exist and contain valid numerical values.
8. **API Key:** The script uses the OpenAI API key stored in a `.env` file. Ensure the `.env` file exists and contains the `OPENAI_API_KEY` variable.

Here's how the core loop of `results_[model].py` should be modified:

```python
# Iterate over each row and make API call
model = 'gpt-4o-mini-2024-07-18'
output_filename = f"experimental_design_results_{model}.csv"
for index, row in df.iterrows():
    try:
        augmented_prompt = row['augmented_prompt']

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
        print(f"Error processing row {index} for model {model}: {e}")
        df.loc[index, 'results'] = f"Error: {e}"  # Store the error message
    except Exception as e:
        print(f"Unexpected error processing row {index} for model {model}: {e}")
        df.loc[index, 'results'] = f"Error: {e}"

# Save the updated DataFrame (optional)
df.to_csv(output_filename, index=False)
```

This revised approach ensures that the script iterates through all rows in your experimental design, uses the fixed model for each API call, and saves all results into a single CSV file.  Remember to remove the line filtering by model to enable this comprehensive processing.
