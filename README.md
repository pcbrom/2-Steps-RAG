# Classical Test Theory (CTT) for analyzing LLM responses

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

### 5. `results_4o_mini.py` and others `results_[model].py`

To ensure all experimental conditions are evaluated and results are captured, the `results_4o_mini.py` script should be updated to iterate through all rows of the experimental design plan. Additionally, error handling can be improved, and the script can be made more robust.  Below is an outline of the necessary modifications:

1.  **Remove Model Filtering:**  Eliminate the line that filters the DataFrame to only include a specific model (e.g., `df = df[df['model'] == 'GPT 4o-mini']`). This ensures that the script processes all models defined in your experimental design.

2.  **Comprehensive Error Handling:** Implement more detailed error handling to catch potential issues during API calls.  This includes logging errors and providing informative messages.

3.  **List of Models:**
    *   'Slim RAFT'
    *   'gpt-4o-mini-2024-07-18'
    *   'o1-mini-2024-09-12'
    *   'o3-mini-2025-01-31'
    *   'deepseek-reasoner'
    *   'gemini-2.0-flash-thinking-exp-01-21'

4.  **Saving All Results:**  The script should save the results for all models into a single CSV file.

Here's how the core loop of `results_[model].py` should be modified:

```python
# Iterate over each row and make API call
model = 'gpt-4o-mini-2024-07-18' # Change this to the desired openai model
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
        print(f"Model: {model}, Result: {generated_text}")
        df.loc[index, 'results'] = generated_text

    except openai.OpenAIError as e:
        print(f"Error processing row {index} for model {model}: {e}")
        df.loc[index, 'results'] = f"Error: {e}"  # Store the error message
    except Exception as e:
        print(f"Unexpected error processing row {index} for model {model}: {e}")
        df.loc[index, 'results'] = f"Error: {e}"

# Save the updated DataFrame (optional)
df.to_csv(f"experimental_design_results_{df['model'].iloc[0]}.csv", index=False)
```

This revised approach ensures that the script iterates through all rows in your experimental design, uses the fixed model for each API call, and saves all results into a single CSV file.  Remember to remove the line filtering by model to enable this comprehensive processing.