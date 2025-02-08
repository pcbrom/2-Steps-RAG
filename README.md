# Classical Test Theory (CTT) for analyzing LLM responses

## Experimental Design

This repository contains a Python script (`experimental_design.py`) that generates an experimental design plan for experiments with large language models (LLMs).  The script defines control variables such as `temperature`, `top_p`, `top_k`, and `model`, creating all possible combinations of these parameters using the `itertools.product` function. This design is then saved to an Excel file (`experimental_design_plan.xlsx`).

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
