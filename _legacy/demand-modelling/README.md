# PyBLP Tests & Discrete Choice Model Comparisons

This repository contains implementations and comparisons of various discrete choice models, specifically focusing on the progression from Simple Logit to Nested Logit and finally to the Random Coefficients Logit (BLP) model.

## Project Structure

The project is organized into three main modules, each corresponding to a model type:

### 1. Logit Model (`logit/`)
Implementation of the standard Multinomial Logit model.
- **Key Feature**: Demonstrates the Independence of Irrelevant Alternatives (IIA) property.
- **Visualizations**: Elasticity heatmaps showing identical cross-elasticities.

### 2. Nested Logit Model (`nested_logit/`)
Implementation of the Nested Logit model which relaxes IIA for products within the same nest.
- **Key Feature**: Allows for correlation in unobserved utility among products in the same group.
- **Visualizations**: Elasticity heatmaps highlighting higher substitution patterns within nests.

### 3. BLP Model (`blp/`)
Implementation of the Berry, Levinsohn, and Pakes (1995) Random Coefficients Logit model.
- **Key Feature**: Allows for flexible substitution patterns based on product characteristics and consumer demographics.
- **Visualizations**: 
    - Elasticity comparisons (Logit vs BLP).
    - Diversion ratios.
    - Merger simulations.

## Documentation

The repository includes detailed markdown files explaining the theory and implementation steps:
- `01_logit.md`: Introduction to the Logit model.
- `02_logit_supply.md`: Supply-side estimation for Logit.
- `03_nested_logit.md`: Nested Logit theory and implementation.
- `04_blp_model.md`: The BLP model framework.
- `05_blp_estimation.md`: Estimation details for BLP (GMM, contraction mapping).
- `06_blp_simulation.md`: Post-estimation simulation (mergers, etc.).
- `07_blp_fake_case_study.md`: A synthetic case study.

## Usage

Each module contains a `main.py` file that runs the estimation and generates visualizations.

To run the BLP model example:
```bash
python blp/main.py
```

To run the Logit model example:
```bash
python logit/main.py
```

To run the Nested Logit model example:
```bash
python nested_logit/main.py
```

## Outputs

Generated figures and tables are saved in the `outputs/` directory.
