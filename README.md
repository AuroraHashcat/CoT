# Causal-CoT: A Framework for Causally-Validated Chain-of-Thought Reasoning

This project implements the **Causal-CoT** framework, a system designed to enhance the reliability of Large Language Models (LLMs) by validating each step of their reasoning process against causal principles and external knowledge.

The framework's core loop identifies causal fallacies (e.g., confounding, spurious correlation) in real-time. When a flawed step is detected, a **reflection-and-regeneration** cycle is triggered to self-correct and build a more robust and trustworthy reasoning path.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Table of Contents

- [Core Design Principles](#core-design-principles)
- [Framework Architecture](#framework-architecture)
- [Supported Datasets](#supported-datasets)
- [Setup and Installation](#setup-and-installation)
- [Configuration](#configuration)
- [How to Run an Experiment](#how-to-run-an-experiment)
- [Evaluation Metrics](#evaluation-metrics)

## Core Design Principles

1.  **Reasoning as Hypothesis Testing:** An LLM's Chain-of-Thought is treated as a **Chain of Hypotheses**. Each step is a claim that is independently scrutinized.
2.  **Causal Validation:** Each step is passed to a **Knowledge Prober** that performs a causal analysis based on Judea Pearl's structural causal model framework, using ConceptNet to identify Causal Chains, Forks, and Colliders.
3.  **Reflective Self-Correction:** When a step is invalidated, the framework enters a **reflection loop**. The LLM is informed of its error and tasked with regenerating a new, more sound reasoning path.

## Framework Architecture

The system operates via a four-phase, iterative pipeline for each question: CoT Generation → Iterative Causal Probing → Reflection & Regeneration → Final Synthesis.

## Supported Datasets

The framework is configured to work with several standard reasoning benchmarks out-of-the-box. All listed datasets are verified to be available on the Hugging Face Hub. Each has a corresponding configuration file in the `configs/` directory.

| Config File                             | Dataset Name   | Hugging Face ID  | Task Type         |
| --------------------------------------- | -------------- | ---------------- | ----------------- |
| `dataset_commonsense_qa.json`           | CommonsenseQA  | `commonsense_qa` | Multiple Choice   |
| `dataset_arc_challenge.json`            | ARC-Challenge  | `ai2_arc`        | Multiple Choice   |
| `dataset_openbookqa.json`               | OpenBookQA     | `openbookqa`     | Multiple Choice   |
| `dataset_piqa.json`                     | PIQA           | `piqa`           | Multiple Choice   |
| `dataset_siqa.json`                     | SocialIQA      | `social_i_qa`    | Multiple Choice   |
| `dataset_boolq.json`                    | BoolQ          | `boolq`          | Yes/No Reasoning  |
| `dataset_gsm8k.json`                    | GSM8K          | `gsm8k`          | Math Word Problem |


## Setup and Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/your-username/causal-cot-final.git
    cd causal-cot-final
    ```

2.  **Create and Activate a Python Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    A `requirements.txt` file is provided.
    ```bash
    pip install -r requirements.txt
    ```
    *Note: For local model usage, you will need to install `torch`, `transformers`, and `accelerate`.*

## Configuration

All experiments are driven by JSON configuration files in the `configs/` directory.

### 1. Model Configuration
Choose between `api` or `local` mode by pointing to the correct file.
-   **API Mode (`configs/model_api.json`):** Set the `api_key_env` to the name of the environment variable holding your API key (e.g., `"DEEPINFRA_API_KEY"`).
-   **Local Mode (`configs/model_local.json`):** Set the `path` to the directory of your locally saved Hugging Face model.

### 2. Dataset Configuration
-   Select a dataset by providing the path to its config file (e.g., `configs/dataset_boolq.json`).
-   The `hf_id` field must match the dataset's ID on the Hugging Face Hub.
-   The `hf_config` field should **only** be present if the dataset has multiple configurations (like ARC). Otherwise, it should be omitted from the JSON file.

## How to Run an Experiment

The `run_experiment.py` script orchestrates the entire process.

### Step 1: Set Environment Variable (API Mode Only)
```bash
# Example for DeepInfra
export DEEPINFRA_API_KEY="your_actual_api_key_here"
```

### Step 2: Execute the Script
Provide the paths to your desired model and dataset configurations.

#### Example: Running BoolQ with an API Model
```bash
python run_experiment.py --model_config configs/model_api.json --dataset_config configs/dataset_boolq.json
```

#### Example: Running GSM8K with a Local Model
```bash
python run_experiment.py --model_config configs/model_local.json --dataset_config configs/dataset_gsm8k.json
```

### Step 3: View Results
-   The console will show a verbose, real-time log of the Causal-CoT process for each sample.
-   Upon completion, a summary of the final metrics is printed.
-   A detailed JSON file is saved in the `results/` directory, named after the experiment configuration (e.g., `results/BoolQ_model_api.json`).

## Evaluation Metrics
-   **`accuracy`**: Final task accuracy after corrections.
-   **`causal_metrics`**:
    -   `intervention_rate`: Average number of self-corrections per problem.
    -   `reasoning_fidelity`: Proportion of initial CoT steps that were valid.
    -   `fallacy_rate`: Percentage of steps identified as a causal fallacy.
    -   `avg_correction_depth_percent`: Average point (%) in the CoT where the first error occurred.
    -   `causal_structure_distribution`: Frequency count of identified causal structures.