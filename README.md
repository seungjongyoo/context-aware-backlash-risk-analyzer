# Context-Aware Backlash Risk Analyzer

> **A hybrid heuristic + LLM framework for context-aware social backlash risk analysis enhanced with Self-Consistency and uncertainty estimation.**

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![LLM](https://img.shields.io/badge/LLM-Qwen3%208B-green.svg)
![Status](https://img.shields.io/badge/Status-Research%20Project-orange.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

---

## Paper

- [Context-Aware Backlash Risk Analyzer Report](paper/Context_Aware_Backlash_Risk_Analyzer_Report.pdf)

## Overview

This project implements a **Context-Aware Backlash Risk Analyzer**, a system designed to estimate the likelihood that a piece of text may trigger **social backlash** under different communication contexts.

Unlike traditional sentiment analysis, which mainly predicts positive, negative, or neutral sentiment, this analyzer evaluates whether a message is likely to receive criticism, misunderstanding, offense, or other forms of negative social reactions.

The system combines:

- **Human-designed heuristic scoring**
- **LLM-based contextual reasoning**
- **Confidence-aware validation**
- **Self-Consistency aggregation**
- **Bounded context adjustment**

to generate **stable, interpretable, and context-aware risk assessments**.

The latest version introduces **Self-Consistency**, allowing the system to perform multiple independent LLM evaluations and estimate both the **final risk score** and the **uncertainty** of the prediction.

---

#  Key Features

-  Context-aware backlash risk assessment
-  Seven interpretable risk dimensions
-  Hybrid heuristic + LLM evaluation
-  Validation layer for reliable LLM integration
-  Self-Consistency using five independent LLM evaluations
-  Uncertainty estimation based on score variance
- Bounded context-aware score adjustment
-  Explainable and interpretable decision process

---

#  Motivation

Online communication increasingly occurs through social media, workplace emails, messaging applications, and online communities.

However, the same sentence may receive completely different reactions depending on **where**, **how**, and **to whom** it is delivered.

For example,

> "That's interesting."

may be interpreted as

- genuine curiosity,
- sarcasm,
- passive aggression,
- or criticism,

depending on its communication context.

Traditional sentiment analysis often fails to capture these subtle context-dependent risks because it focuses primarily on emotional polarity rather than **social interpretation**.

This project addresses this limitation by estimating **social backlash risk** through:

- linguistic analysis,
- contextual reasoning,
- multiple social-risk dimensions,
- and repeated LLM evaluation.

---

#  System Architecture

The runtime system consists of two primary components.

| Component | Description |
|-----------|-------------|
| `app.py` | Lightweight web interface for user interaction |
| `pipeline.py` | Production risk analysis pipeline |

The overall design philosophy is

```text
Human-designed Heuristics
            +
LLM-assisted Evaluation
            +
Validation
            +
Self-Consistency
            +
Context-aware Adjustment
```

Rather than allowing the language model to directly determine the final score, the analyzer constrains LLM outputs using deterministic heuristic scoring, confidence-aware validation, and bounded aggregation.

This hybrid design significantly improves both **robustness** and **interpretability**.

---

#  Overall Pipeline

The upgraded pipeline is shown below.

```text
                Input Text
                    │
                    ▼
             Text Preprocessing
                    │
                    ▼
             Tokenization
                    │
                    ▼
              Cue Extraction
                    │
                    ▼
     Sentence & Context Embedding
                    │
                    ▼
          Heuristic Scoring
                    │
                    ▼
     LLM-based Evaluation (×5)
                    │
                    ▼
          Validation Layer
                    │
                    ▼
     Self-Consistency Aggregation
                    │
                    ▼
            Score Merging
                    │
                    ▼
        Context-aware Adjustment
                    │
                    ▼
             Final Risk Score
                    +
          Uncertainty Estimate
```

Compared with the baseline system, the upgraded version introduces **Self-Consistency**, which performs five independent LLM evaluations before generating the final prediction.

This approach reduces stochastic variation while preserving contextual reasoning.

---

#  Risk Dimensions

Instead of predicting only sentiment polarity, the analyzer evaluates seven dimensions that commonly contribute to social backlash.

| Dimension | Description |
|-----------|-------------|
| **Aggression** | Hostile, insulting, or confrontational language |
| **Group Generalization** | Stereotypes or broad claims targeting social groups |
| **Sarcasm / Mockery** | Sarcastic, ironic, or mocking expressions |
| **Overconfident Judgment** | Excessively certain or absolute statements |
| **Context Inappropriateness** | Expressions unsuitable for the communication context |
| **Misinterpretability** | Statements likely to be misunderstood |
| **Norm Violation** | Language violating common social norms or etiquette |

Each dimension contributes independently to the final backlash risk score.

This multi-dimensional framework provides significantly richer analysis than conventional polarity-based sentiment classification.

---

#  Scoring Method

The analyzer estimates backlash risk through a multi-stage scoring pipeline rather than relying on a single prediction.

Instead of trusting an LLM alone, the final decision is produced by combining deterministic heuristics, LLM reasoning, validation logic, and Self-Consistency aggregation.

The scoring process consists of the following stages.

---

## 1. Heuristic Scoring

The first stage computes an initial risk estimate using deterministic rules and semantic similarity.

The heuristic module evaluates linguistic signals such as:

- Semantic similarity through prototype matching
- Contextual embedding similarity
- Punctuation patterns
- Emoji usage
- Slang-like expressions
- Vague targeting
- Short-post characteristics
- Context-level dimension multipliers

When available, embeddings are generated using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

If optional dependencies are unavailable, the system automatically falls back to a lightweight TF-IDF style vectorizer.

The heuristic score provides a stable baseline that is independent of LLM behavior.

---

## 2. LLM-Based Evaluation

The analyzer uses a Large Language Model as an additional reasoning component.

The default backend is

```text
Ollama
└── qwen3:8b
```

For each social-risk dimension, the LLM predicts:

- Probability
- Severity
- Confidence

The dimension-level LLM risk is computed as

```text
LLM Risk = 0.7 × Probability + 0.3 × Severity
```

Unlike many LLM-only systems, the model **does not directly determine the final score**.

Instead, it acts as a supporting evaluator whose outputs are verified before being incorporated into the final prediction.

---

## 3. Validation Layer

Since LLM outputs can vary across repeated executions, every prediction passes through a validation layer before score aggregation.

The system reduces the influence of the LLM when:

- confidence is low
- heuristic and LLM scores strongly disagree
- the LLM is unavailable
- invalid outputs are produced
- all dimension scores are zero

This validation stage prevents unreliable LLM responses from dominating the final result.

---

#  Self-Consistency

The primary contribution of the upgraded version is the introduction of **Self-Consistency**.

Instead of relying on a single stochastic LLM response, the analyzer performs **five independent evaluations** using the same prompt.

```text
Input
   │
   ▼
LLM Evaluation #1
LLM Evaluation #2
LLM Evaluation #3
LLM Evaluation #4
LLM Evaluation #5
        │
        ▼
 Self-Consistency Aggregation
```

The five independent outputs are aggregated to produce a more reliable prediction.

This approach significantly reduces random variation caused by probabilistic decoding.

---

## Final Risk Score

Let

- s_i be the risk score produced by the i-th evaluation.
- N be the number of evaluations.

The final score is computed as the average of all evaluations.

```text
Final Score = Mean(Risk Scores)
```

Using multiple independent evaluations produces a considerably more stable estimate than relying on a single LLM response.

---

#  Uncertainty Estimation

In addition to the final score, the upgraded analyzer estimates the consistency of the prediction.

Uncertainty is computed as the standard deviation of the five risk scores.

```text
Uncertainty = Standard Deviation(Risk Scores)
```

Interpretation:

| Uncertainty | Interpretation |
|-------------|---------------|
| Low | Stable prediction with high agreement across evaluations |
| Medium | Moderate disagreement between reasoning paths |
| High | Prediction is sensitive to stochastic LLM behavior |

This additional metric provides users with an estimate of **prediction reliability**, which was unavailable in the baseline system.

---

#  Score Merging

The final dimension score is obtained by combining heuristic and validated LLM estimates.

```text
Final Dimension Score

=

(1 − w) × Heuristic

+

w × LLM
```

where the weight **w** depends on:

- LLM confidence
- validation results
- agreement with heuristic scoring

Reliable LLM outputs receive greater influence, while uncertain predictions are automatically down-weighted.

---

#  Context-aware Adjustment

After dimension-level aggregation, the analyzer applies a bounded context adjustment.

Communication contexts include:

- SNS / Public
- SNS / Private
- Email / Public
- Email / Private
- Message / Public
- Message / Private

Internally, these contexts are mapped into three context groups.

| Context | Internal Bucket |
|----------|----------------|
| Public SNS | `public_social` |
| Workplace Email | `workplace` |
| Private Messages | `private_chat` |

Public and workplace environments generally increase social sensitivity.

Private communication reduces the final score, but only within predefined bounds.

The adjustment is intentionally limited so that highly offensive messages cannot be incorrectly classified as low risk simply because they occur in private conversations.

---

# Weight Design Philosophy

The scoring weights are **manually designed** rather than learned from labeled data.

This decision prioritizes:

- Explainability
- Transparency
- Robustness
- Human interpretability

The importance of each dimension is summarized below.

| Priority | Dimensions |
|----------|------------|
| High | Aggression, Norm Violation |
| Medium | Group Generalization, Context Inappropriateness, Sarcasm / Mockery, Misinterpretability |
| Lower | Overconfident Judgment |

The project intentionally favors interpretable scoring over purely data-driven optimization, making it easier to understand how individual factors contribute to the final backlash risk.

---

# Running Locally

## 1. Clone the Repository

```bash
git clone https://github.com/seungjongyoo/context-aware-backlash-risk-analyzer.git
cd context-aware-backlash-risk-analyzer
```

---

## 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the virtual environment.

### Windows (PowerShell)

```powershell
.\.venv\Scripts\Activate.ps1
```

If execution is blocked,

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### Linux / macOS

```bash
source .venv/bin/activate
```

---

## 3. Install Dependencies

Minimum dependencies:

```bash
pip install numpy
```

Recommended dependencies:

```bash
pip install numpy scikit-learn sentence-transformers torch transformers accelerate
```

The analyzer automatically falls back to lightweight implementations when optional dependencies are unavailable, although embedding quality may be reduced.

---

## 4. Run the Local LLM (Optional)

The default LLM backend is **Ollama** using **Qwen3 8B**.

Install Ollama separately and execute

```bash
ollama pull qwen3:8b
ollama run qwen3:8b
```

Keep the Ollama server running while using the analyzer.

If the LLM is unavailable, the pipeline automatically falls back to heuristic-only evaluation.

---

## 5. Start the Web Application

```bash
python app.py
```

Open

```text
http://127.0.0.1:8000
```

using any modern web browser.

---

## 6. Analyze Text

1. Enter the input text.
2. Select the communication category.
3. Select the disclosure scope.
4. Click **Analyze**.

The analyzer returns

- Final Risk Score
- Dimension Scores
- LLM Explanation
- Rewrite Suggestions
- Uncertainty Estimate

---

# Repository Structure

```text
context-aware-backlash-risk-analyzer/
├── app.py
├── pipeline.py
├── README.md
└── paper/
```

---

# Design Principles

The analyzer is designed around four principles.

## Explainability

Every score is derived from interpretable heuristic components together with dimension-level LLM reasoning.

The contribution of each stage can be understood without requiring access to model internals.

---

## Robustness

Instead of relying on a single LLM response, the analyzer combines

- heuristic scoring,
- validation,
- confidence-aware weighting,
- Self-Consistency,

to reduce stochastic variation.

---

## Context Awareness

The same sentence may receive different interpretations depending on where it is communicated.

The analyzer explicitly incorporates communication context throughout the scoring pipeline rather than applying context as a simple post-processing step.

---

## Reliability

The upgraded version estimates not only the final risk score but also the uncertainty of the prediction.

This enables users to distinguish between

- highly consistent predictions,

and

- predictions that vary across multiple reasoning paths.

---

# Limitations

Although the proposed framework improves robustness and interpretability, several limitations remain.

- The scoring weights are manually designed rather than learned from data.
- The system does not rely on large-scale labeled datasets.
- Performance depends on the underlying LLM when LLM evaluation is enabled.
- Cultural differences and domain-specific communication styles are not fully represented.
- Self-Consistency improves stability but increases inference time because multiple LLM evaluations are required.

---

# Demonstration

The analyzer provides an interactive web interface for context-aware social backlash risk assessment.

A typical workflow is:

```text
Input Text
      │
      ▼
Select Context
(SNS / Email / Message)
      │
      ▼
Run Analysis
      │
      ▼
View Results
```

The analyzer reports:

- Final Risk Score
- Dimension-level Scores
- LLM Explanation
- Rewrite Suggestions
- Uncertainty Estimate

---

# Technical Report

This repository accompanies the following technical report.

**Enhancing a Context-Aware Backlash Risk Analyzer Using Self-Consistency**

The report contains:

- Motivation
- Related Work
- System Architecture
- Methods
- Experimental Setup
- Results
- Discussion
- Conclusion

The report can be found at

```text
paper/Context_Aware_Backlash_Risk_Analyzer_Report.pdf
```

---

# Citation

If you use this repository for academic or research purposes, please cite the accompanying report.

```bibtex
@techreport{yoo2026,
  title={Enhancing a Context-Aware Backlash Risk Analyzer Using Self-Consistency},
  author={Yoo, Seungjong},
  institution={Soongsil University},
  year={2026}
}
```

---

# License

This project is released under the MIT License.

See the LICENSE file for details.

---

# Acknowledgements

This project was developed as part of a university research project on context-aware natural language understanding.

The work builds upon previous research in:

- Context-aware toxicity detection
- Contextual abuse detection
- Self-Consistency for LLM reasoning

The project extends these ideas by integrating heuristic scoring, validation, context-aware reasoning, and Self-Consistency into a unified backlash risk analysis framework.