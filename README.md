# Context-Aware Backlash Risk Analyzer

## Overview

This project implements a context-aware social backlash risk analysis system for short text inputs.

The goal is to estimate how likely a given text is to receive negative reactions, such as criticism, backlash, misinterpretation, or inappropriate-context reactions, across communication settings such as public SNS, private messages, and workplace email.

Unlike traditional sentiment analysis, which focuses mainly on positive or negative polarity, this system evaluates multiple dimensions of social risk and applies bounded context-aware adjustments to the final score.

## System Architecture

The runtime system consists of:

- `app.py`: a lightweight local web interface served at `http://127.0.0.1:8000`
- `pipeline.py`: the production scoring pipeline

The pipeline follows this flow:

```text
Input
-> Preprocessing
-> Tokenization
-> Cue Extraction
-> Embedding (sentence + contextual)
-> Heuristic Scoring
-> LLM-based Evaluation
-> Validation
-> Score Merging
-> Context Adjustment
-> Final Risk Score
```

The design principle is:

```text
Human-designed structure + LLM-assisted evaluation
```

## Risk Dimensions

The system evaluates seven social-risk dimensions:

- Aggression
- Group Generalization
- Sarcasm / Mockery
- Overconfident Judgment
- Context Inappropriateness
- Misinterpretability
- Norm Violation

Each dimension represents a different type of social risk that may trigger negative reactions.

## Scoring Method

### 1. Heuristic Scoring

Heuristic scores are computed using:

- semantic similarity through prototype matching
- contextual embedding comparison
- surface signals such as punctuation, vague targeting, emoji, slang-like cues, and short-post structure
- context-level dimension multipliers

The embedding layer uses `sentence-transformers/all-MiniLM-L6-v2` when available. If optional dependencies are unavailable, the pipeline falls back to a lightweight TF-IDF style vectorizer.

### 2. LLM-Based Evaluation

The default LLM backend is Ollama with `qwen3:8b`.

The LLM evaluates each dimension using:

- probability: likelihood that readers perceive the risk
- severity: impact if the risk is perceived
- confidence: reliability of the judgment

LLM risk is computed as:

```text
LLM Risk = 0.7 * probability + 0.3 * severity
```

The LLM does not directly determine the final score.

### 3. Validation Layer

LLM outputs are validated before merging.

The system reduces LLM influence when:

- confidence is low
- there is large disagreement with heuristic scores
- the LLM is unavailable
- the LLM returns zero-valued or unusable scores

This prevents unstable or unreliable LLM behavior from dominating the result.

### 4. Score Merging

Final dimension scores are computed as:

```text
Final = (1 - w) * Heuristic + w * LLM
```

The weight `w` depends on LLM confidence and validation results. Reliable LLM outputs receive more influence, while low-confidence or inconsistent outputs are suppressed.

### 5. Final Risk Score

The overall base risk score is computed using weighted aggregation across all dimensions.

Context then adjusts the score using a bounded bias:

```text
Final Score = 0.8 * Base Score + 0.2 * Context Adjustment
```

A lower bound is enforced:

```text
Final Score >= 0.7 * Base Score
```

This prevents private contexts from unrealistically reducing highly toxic text to near-zero risk.

## Context Handling

The web UI collects:

- Category: `SNS`, `Email`, or `Message`
- Scope of Disclosure: `public` or `private`

These are converted into context strings such as:

```text
SNS / Public
Email / Private
Message / Public
```

Internally, contexts are mapped into buckets:

- `public_social`
- `workplace`
- `private_chat`

Public and workplace contexts increase sensitivity. Private contexts reduce risk, but the adjustment is bounded so content risk is preserved.

## Weight Design Rationale

Weights are manually designed for interpretability.

High-weight dimensions:

- Aggression
- Norm Violation

Medium-weight dimensions:

- Group Generalization
- Context Inappropriateness
- Sarcasm / Mockery
- Misinterpretability

Lower-weight dimension:

- Overconfident Judgment

The weights are not learned from data. They are manually designed based on common social interaction patterns and optimized for transparency, controllability, and robustness across contexts.

## Role of the LLM

The LLM is used as a supporting component. It provides:

- dimension-level estimates
- explanations
- rewrite suggestions

The final decision is constrained by:

- heuristic scoring
- validation logic
- weighted aggregation
- bounded context adjustment

This design reduces over-reliance on LLM output and limits hallucination-driven scoring.

## Running Locally

### 1. Clone the repository

```powershell
git clone https://github.com/seungjongyoo/context-aware-backlash-risk-analyzer.git
cd context-aware-backlash-risk-analyzer
```

### 2. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

Minimum dependencies:

```powershell
pip install numpy
```

Recommended dependencies for better embedding quality:

```powershell
pip install numpy scikit-learn sentence-transformers torch transformers accelerate
```

The app can still import without `scikit-learn`, `sentence-transformers`, or `transformers`, but output quality is better when they are installed.

### 4. Optional: run the local LLM backend

The default LLM backend is Ollama with `qwen3:8b`.

Install Ollama separately, then run:

```powershell
ollama pull qwen3:8b
ollama run qwen3:8b
```

Keep Ollama running while using the web app.

### 5. Start the web app

Start the web app:

```powershell
python app.py
```

Open:

```text
http://127.0.0.1:8000
```

### 6. Use the analyzer

Enter text, choose a category (`SNS`, `Email`, or `Message`), choose disclosure scope (`public` or `private`), and click `Analyze`.

If the LLM or optional embedding dependencies are unavailable, the pipeline falls back to heuristic behavior and records warnings in the result.

## Limitations

- Weights are manually designed and may not generalize perfectly to all domains.
- LLM outputs can vary depending on prompt, backend, and runtime conditions.
- The system does not use large-scale labeled datasets for training.
- Cultural and contextual nuances may not be fully captured.
- Optional dependencies affect embedding quality and backend behavior.

## Design Considerations

### Why not fully rely on the LLM?

LLM-only systems are flexible but can be inconsistent. This system intentionally constrains LLM outputs using validation, weighting, and deterministic components.

### Why are weights manually defined?

The goal is interpretability. Learned weights may improve accuracy but can reduce transparency. This project prioritizes explainability over pure optimization.

### Why is context adjustment limited?

Without constraints, context can distort results. The system enforces bounded adjustment and minimum risk preservation.

## Key Contribution

This project demonstrates that combining structured heuristic scoring with LLM-based reasoning can produce a more stable, interpretable, and context-aware risk analysis system than using either approach alone.
