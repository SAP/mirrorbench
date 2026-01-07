# Human-Judge Correlation Analysis Pipeline

This directory contains a complete pipeline for validating LLM judge metrics against human annotations in MirrorBench.

## Overview

The pipeline performs stratified sampling of evaluation episodes, exports them for human annotation, and computes correlation statistics between human labels and judge scores. This allows you to validate that LLM judges are measuring human-likeness in ways that align with actual human judgments.

## Directory Structure

```
scratch_pad/human_corr/
├── README.md                   # This file
├── sample_episodes.py          # Core sampling logic with stratification
├── export_gteval.py            # Export GTEval samples for annotation
├── export_pi.py                # Export PI pairwise samples for annotation
├── compute_correlation.py      # Correlation analysis (Spearman, Pearson, Kendall)
├── pipeline.py                 # Main orchestrator script
└── runs/
    └── <run_id>/               # Per-run directories (auto-created)
        ├── gteval_samples.json              # GTEval samples for annotation (NO judge scores)
        ├── gteval_judge_scores.json         # Judge scores (kept separate, hidden from annotator)
        ├── pi_pairwise_samples.json         # PI samples for annotation (NO judge scores)
        ├── pi_pairwise_judge_scores.json    # Judge scores + ground truth (kept separate)
        ├── gteval_labeled.json              # After human labeling (rename from samples)
        ├── pi_pairwise_labeled.json         # After human labeling (rename from samples)
        ├── correlation_report.json          # Statistical results
        └── correlation_report.txt           # Human-readable report
```

## Workflow

### Step 1: Export Samples for Annotation

```bash
cd scratch_pad/human_corr

# Export 100 samples for both GTEval and PI metrics
python pipeline.py export <run_id> --num-samples 100

# Example with the chatbot_arena run
python pipeline.py export chatbot_arena_mirror-gemini-2.5-pro-user-gpt-4o-assistant-claude-4-sonnet-judge
```

**What happens:**
- Loads all episodes from the specified run
- Performs stratified sampling based on:
  - Metric score (3 bins: low/medium/high)
  - Conversation length (3 bins: short/medium/long)
- Exports **annotation files** (NO judge scores visible):
  - `gteval_samples.json` - For human labeling
  - `pi_pairwise_samples.json` - For human labeling
- Exports **judge score files** (kept separate, hidden from annotator):
  - `gteval_judge_scores.json` - Merged during correlation analysis
  - `pi_pairwise_judge_scores.json` - Includes ground truth for PI
- Uses different random seeds to ensure different episode sets

**Sampling Parameters:**
- `--num-samples`: Number of samples per metric (default: 100)
- `--gteval-seed`: Random seed for GTEval (default: 42)
- `--pi-seed`: Random seed for PI pairwise (default: 43)

### Step 2: Human Annotation

#### GTEval Annotation Format

Each sample in `gteval_samples.json` contains:
- `real_conversation`: Array of `{role, content}` objects - the actual human-human conversation
- `synthetic_conversation`: Array of `{role, content}` objects - the AI-generated user proxy conversation
- `judge_score`: The LLM judge's score (0.0-1.0)
- `human_label`: **[TO BE FILLED]** - Your score from 0.0 to 1.0

**Task:** Compare the two conversations and rate how similar the synthetic one is to the real one in terms of user behavior, style, and tone. Fill in the `human_label` field with a value between 0.0 (completely different) and 1.0 (identical).

**Example structure:**
```json
{
  "real_conversation": [
    {"role": "user", "content": "What's the weather?"},
    {"role": "assistant", "content": "..."}
  ],
  "synthetic_conversation": [
    {"role": "user", "content": "Can you tell me the weather forecast?"},
    {"role": "assistant", "content": "..."}
  ],
  "human_label": null  // Fill in: 0.0 to 1.0
}
```

#### PI Pairwise Annotation Format

Each sample in `pi_pairwise_samples.json` contains:
- `conversation_a`: Array of `{role, content}` objects - one conversation (either real or synthetic, randomized)
- `conversation_b`: Array of `{role, content}` objects - the other conversation
- `judge_score`: The LLM judge's win rate for synthetic (0.0-1.0)
- `human_label`: **[TO BE FILLED]** - Your choice: "A", "B", or "Tie"

**Task:** Compare the two conversations WITHOUT knowing which is real/synthetic. Decide which user seems more natural and human-like. Fill in the `human_label` field with "A", "B", or "Tie".

**Example structure:**
```json
{
  "conversation_a": [
    {"role": "user", "content": "What's the weather?"},
    {"role": "assistant", "content": "..."}
  ],
  "conversation_b": [
    {"role": "user", "content": "Tell me about the weather"},
    {"role": "assistant", "content": "..."}
  ],
  "human_label": null  // Fill in: "A", "B", or "Tie"
}
```

**Note:** The `ground_truth_synthetic_slot` field is hidden from annotators but used during correlation analysis to convert categorical labels to win rates.

#### Renaming After Annotation

Once you've completed the annotations, rename the files to indicate they're labeled:

```bash
cd runs/<run_id>

# Rename GTEval file
mv gteval_samples.json gteval_labeled.json

# Rename PI file
mv pi_pairwise_samples.json pi_pairwise_labeled.json
```

### Step 3: Compute Correlations

```bash
python pipeline.py analyze <run_id>
```

**What happens:**
- Loads `gteval_labeled.json` and `pi_pairwise_labeled.json`
- Converts PI categorical labels to numeric win rates
- Computes three correlation metrics:
  - **Spearman's ρ**: Rank correlation (robust to outliers, recommended primary metric)
  - **Pearson's r**: Linear correlation
  - **Kendall's τ**: Alternative rank correlation
- Generates:
  - `correlation_report.json`: Structured results with all statistics
  - `correlation_report.txt`: Human-readable interpretation

**Output Example:**

```
================================================================================
HUMAN-JUDGE CORRELATION REPORT
================================================================================

Metric: metric:judge/gteval
Samples: 100

Human Scores:  Mean = 0.652, Std = 0.211
Judge Scores:  Mean = 0.638, Std = 0.198

Correlation Metrics:
  Spearman's ρ:   0.743 (p=0.0000)
  Pearson's r:    0.718 (p=0.0000)
  Kendall's τ:    0.592 (p=0.0000)

Interpretation:
  Strong correlation (significant at α=0.05)
```

## Design Rationale

### Why Stratified Sampling?

Simple random sampling might over-represent common score ranges or conversation lengths. Stratified sampling ensures:
- Coverage across the full score spectrum (low/medium/high)
- Diversity in conversation complexity (short/medium/long)
- Better estimates of correlation across the full distribution

### Why Different Samples for GTEval vs PI?

Using different episode sets (via different random seeds) prevents:
- Annotator fatigue from seeing the same episodes twice
- Bias from remembering prior judgments
- Overfitting to a specific subset of episodes

### Sample Size Recommendation

**100 samples per metric** provides:
- Sufficient statistical power (correlation tests typically need n≥30)
- Reasonable annotation burden (~2-3 hours per metric at 1-2 min/sample)
- ~11 samples per stratification bin (3×3 grid)

For a **full study across 2 datasets**, annotate:
- 100 GTEval samples × 2 datasets = 200 total GTEval annotations
- 100 PI samples × 2 datasets = 200 total PI annotations
- Total: 400 annotations

## Interpreting Results

### Correlation Strength (using |Spearman's ρ|)

- **0.7-1.0**: Strong correlation - Judge aligns well with humans
- **0.5-0.7**: Moderate correlation - Reasonable alignment with gaps
- **0.3-0.5**: Weak correlation - Judge captures some signal but noisy
- **0.0-0.3**: Very weak - Judge may not measure human-likeness well

### Statistical Significance

- **p < 0.05**: Correlation is statistically significant
- **p ≥ 0.05**: Could be due to chance; need more samples or judge may be uncorrelated

### Expected Ranges for Good Judges

Based on LLM-as-judge literature, expect:
- **GTEval**: Spearman ρ = 0.6-0.8 (human-likeness is subjective)
- **PI Pairwise**: Spearman ρ = 0.5-0.7 (more difficult task)

Lower correlations may indicate:
- Judge prompt needs refinement
- Judge model capabilities insufficient
- Task definition ambiguity
- Genuine disagreement between human raters

## Adding New Runs

The pipeline is fully modular. To analyze a different run:

```bash
# Export samples
python pipeline.py export <new_run_id> --num-samples 100

# Annotate the files in runs/<new_run_id>/

# Compute correlations
python pipeline.py analyze <new_run_id>
```

Each run is isolated in its own directory, so you can maintain multiple correlation studies simultaneously.
