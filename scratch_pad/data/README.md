# MirrorBench Dataset Preprocessing Toolkit

This directory contains standalone preprocessors that convert human-authored
conversational datasets into the normalized structure expected by the
MirrorBench evaluation harness. The goal is to distill ≤500 representative
conversations per dataset, attach the metadata required for downstream human
likeness comparisons, and synthesize a lightweight task description that can be
handed to a user-proxy agent.

## Overview

```
 scratch_pad/data/
 ├── common/               # Shared datatypes, sampling, logging, downloads
 ├── oasst1/               # OpenAssistant tree → linear dialog extractor
 ├── chatbot_arena/        # LMSYS Chatbot Arena (winner conversation) sampler
 ├── qulac/                # Query clarification triples from the Qulac corpus
 ├── clariq/               # Multi-turn human clarifications from ClariQ
 └── run_preprocessing.py  # CLI for running any/all preprocessors
   few-shot retrieval    # Optional nearest-neighbour hints for user turns
```

Each dataset-specific module subclasses `BaseDatasetPreprocessor` and focuses
on four steps:

1. **Acquire** the raw source (Hugging Face datasets or public URLs, cached
   under each dataset directory).
2. **Normalize** the conversation into alternating user/assistant `Turn`
   objects with consistent metadata.
3. **Stratify** candidates using a dataset-appropriate key (e.g., language,
   user-turn bucket, topic/facet category) and sample up to the requested
   budget via the shared `stratified_sample` helper.
4. **Describe** the goal for a user proxy by synthesizing a task description
   that mirrors the original user intent and salient metadata.

Outputs are written incrementally as JSONL files named `{dataset}_mirror.jsonl`
inside their respective dataset folders. Processing skips any sample that
fails to normalize or generate a task description, so the pipeline continues
even if individual records encounter errors. Only English-language
conversations are retained; datasets without an explicit language column are
treated as English by construction. Each line follows the
`ConversationRecord` schema:

```json
{
  "dataset": "clariq",
  "conversation_id": "clariq-0",
  "task_description": "...",
  "turns": [
    {"role": "user", "content": "...", "metadata": {...}},
    {"role": "assistant", "content": "...", "metadata": {...}},
    ...
  ],
  "metadata": {
    "facet": "...",
    "clarification_pairs": 3,
    "language": "en",
    "num_turns": 6,
    "few_shot_user_examples": [
      {"conversation_id": "clariq-42", "utterance": "...", "similarity": "0.8123"},
      ...
    ],
    ...
  }
}
```

## Usage

1. Ensure the project virtual environment is active and the core dependencies
   are installed (`pip install -e .`).
2. Configure the task-description LLM (defaults assume Azure OpenAI via LangChain). Set the
   following environment variables if you need to override the defaults:

   - `MIRRORBENCH_TASK_DESC_CLIENT` (defaults to `client:langchain/chat`)
   - `MIRRORBENCH_TASK_DESC_MODEL_IMPORT` (defaults to `langchain_openai.AzureChatOpenAI`)
   - `MIRRORBENCH_TASK_DESC_MODEL_KWARGS` (JSON string forwarded to the model constructor)

   The Azure client expects credentials such as `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`,
   and deployment details to already be present in the environment.

3. Run the preprocessing CLI from the repository root:

```bash
python scratch_pad/data/run_preprocessing.py
```

Useful flags:

- `--datasets oasst1 qulac` to process a subset.
- `--max-samples 300` to reduce the sample budget.
- `--seed 2025` to change the deterministic sampling seed.
- `--output-root /tmp/mirror_outputs` to materialize JSONL files elsewhere.
- `--task-desc-model-import`/`--task-desc-model-kwargs` to override the LLM used for task
  descriptions.
- `--resume` to append to existing JSONL outputs instead of starting from scratch.

Each dataset module caches downloads in `<dataset>/.cached/` by default; delete the
folder to refresh.

### Few-shot user retrieval

The pipeline can enrich each record with up to five similar user utterances
from other processed conversations. Provide an embedding backend by editing
`common/embeddings.py` and returning an object with an `embed(texts: Sequence[str])`
method. When no embedding client is configured, the pipeline skips the
few-shot augmentation without failing the run.

## Dataset-specific Notes

### OASST1
- Loads the `OpenAssistant/oasst1` train and validation splits.
- Reconstructs a single linear path per conversation tree by following ranked
  children and alternating roles.
- Stratifies by language and user-turn count bucket. High-confidence
  moderation labels (≥0.7) are surfaced as task hints.

### Chatbot Arena
- Uses the latest LMSYS arena conversations and selects the conversation for
  the declared winner (falls back to model A on ties).
- Stratifies by number of user turns, and whether the dialog includes
  multiple clarifications.
- Captures judge, winner metadata, and generates stylistic hints based on the
  winning model.

### Qulac
- Downloads the public `qulac.json` column-oriented payload and converts each
  entry into a user → assistant clarification → user acknowledgement triple.
- Stratification relies on `(topic_type, facet_type)` so samples cover both
  ambiguous and faceted queries across informational/navigation intents.
- Facet descriptions seed the task guidance.

### ClariQ
- Fetches the multi-turn human generated TSV and constructs dialogs consisting
  of the initial user request followed by up to three clarification Q/A pairs.
- Buckets topics by `topic_id % 5` to spread coverage, while also balancing the
  number of clarification turns and total dialog length.
- Facet strings inform the task description hints.

## Assumptions & Limitations

- Task descriptions are synthesized via an LLM call. Ensure the configured
  provider credentials are available before running the CLI.
- Conversations without alternation, missing user turns, or lacking sufficient
  metadata are dropped.
- Stratification keys favor broad coverage rather than exact dataset-provided
  distributions; adjust the per-dataset logic if stricter quotas are needed.
- The scripts download directly from Hugging Face or GitHub. If offline access
  is required, place the raw sources in `<dataset>/.cached/` ahead of time and
  skip the download step.
