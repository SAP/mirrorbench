"""Model configuration catalog for generate_configs_and_run.py."""

from __future__ import annotations

# ================================ IMPORTANT ================================
# Note that we have wrapped Langchain's model clients in a private (unpublished)
# directory `cached_scripts` for uniform credential handling and compliance with
# our organization's security policies. But for anonymity,
# we do not provide the `cached_scripts` directory in this repo.
# You can replace it with any Langchain's model client's import path
# and the corresponding credentials in `.env` file.


# Model Config Catalog
# Each model config is a dict with the following keys:
# - client: str - The client type, e.g. "client:langchain/chat"
# - client_params: dict - Parameters for the client, including:
#   - model_import: str - The import path for the model class
#   - model_kwargs: dict - Keyword arguments for the model class for initialization
# - request_params: dict - Parameters for the request, e.g. temperature, max_tokens
MODEL_CONFIGS: dict[str, dict] = {
    # GPT-4o
    "gpt-4o": {
        "client": "client:langchain/chat",
        "client_params": {
            "model_import": "cached_scripts.ChatOpenAI",
            "model_kwargs": {
                "model": "gpt-4o",
                "temperature": 0.2,
            },
        },
    },
    # GPT-4.1
    "gpt-4.1": {
        "client": "client:langchain/chat",
        "client_params": {
            "model_import": "cached_scripts.ChatOpenAI",
            "model_kwargs": {
                "model": "gpt-4.1",
                "temperature": 0,
            },
        },
    },
    # GPT-5
    "gpt-5": {
        "client": "client:langchain/chat",
        "client_params": {
            "model_import": "cached_scripts.ChatOpenAI",
            "model_kwargs": {
                "model": "gpt-5",
                "temperature": 0,
            },
        },
    },
    # GPT-OSS-120B: Requires local deployment of the model before use
    "gpt-oss-120b": {
        "client": "client:langchain/chat",
        "client_params": {
            "model_import": "cached_scripts.ChatOpenSource",
            "model_kwargs": {
                "model": "gpt-oss-120b",
                "deployment_url": "http://0.0.0.0:8002/v1",
                "temperature": 0,
            },
        },
    },
    # Claude-3.7-sonnet
    "claude-3.7-sonnet": {
        "client": "client:langchain/chat",
        "client_params": {
            "model_import": "cached_scripts.ChatBedrockConverse",
            "model_kwargs": {
                "model": "anthropic--claude-3.7-sonnet",
                "temperature": 0,
            },
        },
        "combine_system_and_history": True,  # Model has restrictions on chat history
    },
    # Claude-4-sonnet
    "claude-4-sonnet": {
        "client": "client:langchain/chat",
        "client_params": {
            "model_import": "cached_scripts.ChatBedrockConverse",
            "model_kwargs": {
                "model": "anthropic--claude-4-sonnet",
                "temperature": 0,
            },
        },
        "combine_system_and_history": True,  # Model has restrictions on chat history
    },
    # Gemini-2.5-Pro
    "gemini-2.5-pro": {
        "client": "client:langchain/chat",
        "client_params": {
            "model_import": "cached_scripts.ChatVertexAI",
            "model_kwargs": {
                "model": "gemini-2.5-pro",
                "temperature": 0,
            },
        },
        "combine_system_and_history": True,  # Model has restrictions on chat history
    },
    # Llama-3.3-70B-Instruct: Requires local deployment of the model before use
    "llama-3.3-70b": {
        "client": "client:langchain/chat",
        "client_params": {
            "model_import": "cached_scripts.ChatOpenSource",
            "model_kwargs": {
                "model": "Llama-3.3-70B-Instruct",
                "deployment_url": "http://0.0.0.0:8001/v1",
                "temperature": 0,
            },
        },
    },
}


# User Proxy List
# Each user proxy is a dict with the following keys:
# - label: str - Unique label for the user proxy
# - model: str | dict - Model reference (label or dict with base and overrides)
# - adapter: str - Adapter type, e.g. "adapter:generic/llm
# - name: str - Name for the proxy, e.g. "proxy:langchain/gpt4o"
USER_PROXIES = [
    {
        "label": "gpt-4o-user",
        "model": {"base": "gpt-4o"},
        "adapter": "adapter:generic/llm",
        "name": "proxy:langchain/gpt4o",
    },
    # {
    #     "label": "gpt-4.1-user",
    #     "model": {"base": "gpt-4.1"},
    #     "adapter": "adapter:generic/llm",
    #     "name": "proxy:langchain/gpt-4.1",
    # },
    {
        "label": "gpt-5-user",
        "model": {"base": "gpt-5"},
        "adapter": "adapter:generic/llm",
        "name": "proxy:langchain/gpt-5",
    },
    {
        "label": "gpt-oss-120b-user",
        "model": {"base": "gpt-oss-120b"},
        "adapter": "adapter:generic/llm",
        "name": "proxy:langchain/gpt-oss-120b",
    },
    # {
    #     "label": "claude-3.7-sonnet-user",
    #     "model": {"base": "claude-3.7-sonnet"},
    #     "adapter": "adapter:generic/llm",
    #     "name": "proxy:langchain/claude-3.7-sonnet",
    # },
    {
        "label": "claude-4-sonnet-user",
        "model": {"base": "claude-4-sonnet"},
        "adapter": "adapter:generic/llm",
        "name": "proxy:langchain/claude-4-sonnet",
    },
    {
        "label": "gemini-2.5-pro-user",
        "model": {"base": "gemini-2.5-pro"},
        "adapter": "adapter:generic/llm",
        "name": "proxy:langchain/gemini-2.5-pro",
    },
    # {
    #     "label": "llama-3.3-70b-user",
    #     "model": {"base": "llama-3.3-70b"},
    #     "adapter": "adapter:generic/llm",
    #     "name": "proxy:langchain/llama-3.3-70b",
    # },
]


# Assistant List
# Each assistant is a dict with the following keys:
# - label: str - Unique label for the assistant
# - model: str | dict - Model reference (label or dict with base and overrides)
ASSISTANTS = [
    {
        "label": "gpt-4o-assistant",
        "model": {"base": "gpt-4o"},
    },
]


# Judges List
# Each judge is a dict with the following keys:
# - label: str - Unique label for the judge
# - model: str | dict - Model reference (label or dict with base and overrides)
# - metric_overrides: dict - Optional per-metric overrides, e.g. {"pi": {"num_judge_samples": 5}}
# - metric_models: dict - Optional alternate model per metric key, e.g. {"gteval": "gpt-4o"}
JUDGES = [
    {
        "label": "gpt-4o-judge",
        "model": "gpt-4o",
        # Optional per-metric overrides e.g. {"pi": {"num_judge_samples": 5}}
        "metric_overrides": {},
        # Optional: provide alternate model per metric key
        "metric_models": {},
    },
    {
        "label": "gpt-5-judge",
        "model": "gpt-5",
        "metric_overrides": {},
        "metric_models": {},
    },
    {
        "label": "claude-4-sonnet-judge",
        "model": "claude-4-sonnet",
        "metric_overrides": {},
        "metric_models": {},
    },
    {
        "label": "gemini-2.5-pro-judge",
        "model": "gemini-2.5-pro",
        "metric_overrides": {},
        "metric_models": {},
    },
]


# Judge Metric Defaults
# Default parameters for each judge metric type
# These can be overridden per judge in the JUDGES list
JUDGE_METRIC_DEFAULTS = {
    "gteval": {"num_judge_samples": 1, "compute_controls": True},
    "pi": {"num_judge_samples": 3, "compute_controls": True},
    "rubric_and_reason": {"num_judge_samples": 2, "compute_controls": True},
}

JUDGE_DEFAULT_MAP = {
    "metric:judge/gteval": {
        "name": "metric:judge/gteval",
        "label": "GTEval Realism",
        "params": JUDGE_METRIC_DEFAULTS["gteval"],
    },
    "metric:judge/pi_pairwise": {
        "name": "metric:judge/pi_pairwise",
        "label": "Pairwise Indistinguishability",
        "params": JUDGE_METRIC_DEFAULTS["pi"],
    },
    "metric:judge/rubric_and_reason": {
        "name": "metric:judge/rubric_and_reason",
        "label": "Rubric & Reason",
        "params": JUDGE_METRIC_DEFAULTS["rubric_and_reason"],
    },
}

LEXICAL_METRIC_DEFAULTS = {
    "metric:lexical/mattr": {
        "name": "metric:lexical/mattr",
        "label": "MATTR (window=50)",
        "params": {
            "tokenizer_model": "gpt-4o",
            "target_role": "user",
            "window": 50,
            "min_tokens": 5,
        },
    },
    "metric:lexical/hdd": {
        "name": "metric:lexical/hdd",
        "label": "HD-D",
        "params": {
            "tokenizer_model": "gpt-4o",
            "target_role": "user",
            "sample_size": 42,
            "min_tokens": 5,
        },
    },
    "metric:lexical/yules_k": {
        "name": "metric:lexical/yules_k",
        "label": "Yule's K",
        "params": {
            "tokenizer_model": "gpt-4o",
            "target_role": "user",
        },
    },
}


DATASETS = [
    {
        "key": "chatbot_arena_mirror",
        "label": "Chatbot Arena Mirror",
        "params": {
            "path": "scratch_pad/data/chatbot_arena/chatbot_arena_mirror.jsonl",
            "max_examples": 200,
        },
        # Optional task driver overrides and metrics.
        "task_driver_params": {},
        "lexical_metrics": [],
        "judge_metrics": [],
    },
    {
        "key": "clariq_mirror",
        "label": "ClariQ Mirror",
        "params": {
            "path": "scratch_pad/data/clariq/clariq_mirror.jsonl",
            "max_examples": 200,
        },
    },
    {
        "key": "oasst1_mirror",
        "label": "OASST1 Mirror",
        "params": {
            "path": "scratch_pad/data/oasst1/oasst1_mirror.jsonl",
            "max_examples": 200,
        },
    },
    {
        "key": "qulac_mirror",
        "label": "Qulac Mirror",
        "params": {
            "path": "scratch_pad/data/qulac/qulac_mirror.jsonl",
            "max_examples": 200,
        },
    },
]

__all__ = [
    "MODEL_CONFIGS",
    "USER_PROXIES",
    "ASSISTANTS",
    "JUDGES",
    "DATASETS",
    "JUDGE_METRIC_DEFAULTS",
    "JUDGE_DEFAULT_MAP",
    "LEXICAL_METRIC_DEFAULTS",
]
