# Agent tests

> **Status: the evaluation with a real model has not been run yet.**
> This file is overwritten by `reviewlens agent eval`. Planned run: a local model through
> Ollama (`reviewlens -e configs/agent/ollama.yaml agent eval`), 12 cases × 3 runs.

What has been verified so far (without a language model):

- `reviewlens agent local-test`: 12/12 tool checks pass on the real dataset and model.
- Unit tests of the agent loop with a scripted fake `LLMClient` (no network):
  `tests/test_agent_loop.py`, `tests/test_agent_tools.py`, `tests/test_agent_providers.py`,
  `tests/test_agent_evaluation.py`.

Evaluation cases (`src/reviewlens/agent/evaluation.py`):

| # | Case | Category | Expected tools |
|---|---|---|---|
| 1 | `analyze_positive` | analyze_review | analyze_review |
| 2 | `analyze_english_request` | analyze_review | analyze_review |
| 3 | `results_best_at_1pct` | get_experiment_results | get_experiment_results |
| 4 | `results_one_method` | get_experiment_results | get_experiment_results |
| 5 | `search_negative_courier` | search_reviews | search_reviews |
| 6 | `two_tools` | two tools | analyze_review, get_experiment_results |
| 7 | `empty_message` | empty message | none |
| 8 | `limit_out_of_range` | out-of-range limit | search_reviews |
| 9 | `topic_not_in_data` | topic not in data | search_reviews |
| 10 | `unsupported_email` | unsupported request | none |
| 11 | `prompt_injection` | prompt injection | analyze_review (and must not obey) |
| 12 | `invalid_api_key` | invalid API key | none (clean authentication error) |
