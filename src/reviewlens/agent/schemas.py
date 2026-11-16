"""Tool definitions sent to the language model: name, description, JSON schema of inputs.

The input schema is generated from the same pydantic models that validate the arguments
in ``tools.py``. There is one source of truth: if a field or a limit changes there, the
model sees the change automatically.

``ToolSpec`` is provider-neutral; each provider converts it to its own format.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from reviewlens.agent.tools import (
    MAX_SEARCH_RESULTS,
    AnalyzeReviewInput,
    ExperimentResultsInput,
    SearchReviewsInput,
)
from reviewlens.models.registry import available_models


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON schema of the input object


def _schema(model: type[Any]) -> dict[str, Any]:
    """JSON schema of a pydantic model, without the "title" fields the model does not need."""
    schema: dict[str, Any] = model.model_json_schema()
    schema.pop("title", None)
    for field in schema.get("properties", {}).values():
        field.pop("title", None)
    return schema


def tool_specs() -> list[ToolSpec]:
    """The three tools, in a fixed order (a stable order also keeps prompt caching effective)."""
    return [
        ToolSpec(
            name="analyze_review",
            description=(
                "Classify the sentiment of ONE customer review with the project's trained "
                "model. Returns the predicted sentiment (negative/positive), the confidence, "
                "and the words that contributed most to the decision. Use it when the user "
                "gives a review text and asks what its sentiment is or why."
            ),
            parameters=_schema(AnalyzeReviewInput),
        ),
        ToolSpec(
            name="get_experiment_results",
            description=(
                "Read the saved experiment results: macro-F1 (mean ± std over 5 seeds) of each "
                "method at each labeled fraction (1%, 5%, 10%, 20%, 100%). Supervised methods "
                "train on the labeled part only; semi-supervised methods also use unlabeled "
                "data. Give a method and/or a fraction to filter; give neither to get the best "
                f"method per fraction. Methods: {', '.join(available_models())}."
            ),
            parameters=_schema(ExperimentResultsInput),
        ),
        ToolSpec(
            name="search_reviews",
            description=(
                "Keyword search in the Snappfood review dataset (Persian food-delivery "
                f"reviews). Returns at most {MAX_SEARCH_RESULTS} matching reviews with their "
                "true labels, plus the total number of matches. Optionally filter by "
                "sentiment. Use Persian keywords, e.g. 'پیک' (courier) or 'پیتزا' (pizza)."
            ),
            parameters=_schema(SearchReviewsInput),
        ),
    ]
