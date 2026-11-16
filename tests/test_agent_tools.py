"""Agent tools: plain functions, tested without any language model."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from reviewlens.agent.tools import MAX_REVIEW_CHARS, ReviewTools
from reviewlens.config import AppConfig


@pytest.fixture
def tools(agent_config: AppConfig) -> ReviewTools:
    return ReviewTools(agent_config)


def error_type(output: dict[str, Any]) -> str | None:
    error = output.get("error")
    return error["type"] if isinstance(error, dict) else None


# --------------------------------------------------------------------------- analyze_review


def test_analyze_review_predicts_sentiment_with_explanation(tools: ReviewTools) -> None:
    result = tools.run_tool("analyze_review", {"text": "غذا عالی و خوشمزه بود"})

    assert not result.is_error
    assert result.output["sentiment"] == "positive"
    assert 0.5 <= result.output["confidence"] <= 1.0
    assert sum(result.output["probabilities"].values()) == pytest.approx(1.0, abs=0.01)
    words = {item["word"] for item in result.output["top_words"]}
    assert {"عالی", "خوشمزه"} <= words


def test_analyze_review_top_words_support_the_prediction(tools: ReviewTools) -> None:
    result = tools.run_tool("analyze_review", {"text": "پیک دیر رسید و غذا سرد بود"})
    top = result.output["top_words"]

    assert result.output["sentiment"] == "negative"
    # sorted by size of contribution; the sentiment words push towards "negative" (> 0)
    sizes = [abs(item["contribution"]) for item in top]
    assert sizes == sorted(sizes, reverse=True)
    by_word = {item["word"]: item["contribution"] for item in top}
    assert by_word["سرد"] > 0 and by_word["دیر"] > 0


def test_analyze_review_unknown_words_give_no_top_words(tools: ReviewTools) -> None:
    result = tools.run_tool("analyze_review", {"text": "xyz qqq"})
    assert not result.is_error
    assert result.output["top_words"] == []


@pytest.mark.parametrize(
    "arguments",
    [
        {"text": ""},
        {"text": "    "},  # only spaces: stripped, then empty
        {"text": "x" * (MAX_REVIEW_CHARS + 1)},
        {"text": 42},
        {},
        {"text": "خوب", "extra": 1},  # unknown argument
    ],
)
def test_analyze_review_rejects_invalid_input(tools: ReviewTools, arguments: Any) -> None:
    result = tools.run_tool("analyze_review", arguments)
    assert result.is_error
    assert error_type(result.output) == "invalid_input"


def test_analyze_review_without_trained_model(toy_config: AppConfig) -> None:
    result = ReviewTools(toy_config).run_tool("analyze_review", {"text": "خوب بود"})
    assert error_type(result.output) == "unavailable"


# --------------------------------------------------------------------------- results


def test_results_without_filters_give_best_per_fraction(tools: ReviewTools) -> None:
    output = tools.run_tool("get_experiment_results", {}).output

    best = {row["label_fraction"]: row for row in output["best_per_fraction"]}
    assert best[0.01]["best_method"]["method"] == "logistic_regression"
    assert best[0.01]["best_semi_supervised"]["method"] == "pseudo_labeling"
    assert output["available_fractions"] == [0.01, 1.0]


def test_results_for_one_fraction_are_ranked(tools: ReviewTools) -> None:
    output = tools.run_tool("get_experiment_results", {"label_fraction": 0.01}).output

    methods = [row["method"] for row in output["results"]]
    assert methods == ["logistic_regression", "pseudo_labeling"]
    assert output["results"][1]["kind"] == "semi-supervised"


def test_results_for_one_method_and_fraction(tools: ReviewTools) -> None:
    arguments = {"method": "pseudo_labeling", "label_fraction": 1.0}
    rows = tools.run_tool("get_experiment_results", arguments).output["results"]

    assert len(rows) == 1
    assert rows[0]["macro_f1_mean"] == pytest.approx(0.90)


@pytest.mark.parametrize(
    ("arguments", "kind"),
    [
        ({"method": "magic"}, "not_found"),
        ({"label_fraction": 0.5}, "not_found"),  # valid number, but never run
        ({"label_fraction": 10}, "invalid_input"),  # "10 %" written as 10
        ({"label_fraction": 0}, "invalid_input"),
    ],
)
def test_results_errors(tools: ReviewTools, arguments: dict[str, Any], kind: str) -> None:
    output = tools.run_tool("get_experiment_results", arguments).output
    assert error_type(output) == kind


def test_results_missing_file(toy_config: AppConfig) -> None:
    output = ReviewTools(toy_config).run_tool("get_experiment_results", {}).output
    assert error_type(output) == "unavailable"


# --------------------------------------------------------------------------- search_reviews


def with_reviews(tools: ReviewTools, rows: list[tuple[str, str]]) -> ReviewTools:
    """Replace the dataset (a cached property) with a few hand-made reviews."""
    tools.__dict__["reviews"] = pd.DataFrame(rows, columns=["text", "sentiment"])
    return tools


def test_search_matches_all_words_and_filters_sentiment(tools: ReviewTools) -> None:
    with_reviews(
        tools,
        [
            ("پیک دیر رسید", "negative"),
            ("پیک سریع رسید", "positive"),
            ("غذا دیر رسید", "negative"),
        ],
    )
    output = tools.run_tool("search_reviews", {"query": "پیک دیر"}).output
    assert [r["text"] for r in output["reviews"]] == ["پیک دیر رسید"]

    output = tools.run_tool("search_reviews", {"query": "پیک", "sentiment": "positive"}).output
    assert output["total_matches"] == 1
    assert output["reviews"][0]["sentiment"] == "positive"


def test_search_matches_word_starts_only(tools: ReviewTools) -> None:
    with_reviews(tools, [("مدیریت رستوران ضعیف است", "negative"), ("پیک‌ها دیر آمدند", "negative")])

    assert tools.run_tool("search_reviews", {"query": "دیر"}).output["total_matches"] == 1
    assert tools.run_tool("search_reviews", {"query": "پیک"}).output["total_matches"] == 1


def test_search_treats_query_as_plain_text(tools: ReviewTools) -> None:
    with_reviews(tools, [("غذا خوب بود", "positive")])
    # As a regular expression "خو*" would match "خوب"; as plain text it matches nothing.
    output = tools.run_tool("search_reviews", {"query": "خو*"}).output
    assert output["total_matches"] == 0


def test_search_respects_limit_and_reports_total(tools: ReviewTools) -> None:
    with_reviews(tools, [(f"غذا شماره {i}", "positive") for i in range(8)])
    output = tools.run_tool("search_reviews", {"query": "غذا", "limit": 2}).output

    assert output["total_matches"] == 8
    assert output["returned"] == 2


def test_search_shortens_long_reviews(tools: ReviewTools) -> None:
    with_reviews(tools, [("غذا " + "خیلی " * 200, "positive")])
    text = tools.run_tool("search_reviews", {"query": "غذا"}).output["reviews"][0]["text"]
    assert len(text) <= 301 and text.endswith("…")


def test_search_nothing_found(tools: ReviewTools) -> None:
    output = tools.run_tool("search_reviews", {"query": "تعمیر خودرو"}).output
    assert output["total_matches"] == 0
    assert output["reviews"] == []


@pytest.mark.parametrize(
    ("arguments", "kind"),
    [
        ({"query": "پیک", "limit": 6}, "invalid_input"),
        ({"query": "پیک", "limit": 0}, "invalid_input"),
        ({"query": ""}, "invalid_input"),
        ({"query": "پیک", "sentiment": "angry"}, "invalid_input"),
        ({"query": "!!!"}, "invalid_input"),  # no searchable word after normalization
    ],
)
def test_search_errors(tools: ReviewTools, arguments: dict[str, Any], kind: str) -> None:
    assert error_type(tools.run_tool("search_reviews", arguments).output) == kind


def test_search_on_real_toy_dataset(tools: ReviewTools) -> None:
    output = tools.run_tool("search_reviews", {"query": "عالی", "limit": 5}).output
    assert output["returned"] == 5
    assert all(r["sentiment"] == "positive" for r in output["reviews"])


# --------------------------------------------------------------------------- dispatcher


def test_unknown_tool_is_an_error_not_an_exception(tools: ReviewTools) -> None:
    result = tools.run_tool("send_email", {"to": "boss"})
    assert result.is_error
    assert error_type(result.output) == "unknown_tool"
    assert "analyze_review" in result.output["error"]["message"]


def test_non_object_arguments_are_rejected(tools: ReviewTools) -> None:
    result = tools.run_tool("search_reviews", '{"query": broken json')
    assert error_type(result.output) == "invalid_input"


def test_unexpected_tool_failure_does_not_raise(
    tools: ReviewTools, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(params: Any) -> dict[str, Any]:
        raise RuntimeError("bug")

    monkeypatch.setitem(
        tools._registry, "search_reviews", (tools.input_model("search_reviews"), broken)
    )
    result = tools.run_tool("search_reviews", {"query": "پیک"})

    assert error_type(result.output) == "internal_error"
    assert "bug" not in result.output["error"]["message"]  # internals are not leaked
