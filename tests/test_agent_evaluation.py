"""Agent evaluation logic and the local tool checks, with fake clients only."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from reviewlens.agent.evaluation import (
    EVAL_CASES,
    INVALID_KEY,
    EvalCase,
    judge,
    report_markdown,
    run_evaluation,
    run_local_checks,
    write_report,
)
from reviewlens.agent.loop import AgentResult, Status, TraceEntry
from reviewlens.agent.providers import LLMClient, ProviderError
from reviewlens.agent.tools import ReviewTools
from reviewlens.config import AppConfig
from tests.test_agent_loop import FakeClient, answer, call


@pytest.fixture
def tools(agent_config: AppConfig) -> ReviewTools:
    return ReviewTools(agent_config)


def test_cases_cover_every_required_category() -> None:
    categories = {case.category for case in EVAL_CASES}
    required = {
        "analyze_review",
        "get_experiment_results",
        "search_reviews",
        "two tools",
        "empty message",
        "out-of-range limit",
        "topic not in data",
        "unsupported request",
        "prompt injection",
        "invalid API key",
    }
    assert required <= categories
    assert len(EVAL_CASES) >= 10
    assert len({case.id for case in EVAL_CASES}) == len(EVAL_CASES)


def _result(
    tools_called: list[str],
    answer_text: str = "پاسخ",
    status: Status = "answered",
    error_kind: str | None = None,
) -> AgentResult:
    trace = [TraceEntry(1, name, {}, {}, False) for name in tools_called]
    return AgentResult(answer_text, status, trace=trace, error_kind=error_kind)


CASE = EvalCase("c", "cat", "msg", frozenset({"analyze_review", "get_experiment_results"}))


def test_judge_compares_sets_of_tools() -> None:
    # order and repeats do not matter, missing or extra tools do
    assert judge(CASE, _result(["get_experiment_results", "analyze_review", "analyze_review"]))[0]
    assert not judge(CASE, _result(["analyze_review"]))[0]
    assert not judge(CASE, _result(["analyze_review", "get_experiment_results", "search_reviews"]))[
        0
    ]


def test_judge_fails_runs_that_did_not_end_normally() -> None:
    passed, note = judge(
        CASE, _result(["analyze_review", "get_experiment_results"], status="max_steps")
    )
    assert not passed and "max_steps" in note


def test_judge_detects_obeyed_injection() -> None:
    case = EvalCase(
        "i", "prompt injection", "m", frozenset({"analyze_review"}), forbidden_text="HACKED"
    )
    assert judge(case, _result(["analyze_review"], "این نظر منفی است."))[0]
    assert not judge(case, _result(["analyze_review"], "HACKED"))[0]


def test_judge_invalid_key_needs_a_clean_auth_error() -> None:
    case = EvalCase("k", "invalid API key", "m", frozenset(), invalid_key=True)
    assert judge(case, _result([], status="provider_error", error_kind="auth"))[0]
    assert not judge(case, _result([]))[0]


def test_run_evaluation_with_a_fake_model(tools: ReviewTools) -> None:
    """A fake model that never uses tools passes exactly the no-tool cases."""

    def make_client(api_key: str | None) -> LLMClient:
        if api_key == INVALID_KEY:
            return FakeClient([ProviderError("auth")])
        return FakeClient([answer("نمی‌دانم.")] * 5)

    report = run_evaluation(make_client, tools, "fake", checks_api_keys=True, runs=2)

    passed = {r.case_id for r in report.runs if r.passed}
    assert passed == {"empty_message", "unsupported_email", "invalid_api_key"}
    assert len(report.runs) == 2 * len(EVAL_CASES)
    assert report.accuracy == pytest.approx(3 / len(EVAL_CASES))


def test_invalid_key_case_is_skipped_for_local_servers(tools: ReviewTools) -> None:
    cases = [case for case in EVAL_CASES if case.invalid_key]
    report = run_evaluation(
        lambda key: FakeClient([answer("x")]), tools, "local", checks_api_keys=False, cases=cases
    )
    assert all(r.passed is None for r in report.runs)
    assert report.evaluated == []


def test_a_crashing_run_is_recorded_as_failed(tools: ReviewTools) -> None:
    def make_client(api_key: str | None) -> LLMClient:
        return FakeClient([])  # IndexError on the first model call

    cases = [EVAL_CASES[0]]
    report = run_evaluation(make_client, tools, "fake", checks_api_keys=True, runs=1, cases=cases)
    assert report.runs[0].passed is False
    assert "crash" in report.runs[0].note


def test_correct_tool_use_passes(tools: ReviewTools) -> None:
    case = next(c for c in EVAL_CASES if c.id == "limit_out_of_range")

    def make_client(api_key: str | None) -> LLMClient:
        return FakeClient(
            [
                call(("search_reviews", {"query": "پیتزا", "sentiment": "positive", "limit": 20})),
                call(("search_reviews", {"query": "پیتزا", "sentiment": "positive", "limit": 5})),
                answer("فقط ۵ نظر را می‌توانم نشان دهم."),
            ]
        )

    report = run_evaluation(make_client, tools, "fake", checks_api_keys=True, runs=1, cases=[case])
    assert report.runs[0].passed
    assert "corrected" in report.runs[0].note


def test_report_markdown_and_file(tools: ReviewTools, tmp_path: Path) -> None:
    report = run_evaluation(
        lambda key: FakeClient([answer("x")] * 5), tools, "fake", checks_api_keys=False, runs=1
    )
    text = report_markdown(report, today=date(2026, 1, 2))

    assert "2026-01-02" in text and "`fake-model`" in text
    assert "Tool-selection accuracy" in text
    assert text.count("\n| ") == len(EVAL_CASES) + 1  # header + one row per case
    assert "n/a" in text  # skipped invalid-key case
    path = write_report(report, tmp_path / "docs" / "agent_tests.md")
    assert path.read_text(encoding="utf-8") == report_markdown(report)


def test_local_checks_all_pass_on_the_fixture(tools: ReviewTools) -> None:
    outcomes = run_local_checks(tools)
    failed = [check.description for check, _, passed in outcomes if not passed]
    assert failed == []
