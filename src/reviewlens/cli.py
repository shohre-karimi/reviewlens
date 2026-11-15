"""Command-line entry point: ``reviewlens [OPTIONS] COMMAND``.

The top-level callback loads and validates the config once and stores it on the typer
context, so every command receives the same ``AppConfig`` through ``ctx.obj``.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
import yaml
from pydantic import ValidationError

from reviewlens import __version__
from reviewlens.config import DEFAULT_CONFIG_PATH, AppConfig, load_config
from reviewlens.data.download import DownloadError, download_raw_files
from reviewlens.data.prepare import load_prepared, prepare_dataset, save_prepared
from reviewlens.experiments import runner
from reviewlens.experiments.train import build_experiment_features, train_and_evaluate
from reviewlens.logging_utils import setup_logging
from reviewlens.models.registry import available_models, get_spec
from reviewlens.models.text_classifier import TextClassifier, save_text_classifier

logger = logging.getLogger(__name__)

app = typer.Typer(
    help="ReviewLens: Persian review sentiment, semi-supervised learning and an AI agent.",
    no_args_is_help=True,
    add_completion=False,
)
data_app = typer.Typer(help="Dataset commands.", no_args_is_help=True)
agent_app = typer.Typer(help="Agent commands.", no_args_is_help=True)
app.add_typer(data_app, name="data")
app.add_typer(agent_app, name="agent")


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"reviewlens {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    ctx: typer.Context,
    config: Annotated[
        Path, typer.Option("--config", "-c", help="Main YAML config file.")
    ] = DEFAULT_CONFIG_PATH,
    experiment: Annotated[
        Path | None,
        typer.Option("--experiment", "-e", help="Experiment YAML merged over the main config."),
    ] = None,
    log_level: Annotated[
        str | None, typer.Option("--log-level", help="Override the configured log level.")
    ] = None,
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Show version."),
    ] = False,
) -> None:
    """Load the configuration and set up logging before any command runs."""
    overrides = [experiment] if experiment is not None else []
    try:
        cfg = load_config(config, overrides)
    except (FileNotFoundError, ValueError, ValidationError) as error:
        # ValidationError is a ValueError subclass; listed for readability.
        typer.echo(f"Invalid configuration: {error}", err=True)
        raise typer.Exit(code=2) from error

    setup_logging(log_level or cfg.logging.level)
    logger.debug("Config loaded path=%s experiment=%s", config, cfg.experiment.name)
    ctx.obj = cfg


def _not_implemented(command: str, phase: int) -> None:
    typer.echo(f"'{command}' is not implemented yet (planned for phase {phase}).", err=True)
    raise typer.Exit(code=1)


@app.command("show-config")
def show_config(ctx: typer.Context) -> None:
    """Print the fully resolved configuration as YAML."""
    cfg: AppConfig = ctx.obj
    typer.echo(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False, allow_unicode=True))


@data_app.command("download")
def data_download(
    ctx: typer.Context,
    force: Annotated[bool, typer.Option("--force", help="Re-download existing files.")] = False,
) -> None:
    """Download the raw dataset files into paths.raw_dir."""
    cfg: AppConfig = ctx.obj
    try:
        paths = download_raw_files(cfg.data, cfg.paths.raw_dir, force=force)
    except DownloadError as error:
        typer.echo(f"Download failed: {error}", err=True)
        raise typer.Exit(code=1) from error
    for path in paths:
        typer.echo(f"{path}  ({path.stat().st_size:,} bytes)")


@data_app.command("prepare")
def data_prepare(ctx: typer.Context) -> None:
    """Clean and validate the raw data and save train/test to paths.processed_dir."""
    cfg: AppConfig = ctx.obj
    try:
        prepared = prepare_dataset(cfg)
    except (FileNotFoundError, ValueError) as error:  # DataValidationError is a ValueError
        typer.echo(f"Preparation failed: {error}", err=True)
        raise typer.Exit(code=1) from error
    train_path, test_path = save_prepared(prepared, cfg.paths.processed_dir)

    for name, report in (("train", prepared.train_report), ("test", prepared.test_report)):
        typer.echo(f"[{name}]")
        for key, value in asdict(report).items():
            typer.echo(f"  {key:16} {value}")
    typer.echo(f"Saved {train_path} and {test_path}")


FractionOption = Annotated[
    float, typer.Option("--fraction", "-f", min=0.0, max=1.0, help="Labeled fraction (0, 1].")
]
SeedOption = Annotated[int, typer.Option("--seed", "-s", help="Seed for label masking and model.")]


def _load_data_or_exit(cfg: AppConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        return load_prepared(cfg)
    except FileNotFoundError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error


@app.command("train")
def train(
    ctx: typer.Context,
    method: Annotated[str, typer.Option("--method", "-m", help="Model name from the registry.")],
    fraction: FractionOption = 1.0,
    seed: SeedOption = 0,
    save: Annotated[
        bool, typer.Option("--save/--no-save", help="Save the model to paths.models_dir.")
    ] = False,
) -> None:
    """Train one model on a labeled fraction and evaluate it on the test set."""
    cfg: AppConfig = ctx.obj
    train_df, test_df = _load_data_or_exit(cfg)
    text, label = cfg.task.text_column, cfg.task.label_column
    try:
        features = build_experiment_features(train_df[text], test_df[text], [method], cfg)
        model, result = train_and_evaluate(
            method, features, train_df[label], test_df[label], fraction, seed, cfg
        )
    except ValueError as error:  # unknown method, bad hyperparameters, invalid fraction
        typer.echo(f"Training failed: {error}", err=True)
        raise typer.Exit(code=1) from error

    for key, value in result.as_row().items():
        typer.echo(f"{key:20} {value:.4f}" if isinstance(value, float) else f"{key:20} {value}")
    typer.echo(f"{'confusion':20} {result.confusion}  (rows = true {cfg.task.labels})")

    if save:
        classifier = TextClassifier(
            method=method,
            labels=list(cfg.task.labels),
            vectorizer=features.vectorizer,
            model=model,
            lsa=features.lsa if get_spec(method).features == "lsa" else None,
            metadata=result.as_row(),
        )
        path = save_text_classifier(classifier, cfg.paths.models_dir / f"{method}.joblib")
        typer.echo(f"Saved model to {path}")


@app.command("baselines")
def baselines(
    ctx: typer.Context,
    fraction: FractionOption = 1.0,
    seed: SeedOption = 0,
    method: Annotated[
        list[str] | None,
        typer.Option("--method", "-m", help="Only these methods (repeatable). Default: all."),
    ] = None,
) -> None:
    """Train the supervised baselines on one labeled fraction and compare them."""
    cfg: AppConfig = ctx.obj
    train_df, test_df = _load_data_or_exit(cfg)
    text, label = cfg.task.text_column, cfg.task.label_column
    supervised = [name for name in available_models() if not get_spec(name).semi_supervised]
    methods = method or supervised
    try:
        features = build_experiment_features(train_df[text], test_df[text], methods, cfg)
        rows = [
            train_and_evaluate(
                method, features, train_df[label], test_df[label], fraction, seed, cfg
            )[1].as_row()
            for method in methods
        ]
    except ValueError as error:
        typer.echo(f"Training failed: {error}", err=True)
        raise typer.Exit(code=1) from error

    table = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    columns = ["method", "n_labeled", "macro_f1", "accuracy", "fit_seconds"]
    typer.echo(table[columns].to_string(index=False, float_format="{:.4f}".format))

    cfg.paths.results_dir.mkdir(parents=True, exist_ok=True)
    out = cfg.paths.results_dir / f"baselines_{cfg.features.analyzer}_f{fraction}_s{seed}.csv"
    table.to_csv(out, index=False)
    typer.echo(f"Saved {out}")


@app.command("run-experiments")
def run_experiments(
    ctx: typer.Context,
    resume: Annotated[
        bool,
        typer.Option("--resume/--fresh", help="Skip runs already saved, or start over."),
    ] = True,
) -> None:
    """Run every method x label fraction x seed and save raw results."""
    cfg: AppConfig = ctx.obj
    train_df, test_df = _load_data_or_exit(cfg)
    try:
        output = runner.run_experiments(cfg, train_df, test_df, resume=resume)
    except ValueError as error:
        typer.echo(f"Experiment failed: {error}", err=True)
        raise typer.Exit(code=1) from error

    typer.echo("macro-F1 (mean ± std over seeds); columns = labeled fraction")
    typer.echo(runner.format_summary_table(output.summary).to_string())
    typer.echo(f"Results saved in {output.output_dir}")


@agent_app.command("local-test")
def agent_local_test(ctx: typer.Context) -> None:
    """Test all agent tools without any model or API key."""
    _not_implemented("agent local-test", phase=6)


@agent_app.command("eval")
def agent_eval(ctx: typer.Context) -> None:
    """Evaluate the agent's tool selection and write docs/agent_tests.md."""
    _not_implemented("agent eval", phase=6)
