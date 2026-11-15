"""Command-line entry point: ``reviewlens [OPTIONS] COMMAND``.

The top-level callback loads and validates the config once and stores it on the typer
context, so every command receives the same ``AppConfig`` through ``ctx.obj``.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import typer
import yaml
from pydantic import ValidationError

from reviewlens import __version__
from reviewlens.config import DEFAULT_CONFIG_PATH, AppConfig, load_config
from reviewlens.data.download import DownloadError, download_raw_files
from reviewlens.data.prepare import prepare_dataset, save_prepared
from reviewlens.logging_utils import setup_logging

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


@app.command("run-experiments")
def run_experiments(ctx: typer.Context) -> None:
    """Run every method x label fraction x seed and save raw results."""
    _not_implemented("run-experiments", phase=4)


@agent_app.command("local-test")
def agent_local_test(ctx: typer.Context) -> None:
    """Test all agent tools without any model or API key."""
    _not_implemented("agent local-test", phase=6)


@agent_app.command("eval")
def agent_eval(ctx: typer.Context) -> None:
    """Evaluate the agent's tool selection and write docs/agent_tests.md."""
    _not_implemented("agent eval", phase=6)
