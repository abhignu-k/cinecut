"""Command-line interface for cinecut."""

from __future__ import annotations

from typing import Annotated

import typer

from cinecut import __version__

app = typer.Typer(
    name="cinecut",
    help="cinecut: Context-aware AI cinematic vertical edit engine.",
    no_args_is_help=True,
    add_completion=False,
)


def _version_callback(value: bool) -> None:
    """Print the package version and exit when --version is passed."""
    if value:
        typer.echo(f"cinecut {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show the cinecut version and exit.",
        ),
    ] = False,
) -> None:
    """Top-level CLI callback for cinecut."""
    _ = version
