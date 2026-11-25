from typer.testing import CliRunner

import cinecut
from cinecut.cli import app

runner = CliRunner()


def test_package_version() -> None:
    assert cinecut.__version__ == "0.1.0"


def test_cli_help_shows_cinecut() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "cinecut" in result.output.lower()


def test_cli_version_flag() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output
