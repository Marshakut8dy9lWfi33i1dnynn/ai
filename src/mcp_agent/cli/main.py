import typer
from mnemo_agent.cli.terminal import Application
from mnemo_agent.cli.commands import config

app = typer.Typer()

# Subcommands
app.add_typer(config.app, name="config")

# Shared application context
application = Application()


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose mode"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Disable output"),
    color: bool = typer.Option(
        True, "--color/--no-color", help="Enable/disable color output"
    ),
):
    """Main entry point for the MCP Agent CLI."""
    application.verbosity = 1 if verbose else 0 if not quiet else -1
    application.console = application.console if color else None
