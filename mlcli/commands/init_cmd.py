"""Initialize command for ML Assistant CLI."""

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.prompt import Confirm, Prompt

from mlcli.core.config import MLCLIConfig, save_config
from mlcli.core.exceptions import ConfigurationError
from mlcli.utils.logging import get_logger

logger = get_logger(__name__)
console = Console()

app = typer.Typer()


def is_interactive() -> bool:
    """Check if running in interactive mode."""
    return sys.stdin.isatty()


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Project name"),
    description: Optional[str] = typer.Option(None, "--description", "-d", help="Project description"),
    plugin: str = typer.Option("tabular", "--plugin", "-p", help="Project template (tabular, chatbot, image-classification)"),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing configuration"),
) -> None:
    """Initialize a new ML project with configuration and directory structure."""
    
    project_dir = ctx.obj["project_dir"]
    config_file = project_dir / "mlcli.yaml"
    
    if config_file.exists() and not force:
        if is_interactive():
            if not Confirm.ask(f"Project configuration already exists at {config_file}. Overwrite?"):
                console.print("[yellow]Initialization cancelled.[/yellow]")
                raise typer.Exit(0)
        else:
            console.print("[yellow]Configuration exists, use --force to overwrite[/yellow]")
            raise typer.Exit(0)
    
    from mlcli.plugins.registry import PluginRegistry
    
    registry = PluginRegistry()
    
    try:
        selected_plugin = registry.get(plugin)
        console.print(f"[cyan]Using plugin:[/cyan] {selected_plugin.description}")
    except Exception as e:
        console.print(f"[red]Error loading plugin '{plugin}':[/red] {e}")
        available = registry.list_available()
        if available:
            console.print(f"[yellow]Available plugins:[/yellow] {', '.join(available)}")
        raise typer.Exit(1)
    
    project_name = name or project_dir.name
    
    if not name and is_interactive():
        project_name = Prompt.ask(
            "Project name", 
            default=project_dir.name,
            show_default=True
        )
    
    if not description and is_interactive():
        description = Prompt.ask(
            "Project description (optional)", 
            default="",
            show_default=False
        )
    else:
        description = description or ""
    
    try:
        selected_plugin.create_project(project_dir, project_name)
        
        config_dict = {
            "project_name": project_name,
            "description": description or None,
            "version": "0.1.0",
        }
        config_dict.update(selected_plugin.get_config_template())
        
        config = MLCLIConfig(**config_dict)
        save_config(config, config_file)
        
        console.print(f"[green]✓[/green] Initialized [bold]{plugin}[/bold] project: [bold]{project_name}[/bold]")
        console.print(f"[green]✓[/green] Configuration saved to: {config_file}")
        console.print(f"[green]✓[/green] Project structure created in: {project_dir}")
        
        _print_next_steps(plugin)
        
    except Exception as e:
        raise ConfigurationError(f"Failed to initialize project: {e}")


def _print_next_steps(plugin: str) -> None:
    """Print next steps based on plugin type."""
    console.print("\n[bold]Next steps:[/bold]")
    
    if plugin == "chatbot":
        console.print("1. Copy [cyan].env.example[/cyan] to [cyan].env[/cyan] and add your OPENAI_API_KEY")
        console.print("2. Add documents to [cyan]data/knowledge_base/[/cyan]")
        console.print("3. Run [cyan]python src/app.py[/cyan] to start the chatbot")
    elif plugin == "image-classification":
        console.print("1. Add your images to [cyan]data/raw/images/[/cyan]")
        console.print("2. Update [cyan]mlcli.yaml[/cyan] with your class labels")
        console.print("3. Run [cyan]mlcli train[/cyan] to start training")
    else:
        console.print("1. Add your dataset to [cyan]data/raw/[/cyan]")
        console.print("2. Run [cyan]mlcli preprocess --input data/raw/your_data.csv[/cyan]")
        console.print("3. Run [cyan]mlcli train[/cyan] to start training models")
