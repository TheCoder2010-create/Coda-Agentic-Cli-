"""
Main CLI interface for Coda AI coding assistant
"""

import os
import sys
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.prompt import Confirm

from coda.core.config import CodaConfig
from coda.adapters.llm_adapter import LLMAdapter
from coda.storage.embeddings import EmbeddingManager
from coda.core.planner import StructuredPlanner
from coda.core.patcher import PatchManager

console = Console()

@click.group()
@click.version_option(version='0.1.0')
@click.pass_context
def main(ctx):
    """Coda - A local-first terminal-based AI coding assistant"""
    ctx.ensure_object(dict)
    
    # Initialize configuration
    try:
        ctx.obj['config'] = CodaConfig()
    except Exception as e:
        console.print(f"[red]Error loading configuration: {e}[/red]")
        sys.exit(1)

@main.command()
@click.option('--path', default='.', help='Path to index (default: current directory)')
@click.option('--force', is_flag=True, help='Force re-indexing of all files')
@click.option('--exclude', multiple=True, help='Additional patterns to exclude')
@click.pass_context
def index(ctx, path, force, exclude):
    """Index project files into embeddings for repo-aware queries"""
    console.print(Panel.fit("🔍 Indexing Project Files", style="bold blue"))
    
    config = ctx.obj['config']
    
    # Initialize embedding manager
    embedding_manager = EmbeddingManager(config)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Scanning files...", total=None)
        
        try:
            # Index the repository
            stats = embedding_manager.index_repository(path, force=force, exclude_patterns=exclude)
            
            progress.update(task, description="Indexing complete!")
            
            # Display statistics
            table = Table(title="Indexing Results")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            
            table.add_row("Files processed", str(stats.get('files_processed', 0)))
            table.add_row("Chunks created", str(stats.get('chunks_created', 0)))
            table.add_row("Embeddings stored", str(stats.get('embeddings_stored', 0)))
            table.add_row("Total size", f"{stats.get('total_size_mb', 0):.2f} MB")
            
            console.print(table)
            console.print("[green]✓ Repository indexing completed successfully![/green]")
            
        except Exception as e:
            console.print(f"[red]Error during indexing: {e}[/red]")
            sys.exit(1)

@main.command()
@click.argument('question', required=True)
@click.option('--provider', default=None, help='LLM provider (openai, anthropic, ollama)')
@click.option('--model', default=None, help='Specific model to use')
@click.option('--context-files', multiple=True, help='Specific files to include in context')
@click.option('--max-context', default=8000, help='Maximum context tokens')
@click.pass_context
def ask(ctx, question, provider, model, context_files, max_context):
    """Ask repository-aware questions"""
    console.print(Panel.fit(f"💭 Asking: {question}", style="bold green"))
    
    config = ctx.obj['config']
    
    # Initialize components
    llm_adapter = LLMAdapter()
    embedding_manager = EmbeddingManager(config)
    
    # Get available providers
    available_providers = llm_adapter.get_available_providers()
    if not available_providers:
        console.print("[red]No LLM providers available. Please set API keys.[/red]")
        console.print("Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variables.")
        sys.exit(1)
    
    # Use default provider if not specified
    if not provider:
        provider = config.get('llm.default_provider', available_providers[0])
    
    if provider not in available_providers:
        console.print(f"[red]Provider {provider} not available. Available: {', '.join(available_providers)}[/red]")
        sys.exit(1)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Gathering context...", total=None)
        
        try:
            # Get relevant context
            if context_files:
                context = embedding_manager.get_file_context(list(context_files))
            else:
                context = embedding_manager.search_relevant_context(question, max_tokens=max_context)
            
            progress.update(task, description="Querying LLM...")
            
            # Prepare messages
            system_prompt = f"""You are Coda, an AI coding assistant with access to the current repository context.
Use the following repository context to answer questions accurately and helpfully.

Repository Context:
{context}

Provide detailed, accurate answers based on the actual code and files in this repository.
If you need to reference specific files or code, cite them directly.
"""
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ]
            
            # Query LLM
            response = llm_adapter.chat(messages, provider=provider, model=model)
            
            progress.update(task, description="Response received!")
            
            # Display response
            console.print(Panel(response.content, title="🤖 Response", style="blue"))
            
            # Display token usage
            usage_table = Table(title="Token Usage & Cost")
            usage_table.add_column("Metric", style="cyan")
            usage_table.add_column("Value", style="yellow")
            
            usage_table.add_row("Provider", response.provider)
            usage_table.add_row("Model", response.model)
            usage_table.add_row("Input tokens", str(response.token_usage.prompt_tokens))
            usage_table.add_row("Output tokens", str(response.token_usage.completion_tokens))
            usage_table.add_row("Total tokens", str(response.token_usage.total_tokens))
            usage_table.add_row("Estimated cost", f"${response.token_usage.estimated_cost:.4f}")
            
            console.print(usage_table)
            
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            sys.exit(1)

@main.command()
@click.argument('task_description', required=True)
@click.option('--provider', default=None, help='LLM provider')
@click.option('--model', default=None, help='Specific model to use')
@click.option('--safe', is_flag=True, help='Safe mode: only preview, don\'t apply')
@click.option('--apply', is_flag=True, help='Apply patches directly (use with caution)')
@click.pass_context
def patch(ctx, task_description, provider, model, safe, apply):
    """Generate and preview code patches"""
    console.print(Panel.fit(f"🔧 Generating patch for: {task_description}", style="bold yellow"))
    
    config = ctx.obj['config']
    
    # Initialize components
    llm_adapter = LLMAdapter()
    embedding_manager = EmbeddingManager(config)
    planner = StructuredPlanner(llm_adapter, embedding_manager)
    patcher = PatchManager(config)
    
    # Get available providers
    available_providers = llm_adapter.get_available_providers()
    if not available_providers:
        console.print("[red]No LLM providers available. Please set API keys.[/red]")
        sys.exit(1)
    
    if not provider:
        provider = config.get('llm.default_provider', available_providers[0])
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        # Step 1: Create structured plan
        task1 = progress.add_task("Creating structured plan...", total=None)
        plan = planner.create_plan(task_description, provider=provider, model=model)
        
        console.print(Panel(plan.formatted_plan, title="📋 Execution Plan", style="green"))
        
        progress.update(task1, description="Plan created!")
        
        # Step 2: Generate patches
        task2 = progress.add_task("Generating patches...", total=None)
        patches = patcher.generate_patches(task_description, plan, provider=provider, model=model)
        
        progress.update(task2, description="Patches generated!")
        
        # Display patches
        for i, patch in enumerate(patches, 1):
            console.print(f"\n[bold]Patch {i}: {patch.filename}[/bold]")
            console.print(Syntax(patch.diff, "diff", theme="monokai", line_numbers=True))
        
        # Decide whether to apply
        should_apply = apply
        if not safe and not apply:
            # Ask user
            should_apply = Confirm.ask("Apply these patches?")
        elif safe:
            console.print("[yellow]Safe mode: patches not applied[/yellow]")
            return
        
        if should_apply:
            task3 = progress.add_task("Applying patches...", total=None)
            
            try:
                results = patcher.apply_patches(patches)
                
                success_count = sum(1 for r in results if r.success)
                total_count = len(results)
                
                if success_count == total_count:
                    console.print(f"[green]✓ Successfully applied {success_count} patches![/green]")
                else:
                    console.print(f"[yellow]Applied {success_count}/{total_count} patches[/yellow]")
                    for result in results:
                        if not result.success:
                            console.print(f"[red]Failed to apply {result.filename}: {result.error}[/red]")
                
                progress.update(task3, description="Patches applied!")
                
            except Exception as e:
                console.print(f"[red]Error applying patches: {e}[/red]")

@main.command()
@click.argument('task_description', required=True)
@click.option('--provider', default=None, help='LLM provider')
@click.option('--model', default=None, help='Specific model to use')
@click.option('--branch', default=None, help='Branch name (auto-generated if not provided)')
@click.option('--draft', is_flag=True, help='Create as draft PR')
@click.pass_context
def pr(ctx, task_description, provider, model, branch, draft):
    """Create branch and draft pull request"""
    console.print(Panel.fit(f"🔀 Creating PR for: {task_description}", style="bold magenta"))
    
    # This is a simplified version - full git integration would be more complex
    console.print("[yellow]PR creation feature coming soon![/yellow]")
    console.print("For now, use 'coda patch' and manually create PRs.")

@main.command()
@click.pass_context
def status(ctx):
    """Show Coda status and configuration"""
    config = ctx.obj['config']
    
    console.print(Panel.fit("📊 Coda Status", style="bold cyan"))
    
    # Check LLM providers
    llm_adapter = LLMAdapter()
    providers = llm_adapter.get_available_providers()
    
    status_table = Table(title="System Status")
    status_table.add_column("Component", style="cyan")
    status_table.add_column("Status", style="green")
    status_table.add_column("Details", style="blue")
    
    status_table.add_row("Configuration", "✓ Loaded", config.config_path)
    status_table.add_row("Storage", "✓ Ready", config.coda_dir)
    
    if providers:
        status_table.add_row("LLM Providers", "✓ Available", ", ".join(providers))
    else:
        status_table.add_row("LLM Providers", "⚠ None", "Set API keys")
    
    # Check embeddings
    embedding_dir = config.get('storage.embeddings_dir')
    if os.path.exists(embedding_dir) and os.listdir(embedding_dir):
        status_table.add_row("Embeddings", "✓ Indexed", f"Files in {embedding_dir}")
    else:
        status_table.add_row("Embeddings", "⚠ Empty", "Run 'coda index' first")
    
    console.print(status_table)

if __name__ == '__main__':
    main()