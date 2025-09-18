# Coda - AI Coding Assistant

A local-first terminal-based AI coding assistant that addresses the limitations of existing tools like Claude Code and OpenAI Codex.

## Features

- **Local-first privacy**: All embeddings and logs stored locally in `.coda/` directory
- **Multi-model support**: OpenAI, Anthropic, and Ollama integration
- **Scoped automation**: Interactive patch preview with `--safe` mode
- **Cost control**: Token usage tracking and context optimization
- **Repository awareness**: Vector embeddings for intelligent code understanding
- **Structured planning**: Step-by-step execution plans for code changes

## Installation

```bash
pip install -e .
```

## Setup

1. Set your API keys:
```bash
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"
```

2. Initialize your project:
```bash
coda index
```

## Usage

### Index your repository
```bash
coda index                    # Index current directory
coda index --path /my/repo    # Index specific path
coda index --force            # Force re-indexing
```

### Ask repository-aware questions
```bash
coda ask "How does authentication work in this codebase?"
coda ask "What are the main API endpoints?" --provider anthropic
```

### Generate and apply code patches
```bash
coda patch "Add input validation to the login form"
coda patch "Fix the memory leak in data processing" --safe  # Preview only
coda patch "Optimize database queries" --apply             # Apply directly
```

### Check system status
```bash
coda status
```

## Configuration

Coda creates a `.coda/config.yaml` file with default settings. You can customize:

- Default LLM provider and models
- Embedding chunk size and overlap
- Token limits and safety settings
- Plugin configuration

## Local Storage

- `.coda/embeddings/` - Vector embeddings of your code
- `.coda/logs/` - Query and operation logs
- `.coda/cache/` - Cached responses and computations
- `.coda/backups/` - File backups before applying patches

## Examples

Query your codebase:
```bash
coda ask "Show me all the database models and their relationships"
```

Make targeted improvements:
```bash
coda patch "Add error handling to the payment processing function"
```

Preview changes safely:
```bash
coda patch "Refactor the user authentication system" --safe
```

